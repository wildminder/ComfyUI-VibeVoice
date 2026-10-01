"""Phase 1 tests: streaming module wrappers (modules/comfy_stream.py).

Parity against the ORIGINAL classes, conversion census/idempotence,
meta-safety, fast-path vs cast-path plumbing.
"""

import logging
import sys
from unittest.mock import patch

import pytest
import torch
from torch import nn

from modules import comfy_stream as CS


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------

def _build(kind, **kwargs):
    if kind == "linear":
        return nn.Linear(8, 6)
    if kind == "embedding":
        return nn.Embedding(16, 6)
    if kind == "conv1d":
        return nn.Conv1d(3, 5, kernel_size=3, padding=1)
    if kind == "convtranspose1d":
        return nn.ConvTranspose1d(3, 5, kernel_size=3, padding=1)
    if kind == "layernorm":
        return nn.LayerNorm(6)
    raise ValueError(kind)


def _input_for(kind):
    if kind in ("conv1d", "convtranspose1d"):
        return torch.randn(2, 3, 10)
    if kind == "embedding":
        return torch.randint(0, 15, (2, 4))
    if kind == "linear":
        return torch.randn(2, 8)
    return torch.randn(2, 6)


# ---------------------------------------------------------------------
# Built-in kinds
# ---------------------------------------------------------------------

class TestBuiltinKinds:
    @pytest.mark.parametrize("kind", ["linear", "embedding", "conv1d",
                                      "convtranspose1d", "layernorm"])
    def test_parity_with_original(self, kind):
        torch.manual_seed(0)
        orig = _build(kind).eval()
        twin = _build(kind)
        twin.load_state_dict(orig.state_dict())

        compute = CS._BUILTIN_COMPUTE[type(orig)]
        twin.__class__ = CS.make_streaming(type(orig), compute)

        x = _input_for(kind)
        with torch.no_grad():
            assert torch.equal(orig(x), twin(x))

    def test_fast_path_skips_cast_bias_weight(self):
        torch.manual_seed(0)
        lin = nn.Linear(8, 4)
        lin.__class__ = CS.make_streaming(nn.Linear, CS._compute_linear)
        x = torch.randn(2, 8)

        import comfy.ops
        with patch.object(comfy.ops, "cast_bias_weight",
                          side_effect=AssertionError("cast on fast path")):
            y = lin(x)
        assert torch.equal(y, torch.nn.functional.linear(
            x, lin.weight, lin.bias))

    def test_streaming_path_uses_cast_when_stripped(self):
        """Simulate core's lowvram strip: weight_function attached -> the
        wrapper must route through cast_bias_weight/uncast_bias_weight."""
        torch.manual_seed(0)
        lin = nn.Linear(8, 4)
        lin.__class__ = CS.make_streaming(nn.Linear, CS._compute_linear)
        lin.weight_function = [object()]  # LowVramPatch stand-in
        x = torch.randn(2, 8)

        calls = {"cast": 0, "uncast": 0}

        def fake_cast(module, inp, offloadable=False, **kw):
            calls["cast"] += 1
            return (module.weight.to(inp.device),
                    module.bias.to(inp.device)
                    if module.bias is not None else None,
                    None)

        def fake_uncast(module, w, b, stream):
            calls["uncast"] += 1

        import comfy.ops
        with patch.object(comfy.ops, "cast_bias_weight", fake_cast), \
             patch.object(comfy.ops, "uncast_bias_weight", fake_uncast):
            y = lin(x)

        assert calls == {"cast": 1, "uncast": 1}
        ref = torch.nn.functional.linear(
            x, lin.weight.to(x.device), lin.bias.to(x.device))
        assert torch.equal(y, ref)


# ---------------------------------------------------------------------
# Vendored / HF norm parity
# ---------------------------------------------------------------------

class TestNormWrappers:
    """Stand-in norm classes with math identical to the vendored ones
    (conftest mocks the real vendored modules for suite speed). The wrapper
    mechanics validated here are exactly what production registers."""

    @pytest.fixture(autouse=True)
    def _register(self):
        CS.register_streaming_type(_TokRMSNorm, CS._compute_rmsnorm)
        CS.register_streaming_type(_ConvRMSNorm, CS._compute_convrmsnorm)
        CS.register_streaming_type(_ConvLayerNorm, CS._compute_convlayernorm)

    def _cases(self):
        return [
            (_TokRMSNorm, dict(dim=8), torch.randn(2, 5, 8)),
            (_ConvRMSNorm, dict(dim=8), torch.randn(2, 8, 5)),
            (_ConvLayerNorm, dict(normalized_shape=8), torch.randn(2, 8, 5)),
        ]

    @pytest.mark.parametrize("idx", range(3))
    def test_norm_parity_after_conversion(self, idx):
        torch.manual_seed(0)
        base, kw, x = self._cases()[idx]
        orig, twin = base(**kw), base(**kw)
        twin.load_state_dict(orig.state_dict())
        twin.__class__ = CS.make_streaming(base, CS._EXTRA_COMPUTE[base])
        with torch.no_grad():
            before, after = orig(x), twin(x)
        assert torch.equal(before, after), base.__name__

    def test_elementwise_affine_false_passthrough(self):
        orig = _TokRMSNorm(8, elementwise_affine=False)
        twin = _TokRMSNorm(8, elementwise_affine=False)
        twin.__class__ = CS.make_streaming(_TokRMSNorm, CS._compute_rmsnorm)
        x = torch.randn(3, 8)
        assert torch.equal(orig(x), twin(x))

    def test_qwen2rmsnorm_parity_real_class(self):
        # transformers is NOT mocked by conftest -> exercise the real HF class.
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

        CS.register_streaming_type(Qwen2RMSNorm, CS._compute_qwen2rmsnorm)
        torch.manual_seed(0)
        orig, twin = Qwen2RMSNorm(8), Qwen2RMSNorm(8)
        twin.load_state_dict(orig.state_dict())
        twin.__class__ = CS.make_streaming(Qwen2RMSNorm,
                                           CS._compute_qwen2rmsnorm)
        x = torch.randn(2, 5, 8)
        with torch.no_grad():
            assert torch.allclose(orig(x), twin(x))


class _TokRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        out = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            out = out * self.weight
        return out


class _ConvRMSNorm(_TokRMSNorm):
    def forward(self, x):
        x = x.transpose(1, 2)
        out = super().forward(x)
        return out.transpose(1, 2)


class _ConvLayerNorm(nn.LayerNorm):
    def forward(self, x):
        x = x.transpose(1, 2)
        x = nn.functional.layer_norm(
            x.float(), self.normalized_shape,
            self.weight.float(), self.bias.float(), self.eps).type_as(x)
        return x.transpose(1, 2)


# ---------------------------------------------------------------------
# Conversion sweep
# ---------------------------------------------------------------------

class _QuantResident(nn.Linear):
    _quant_resident = True


class _CustomWithParams(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(3))

    def forward(self, x):
        return x * self.scale


class TestConvertTree:
    def _tree(self):
        t = nn.Module()
        t.lin = nn.Linear(4, 4)
        t.emb = nn.Embedding(8, 4)
        t.resident = _QuantResident(4, 4)
        t.seq = nn.Sequential(nn.LayerNorm(4))
        t.custom = _CustomWithParams()
        return t

    def test_census_resident_skip_and_idempotence(self, caplog):
        t = self._tree()
        with caplog.at_level(logging.INFO, logger="modules.comfy_stream"):
            c1 = CS.convert_tree_for_streaming(t)
        # _CustomWithParams owns a direct Parameter (scale) with no leaf
        # compute -> converted via the streaming-container path (NOT skipped,
        # otherwise its param would strand on CPU under lowvram).
        assert c1 == {"Linear": 1, "Embedding": 1, "LayerNorm": 1,
                      "_CustomWithParams": 1}
        # Resident skipped entirely (streams natively via Phase 3).
        assert type(t.resident) is _QuantResident
        assert t.resident._quant_resident is True
        # Direct-param module is now a streaming subclass; its forward still
        # runs and its param rides with the activation device.
        assert getattr(t.custom, "comfy_cast_weights", False) is True
        out = t.custom(torch.randn(2, 3))
        assert out.shape == (2, 3)
        # No longer warned-as-unknown (it was converted).
        assert not any("without a registered streaming forward" in r.message
                       for r in caplog.records)

        c2 = CS.convert_tree_for_streaming(t)
        assert c2 == {}  # idempotent: already streaming

    def test_isinstance_base_preserved_and_flag_set(self):
        t = self._tree()
        CS.convert_tree_for_streaming(t)
        assert isinstance(t.lin, nn.Linear)
        assert isinstance(t.emb, nn.Embedding)
        assert isinstance(t.seq[0], nn.LayerNorm)
        assert t.lin.comfy_cast_weights is True
        assert t.seq[0].comfy_cast_weights is True

    def test_state_dict_unchanged_by_conversion(self):
        t = self._tree()
        sd_before = {k: v.clone() for k, v in t.state_dict().items()}
        CS.convert_tree_for_streaming(t)
        sd_after = t.state_dict()
        assert set(sd_after) == set(sd_before)
        for k in sd_before:
            assert torch.equal(sd_after[k], sd_before[k])

    def test_meta_safe_conversion(self):
        class _Tree(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 4)

        with torch.device("meta"):
            t = _Tree()
            CS.convert_tree_for_streaming(t)
        assert t.lin.weight.is_meta
        assert t.lin.comfy_cast_weights is True

    def test_skip_list_honored(self):
        t = self._tree()
        keep = t.lin
        CS.convert_tree_for_streaming(t, skip=[keep])
        assert getattr(keep, "comfy_cast_weights", False) is False

    def test_get_key_weight_contract(self):
        """Core's `_load_list` calls `comfy.model_patcher.get_key_weight`
        with ``"{n}.weight"`` and ``"{n}.bias"`` on EVERY converted module.
        A streaming subclass that lacks a real ``bias`` (norms, embeddings,
        containers) used to raise AttributeError and abort placement. This
        mirrors the exact core call so the regression is caught in CI."""
        import comfy.model_patcher

        t = self._tree()
        CS.convert_tree_for_streaming(t)
        t.cpu()

        for name, module in t.named_modules():
            if module is t:
                continue
            if not getattr(module, "comfy_cast_weights", False):
                continue
            for suffix in ("weight", "bias"):
                key = "{}.{}".format(name, suffix)
                # Must not raise; returns (tensor|None, set_func, convert_func)
                weight, set_func, convert_func = (
                    comfy.model_patcher.get_key_weight(t, key)
                )
                assert set_func is None and convert_func is None
                if suffix == "weight":
                    # Every converted leaf/container keeps a real weight or
                    # None; either way core's size estimate must cope.
                    assert weight is None or isinstance(weight, torch.Tensor)
                else:
                    # Bias is optional; None must be returned, not a raise.
                    assert weight is None or isinstance(weight, torch.Tensor)

    def test_container_relocation_cpu(self):
        """A direct-Parameter module (e.g. Block1D's layer-scale gamma) must
        forward correctly after streaming conversion, and its direct param
        must end on the activation device."""
        t = self._tree()
        CS.convert_tree_for_streaming(t)
        x = torch.randn(2, 3)
        out = t.custom(x)
        assert out.shape == (2, 3)
        assert t.custom.scale.device == x.device

    def test_container_forward_passes_kwargs_through(self):
        """Regression (native ASR): the container wrapper must keep the base
        forward's full signature. VibeVoiceAcousticTokenizerConvNext1dLayer
        takes padding_cache= and the ASR model passes it — a bare
        ``(self, x)`` wrapper raised "unexpected keyword argument
        'padding_cache'" mid-transcription."""

        class _KwargsModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(3))

            def forward(self, x, padding_cache=None, use_cache=False):
                if padding_cache is not None:
                    x = x + 1  # visible marker that the kwarg arrived
                return x * self.scale

        t = nn.Module()
        t.mod = _KwargsModule()
        CS.convert_tree_for_streaming(t)
        x = torch.zeros(2, 3)
        out = t.mod(x, padding_cache="sentinel", use_cache=True)
        # kwarg reached the base forward (the +1 branch fired)
        assert torch.equal(out, torch.ones(2, 3))

    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason="relocation cross-device needs CUDA")
    def test_container_relocation_cross_device(self):
        t = self._tree()
        CS.convert_tree_for_streaming(t)
        # Force the direct param to the offload (CPU) device.
        t.custom.scale.data = t.custom.scale.data.to("cpu")
        x = torch.randn(2, 3, device="cuda")
        out = t.custom(x)
        assert out.device.type == "cuda"
        assert t.custom.scale.device.type == "cuda"

    def test_forward_features_direct_param_access(self):
        """Regression for the real bug: the tokenizer's ``forward_features``
        streams a child leaf op (``block.mixer.conv``) to the compute device
        and THEN reads a direct param (``block.gamma``) WITHOUT calling
        ``block.forward``. Relocating only inside ``block.forward`` (or in a
        root entry wrapper) left gamma on CPU for nested entry points ->
        "expected all tensors on the same device". The container's
        ``__getattr__`` relocates direct params on access instead."""
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(3, 3, 3, padding=1)  # streamed leaf
                self.gamma = nn.Parameter(torch.ones(3))

            def forward(self, x):  # not used by forward_features path
                return x * self.gamma

        class Tokenizer(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = Block()

            def forward_features(self, x):
                x = self.block.conv(x)             # streams conv.weight -> compute dev
                return x * self.block.gamma.unsqueeze(-1)  # direct read; must relocate

            def forward(self, x):
                return self.forward_features(x)

        tok = Tokenizer()
        CS.convert_tree_for_streaming(tok)
        # Block is now a streaming container with access-time relocation.
        assert getattr(tok.block, "comfy_cast_weights", False) is True
        assert "__getattr__" in type(tok.block).__dict__

        if torch.cuda.is_available():
            # Simulate lowvram: weights start on the offload (CPU) device.
            tok.block.conv.weight.data = tok.block.conv.weight.data.to("cpu")
            tok.block.gamma.data = tok.block.gamma.data.to("cpu")
            x = torch.randn(2, 3, 10, device="cuda")
            out = tok.forward_features(x)
            assert out.device.type == "cuda"
            assert tok.block.gamma.device.type == "cuda"
        else:
            x = torch.randn(2, 3, 10)
            out = tok.forward_features(x)
            assert out.shape == (2, 3, 10)
            assert tok.block.gamma.device.type == "cpu"

    def test_direct_param_read_relocates_without_any_forward(self, monkeypatch):
        """A direct param read BEFORE any streaming forward (no _LAST_DEVICE
        yet) must still land on the runtime's compute device via the
        get_torch_device() fallback — an entry wrapper could never cover
        this."""
        if not torch.cuda.is_available():
            pytest.skip("cross-device relocation needs CUDA")
        t = self._tree()
        CS.convert_tree_for_streaming(t)
        target = torch.device("cuda")
        t.custom.scale.data = t.custom.scale.data.to("cpu")
        monkeypatch.setattr(CS, "_LAST_DEVICE", None)

        import comfy.model_management as mm
        with patch.object(mm, "get_torch_device", return_value=target):
            scale = t.custom.scale  # plain attribute read, no forward
        assert scale.device.type == "cuda"
        assert t.custom.scale.device.type == "cuda"

    def test_current_compute_device_fallback_without_comfy(self, monkeypatch):
        """Standalone (non-ComfyUI) use falls back to the last activation
        device seen by a streaming forward."""
        dev = torch.device("cpu")
        monkeypatch.setattr(CS, "_LAST_DEVICE", dev)
        monkeypatch.setitem(sys.modules, "comfy.model_management", None)
        assert CS._current_compute_device() == dev

    def test_getattr_missing_attr_still_raises(self):
        t = self._tree()
        CS.convert_tree_for_streaming(t)
        with pytest.raises(AttributeError):
            _ = t.custom.does_not_exist
        # None-valued params (register_parameter(name, None)) resolve to None.
        t.custom.register_parameter("ghost", None)
        assert t.custom.ghost is None


# ---------------------------------------------------------------------
# streaming_conversion gate
# ---------------------------------------------------------------------

class TestStreamingConversionGate:
    """``streaming_conversion(False)`` must suppress the sweep and nothing
    else. No caller opts in yet, so the default path has to be unchanged."""

    def test_conversion_runs_by_default(self):
        assert CS._STREAMING_CONVERSION_ENABLED is True
        t = nn.Module()
        t.lin = nn.Linear(4, 4)
        assert CS.convert_tree_for_streaming(t) == {"Linear": 1}
        assert t.lin.comfy_cast_weights is True

    def test_conversion_suppressed_inside_context(self):
        t = nn.Module()
        t.lin = nn.Linear(4, 4)
        with CS.streaming_conversion(False):
            assert CS.convert_tree_for_streaming(t) == {}
        # Suppressed, not converted: class and attributes are untouched.
        assert type(t.lin) is nn.Linear
        assert getattr(t.lin, "comfy_cast_weights", False) is False
        # Restored on exit: a later sweep still converts.
        assert CS._STREAMING_CONVERSION_ENABLED is True
        assert CS.convert_tree_for_streaming(t) == {"Linear": 1}

    def test_context_restores_previous_value_on_exception(self):
        with pytest.raises(RuntimeError):
            with CS.streaming_conversion(False):
                raise RuntimeError("boom")
        assert CS._STREAMING_CONVERSION_ENABLED is True
        t = nn.Module()
        t.lin = nn.Linear(4, 4)
        assert CS.convert_tree_for_streaming(t) == {"Linear": 1}

    def test_nested_contexts_restore_correctly(self):
        t = nn.Module()
        t.lin = nn.Linear(4, 4)
        with CS.streaming_conversion(False):
            assert CS._STREAMING_CONVERSION_ENABLED is False
            with CS.streaming_conversion(False):
                assert CS._STREAMING_CONVERSION_ENABLED is False
                assert CS.convert_tree_for_streaming(t) == {}
            # Inner exit restores the OUTER value, not the module default.
            assert CS._STREAMING_CONVERSION_ENABLED is False
            assert CS.convert_tree_for_streaming(t) == {}
        assert CS._STREAMING_CONVERSION_ENABLED is True
        assert CS.convert_tree_for_streaming(t) == {"Linear": 1}

    def test_nested_context_restores_enabling_value(self):
        """A nested ``True`` inside ``False`` must re-enable, and the outer
        ``False`` must come back — save/restore, not a boolean OR."""
        with CS.streaming_conversion(False):
            with CS.streaming_conversion(True):
                assert CS._STREAMING_CONVERSION_ENABLED is True
            assert CS._STREAMING_CONVERSION_ENABLED is False
        assert CS._STREAMING_CONVERSION_ENABLED is True

    @pytest.mark.parametrize("enabled", [True, False])
    def test_quant_resident_still_excluded(self, enabled):
        """GGUF / convrot-int8 / fp8 residents stream natively and are never
        rewrapped, whichever way the gate is set."""
        from gguf.constants import GGMLQuantizationType as T

        from modules.convrot_quant import ConvRotInt8Linear
        from modules.fp8_quant import FP8Linear
        from modules.gguf_quant import GGUFLinear

        t = nn.Module()
        t.gguf = GGUFLinear(32, 32, bias=False, ggml_type=T.Q8_0)
        t.convrot = ConvRotInt8Linear(4, 4, bias=False, group_size=32)
        t.fp8 = FP8Linear(4, 4, bias=False, fp8_dtype=torch.float8_e4m3fn)
        # The local stand-in from the sweep tests, same exclusion path.
        t.resident = _QuantResident(4, 4)

        before = {k: (type(v), getattr(v, "comfy_cast_weights", "unset"),
                      v.weight_function if hasattr(v, "weight_function")
                      else "unset")
                  for k, v in t.named_children()}
        with CS.streaming_conversion(enabled):
            census = CS.convert_tree_for_streaming(t)
        assert census == {}
        for name, (cls, cast, wf) in before.items():
            child = getattr(t, name)
            assert type(child) is cls
            assert getattr(child, "comfy_cast_weights", "unset") == cast
            if wf != "unset":
                assert child.weight_function is wf


# ---------------------------------------------------------------------
# Regression: index-input ops must not derive the weight cast from x
# ---------------------------------------------------------------------

class TestEmbeddingIndexDtype:
    """``cast_bias_weight`` takes its target dtype from ``input.dtype``.

    For nn.Embedding the input is an int64 index tensor, so passing it as
    the dtype source cast the WHOLE embedding table to int64 and made the
    lookup return int64 embeddings. That silently poisoned every downstream
    consumer: the LM's ``inputs_embeds`` became int64, and
    ``speech_tensors.type_as(x)`` then fed int64 audio into the acoustic
    tokenizer, crashing with "Input type (__int64) and bias type
    (BFloat16) should be the same" at the encoder stem conv. Core never
    does this — it passes only ``device=input.device`` (comfy/ops.py:793).
    """

    def _converted_embedding(self, dtype=torch.bfloat16):
        t = nn.Module()
        t.emb = nn.Embedding(32, 8, dtype=dtype)
        CS.convert_tree_for_streaming(t)
        return t

    def test_embedding_lookup_keeps_float_dtype(self):
        """int64 ids in -> float embeddings out (the reported crash).

        A non-empty ``weight_function`` forces the CAST path — the fast path
        yields the already-resident weight untouched, and the streaming /
        vbar route the bug was reported from is exactly the cast path.
        """
        t = self._converted_embedding()
        t.emb.weight_function = [lambda w: w]
        ids = torch.tensor([1, 5, 31], dtype=torch.int64)
        out = t.emb(ids)
        assert out.dtype.is_floating_point, f"embedding returned {out.dtype}"
        assert out.shape == (3, 8)
        # Values must still be the table rows (no truncated-to-int cast).
        assert torch.equal(out.float(), t.emb.weight.detach()[ids].float())

    def test_embedding_accepts_int32_indices_too(self):
        t = self._converted_embedding()
        t.emb.weight_function = [lambda w: w]
        out = t.emb(torch.tensor([0, 2], dtype=torch.int32))
        assert out.dtype.is_floating_point

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_embedding_across_devices_keeps_weight_dtype(self):
        """The slow (cast) path: module on CPU, ids on CUDA."""
        t = self._converted_embedding()
        t.emb.to("cpu")
        ids = torch.tensor([4, 9], dtype=torch.int64, device="cuda")
        out = t.emb(ids)
        assert out.dtype == torch.bfloat16, f"got {out.dtype}"
        assert out.device.type == "cuda"
        ref = torch.nn.functional.embedding(
            ids, t.emb.weight.detach().to("cuda", torch.bfloat16))
        assert torch.equal(out, ref)


class TestPullStats:
    """The [vvpull] line is the contract with a live run: it must name the
    path that served each weight (vbar = core's file->VRAM paging read,
    nonvbar = the cast-buffer path that can copy a file view HOST-side) and
    the bytes moved. A silent counter is how the 'insanely long + RAM fill'
    report goes unattributed again, so the format is pinned here."""

    def test_empty_after_reset(self):
        from ComfyUI_VibeVoice.modules.comfy_stream import (
            pull_stats_line, reset_pull_stats,
        )

        reset_pull_stats()
        assert "no streaming leaf pulls" in pull_stats_line()

    def test_records_path_bytes_and_resets(self):
        from ComfyUI_VibeVoice.modules.comfy_stream import (
            _record_pull, pull_stats_line, reset_pull_stats,
        )

        reset_pull_stats()
        _record_pull("Linear", "vbar", 1024 ** 2)
        _record_pull("Linear", "vbar", 1024 ** 2)
        _record_pull("Conv1d", "nonvbar", 4096)
        line = pull_stats_line()
        assert "vbar/Linear=2(2MB)" in line, line
        assert "nonvbar/Conv1d=1(0MB)" in line, line

        reset_pull_stats()
        assert "no streaming leaf pulls" in pull_stats_line()

    def test_acquire_counts_the_colocated_fast_path(self):
        import torch
        from ComfyUI_VibeVoice.modules.comfy_stream import (
            pull_stats_line, reset_pull_stats,
        )
        from ComfyUI_VibeVoice.modules.comfy_stream import _acquire

        reset_pull_stats()
        lin = torch.nn.Linear(4, 4)
        with _acquire(lin, torch.randn(2, 4)) as (w, b):
            assert w is lin.weight
        line = pull_stats_line()
        assert "colocated/Linear=1" in line, line


class TestVbarResidencyObserver:
    """``[vvpull]`` must separate a cheap arena hit from a disk re-read.

    Finding F7 (tests/probe_vbar_residency.py): when the model does not fit in
    free VRAM, every forward re-reads the weights from the checkpoint file,
    which is the reported "~10x slower inference". The per-kind byte totals
    above CANNOT show that -- a resident pull and a re-read both "serve" a
    weight -- so the verdict core already computed is reported separately.
    """

    @staticmethod
    def _install_over(monkeypatch, delegate):
        """Install the observer on top of a stand-in for core's resolver.

        Installing over a fake lets the assertions cover what the wrapper does
        to core's call (counts, then delegates the SAME arguments) without
        dragging core's real vbar machinery into a CPU-only test. The
        module-level ``_VBAR_OBSERVER`` guard is cleared so the install path
        itself is exercised rather than short-circuited. The observer is a
        diagnostic, off in production, so its gate is switched on here.
        """
        import comfy.ops

        monkeypatch.setenv("VIBEVOICE_VBAR_OBSERVER", "1")
        monkeypatch.setattr(comfy.ops, "resolve_cast_module_with_vbar", delegate)
        monkeypatch.setattr(CS, "_VBAR_OBSERVER", None)
        CS._install_vbar_observer()
        return comfy.ops.resolve_cast_module_with_vbar

    class _FakeModule:
        def __init__(self, resident, weight):
            self._prefetch = {"signature": object(), "resident": resident}
            self.weight = weight

    def test_splits_resident_from_reread(self, monkeypatch):
        calls = []

        def delegate(s, *args, **kwargs):
            calls.append((s, args, kwargs))
            return "delegated"

        observed = self._install_over(monkeypatch, delegate)
        CS.reset_pull_stats()
        lin = torch.nn.Linear(4, 4)

        assert observed(self._FakeModule(True, lin.weight)) == "delegated"
        assert observed(self._FakeModule(False, lin.weight)) == "delegated"

        split = CS._VBAR_SPLIT
        assert split["resident_calls"] == 1
        assert split["reread_calls"] == 1
        assert split["reread_bytes"] == 4 * 4 * 4
        assert len(calls) == 2, "core's call must not be skipped"
        CS.reset_pull_stats()

    def test_forwards_arguments_untouched(self, monkeypatch):
        seen = {}
        delegate = lambda s, *a, **k: seen.update(args=a, kwargs=k) or "ok"

        observed = self._install_over(monkeypatch, delegate)
        marker = self._FakeModule(True, torch.nn.Linear(4, 4).weight)
        assert observed(marker, 1, 2, three=3) == "ok"
        assert seen["args"] == (1, 2)
        assert seen["kwargs"] == {"three": 3}

    def test_module_without_prefetch_does_not_raise(self, monkeypatch):
        observed = self._install_over(monkeypatch, lambda s, *a, **k: "ok")
        CS.reset_pull_stats()
        assert observed(torch.nn.Linear(4, 4), 1) == "ok"
        assert CS._VBAR_SPLIT["resident_calls"] == 0
        assert CS._VBAR_SPLIT["reread_calls"] == 0

    def test_installs_once_and_is_idempotent(self, monkeypatch):
        observed = self._install_over(monkeypatch, lambda s, *a, **k: "ok")
        CS._install_vbar_observer()
        assert CS._VBAR_OBSERVER is observed
        import comfy.ops

        assert comfy.ops.resolve_cast_module_with_vbar is observed, (
            "a second install would stack wrappers onto core's resolver")

    def test_pull_line_reports_the_split_and_its_share(self):
        CS.reset_pull_stats()
        assert "resident=" not in CS.pull_stats_line()

        CS._record_pull("Linear", "vbar", 1024 ** 2)
        CS._VBAR_SPLIT["resident_calls"] = 90
        CS._VBAR_SPLIT["reread_calls"] = 10
        CS._VBAR_SPLIT["reread_bytes"] = 5 * 1024 ** 2
        line = CS.pull_stats_line()
        assert "resident=90" in line, line
        assert "reread=10(10.0%,5MB)" in line, line

        CS.reset_pull_stats()
        assert CS._VBAR_SPLIT["resident_calls"] == 0
        assert CS._VBAR_SPLIT["reread_calls"] == 0
        assert CS._VBAR_SPLIT["reread_bytes"] == 0

    def test_reset_clears_the_split(self):
        CS._VBAR_SPLIT["reread_calls"] = 7
        CS.reset_pull_stats()
        assert CS._VBAR_SPLIT["reread_calls"] == 0

    def test_observer_is_not_installed_in_production(self, monkeypatch):
        """No env var set => the observer must leave core's resolver alone."""
        import comfy.ops

        monkeypatch.delenv("VIBEVOICE_DIAGNOSTICS", raising=False)
        monkeypatch.delenv("VIBEVOICE_VBAR_OBSERVER", raising=False)
        original = comfy.ops.resolve_cast_module_with_vbar
        monkeypatch.setattr(CS, "_VBAR_OBSERVER", None)
        CS._install_vbar_observer()
        assert CS._VBAR_OBSERVER is False
        assert comfy.ops.resolve_cast_module_with_vbar is original
        monkeypatch.setattr(CS, "_VBAR_OBSERVER", None)
