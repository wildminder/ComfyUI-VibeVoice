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
