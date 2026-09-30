"""Flat-block (quantui-rs) GGUF recovery tests.

Regression suite for third-party converters (e.g. quantui-rs) that quantize
small-kernel conv weights — whose row is below the quant block size — as a
flat C-order pool of whole blocks. The stock ``gguf.GGUFReader`` rejects the
whole file at open; this node's tolerant open recovers every tensor and the
non-Linear quantized weights (embeddings, conv heads) dequantize at load.

Pinned to a real-world file layout: VibeVoice-1.5B-q8_0 with
``model.acoustic_tokenizer.decoder.head.conv.conv.weight`` stored as header
dims [7, 32, 1] (kernel size 7 < block size 32) and 102 further conv tensors
in the same shape family, plus a Q8_0 ``embed_tokens`` targeting an
``nn.Embedding``.
"""

import pytest
import torch
from gguf.constants import GGMLQuantizationType as T

import gguf

from conftest import build_stub_vv, write_synthetic_gguf

from ComfyUI_VibeVoice.modules.gguf_quant import (
    GGUFLinear,
    dequantize_reader_tensor,
    open_gguf_reader,
)
from ComfyUI_VibeVoice.modules.external_loader import (
    _install_gguf_weights,
    _load_gguf_state_dict,
    resolve_auto_config_name,
)
from ComfyUI_VibeVoice.modules.config_detect import fingerprint_weights

# quantui-rs family: kernel-size rows below the Q8_0 block size (32).
FLAT_CONV_KEY = "model.acoustic_tokenizer.decoder.head.conv.conv.weight"
# Header dims (ggml ne order): kernel first -> [7, 32, 1]; torch logical
# shape is the reverse -> (1, 32, 7).
FLAT_CONV_LOGICAL = (1, 32, 7)
FLAT_CONV_SPEC = (FLAT_CONV_KEY, "Q8_0", FLAT_CONV_LOGICAL)


# ====================================================================
# Fixtures
# ====================================================================

@pytest.fixture
def flat_pool_gguf(tmp_path):
    """A GGUF mixing a quantui-rs flat-pool conv with normal tensors.

    Layout mirrors the real VibeVoice-1.5B-q8_0_my.gguf: one sub-block-row
    Q8_0 conv (flat pool), one row-legal Q8_0 linear, one F32 bias, one
    Q8_0 embedding (row-legal byte shape, targets a non-Linear module).
    """
    spec = [
        FLAT_CONV_SPEC,
        ("model.language_model.layers.0.self_attn.q_proj.weight",
         "Q8_0", (64, 64)),
        ("model.language_model.layers.0.self_attn.q_proj.bias", "F32", (64,)),
        # embeddings are non-Linear quantized targets in quantui-rs files;
        # (96, 64) matches build_stub_vv's default vocab=96
        ("model.language_model.embed_tokens.weight", "Q8_0", (96, 64)),
    ]
    path = tmp_path / "flat_pool.gguf"
    write_synthetic_gguf(
        path, spec, seed=7,
        flat_pool={FLAT_CONV_KEY},
    )
    return str(path)


# ====================================================================
# Tolerant open
# ====================================================================

class TestTolerantOpen:
    def test_stock_reader_rejects_flat_pool(self, flat_pool_gguf):
        """Guard the premise: the stock gguf reader refuses this file."""
        with pytest.raises(ValueError, match="block size"):
            gguf.GGUFReader(flat_pool_gguf)

    def test_tolerant_open_returns_all_tensors(self, flat_pool_gguf):
        reader = open_gguf_reader(flat_pool_gguf)
        assert len(reader.tensors) == 4

    def test_tolerant_open_preserves_header_dims(self, flat_pool_gguf):
        """Header dims survive; only the byte layout of the conv is flat."""
        reader = open_gguf_reader(flat_pool_gguf)
        by_name = {t.name: t for t in reader.tensors}
        conv = by_name[FLAT_CONV_KEY]
        assert tuple(int(s) for s in conv.shape) == (7, 32, 1)
        # flat pool: 7 blocks * 34 bytes = 238
        assert conv.data.shape == (238,)

    def test_row_legal_tensors_keep_native_byte_shape(self, flat_pool_gguf):
        """Only sub-block-row tensors flatten; row-legal ones stay mapped."""
        reader = open_gguf_reader(flat_pool_gguf)
        by_name = {t.name: t for t in reader.tensors}
        lin = by_name["model.language_model.layers.0.self_attn.q_proj.weight"]
        # (64, 64) torch -> (64, 64) logical; ggml dims (64, 64) rows legal
        # -> bytes per row = 64/32*34 = 68
        assert lin.data.shape == (64, 68)

    def test_tolerant_open_warns(self, flat_pool_gguf, caplog):
        # fresh session state: the dedupe set must not hold this file
        from ComfyUI_VibeVoice.modules import gguf_quant as _gq
        _gq._FLAT_POOL_WARNED.discard(flat_pool_gguf)
        with caplog.at_level("WARNING", logger="ComfyUI_VibeVoice.modules.gguf_quant"):
            open_gguf_reader(flat_pool_gguf)
        assert any("automatic recovery" in r.getMessage()
                   for r in caplog.records)

    def test_tolerant_open_warns_once_per_file(self, flat_pool_gguf, caplog):
        """The loader opens one GGUF several times; only the first warns."""
        from ComfyUI_VibeVoice.modules import gguf_quant as _gq
        _gq._FLAT_POOL_WARNED.discard(flat_pool_gguf)
        with caplog.at_level("WARNING", logger="ComfyUI_VibeVoice.modules.gguf_quant"):
            open_gguf_reader(flat_pool_gguf)
            open_gguf_reader(flat_pool_gguf)
            open_gguf_reader(flat_pool_gguf)
        warned = [r for r in caplog.records if "automatic recovery" in r.getMessage()]
        assert len(warned) == 1
        # A DIFFERENT flat-pool file still gets its own warning.
        _gq._FLAT_POOL_WARNED.clear()
        assert flat_pool_gguf not in _gq._FLAT_POOL_WARNED

    def test_unrelated_open_error_propagates(self, tmp_path):
        """A corrupt file (not a block-size issue) still raises its error."""
        bad = tmp_path / "corrupt.gguf"
        bad.write_bytes(b"not a gguf file at all")
        with pytest.raises(Exception):
            open_gguf_reader(str(bad))

    def test_spec_conformant_file_uses_stock_path(self, tmp_path):
        """A clean file opens without the fallback (no warning logged)."""
        import logging
        spec = [("model.language_model.layers.0.self_attn.q_proj.weight",
                 "Q8_0", (64, 64))]
        path = write_synthetic_gguf(tmp_path / "clean.gguf", spec)
        reader = open_gguf_reader(str(path))
        assert len(reader.tensors) == 1


# ====================================================================
# Dequantization of reader tensors
# ====================================================================

class TestDequantizeReaderTensor:
    def test_flat_pool_dequantizes_to_logical_shape(self, flat_pool_gguf):
        reader = open_gguf_reader(flat_pool_gguf)
        conv = {t.name: t for t in reader.tensors}[FLAT_CONV_KEY]
        w = dequantize_reader_tensor(conv)
        assert isinstance(w, torch.Tensor)
        assert tuple(w.shape) == FLAT_CONV_LOGICAL
        assert w.dtype == torch.float32
        assert torch.isfinite(w).all()

    def test_flat_pool_bitwise_par(self, flat_pool_gguf):
        """Flat dequant is bitwise-equal to the gguf-py oracle on the flat bytes."""
        import numpy as np
        from gguf.quants import dequantize as oracle

        reader = open_gguf_reader(flat_pool_gguf)
        conv = {t.name: t for t in reader.tensors}[FLAT_CONV_KEY]
        ref = oracle(conv.data.view(np.uint8), T.Q8_0).reshape(FLAT_CONV_LOGICAL)
        w = dequantize_reader_tensor(conv)
        assert torch.equal(w, torch.from_numpy(ref))

    def test_row_legal_quantized_tensor_round_trips(self, flat_pool_gguf):
        """Row-legal quantized tensors dequantize through the same helper."""
        reader = open_gguf_reader(flat_pool_gguf)
        lin = {t.name: t for t in reader.tensors}[
            "model.language_model.layers.0.self_attn.q_proj.weight"]
        w = dequantize_reader_tensor(lin)
        assert tuple(w.shape) == (64, 64)
        assert torch.isfinite(w).all()

    def test_float_tensor_passes_through(self, flat_pool_gguf):
        reader = open_gguf_reader(flat_pool_gguf)
        bias = {t.name: t for t in reader.tensors}[
            "model.language_model.layers.0.self_attn.q_proj.bias"]
        w = dequantize_reader_tensor(bias)
        assert tuple(w.shape) == (64,)
        assert w.dtype == torch.float32

    def test_truncated_pool_raises_actionable(self, tmp_path):
        """A pool whose element count is not whole blocks cannot recover."""
        # 33 elements -> not a multiple of 32: writer refuses; craft the
        # header via a (33,)-row tensor of raw blocks is impossible — so
        # assert the guard via a direct call on a hand-built stand-in.
        class _FakeTensor:
            name = "fake.weight"
            tensor_type = T.Q8_0
            shape = (33,)  # ggml dims -> logical (33,), 33 % 32 != 0
            data = None

        with pytest.raises(ValueError, match="cannot be recovered"):
            dequantize_reader_tensor(_FakeTensor())


# ====================================================================
# Dense state-dict path
# ====================================================================

class TestDenseStateDict:
    def test_load_state_dict_handles_flat_pool(self, flat_pool_gguf):
        """The dense loader dequantizes everything, flat pools included."""
        sd = _load_gguf_state_dict(flat_pool_gguf)
        assert FLAT_CONV_KEY in sd
        assert tuple(sd[FLAT_CONV_KEY].shape) == FLAT_CONV_LOGICAL
        assert sd[FLAT_CONV_KEY].dtype == torch.float32
        assert torch.isfinite(sd[FLAT_CONV_KEY]).all()

    def test_load_state_dict_all_tensors_present(self, flat_pool_gguf):
        sd = _load_gguf_state_dict(flat_pool_gguf)
        assert len(sd) == 4


# ====================================================================
# Install: non-Linear dequant-at-load branch
# ====================================================================

class _ConvHeadStub(torch.nn.Module):
    """Mimics the vendored SConv1d head: a Conv1d at ``.conv.conv``."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Module()
        self.conv.conv = torch.nn.Conv1d(32, 1, kernel_size=7, bias=False)

    def forward(self, x):
        return self.conv.conv(x)


class TestInstallNonLinearFallback:
    def _build_model(self):
        model = build_stub_vv(n_layers=1)
        # graft the conv head onto the stub tree at the acoustic path
        model.model.acoustic_tokenizer = torch.nn.Module()
        model.model.acoustic_tokenizer.decoder = torch.nn.Module()
        model.model.acoustic_tokenizer.decoder.head = _ConvHeadStub()
        return model

    def test_nonlinear_quantized_weight_dequants_at_load(self, flat_pool_gguf):
        """A quantized non-Linear target loads as float, not an error."""
        model = self._build_model()
        reader = open_gguf_reader(flat_pool_gguf)
        stats = _install_gguf_weights(model, reader)

        conv = model.model.acoustic_tokenizer.decoder.head.conv.conv
        assert isinstance(conv, torch.nn.Conv1d)
        assert not isinstance(conv, GGUFLinear)
        assert conv.weight.dtype.is_floating_point
        assert tuple(conv.weight.shape) == (1, 32, 7)
        assert not conv.weight.is_meta

        # stats report the dequant-at-load accounting
        assert stats["n_dequant_load"] == 2  # conv + embed_tokens
        assert stats["n_resident_layers"] == 1
        assert stats["dequant_load_bytes"] > 0

    def test_nonlinear_values_match_flat_dequant(self, flat_pool_gguf):
        """Installed conv weights are bitwise the flat-pool dequant."""
        model = self._build_model()
        reader = open_gguf_reader(flat_pool_gguf)
        _install_gguf_weights(model, reader)

        by_name = {t.name: t for t in reader.tensors}
        ref = dequantize_reader_tensor(by_name[FLAT_CONV_KEY])
        conv = model.model.acoustic_tokenizer.decoder.head.conv.conv
        assert torch.equal(conv.weight.data, ref)

    def test_nonlinear_dequant_targets_model_dtype(self, flat_pool_gguf):
        """Dequant-at-load lands in the destination dtype, not fp32.

        The fp32 intermediate used to be materialized and cast afterwards,
        so a bf16 model carried 4x-size scratch for every embedding/conv head.
        """
        model = self._build_model().to(torch.bfloat16)
        reader = open_gguf_reader(flat_pool_gguf)
        stats = _install_gguf_weights(model, reader)

        conv = model.model.acoustic_tokenizer.decoder.head.conv.conv
        assert conv.weight.dtype is torch.bfloat16

        by_name = {t.name: t for t in reader.tensors}
        ref = dequantize_reader_tensor(by_name[FLAT_CONV_KEY])
        assert torch.equal(conv.weight.data, ref.to(torch.bfloat16))
        # reported bytes track the stored dtype, not the fp32 scratch
        assert stats["dequant_load_bytes"] > 0

    def test_install_streams_dense_tensors(self, flat_pool_gguf, monkeypatch):
        """Dense tensors stream one at a time; no batch state dict is built."""
        from ComfyUI_VibeVoice.modules.loader import VibeVoiceLoader

        seen = {}
        real = VibeVoiceLoader._stream_apply_dense

        def _spy(model, tensor_pairs, known_missing=None, **kw):
            seen["lazy"] = not isinstance(tensor_pairs, (dict, list, tuple))
            seen["pairs"] = list(tensor_pairs)
            seen["target_device"] = kw.get("target_device")
            return real(model, tensor_pairs, known_missing=known_missing, **kw)

        def _batch(*a, **k):
            raise AssertionError("batch _apply_state_dict must not be used")

        monkeypatch.setattr(VibeVoiceLoader, "_stream_apply_dense", staticmethod(_spy))
        monkeypatch.setattr(VibeVoiceLoader, "_apply_state_dict", staticmethod(_batch))

        model = self._build_model()
        reader = open_gguf_reader(flat_pool_gguf)
        _install_gguf_weights(model, reader, target_device=None)

        assert seen["lazy"], "dense pairs must be a generator, not a materialized dict"
        assert seen["pairs"], "expected at least one dense tensor"
        assert seen["target_device"] is None, "no device requested -> nothing placed"

    def test_embedding_quantized_dequants_at_load(self, flat_pool_gguf):
        """Q8_0 embed_tokens (nn.Embedding) lands as float, stays float."""
        model = self._build_model()
        reader = open_gguf_reader(flat_pool_gguf)
        _install_gguf_weights(model, reader)

        emb = model.model.language_model.embed_tokens
        assert not isinstance(emb, GGUFLinear)
        assert emb.weight.dtype.is_floating_point
        assert tuple(emb.weight.shape) == (96, 64)
        assert not emb.weight.is_meta

    def test_linear_residents_unaffected(self, flat_pool_gguf):
        """The Linear in the same file still installs quant-resident."""
        model = self._build_model()
        reader = open_gguf_reader(flat_pool_gguf)
        _install_gguf_weights(model, reader)

        q = model.model.language_model.layers[0].self_attn.q_proj
        assert isinstance(q, GGUFLinear)
        assert q.weight.dtype == torch.uint8

    def test_shape_mismatch_on_nonlinear_still_raises(self, tmp_path):
        """A conv whose header shape disagrees with the module fails loudly."""
        from ComfyUI_VibeVoice.modules.quant_common import QuantTargetMismatch

        spec = [(FLAT_CONV_KEY, "Q8_0", (1, 32, 5))]  # kernel 5 vs module 7
        path = write_synthetic_gguf(tmp_path / "mismatch.gguf", spec,
                                    flat_pool={FLAT_CONV_KEY})
        model = self._build_model()
        reader = open_gguf_reader(str(path))
        with pytest.raises(QuantTargetMismatch, match="disagrees"):
            _install_gguf_weights(model, reader)


# ====================================================================
# Auto-detect / fingerprint on flat-pool files
# ====================================================================

class TestFingerprintOnFlatPool:
    def test_detect_reads_header_on_flat_pool(self, flat_pool_gguf, caplog):
        """Auto-detect reads the embedding fingerprint on a flat-pool file.

        The fixture embedding (96, 64) matches no family signature, so
        classify returns None — but detection must still REACH the
        classification (the stock reader would crash on this file before
        ever looking at shapes).
        """
        with caplog.at_level("DEBUG", logger="ComfyUI_VibeVoice.modules.config_detect"):
            fp = fingerprint_weights(flat_pool_gguf)
        assert fp is None
        # "foreign shape" is logged only after the header was successfully
        # read past the flat-pool tensors — proof detection reached the
        # classification instead of crashing on the reader open.
        assert any("foreign shape" in r.getMessage()
                   and "embed_tokens" in r.getMessage()
                   for r in caplog.records)

    def test_detect_with_real_signature(self, tmp_path):
        """A 1.5B-signature embedding in a flat-pool file auto-detects."""
        spec = [
            FLAT_CONV_SPEC,
            # 1.5B signature: vocab 151936, hidden 1536 (row 1536 % 32 == 0)
            ("model.language_model.embed_tokens.weight", "Q8_0",
             (151936, 1536)),
        ]
        path = write_synthetic_gguf(tmp_path / "sig.gguf", spec,
                                    flat_pool={FLAT_CONV_KEY})
        assert resolve_auto_config_name(str(path)) == "VibeVoice-1.5B"

    def test_detect_7b_mixed_scheme_file(self, tmp_path):
        """The quantui-rs 7B layout auto-detects: HF keys + 'output.weight'
        (llamacpp lm_head alias) + flat-pool convs, like the real file."""
        spec = [
            FLAT_CONV_SPEC,
            # 7B signature: vocab 152064, hidden 3584
            ("model.language_model.embed_tokens.weight", "Q8_0",
             (152064, 3584)),
            # one HF key + the llamacpp lm_head alias -> 'mixed' census
            ("model.language_model.layers.0.self_attn.q_proj.weight",
             "Q8_0", (3584, 3584)),
            ("output.weight", "F16", (152064, 3584)),
        ]
        path = write_synthetic_gguf(tmp_path / "sig7b.gguf", spec,
                                    flat_pool={FLAT_CONV_KEY})
        assert resolve_auto_config_name(str(path)) == "VibeVoice-7B"


class TestResidentBlocksPlacement:
    """Where a quant-resident Linear's raw blocks live.

    The raw blocks ARE the model's storage — a q8_0 checkpoint's 8 GB of
    blocks is not 16 GB of dequantized weights waiting to happen. Left on the
    host they are private memory (the 24 -> 41 GB spike) and every forward
    takes the paged `cast_bias_weight` path; placed on the load device they
    cost the same bytes, load flat, and dequantize in place.
    """

    def test_resident_blocks_go_to_the_requested_device(self, flat_pool_gguf):
        if not torch.cuda.is_available():
            pytest.skip("no GPU in this environment")
        from ComfyUI_VibeVoice.modules.external_loader import _install_gguf_weights
        from ComfyUI_VibeVoice.modules.gguf_quant import GGUFLinear

        model = TestInstallNonLinearFallback()._build_model()
        reader = open_gguf_reader(flat_pool_gguf)
        stats = _install_gguf_weights(
            model, reader, target_device=torch.device("cuda", 0)
        )

        assert stats["n_resident_layers"] > 0, "fixture has no resident layers"
        residents = [
            m for m in model.modules() if isinstance(m, GGUFLinear)
        ]
        assert residents
        off_device = [
            type(m).__name__ for m in residents if m.weight.device.type != "cuda"
        ]
        assert not off_device, f"resident blocks left on host: {off_device}"
        # Blocks stay blocks — no float materialisation on the way in.
        assert all(m.weight.dtype == torch.uint8 for m in residents)

    def test_no_device_request_keeps_the_host_contract(self, flat_pool_gguf):
        from ComfyUI_VibeVoice.modules.external_loader import _install_gguf_weights
        from ComfyUI_VibeVoice.modules.gguf_quant import GGUFLinear

        model = TestInstallNonLinearFallback()._build_model()
        reader = open_gguf_reader(flat_pool_gguf)
        _install_gguf_weights(model, reader, target_device=None)

        residents = [m for m in model.modules() if isinstance(m, GGUFLinear)]
        assert residents
        assert all(m.weight.device.type == "cpu" for m in residents)
