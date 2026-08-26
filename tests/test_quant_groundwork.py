"""Phase 0 groundwork tests: synthetic GGUF fixtures, type matrix, backend probe."""

import pytest
import torch
import gguf
from gguf.constants import GGMLQuantizationType as T

from ComfyUI_VibeVoice.modules import gguf_quant as G
from ComfyUI_VibeVoice.modules.convrot_quant import (
    CONVROT_FORMAT,
    assert_convrot_backend,
)


class TestSyntheticFixtureRoundtrip:
    def _spec(self):
        return [
            ("model.language_model.layers.0.self_attn.q_proj.weight", "Q8_0", (16, 32)),
            ("model.language_model.layers.0.mlp.down_proj.weight", "Q4_K", (32, 256)),
            ("model.language_model.layers.0.mlp.up_proj.weight", "Q5_K", (32, 256)),
            ("model.language_model.layers.1.self_attn.q_proj.weight", "Q6_K", (32, 256)),
            ("model.language_model.embed_tokens.weight", "BF16", (64, 32)),
            ("model.language_model.norm.weight", "F32", (32,)),
            ("lm_head.weight", "F16", (16, 8)),
        ]

    def test_writer_reader_roundtrip(self, make_gguf_file):
        path = make_gguf_file(self._spec(), tag="roundtrip")
        reader = gguf.GGUFReader(str(path))
        by_name = {t.name: t for t in reader.tensors}
        assert len(reader.tensors) == 7

        checks = {
            "model.language_model.layers.0.self_attn.q_proj.weight": (T.Q8_0, (32, 16)),
            "model.language_model.layers.0.mlp.down_proj.weight": (T.Q4_K, (256, 32)),
            "model.language_model.layers.0.mlp.up_proj.weight": (T.Q5_K, (256, 32)),
            "model.language_model.layers.1.self_attn.q_proj.weight": (T.Q6_K, (256, 32)),
            "model.language_model.embed_tokens.weight": (T.BF16, (32, 64)),
            "model.language_model.norm.weight": (T.F32, (32,)),
            "lm_head.weight": (T.F16, (8, 16)),
        }
        for name, (tt, shape) in checks.items():
            t = by_name[name]
            assert t.tensor_type == tt, name
            assert tuple(int(s) for s in t.shape) == shape, name

    def test_byte_counts_match_expected_numel(self, make_gguf_file):
        """A3 cross-check: expected_numel matches actual reader byte counts."""
        spec = self._spec()
        path = make_gguf_file(spec, tag="numel")
        reader = gguf.GGUFReader(str(path))
        by_name = {t.name: t for t in reader.tensors}
        for name, kind, shape in spec:
            n_elem = 1
            for s in shape:
                n_elem *= s
            tt = getattr(T, kind)
            assert by_name[name].n_bytes == G.expected_numel(n_elem, tt), name

    def test_builder_rejects_undivisible_shape(self, make_gguf_file):
        with pytest.raises(ValueError, match="multiple of"):
            make_gguf_file(
                [("x.weight", "Q4_K", (8, 100))], tag="bad"
            )


class TestTypeMatrix:
    def test_supported_matrix_exported(self):
        names = {t.name for t in G.SUPPORTED_GGML_TYPES}
        assert {"Q8_0", "Q4_K", "Q5_K", "Q6_K"} <= names

    def test_float_matrix_includes_bf16(self):
        """The real VibeVoice GGUF is dense-BF16 heavy; BF16 must be supported."""
        names = {t.name for t in G.FLOAT_GGML_TYPES}
        assert {"F32", "F16", "BF16"} <= names

    def test_convrot_format_constant(self):
        assert CONVROT_FORMAT == "int8_tensorwise"


class TestConvRotBackendProbe:
    def test_probe_passes_in_embedded_env(self):
        backend = assert_convrot_backend()
        assert backend in ("triton", "cuda", "eager")

    def test_probe_fails_without_kitchen(self, monkeypatch):
        import sys

        saved = sys.modules.get("comfy_kitchen")
        monkeypatch.setitem(sys.modules, "comfy_kitchen", None)
        try:
            with pytest.raises(RuntimeError, match="comfy_kitchen"):
                assert_convrot_backend()
        finally:
            if saved is not None:
                sys.modules["comfy_kitchen"] = saved

    def test_probe_fails_when_capability_missing(self, monkeypatch):
        import sys
        import types

        fake = types.ModuleType("comfy_kitchen")
        fake.list_backends = lambda: {
            "eager": {"available": True,
                      "capabilities": ["int8_linear"]},  # dequant capability absent
        }
        monkeypatch.setitem(sys.modules, "comfy_kitchen", fake)
        with pytest.raises(RuntimeError, match="capabilities"):
            assert_convrot_backend()
