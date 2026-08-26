"""Phase F1: cross-family numerical parity from IDENTICAL weight bytes.

The dense reference is built by dequantizing the SAME GGML blocks through the
gguf-py oracle; the resident model loads those blocks through the real loader.
Q8_0 output must match the float reference near-bitwise; K-quants within a
small relative tolerance (scale arithmetic is exact, quantization error is in
the stored data itself).
"""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

import gguf
from gguf.constants import GGMLQuantizationType as T
from gguf.quants import dequantize as oracle_dequantize

from ComfyUI_VibeVoice.modules import external_loader as EL
from ComfyUI_VibeVoice.modules.external_loader import load_external_vibevoice_model
from ComfyUI_VibeVoice.modules.gguf_quant import GGUFLinear
from conftest import build_stub_vv, stub_vv_gguf_spec


class _FakeStreamingCfg:
    pass


def _dims_for(qtype):
    """K-quants need in_features % 256 == 0."""
    if qtype in ("Q4_K", "Q5_K", "Q6_K"):
        return dict(n_layers=1, hidden=256, ffn=512)
    return dict(n_layers=1, hidden=64, ffn=128)


def _load_resident(make_gguf_file, tag, qtype):
    dims = _dims_for(qtype)

    def _instantiate(config, is_streaming, attn_implementation,
                     final_load_dtype, use_meta=True):
        return build_stub_vv(**dims)

    path = make_gguf_file(stub_vv_gguf_spec(qtype=qtype, **dims), tag=tag)
    with patch.object(EL.VibeVoiceLoader, "_load_config", return_value=MagicMock()), \
         patch.object(EL.VibeVoiceLoader, "_load_tokenizer", return_value=MagicMock()), \
         patch.object(EL.VibeVoiceLoader, "_load_processor", return_value=MagicMock()), \
         patch.object(EL.VibeVoiceLoader, "_instantiate_model",
                      side_effect=_instantiate), \
         patch.object(EL, "resolve_sidecar_config", return_value="/fake/config.json"), \
         patch.object(EL, "resolve_sidecar_preprocessor", return_value=""), \
         patch.object(EL, "resolve_sidecar_tokenizer_dir", return_value="/fake/dir"), \
         patch.object(EL, "resolve_dtype", return_value=torch.float32), \
         patch.object(EL, "resolve_attention_mode", side_effect=lambda m, q: m), \
         patch.object(EL, "get_attn_implementation_for_load", return_value="eager"), \
         patch.object(EL, "VibeVoiceStreamingConfig", _FakeStreamingCfg), \
         patch.object(EL.model_management, "get_torch_device",
                      return_value=torch.device("cpu")):
        bundle = load_external_vibevoice_model(
            weight_path=str(path), config_name="VibeVoice-1.5B",
            attention_mode="sdpa", use_llm_4bit=False, dtype_str="auto",
        )
    return bundle["model"]


def _oracle_dense_weight(gguf_path, key):
    """Dequantize one tensor through gguf-py -> the shared float ground truth."""
    reader = gguf.GGUFReader(str(gguf_path))
    rt = next(t for t in reader.tensors if t.name == key)
    shape = tuple(int(s) for s in reversed(rt.shape))
    ref = oracle_dequantize(rt.data, rt.tensor_type).reshape(shape)
    return torch.from_numpy(np.ascontiguousarray(ref))


class TestEndToEndParity:
    @pytest.mark.parametrize("qtype,rtol,atol_scale", [
        ("Q8_0", 1e-4, 1e-4),
        ("Q4_K", None, 3e-3),
    ])
    def test_resident_matches_oracle_dense(self, make_gguf_file, qtype, rtol,
                                           atol_scale):
        model = _load_resident(make_gguf_file, tag=f"parity-{qtype}", qtype=qtype)
        res = model.model.language_model.layers[0].self_attn.q_proj
        assert isinstance(res, GGUFLinear)

        w_ref = _oracle_dense_weight(
            make_gguf_file(stub_vv_gguf_spec(qtype=qtype, **_dims_for(qtype)),
                           tag=f"parity-ref-{qtype}"),
            "model.language_model.layers.0.self_attn.q_proj.weight",
        )

        torch.manual_seed(11)
        x = torch.randn(4, res.in_features)
        y = res(x)
        y_ref = torch.nn.functional.linear(x, w_ref)

        # Deterministic CPU math + bitwise-equal kernels -> near-bitwise for
        # Q8_0; tolerance scales with |y| for K-quants.
        atol = atol_scale * max(1.0, float(y_ref.norm()))
        close = torch.allclose(y, y_ref, rtol=rtol or 0.0, atol=atol)
        if not close:
            rel = float((y - y_ref).norm() / y_ref.norm())
            pytest.fail(f"{qtype}: relative output error {rel:.5f}")
        assert torch.isfinite(y).all()

    def test_convrot_module_contract_on_eager(self):
        """ConvRot executes through kitchen on eager (CPU) — see
        test_convrot_module for why a CPU float reference is not derivable."""
        pytest.importorskip("comfy_kitchen")
        from ComfyUI_VibeVoice.modules.convrot_quant import ConvRotInt8Linear

        layer = ConvRotInt8Linear(64, 32, bias=False, group_size=16)
        layer.weight.data = torch.randint(-100, 100, (32, 64), dtype=torch.int8)
        layer.weight_scale.data = torch.full((32, 1), 0.01)
        y = layer(torch.randn(2, 64))
        assert y.shape == (2, 32)
        assert torch.isfinite(y).all()
