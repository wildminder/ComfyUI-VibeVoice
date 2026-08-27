"""Unit tests for the FP8-resident linear runtime (plan 2026-08-27, Phase 1).

Parity references are the MANUAL dequant formula the kitchen eager backend
implements — ``(w.float() * scale).to(out_dtype)`` — verified bit-exact
against ``comfy_kitchen.dequantize_per_tensor_fp8`` on this box. All tests
run on CPU (eager backend); no GPU required.
"""

import json
import sys
import types
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ComfyUI_VibeVoice.modules.fp8_quant import (
    FP8Linear,
    make_fp8_linear,
    probe_fp8_backend,
)
from ComfyUI_VibeVoice.modules.convrot_quant import QuantLayerInfo


def _fill(m, seed=0, scale=2.0, bias_val=None):
    """Install deterministic fp8 storage + scalar scale into a fresh module."""
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(m.out_features, m.in_features, generator=g) * 0.3).to(m.fp8_dtype)
    m.weight.data.copy_(w)
    m.weight_scale.data.copy_(torch.tensor(scale, dtype=torch.float32))
    if m.bias is not None:
        b = torch.randn(m.out_features, generator=g) * 0.1
        if bias_val is not None:
            b = torch.full((m.out_features,), bias_val)
        m.bias.data.copy_(b)
    return m


def _reference(m, x):
    ref_w = (m.weight.float() * m.weight_scale).to(x.dtype)
    bias = m.bias.to(x.dtype) if m.bias is not None else None
    return F.linear(x, ref_w, bias)


class TestProbeFP8Backend:
    def test_returns_backend_on_this_box(self):
        assert probe_fp8_backend() in ("triton", "cuda", "eager")

    def test_none_when_kitchen_missing(self):
        with patch.dict(sys.modules, {"comfy_kitchen": None}):
            assert probe_fp8_backend() is None

    def test_none_when_callable_missing_despite_capability(self):
        # D-3 trap guard: capability strings have shipped without callables.
        fake = types.ModuleType("comfy_kitchen")
        fake.list_backends = lambda: {
            "eager": {"available": True,
                      "capabilities": ["dequantize_per_tensor_fp8"]}
        }
        with patch.dict(sys.modules, {"comfy_kitchen": fake}):
            assert probe_fp8_backend() is None

    def test_none_when_no_backend_available(self):
        import comfy_kitchen

        fake = types.ModuleType("comfy_kitchen")
        fake.dequantize_per_tensor_fp8 = comfy_kitchen.dequantize_per_tensor_fp8
        fake.list_backends = lambda: {
            "eager": {"available": False,
                      "capabilities": ["dequantize_per_tensor_fp8"]}
        }
        with patch.dict(sys.modules, {"comfy_kitchen": fake}):
            assert probe_fp8_backend() is None


class TestFP8LinearConstruction:
    def test_rejects_non_fp8_dtype(self):
        with pytest.raises(ValueError):
            FP8Linear(4, 4, False, torch.float16)

    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_storage_dtypes_and_markers(self, fp8_dtype):
        m = FP8Linear(6, 8, True, fp8_dtype)
        assert m.weight.dtype == fp8_dtype
        assert m.weight_scale.dtype == torch.float32
        assert m.weight_scale.shape == ()
        assert m.bias is not None and m.bias.dtype == torch.float32
        assert m._quant_resident is True
        assert m.comfy_cast_weights is True
        assert m.weight_comfy_model_dtype == fp8_dtype

    def test_no_bias(self):
        m = FP8Linear(6, 8, False, torch.float8_e4m3fn)
        assert m.bias is None

    def test_meta_init_compatible(self):
        with torch.device("meta"):
            m = FP8Linear(6, 8, False, torch.float8_e4m3fn)
        assert m.weight.is_meta
        assert m.weight_scale.is_meta
        assert tuple(m.weight.shape) == (8, 6)


class TestFP8LinearForward:
    @pytest.mark.parametrize("act_dtype", [torch.bfloat16, torch.float32])
    def test_parity_bit_exact(self, act_dtype):
        m = _fill(FP8Linear(6, 8, False, torch.float8_e4m3fn), seed=1)
        x = torch.randn(3, 6, dtype=act_dtype)
        out = m(x)
        assert out.dtype == act_dtype
        assert torch.equal(out, _reference(m, x))

    def test_parity_e5m2(self):
        m = _fill(FP8Linear(6, 8, False, torch.float8_e5m2), seed=2)
        x = torch.randn(3, 6, dtype=torch.bfloat16)
        assert torch.equal(m(x), _reference(m, x))

    def test_parity_with_bias(self):
        m = _fill(FP8Linear(6, 8, True, torch.float8_e4m3fn), seed=3)
        x = torch.randn(2, 5, 6, dtype=torch.bfloat16)  # batched 3-D input
        assert torch.equal(m(x), _reference(m, x))

    def test_non_float_activation_raises(self):
        m = _fill(FP8Linear(6, 8, False, torch.float8_e4m3fn))
        with pytest.raises(TypeError):
            m(torch.randint(0, 4, (2, 6)))

    def test_streamed_path_parity(self):
        """A non-empty weight_function forces _forward_streamed; the pulled
        weight must stay fp8 (dtype-preserving) and the result must match."""
        m = _fill(FP8Linear(6, 8, True, torch.float8_e4m3fn), seed=4)
        seen = []

        def _identity(w):
            seen.append(w.dtype)
            return w

        m.weight_function = [_identity]
        x = torch.randn(3, 6, dtype=torch.bfloat16)
        assert torch.equal(m(x), _reference(m, x))
        assert seen == [torch.float8_e4m3fn], "streamed pull must preserve fp8 dtype"
        m.weight_function = []


class TestFP8LinearStateDict:
    def test_load_consumes_comfy_quant_meta(self):
        m = FP8Linear(6, 8, False, torch.float8_e4m3fn)
        g = torch.Generator().manual_seed(5)
        w = (torch.randn(8, 6, generator=g) * 0.3).to(torch.float8_e4m3fn)
        meta = json.dumps(
            {"format": "float8_e4m3fn", "orig_dtype": "torch.bfloat16"}
        ).encode()
        sd = {
            "weight": w,
            "weight_scale": torch.tensor(1.5),
            "comfy_quant": torch.frombuffer(bytearray(meta), dtype=torch.uint8),
        }
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert list(missing) == []
        assert list(unexpected) == []
        assert torch.equal(m.weight, w)
        assert m.weight_scale.item() == pytest.approx(1.5)

    def test_state_dict_roundtrip_keeps_storage_dtypes(self):
        m = _fill(FP8Linear(6, 8, True, torch.float8_e4m3fn), seed=6)
        sd = m.state_dict()
        assert set(sd.keys()) == {"weight", "weight_scale", "bias"}
        assert sd["weight"].dtype == torch.float8_e4m3fn
        assert sd["weight_scale"].dtype == torch.float32


class TestMakeFP8Linear:
    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_factory_builds_matching_module(self, fp8_dtype):
        info = QuantLayerInfo(
            prefix="model.language_model.layers.0.self_attn.q_proj",
            group_size=0,
            in_features=6,
            out_features=8,
            convrot=False,
            rowwise_dtype=fp8_dtype,
        )
        mod = make_fp8_linear(info)(6, 8, True)
        assert isinstance(mod, FP8Linear)
        assert mod.fp8_dtype == fp8_dtype
        assert mod.bias is not None
