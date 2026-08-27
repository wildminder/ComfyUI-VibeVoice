"""Phase 3 tests: quant residents under comfy streaming.

The dtype pin (weight_comfy_model_dtype) and the streamed forwards must
preserve RAW storage bit-exactly while pulling offloaded weights back.
"""

import pytest
import torch

import numpy as np
from gguf.constants import GGMLQuantizationType as T
from gguf.quants import dequantize as oracle_dequantize, quantize as oracle_quantize

from ComfyUI_VibeVoice.modules.gguf_quant import GGUFLinear
from ComfyUI_VibeVoice.modules.convrot_quant import ConvRotInt8Linear


def _q8_resident():
    x_np = (np.random.default_rng(3).standard_normal((16, 32)) * 0.05).astype(np.float32)
    raw = oracle_quantize(x_np, T.Q8_0)
    lin = GGUFLinear(32, 16, bias=True, ggml_type=T.Q8_0)
    lin.set_raw_weight(
        torch.from_numpy(np.ascontiguousarray(raw)).view(torch.uint8).reshape(-1)
    )
    lin.bias.data = torch.randn(16) * 0.01
    return lin


class TestGGUFResidentStreaming:
    def test_dtype_pin_and_flags(self):
        lin = _q8_resident()
        assert lin.weight_comfy_model_dtype == torch.uint8
        assert lin.comfy_cast_weights is True
        assert lin.weight_function == []
        assert lin._quant_resident is True

    def test_streamed_forward_bitwise_matches_fast_path(self):
        lin = _q8_resident()
        x = torch.randn(4, 32)

        y_fast = lin(x)

        # Core's strip contract: LowVramPatch-style callable attached.
        pulls = []

        class _FakeLowVramPatch:
            def __init__(self, target):
                self.target = target

            def __call__(self, t):
                pulls.append(t.dtype)
                return t.to(self.target)

        lin.weight_function = [_FakeLowVramPatch(torch.device("cpu"))]
        y_slow = lin(x)
        lin.weight_function = []

        assert pulls == [torch.uint8], "pull must receive RAW storage"
        assert torch.equal(y_fast, y_slow), "streamed path must be bitwise"

    def test_raw_storage_never_recast_to_float(self):
        """Guard against silent uint8->float recasts: drive the REAL
        comfy.ops.cast_bias_weight with a LowVramPatch-style callable and
        spy on what our forward feeds to dequantize_blocks."""
        lin = _q8_resident()
        x = torch.randn(2, 32)
        seen_dtypes = []

        class _FakeLowVramPatch:
            """Mirrors core LowVramPatch contract: pull tensor to the
            activation device WITHOUT touching its dtype."""

            def __init__(self, target):
                self.target = target

            def __call__(self, t):
                return t.to(self.target)

        import ComfyUI_VibeVoice.modules.gguf_quant as G

        real_dequant = G.dequantize_blocks

        def spy_dequant(raw, ggml_type, out_dtype, shape):
            seen_dtypes.append((raw.dtype, out_dtype))
            return real_dequant(raw, ggml_type, out_dtype, shape)

        G.dequantize_blocks = spy_dequant
        try:
            lin.weight_function = [_FakeLowVramPatch(torch.device("cpu"))]
            y = lin(x)
        finally:
            G.dequantize_blocks = real_dequant
            lin.weight_function = []

        assert seen_dtypes == [(torch.uint8, torch.float32)]
        assert torch.isfinite(y).all()

    def test_pull_helper_applies_functions_in_order(self):
        """Contract: attached LowVramPatch-style callables run in order and
        their result is what reaches dequantize (device-move branch itself
        is a plain Tensor.to, covered by the manual GPU gate)."""
        lin = _q8_resident()
        calls = []

        def fn_a(t):
            calls.append(("a", t.dtype))
            return t

        def fn_b(t):
            calls.append(("b", t.dtype))
            return t

        lin.weight_function = [fn_a, fn_b]
        out = lin._pull_to_device(lin.weight, torch.device("cpu"))
        assert out is lin.weight
        assert calls == [("a", torch.uint8), ("b", torch.uint8)]


class TestConvRotResidentStreaming:
    def test_dtype_pin_and_flags(self):
        layer = ConvRotInt8Linear(64, 32, bias=False, group_size=16)
        assert layer.weight_comfy_model_dtype == torch.int8
        assert layer.comfy_cast_weights is True
        assert layer._quant_resident is True

    def test_streamed_forward_runs_and_finite(self):
        pytest.importorskip("comfy_kitchen")

        torch.manual_seed(0)
        layer = ConvRotInt8Linear(64, 32, bias=True, group_size=16)
        layer.weight.data = torch.randint(-100, 100, (32, 64), dtype=torch.int8)
        layer.weight_scale.data = torch.full((32, 1), 0.01)
        layer.bias.data = torch.randn(32) * 0.01

        x = torch.randn(2, 64)
        y_fast = layer(x)

        class _FakeLowVramPatch:
            def __call__(self, t):
                return t.to(x.device)

        layer.weight_function = [_FakeLowVramPatch()]
        y_slow = layer(x)
        layer.weight_function = []

        assert y_fast.shape == y_slow.shape == (2, 32)
        # Same kernels on identical (device-moved) storage: deterministic.
        assert torch.equal(y_fast, y_slow)
