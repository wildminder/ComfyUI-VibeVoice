"""Tests for modules/dtype_utils.py - Dtype resolution and casting."""

import torch
import pytest
from unittest.mock import patch, MagicMock

from ComfyUI_VibeVoice.modules.dtype_utils import (
    DTYPE_AUTO,
    DTYPE_BF16,
    DTYPE_FP16,
    DTYPE_FP32,
    get_dtype_options,
    resolve_dtype,
    get_dtype_str,
    cast_model_to_dtype,
    cast_model_to_dtype_if_needed,
)


class TestGetDtypeOptions:
    """Test get_dtype_options function."""

    def test_contains_auto(self):
        options = get_dtype_options()
        assert DTYPE_AUTO in options

    def test_contains_bf16(self):
        options = get_dtype_options()
        assert DTYPE_BF16 in options

    def test_contains_fp16(self):
        options = get_dtype_options()
        assert DTYPE_FP16 in options

    def test_contains_fp32(self):
        options = get_dtype_options()
        assert DTYPE_FP32 in options

    def test_auto_is_first(self):
        options = get_dtype_options()
        assert options[0] == DTYPE_AUTO


class TestResolveDtype:
    """Test resolve_dtype function."""

    def test_resolve_explicit_bf16(self):
        dtype = resolve_dtype(DTYPE_BF16)
        assert dtype == torch.bfloat16

    def test_resolve_explicit_fp16(self):
        dtype = resolve_dtype(DTYPE_FP16)
        assert dtype == torch.float16

    def test_resolve_explicit_fp32(self):
        dtype = resolve_dtype(DTYPE_FP32)
        assert dtype == torch.float32

    def test_resolve_auto_bf16(self):
        with patch("ComfyUI_VibeVoice.modules.dtype_utils._get_model_management") as mock_mm:
            mock_mm.return_value.get_torch_device.return_value = torch.device("cpu")
            with patch("ComfyUI_VibeVoice.modules.dtype_utils.should_use_bf16", return_value=True):
                dtype = resolve_dtype(DTYPE_AUTO)
                assert dtype == torch.bfloat16

    def test_resolve_auto_fp16(self):
        with patch("ComfyUI_VibeVoice.modules.dtype_utils._get_model_management") as mock_mm:
            mock_mm.return_value.get_torch_device.return_value = torch.device("cpu")
            with patch("ComfyUI_VibeVoice.modules.dtype_utils.should_use_bf16", return_value=False), \
                 patch("ComfyUI_VibeVoice.modules.dtype_utils.should_use_fp16", return_value=True):
                dtype = resolve_dtype(DTYPE_AUTO)
                assert dtype == torch.float16

    def test_resolve_auto_fp32(self):
        with patch("ComfyUI_VibeVoice.modules.dtype_utils._get_model_management") as mock_mm:
            mock_mm.return_value.get_torch_device.return_value = torch.device("cpu")
            with patch("ComfyUI_VibeVoice.modules.dtype_utils.should_use_bf16", return_value=False), \
                 patch("ComfyUI_VibeVoice.modules.dtype_utils.should_use_fp16", return_value=False):
                dtype = resolve_dtype(DTYPE_AUTO)
                assert dtype == torch.float32

    def test_resolve_none_returns_auto(self):
        with patch("ComfyUI_VibeVoice.modules.dtype_utils._get_model_management") as mock_mm:
            mock_mm.return_value.get_torch_device.return_value = torch.device("cpu")
            with patch("ComfyUI_VibeVoice.modules.dtype_utils.should_use_bf16", return_value=False), \
                 patch("ComfyUI_VibeVoice.modules.dtype_utils.should_use_fp16", return_value=False):
                dtype = resolve_dtype(None)
                assert dtype == torch.float32

    def test_resolve_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            resolve_dtype("invalid_dtype")


class TestGetDtypeStr:
    """Test get_dtype_str function."""

    def test_bf16_str(self):
        assert get_dtype_str(torch.bfloat16) == DTYPE_BF16

    def test_fp16_str(self):
        assert get_dtype_str(torch.float16) == DTYPE_FP16

    def test_fp32_str(self):
        assert get_dtype_str(torch.float32) == DTYPE_FP32

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown torch.dtype"):
            get_dtype_str(torch.int64)


class TestCastModelToDtype:
    """Test cast_model_to_dtype function."""

    def test_cast_floats_only(self):
        """cast_model_to_dtype casts floating params; raw uint8/int8 storage
        (quant residents) must never be converted (plan 2026-08-24, E1)."""
        class _M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.float_lin = torch.nn.Linear(4, 4)
                self.raw = torch.nn.Parameter(
                    torch.zeros(7, dtype=torch.uint8), requires_grad=False
                )

        model = _M()
        model.raw._quant_resident = True  # mark via param attr is not used; module marker below
        cast_model_to_dtype(model, torch.float16)
        assert model.float_lin.weight.dtype == torch.float16
        assert model.raw.dtype == torch.uint8, "raw bytes recast!"

    def test_cast_none_dtype_noop(self):
        model = MagicMock()
        cast_model_to_dtype(model, None)
        model.to.assert_not_called()

    def test_cast_protects_quant_resident_module(self):
        from ComfyUI_VibeVoice.modules.gguf_quant import GGUFLinear

        class _M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 4)
                from gguf.constants import GGMLQuantizationType as _T
                self.res = GGUFLinear(8, 8, bias=True,
                                      ggml_type=_T.Q8_0)
                self.res.set_raw_weight(torch.full((self.res.weight.numel(),),
                                                   17, dtype=torch.uint8))
                self.res.bias.data = torch.ones(8, dtype=torch.float32)

        model = _M()
        cast_model_to_dtype(model, torch.float16)
        # Resident weight stays raw uint8; its bias follows model dtype.
        assert model.res.weight.dtype == torch.uint8
        assert model.res.bias.dtype == torch.float16
        assert model.lin.weight.dtype == torch.float16


# ====================================================================
# Plan 2026-08-18, D4/RC-3: conditional cast (skip when already matching)
# ====================================================================
class _TinyModel(torch.nn.Module):
    """Minimal real module for cast tests."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(4, 4)
        self.b = torch.nn.Linear(4, 4)


class _TiedModel(torch.nn.Module):
    """Module with tied input/output embeddings and a tie_weights() hook."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.config = MagicMock()
        self.config.tie_word_embeddings = True
        self.config.decoder_config = None
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.embed.weight


class TestCastModelToDtypeIfNeeded:
    """Test cast_model_to_dtype_if_needed (plan 2026-08-18, D4/RC-3)."""

    def test_cast_skipped_when_all_match(self):
        """All params already at target dtype → fast path, no tensor replaced.

        The contract is storage stability: every parameter keeps its exact
        data_ptr (no cast pass ran, no new tensors allocated — RC-3).
        """
        model = _TinyModel().to(torch.bfloat16)
        ptrs_before = {n: p.data_ptr() for n, p in model.named_parameters()}

        cast_model_to_dtype_if_needed(model, torch.bfloat16)

        ptrs_after = {n: p.data_ptr() for n, p in model.named_parameters()}
        assert ptrs_after == ptrs_before, "fast path must not replace any storage"
        assert all(p.dtype == torch.bfloat16 for p in model.parameters())

    def test_cast_only_mismatched_params(self):
        """Mixed dtypes → only mismatched params are cast, others untouched."""
        model = _TinyModel()
        model.a = model.a.to(torch.bfloat16)  # a: bf16, b: fp32
        ptr_b_before = model.b.weight.data_ptr()

        ptr_a_before = model.a.weight.data_ptr()

        cast_model_to_dtype_if_needed(model, torch.bfloat16)

        assert model.a.weight.dtype == torch.bfloat16
        assert model.b.weight.dtype == torch.bfloat16
        # The already-matching param must keep its original storage.
        assert model.a.weight.data_ptr() == ptr_a_before
        # The mismatched param was replaced with a new cast tensor.
        assert model.b.weight.data_ptr() != ptr_b_before

    def test_cast_preserves_tying(self):
        """Casting a tied model re-ties the pair (shared data_ptr after cast)."""
        model = _TiedModel()  # fp32
        assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()

        cast_model_to_dtype_if_needed(model, torch.bfloat16)

        assert model.embed.weight.dtype == torch.bfloat16
        assert model.lm_head.weight.dtype == torch.bfloat16
        assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr(), (
            "tied pair must share storage after the conditional cast")

    def test_cast_none_dtype_noop(self):
        model = _TinyModel()
        ptr_before = model.a.weight.data_ptr()
        cast_model_to_dtype_if_needed(model, None)
        assert model.a.weight.data_ptr() == ptr_before
        assert model.a.weight.dtype == torch.float32
