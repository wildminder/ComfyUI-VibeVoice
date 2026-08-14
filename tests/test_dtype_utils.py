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

    def test_cast_calls_to(self):
        model = MagicMock()
        cast_model_to_dtype(model, torch.float16)
        model.to.assert_called_once_with(dtype=torch.float16)

    def test_cast_none_dtype_noop(self):
        model = MagicMock()
        cast_model_to_dtype(model, None)
        model.to.assert_not_called()
