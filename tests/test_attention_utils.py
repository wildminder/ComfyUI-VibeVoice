"""Tests for modules/attention_utils.py - Attention mode detection and resolution."""

import torch
import pytest
from unittest.mock import patch

from ComfyUI_VibeVoice.modules.attention_utils import (
    ATTENTION_MODES,
    SAGE_ATTENTION_AVAILABLE,
    check_sage_attention_compatible,
    check_flash_attention_available,
    get_available_attention_modes,
    resolve_attention_mode,
    get_attn_implementation_for_load,
)


class TestAttentionModes:
    """Test ATTENTION_MODES constant."""

    def test_contains_eager(self):
        assert "eager" in ATTENTION_MODES

    def test_contains_sdpa(self):
        assert "sdpa" in ATTENTION_MODES

    def test_contains_flash(self):
        assert "flash_attention_2" in ATTENTION_MODES


class TestCheckFlashAttentionAvailable:
    """IMP-001: flash_attention_2 availability probe."""

    def test_flash_available_true(self):
        import sys
        fake_flash = type(sys)("flash_attn")
        with patch.dict(sys.modules, {"flash_attn": fake_flash}), \
             patch("torch.cuda.is_available", return_value=True):
            assert check_flash_attention_available() is True

    def test_flash_available_false_no_module(self):
        import sys
        saved = sys.modules.pop("flash_attn", None)
        try:
            with patch("torch.cuda.is_available", return_value=True):
                assert check_flash_attention_available() is False
        finally:
            if saved is not None:
                sys.modules["flash_attn"] = saved

    def test_flash_available_false_no_cuda(self):
        import sys
        fake_flash = type(sys)("flash_attn")
        with patch.dict(sys.modules, {"flash_attn": fake_flash}), \
             patch("torch.cuda.is_available", return_value=False):
            assert check_flash_attention_available() is False


class TestCheckSageAttentionCompatible:
    """Test check_sage_attention_compatible function."""

    def test_no_sage_module(self):
        with patch("ComfyUI_VibeVoice.modules.attention_utils.SAGE_ATTENTION_AVAILABLE", False):
            assert check_sage_attention_compatible() is False

    def test_no_cuda(self):
        with patch("ComfyUI_VibeVoice.modules.attention_utils.SAGE_ATTENTION_AVAILABLE", True), \
             patch("torch.cuda.is_available", return_value=False):
            assert check_sage_attention_compatible() is False

    def test_low_compute_capability(self):
        with patch("ComfyUI_VibeVoice.modules.attention_utils.SAGE_ATTENTION_AVAILABLE", True), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_capability", return_value=(7, 5)):
            assert check_sage_attention_compatible() is False

    def test_high_compute_capability(self):
        with patch("ComfyUI_VibeVoice.modules.attention_utils.SAGE_ATTENTION_AVAILABLE", True), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_capability", return_value=(8, 6)):
            assert check_sage_attention_compatible() is True

    def test_compute_capability_9(self):
        with patch("ComfyUI_VibeVoice.modules.attention_utils.SAGE_ATTENTION_AVAILABLE", True), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_capability", return_value=(9, 0)):
            assert check_sage_attention_compatible() is True


class TestGetAvailableAttentionModes:
    """Test get_available_attention_modes function."""

    def test_always_includes_eager(self):
        modes = get_available_attention_modes()
        assert "eager" in modes

    def test_always_includes_sdpa(self):
        modes = get_available_attention_modes()
        assert "sdpa" in modes

    def test_always_includes_flash_when_available(self):
        with patch("ComfyUI_VibeVoice.modules.attention_utils.check_flash_attention_available", return_value=True):
            modes = get_available_attention_modes()
        assert "flash_attention_2" in modes

    def test_includes_sage_when_compatible(self):
        with patch("ComfyUI_VibeVoice.modules.attention_utils.check_sage_attention_compatible", return_value=True):
            modes = get_available_attention_modes()
            assert "sage" in modes

    def test_excludes_sage_when_incompatible(self):
        with patch("ComfyUI_VibeVoice.modules.attention_utils.check_sage_attention_compatible", return_value=False):
            modes = get_available_attention_modes()
            assert "sage" not in modes

    def test_excludes_flash_when_unavailable(self):
        with patch("ComfyUI_VibeVoice.modules.attention_utils.check_flash_attention_available", return_value=False), \
             patch("ComfyUI_VibeVoice.modules.attention_utils.check_sage_attention_compatible", return_value=False):
            modes = get_available_attention_modes()
        assert "flash_attention_2" not in modes
        assert "eager" in modes
        assert "sdpa" in modes

    def test_includes_flash_when_available_flag(self):
        with patch("ComfyUI_VibeVoice.modules.attention_utils.check_flash_attention_available", return_value=True), \
             patch("ComfyUI_VibeVoice.modules.attention_utils.check_sage_attention_compatible", return_value=False):
            modes = get_available_attention_modes()
        assert "flash_attention_2" in modes


class TestResolveAttentionMode:
    """Test resolve_attention_mode function."""

    def test_sdpa_unchanged(self):
        assert resolve_attention_mode("sdpa") == "sdpa"

    def test_eager_unchanged(self):
        assert resolve_attention_mode("eager") == "eager"

    def test_flash_unchanged_when_available(self):
        with patch("ComfyUI_VibeVoice.modules.attention_utils.check_flash_attention_available", return_value=True):
            assert resolve_attention_mode("flash_attention_2") == "flash_attention_2"

    def test_flash_falls_back_when_unavailable(self):
        with patch("ComfyUI_VibeVoice.modules.attention_utils.check_flash_attention_available", return_value=False):
            assert resolve_attention_mode("flash_attention_2") == "sdpa"

    def test_4bit_eager_fallback_to_sdpa(self):
        assert resolve_attention_mode("eager", quantize_4bit=True) == "sdpa"

    def test_4bit_flash_fallback_to_sdpa(self):
        assert resolve_attention_mode("flash_attention_2", quantize_4bit=True) == "sdpa"

    def test_4bit_sdpa_unchanged(self):
        assert resolve_attention_mode("sdpa", quantize_4bit=True) == "sdpa"

    def test_4bit_sage_unchanged(self):
        with patch("ComfyUI_VibeVoice.modules.attention_utils.ATTENTION_MODES", ["eager", "sdpa", "flash_attention_2", "sage"]):
            assert resolve_attention_mode("sage", quantize_4bit=True) == "sage"

    def test_unknown_mode_fallback_to_eager(self):
        assert resolve_attention_mode("unknown_mode") == "eager"


class TestGetAttnImplementationForLoad:
    """Test get_attn_implementation_for_load function."""

    def test_eager(self):
        assert get_attn_implementation_for_load("eager") == "eager"

    def test_sdpa(self):
        assert get_attn_implementation_for_load("sdpa") == "sdpa"

    def test_flash(self):
        assert get_attn_implementation_for_load("flash_attention_2") == "flash_attention_2"

    def test_sage_returns_sdpa(self):
        """Sage is applied post-load, so loading uses sdpa."""
        assert get_attn_implementation_for_load("sage") == "sdpa"
