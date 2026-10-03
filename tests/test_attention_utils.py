"""Tests for modules/attention_utils.py - Attention mode detection and resolution."""

import torch
import pytest
from unittest.mock import patch

from ComfyUI_VibeVoice.modules.attention_utils import (
    ATTENTION_MODES,
    SAGE_ATTENTION_AVAILABLE,
    SAGE_SUPPORTED_ARCHS,
    ASR_ATTENTION_FALLBACK,
    ASR_EXCLUDED_ATTENTION_MODES,
    check_sage_attention_compatible,
    check_flash_attention_available,
    check_dtype_attention_compatible,
    get_available_attention_modes,
    resolve_attention_mode,
    resolve_asr_attention_mode,
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
        # `sys.modules[name] = None` makes `import name` raise ImportError,
        # which is exactly what the probe sees on a machine without
        # flash-attn. Popping the entry does NOT work: the package is on disk
        # and already imported here, so the import statement resolves it again
        # — which is why this test used to pass or fail on whatever the
        # embedded environment happens to have installed.
        with patch.dict(sys.modules, {"flash_attn": None}), \
             patch("torch.cuda.is_available", return_value=True):
            assert check_flash_attention_available() is False

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


class TestSageArchSetMatchesSage:
    """SAGE_SUPPORTED_ARCHS must be the set sageattention 2.2.0 actually
    dispatches, not "compute capability major >= 8".

    The old check admitted sm100/sm103 (Blackwell datacenter, CC 10.x): the
    project's ``arch_code >= 90`` branch then selected the SM90 kernel, and
    sage's own ``sageattn()`` has no sm100/sm103 branch at all — it raises
    ``ValueError: Unsupported CUDA architecture``.
    """

    def test_supported_set_is_exactly_sages_dispatch(self):
        assert SAGE_SUPPORTED_ARCHS == frozenset(
            {"sm80", "sm86", "sm89", "sm90", "sm120"}
        ), (
            "sageattention 2.2.0's core.sageattn branches on sm80/sm86, sm75 "
            "(triton), sm89, sm90, sm120 and raises ValueError otherwise; this "
            "project pins that set minus sm75. If a future sage adds an arch, "
            "update this set and "
            "get_sage_attention_function_and_params together"
        )

    @staticmethod
    def _capability(arch):
        """`smXY` -> torch's (major, minor). sm120 is major 12, minor 0 —
        splitting the string the obvious way would read it as (1, 20)."""
        code = int(arch[2:])
        return code // 10, code % 10

    @pytest.mark.parametrize("arch", sorted(SAGE_SUPPORTED_ARCHS))
    def test_supported_arch_is_accepted(self, arch):
        major, minor = self._capability(arch)
        with patch("ComfyUI_VibeVoice.modules.attention_utils.SAGE_ATTENTION_AVAILABLE", True), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_capability", return_value=(major, minor)):
            assert check_sage_attention_compatible() is True

    @pytest.mark.parametrize(
        "arch", ["sm75", "sm87", "sm88", "sm100", "sm103", "sm110"]
    )
    def test_unsupported_arch_is_rejected(self, arch):
        major, minor = self._capability(arch)
        with patch("ComfyUI_VibeVoice.modules.attention_utils.SAGE_ATTENTION_AVAILABLE", True), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_capability", return_value=(major, minor)):
            assert check_sage_attention_compatible() is False, (
                f"{arch} is not in the sage dispatch table; admitting it would "
                f"route the SM90 kernel onto silicon sage does not support"
            )


class TestResolveAttentionModeHonoursSageAvailability:
    """A saved workflow carries the mode string verbatim; the dropdown is
    gated on get_available_attention_modes() at build time. Before this fix
    resolve_attention_mode() re-checked flash but never sage, so a workflow
    naming "sage" on a machine without it survived every guard and then died
    deep in the loader with RuntimeError("Incompatible hardware/setup").
    """

    _WITH_SAGE = ["eager", "sdpa", "flash_attention_2", "sage"]

    def test_sage_falls_back_when_sageattention_is_missing(self):
        with patch("ComfyUI_VibeVoice.modules.attention_utils.check_sage_attention_compatible",
                   return_value=False), \
             patch("ComfyUI_VibeVoice.modules.attention_utils.ATTENTION_MODES",
                   self._WITH_SAGE):
            assert resolve_attention_mode("sage") == "sdpa"

    def test_sage_falls_back_on_unsupported_arch(self):
        with patch("ComfyUI_VibeVoice.modules.attention_utils.SAGE_ATTENTION_AVAILABLE", True), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_capability", return_value=(10, 0)), \
             patch("ComfyUI_VibeVoice.modules.attention_utils.ATTENTION_MODES",
                   self._WITH_SAGE):
            assert resolve_attention_mode("sage") == "sdpa"

    def test_sage_is_kept_when_actually_usable(self):
        with patch("ComfyUI_VibeVoice.modules.attention_utils.check_sage_attention_compatible",
                   return_value=True), \
             patch("ComfyUI_VibeVoice.modules.attention_utils.ATTENTION_MODES",
                   self._WITH_SAGE):
            assert resolve_attention_mode("sage") == "sage"

    def test_fallback_warns_with_the_reason(self, caplog):
        with patch("ComfyUI_VibeVoice.modules.attention_utils.check_sage_attention_compatible",
                   return_value=False), \
             patch("ComfyUI_VibeVoice.modules.attention_utils.ATTENTION_MODES",
                   self._WITH_SAGE), \
             caplog.at_level("WARNING"):
            resolve_attention_mode("sage")
        assert "sage" in caplog.text
        assert "falling back" in caplog.text

    def test_flash_fallback_is_unchanged(self):
        """The pre-existing flash rule must stay byte-for-byte compatible."""
        with patch("ComfyUI_VibeVoice.modules.attention_utils.check_flash_attention_available",
                   return_value=False):
            assert resolve_attention_mode("flash_attention_2") == "sdpa"

    def test_4bit_eager_and_flash_still_go_to_sdpa(self):
        assert resolve_attention_mode("eager", quantize_4bit=True) == "sdpa"
        with patch("ComfyUI_VibeVoice.modules.attention_utils.check_flash_attention_available",
                   return_value=True):
            assert resolve_attention_mode("flash_attention_2", quantize_4bit=True) == "sdpa"

    def test_4bit_sage_still_goes_to_sage(self):
        with patch("ComfyUI_VibeVoice.modules.attention_utils.check_sage_attention_compatible",
                   return_value=True), \
             patch("ComfyUI_VibeVoice.modules.attention_utils.ATTENTION_MODES",
                   self._WITH_SAGE):
            assert resolve_attention_mode("sage", quantize_4bit=True) == "sage"

    def test_unknown_mode_still_falls_back_to_eager(self):
        """The new branch must not swallow the trailing unknown-mode rule."""
        with patch("ComfyUI_VibeVoice.modules.attention_utils.check_sage_attention_compatible",
                   return_value=True), \
             patch("ComfyUI_VibeVoice.modules.attention_utils.ATTENTION_MODES",
                   self._WITH_SAGE):
            assert resolve_attention_mode("sage_attn") == "eager"


class TestASRExcludesSage:
    """The ASR processor left-pads every batch to the longest utterance, so a
    prefill step hands the decoder a real additive (B,1,S,S) mask. sageattn has
    no attn_mask parameter, and the kernel path used the mask only to pick
    is_causal and then dropped it — every query attended to the pad columns.
    """

    def test_sage_is_downgraded_on_the_asr_path(self):
        assert resolve_asr_attention_mode("sage") == ASR_ATTENTION_FALLBACK

    def test_the_registry_carries_the_reason(self):
        # The registry holds the one-line user-facing cause; the full kernel
        # analysis stays in the module comment above it.
        assert "sage" in ASR_EXCLUDED_ATTENTION_MODES
        assert "mask" in ASR_EXCLUDED_ATTENTION_MODES["sage"].lower()

    def test_downgrade_warns(self, caplog):
        with caplog.at_level("WARNING"):
            resolve_asr_attention_mode("sage")
        assert "sage" in caplog.text
        assert "ASR" in caplog.text

    @pytest.mark.parametrize("mode", ["eager", "sdpa", "flash_attention_2"])
    def test_other_modes_pass_through_unchanged(self, mode):
        assert resolve_asr_attention_mode(mode) == mode

    def test_realtime_policy_is_separate_and_still_in_place(self):
        """Excluding sage from ASR must not quietly widen into a repo-wide
        removal: the TTS family still uses it."""
        from ComfyUI_VibeVoice.modules.attention_utils import (
            REALTIME_EXCLUDED_ATTENTION_MODES,
            resolve_realtime_attention_mode,
        )
        assert ASR_ATTENTION_FALLBACK == "sdpa"
        assert resolve_realtime_attention_mode("sage") == "sdpa"
        assert set(REALTIME_EXCLUDED_ATTENTION_MODES) == {"sage"}
        # resolve_attention_mode (the TTS path) is NOT ASR-aware.
        with patch("ComfyUI_VibeVoice.modules.attention_utils.check_sage_attention_compatible",
                   return_value=True), \
             patch("ComfyUI_VibeVoice.modules.attention_utils.ATTENTION_MODES",
                   ["eager", "sdpa", "flash_attention_2", "sage"]):
            assert resolve_attention_mode("sage") == "sage"


class TestDtypeAttentionCrossCheck:
    """The sage kernels hard-assert fp16/bf16 and resolve_sage_target_dtype
    returns the stored weight dtype for a plain float linear, so
    dtype='fp32' + attention_mode='sage' crashed inside the kernel. The two
    widgets are independent inputs with no cross-check anywhere."""

    def test_fp32_with_sage_is_rejected(self):
        message = check_dtype_attention_compatible("fp32", "sage")
        assert message is not None
        assert "fp32" in message and "sage" in message
        # Actionable: it must name both ways out.
        assert "bf16" in message and "sdpa" in message

    @pytest.mark.parametrize("mode", ["eager", "sdpa", "flash_attention_2", None])
    def test_fp32_is_fine_for_every_other_backend(self, mode):
        assert check_dtype_attention_compatible("fp32", mode) is None

    @pytest.mark.parametrize("dtype_str", ["auto", "bf16", "fp16"])
    def test_sage_accepts_the_half_dtypes(self, dtype_str):
        assert check_dtype_attention_compatible(dtype_str, "sage") is None

    def test_4bit_exempts_fp32(self):
        """4-bit forces an fp32 bnb compute dtype in the loader, but every
        quantized linear carries a quant_state so sage still reads bf16."""
        assert check_dtype_attention_compatible(
            "fp32", "sage", quantized_4bit=True
        ) is None
