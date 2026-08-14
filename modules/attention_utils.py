"""Attention mode utilities for VibeVoice nodes.

Provides attention mode detection, GPU capability checks, and resolution
of attention modes based on hardware and quantization settings.
"""

import torch
import logging

logger = logging.getLogger(__name__)

# Try to import sageattention
try:
    import sageattention
    SAGE_ATTENTION_AVAILABLE = True
except ImportError:
    SAGE_ATTENTION_AVAILABLE = False

# Base attention modes always available
ATTENTION_MODES = ["eager", "sdpa", "flash_attention_2"]

# Add sage if available
if SAGE_ATTENTION_AVAILABLE:
    ATTENTION_MODES.append("sage")


def check_sage_attention_compatible() -> bool:
    """Check if the current GPU supports SageAttention.

    SageAttention requires CUDA with compute capability >= 8.0.

    Returns:
        True if SageAttention can be used, False otherwise.
    """
    if not SAGE_ATTENTION_AVAILABLE:
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        logger.warning(
            f"Your GPU (compute capability {major}.x) does not support "
            f"SageAttention, which requires CC 8.0+. Sage option will be disabled."
        )
        return False
    return True


def check_flash_attention_available() -> bool:
    """Check if Flash Attention 2 is usable on this hardware.

    Flash Attention 2 requires the ``flash_attn`` package and a CUDA GPU.
    When either is missing, the mode would fail at model load time, so we hide
    it from the available options and fall back gracefully when requested.

    Returns:
        True if flash_attention_2 can be used, False otherwise.
    """
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available()


def get_available_attention_modes() -> list[str]:
    """Get list of attention modes available on this hardware.

    Returns:
        List of attention mode strings. Always includes "eager" and "sdpa".
    """
    modes = ["eager", "sdpa"]
    # Only offer flash_attention_2 when the flash-attn package and a CUDA GPU are
    # actually present; otherwise selecting it fails deep in the loader.
    if check_flash_attention_available():
        modes.append("flash_attention_2")
    if check_sage_attention_compatible():
        modes.append("sage")
    return modes


def resolve_attention_mode(
    requested_mode: str,
    quantize_4bit: bool = False,
) -> str:
    """Resolve the effective attention mode based on hardware and quantization.

    Applies fallback logic:
    - 4-bit quantization + eager/flash → sdpa (for stability)
    - Unknown mode → eager

    Args:
        requested_mode: The attention mode requested by the user.
        quantize_4bit: Whether 4-bit quantization is enabled.

    Returns:
        The resolved attention mode string.
    """
    mode = requested_mode

    if quantize_4bit and mode in ["eager", "flash_attention_2"]:
        logger.warning(
            f"Attention mode '{mode}' is not recommended with 4-bit quantization. "
            f"Falling back to 'sdpa' for stability and performance."
        )
        mode = "sdpa"

    if mode == "flash_attention_2" and not check_flash_attention_available():
        logger.warning(
            f"flash_attention_2 is not available on this hardware; "
            f"falling back to 'sdpa'."
        )
        mode = "sdpa"

    if mode not in ATTENTION_MODES:
        logger.warning(f"Unknown attention mode '{mode}', falling back to eager")
        mode = "eager"

    return mode


def get_attn_implementation_for_load(attention_mode: str) -> str:
    """Get the attn_implementation string for model loading.

    SageAttention is applied post-load via patching, so during loading
    we use "sdpa" as the implementation.

    Args:
        attention_mode: The resolved attention mode.

    Returns:
        The attn_implementation string for from_pretrained().
    """
    if attention_mode == "sage":
        return "sdpa"
    return attention_mode
