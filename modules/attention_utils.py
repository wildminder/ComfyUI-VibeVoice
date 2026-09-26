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


# ---------------------------------------------------------------------------
# Realtime (streaming) attention policy
# ---------------------------------------------------------------------------
# Measured on VibeVoice-Realtime-0.5B, RTX SM89, bf16, one fixed voice prompt,
# through the node load path (plan 2026-09-26, step S3.2). The compared
# quantity is the conditioning vector of the first text window -- the tensor the
# diffusion head actually consumes -- as cosine/relative-L2 against eager:
#
#   sdpa               cos=0.999962  rel_l2=0.0088
#   flash_attention_2  cos=0.999940  rel_l2=0.0110
#   sage               cos=0.994651  rel_l2=0.1033   <- fails the 0.999 gate
#
# Two independent causes, both measured on the same prompt:
#   1. The sage kernel ignores the additive attention mask
#      (``sage_attention_forward`` sets ``is_causal = attention_mask is None and
#      q_len > 1``). With the realtime loop's 5-token text window over the
#      316-token voice prefill, causal masking lets query i see keys <= 316+i,
#      while sage lets every query see all 321 keys -- a lookahead leak over the
#      rest of the window. Shrinking the query to 1 token (where causal and
#      non-causal coincide) drops sage's error from rel_l2=0.103 to 0.043.
#   2. What remains at q_len=1 is the int8-QK / fp8-PV kernel's own error,
#      still 4-5x the sdpa/flash gap (0.043 vs 0.009).
#
# A backend that diverges is excluded here rather than offered and quietly
# producing different conditioning than every other backend. This is a
# realtime-only decision: the standard TTS family generates without a cached
# prefill window, and the change is visible to users, so the README has to say
# so (plan step S6.1).
REALTIME_ATTENTION_FALLBACK = "sdpa"

REALTIME_EXCLUDED_ATTENTION_MODES: dict[str, str] = {
    "sage": (
        "its conditioning diverges from every other backend (cos=0.9947 vs "
        "eager, gate is 0.999): the sage kernel ignores the attention mask, so "
        "a text window over the voice prefill attends ahead of its own "
        "positions, and its int8/fp8 quantisation adds a further rel_l2=0.043"
    ),
}


def resolve_realtime_attention_mode(attention_mode: str) -> str:
    """Downgrade a backend excluded from the realtime path, with a log line.

    Call this where the model family is known (the loader, the external loader
    node). Excluded backends are replaced by
    :data:`REALTIME_ATTENTION_FALLBACK` and the reason is logged at WARNING
    level, so a user who picked sage sees why the run used sdpa instead.

    Args:
        attention_mode: The resolved attention mode.

    Returns:
        The attention mode to actually use, or ``attention_mode`` unchanged
        when it is not excluded.
    """
    reason = REALTIME_EXCLUDED_ATTENTION_MODES.get(attention_mode)
    if reason is None:
        return attention_mode
    logger.warning(
        "Attention mode '%s' is not used for realtime (VibeVoice-Realtime) "
        "models: %s. Falling back to '%s' for this load. The standard TTS "
        "family is unaffected.",
        attention_mode,
        reason,
        REALTIME_ATTENTION_FALLBACK,
    )
    return REALTIME_ATTENTION_FALLBACK
