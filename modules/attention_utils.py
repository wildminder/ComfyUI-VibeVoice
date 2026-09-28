"""Attention mode utilities for VibeVoice nodes.

Provides attention mode detection, GPU capability checks, and resolution
of attention modes based on hardware and quantization settings.
"""

import torch
import logging
from typing import Optional

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


# CUDA architectures this project is willing to run SageAttention on, pinned as
# a literal set of ``sm<major><minor>`` strings. The spelling is sage's own
# (``sageattention.core.get_cuda_arch_versions``), so the membership test here
# and the exact-``arch_code`` dispatch in
# ``sage_attention_patch.get_sage_attention_function_and_params`` cannot drift
# apart. Deriving it from the installed library instead would couple the check
# to whatever happens to be on disk, which may not be what ships.
#
# Two entries are deliberately absent from sage's own ``sageattn()`` branch set
# (sm80/sm86, sm75, sm89, sm90, sm120, else ValueError):
#   * sm75 -- this project already rejected it (the check it replaces was
#     "compute capability major < 8", which excludes sm75); sage routes it to a
#     Triton kernel, which the vendored dispatcher never calls.
#   * sm100/sm103 (Blackwell datacenter, CC 10.x) -- sage has no branch for
#     them, so the dispatcher must refuse rather than fall through its old
#     "arch_code >= 90 means Hopper" threshold and hand the SM90 kernel to
#     silicon it was not built for.
SAGE_SUPPORTED_ARCHS = frozenset({"sm80", "sm86", "sm89", "sm90", "sm120"})


def sage_arch_code() -> int:
    """Return the current device as ``major * 10 + minor`` (80, 86, 89, 90, 120).

    Spelled as an integer because that is the form both the vendored
    dispatcher and the ``smXY`` membership test below consume.
    """
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def check_sage_attention_compatible() -> bool:
    """Check if the current GPU supports SageAttention.

    SageAttention needs CUDA *and* one of :data:`SAGE_SUPPORTED_ARCHS` — the
    architectures the vendored dispatcher can actually serve. Testing only
    "CC major >= 8" let sm100/sm103 through: sage's own ``sageattn()`` refuses
    them with ``ValueError: Unsupported CUDA architecture``, and the vendored
    ``arch_code >= 90`` branch would have selected the SM90 kernel for them.

    Returns:
        True if SageAttention can be used, False otherwise.
    """
    if not SAGE_ATTENTION_AVAILABLE:
        return False
    if not torch.cuda.is_available():
        return False
    arch = f"sm{sage_arch_code()}"
    if arch not in SAGE_SUPPORTED_ARCHS:
        logger.warning(
            f"Your GPU (compute capability {torch.cuda.get_device_capability()[0]}."
            f"{torch.cuda.get_device_capability()[1]}, {arch}) is not one of the "
            f"architectures SageAttention ships kernels for "
            f"({', '.join(sorted(SAGE_SUPPORTED_ARCHS))}). "
            f"Sage option will be disabled."
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
    - Requested backend not actually usable on this machine → sdpa
      (``flash_attention_2`` via :func:`check_flash_attention_available`,
      ``sage`` via :func:`check_sage_attention_compatible`)
    - Unknown mode → eager

    The availability check exists because a *saved workflow* carries the mode
    string verbatim, while the node dropdown is gated on
    :func:`get_available_attention_modes` at build time. Without it, a
    workflow naming "sage" on a machine without sageattention survives every
    guard here and then dies deep inside the loader with
    ``RuntimeError("Incompatible hardware/setup for SageAttention.")``.

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

    if mode == "sage" and not check_sage_attention_compatible():
        logger.warning(
            f"sage is not usable on this machine (SageAttention missing, no "
            f"CUDA device, or an unsupported GPU architecture); "
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


# ---------------------------------------------------------------------------
# ASR attention policy
# ---------------------------------------------------------------------------
# Same kernel defect as the realtime exclusion, but reached unconditionally
# instead of through a quality gate. The ASR processor left-pads every batch to
# the longest utterance (`vibevoice_asr_processor.py`, and the generation path
# calls it with `padding=True`), so a prefill step hands the decoder a real
# (B, 1, S, S) additive mask. `sage_attention_forward` uses that mask ONLY to
# decide causality and then drops it (`is_causal = attention_mask is None and
# q_len > 1`), so every query attends to the pad columns. There is no "no mask
# means causal" case to lean on: with a real mask the kernel runs
# non-causally over the whole padded row and the transcribe result is silently
# wrong.
#
# sageattention 2.2.0's `sageattn` has no attn_mask parameter at all, so the
# mask cannot simply be forwarded the way ComfyUI core forwards it
# (comfy/ldm/modules/attention.py:708-710, which falls back to pytorch when
# the installed kernel cannot take a mask). Excluding sage from ASR is the
# same shape as the realtime exclusion and keeps one rule: a backend that
# cannot honour the mask is not offered on a path that has one.
ASR_ATTENTION_FALLBACK = "sdpa"

ASR_EXCLUDED_ATTENTION_MODES: dict[str, str] = {
    "sage": (
        "the ASR processor left-pads each batch to the longest utterance, so "
        "prefill arrives with a real additive (B,1,S,S) attention mask, and "
        "the sage kernel cannot take one -- it uses the mask only to decide "
        "causality and then discards it, so every query attends to the pad "
        "columns and the transcript is silently wrong"
    ),
}


def resolve_asr_attention_mode(attention_mode: str) -> str:
    """Downgrade a backend excluded from the ASR path, with a log line.

    Mirror of :func:`resolve_realtime_attention_mode`. Call it wherever the
    resolved mode is turned into a cache key and a patcher for an ASR model
    (``asr_loader``, the ASR branch of ``external_loader``, and both
    ``asr_generation`` resolvers) so the exclusion actually reaches the
    weights that get built -- a downgrade that misses the cache key would
    leave a sage-loaded model cached under an sdpa key.

    Args:
        attention_mode: The resolved attention mode.

    Returns:
        The attention mode to actually use, or ``attention_mode`` unchanged
        when it is not excluded.
    """
    reason = ASR_EXCLUDED_ATTENTION_MODES.get(attention_mode)
    if reason is None:
        return attention_mode
    logger.warning(
        "Attention mode '%s' is not used for ASR models: %s. Falling back to "
        "'%s' for this load. The TTS family is unaffected.",
        attention_mode,
        reason,
        ASR_ATTENTION_FALLBACK,
    )
    return ASR_ATTENTION_FALLBACK


# ---------------------------------------------------------------------------
# dtype / attention cross-checks
# ---------------------------------------------------------------------------
# The sage kernels hard-assert `dtype in [torch.float16, torch.bfloat16]`
# (sageattn_qk_int8_pv_fp8_cuda_sm90), and `resolve_sage_target_dtype` returns
# the *stored weight dtype* for a plain float linear — so a user who picks
# "fp32" in the node's dtype widget (offered by dtype_utils.get_dtype_options)
# and "sage" in the attention widget crashes inside the kernel with an
# unrelated-looking assert. The two widgets are independent inputs with no
# cross-check anywhere, so this has to be caught before the model is built.
#
# "auto" is exempt: on any GPU sage supports (sm80+) ComfyUI's
# should_use_bf16/should_use_fp16 resolve it to a half dtype. The 4-bit case
# is handled by the loader instead (it forces bnb_compute_dtype=float32 and
# sage reads bf16 out of the quantized linears), so callers on a node that
# exposes 4-bit should pass the *effective* model dtype, not the widget.
SAGE_UNSUPPORTED_DTYPES = frozenset({"fp32"})


def check_dtype_attention_compatible(
    dtype_str: str,
    attention_mode: Optional[str],
    quantized_4bit: bool = False,
) -> str | None:
    """Cross-check a dtype choice against an attention backend.

    Call from a node's ``validate_inputs`` so the user gets a queue-time
    message naming the two widgets, rather than an assert from inside a CUDA
    kernel several minutes into a model load.

    Args:
        dtype_str: The dtype widget value ("auto", "bf16", "fp16", "fp32").
        attention_mode: The attention widget value, as the user picked it
            (pre-resolution is fine — "sage" is caught either way). ``None``
            means the widget was not part of the prompt, so there is nothing
            to cross-check.
        quantized_4bit: Whether 4-bit quantization is on. It exempts the
            check: the loader forces bnb to an fp32 compute dtype for 4-bit +
            sage, but every quantized linear carries a ``quant_state``, so
            ``resolve_sage_target_dtype`` still hands the kernel bf16.

    Returns:
        An actionable error message, or ``None`` when the pair is fine.
    """
    if attention_mode != "sage":
        return None
    if dtype_str not in SAGE_UNSUPPORTED_DTYPES:
        return None
    if quantized_4bit:
        return None
    return (
        f"dtype '{dtype_str}' cannot run with attention_mode 'sage': the "
        f"SageAttention kernels require fp16 or bf16 inputs. Pick dtype "
        f"'bf16' (or 'fp16'), or switch attention_mode to 'sdpa' / "
        f"'flash_attention_2'."
    )
