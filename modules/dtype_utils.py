"""Dtype utilities for VibeVoice nodes.

Provides dtype resolution and casting helpers using ComfyUI's
model_management functions for consistent precision selection.
"""

import torch
import logging
from typing import List

from .device_utils import should_use_bf16, should_use_fp16, _get_model_management

logger = logging.getLogger(__name__)

# Dtype string constants
DTYPE_AUTO = "auto"
DTYPE_BF16 = "bf16"
DTYPE_FP16 = "fp16"
DTYPE_FP32 = "fp32"

# Mapping from string to torch.dtype
_DTYPE_MAP = {
    DTYPE_BF16: torch.bfloat16,
    DTYPE_FP16: torch.float16,
    DTYPE_FP32: torch.float32,
}

# Reverse mapping
_DTYPE_STR_MAP = {
    torch.bfloat16: DTYPE_BF16,
    torch.float16: DTYPE_FP16,
    torch.float32: DTYPE_FP32,
}


def get_dtype_options() -> List[str]:
    """Get list of available dtype options for node dropdown.

    Returns:
        List of dtype option strings.
    """
    return [DTYPE_AUTO, DTYPE_BF16, DTYPE_FP16, DTYPE_FP32]


def resolve_dtype(dtype_str: str, device: torch.device = None) -> torch.dtype:
    """Resolve a dtype string to a torch.dtype.

    Args:
        dtype_str: Dtype string ("auto", "bf16", "fp16", "fp32").
        device: torch.device for "auto" resolution. If None, uses default device.

    Returns:
        torch.dtype instance.

    Raises:
        ValueError: If dtype_str is not recognized.
    """
    if dtype_str == DTYPE_AUTO or dtype_str is None:
        mm = _get_model_management()
        if device is None:
            device = mm.get_torch_device()
        if should_use_bf16(device):
            return torch.bfloat16
        elif should_use_fp16(device):
            return torch.float16
        else:
            return torch.float32

    if dtype_str in _DTYPE_MAP:
        return _DTYPE_MAP[dtype_str]

    raise ValueError(f"Unknown dtype: {dtype_str}. Valid options: {list(_DTYPE_MAP.keys())}")


def get_dtype_str(dtype: torch.dtype) -> str:
    """Convert a torch.dtype to its string representation.

    Args:
        dtype: torch.dtype instance.

    Returns:
        Dtype string (e.g., "bf16", "fp16", "fp32").

    Raises:
        ValueError: If dtype is not recognized.
    """
    if dtype in _DTYPE_STR_MAP:
        return _DTYPE_STR_MAP[dtype]
    raise ValueError(f"Unknown torch.dtype: {dtype}. Supported: {list(_DTYPE_STR_MAP.values())}")


def _quant_protected_names(model) -> set:
    """Parameter names excluded from bulk dtype casts (plan 2026-08-24, E1).

    - Every parameter of modules marked ``_quant_resident`` (GGUFLinear /
      ConvRotInt8Linear): their weights are RAW BYTE / INT8 storage and the
      fp32 ``weight_scale`` must stay fp32.
    - Defense-in-depth: any parameter whose leaf name is ``weight_scale``.
    """
    protected = set()
    for name, mod in model.named_modules():
        if getattr(mod, "_quant_resident", False):
            prefix = f"{name}." if name else ""
            protected.add(f"{prefix}weight")
            protected.add(f"{prefix}weight_scale")
    return protected


def representative_dtype(model):
    """First FLOATING parameter dtype (None when the tree has none).

    Unlike transformers' ``dtype`` property this never reports an integer
    raw-storage param (uint8 GGUF blocks), so callers comparing dtypes don't
    spuriously schedule casts. Falls back to the model's ``dtype`` attribute
    when the parameter tree cannot be walked (e.g. test doubles).
    """
    try:
        for p in model.parameters():
            if p.dtype.is_floating_point:
                return p.dtype
    except Exception:
        pass
    return getattr(model, "dtype", None)


def _cast_mismatched_params(model, dtype: torch.dtype) -> None:
    """Cast only floating, unprotected, mismatched parameters, then re-tie."""
    protected = _quant_protected_names(model)
    for name, param in model.named_parameters():
        if name in protected:
            continue
        if not param.dtype.is_floating_point:
            continue  # raw int8/uint8 quant storage, index tensors, etc.
        if param.dtype != dtype:
            param.data = param.data.to(dtype)

    # Re-tie if the config marks weights as tied (a cast of one member of a
    # tied pair replaces its storage and breaks the sharing).
    config = getattr(model, "config", None)
    if config is not None and hasattr(model, "tie_weights"):
        decoder_config = getattr(config, "decoder_config", None)
        tied = bool(getattr(decoder_config, "tie_word_embeddings", False)) or \
            bool(getattr(config, "tie_word_embeddings", False))
        if tied:
            model.tie_weights()


def cast_model_to_dtype(model, dtype: torch.dtype) -> None:
    """Cast a model's FLOATING parameters to the specified dtype.

    Quant-resident storage (raw uint8 GGUF blocks, int8 ConvRot weights,
    fp32 ``weight_scale``) is NEVER touched. Torch's own ``Module.to(dtype)``
    happens to skip integer tensors, but it WOULD recast fp32 scales — hence
    this filtered walk instead of a bulk ``.to()``.

    Args:
        model: A torch.nn.Module to cast.
        dtype: Target torch.dtype.
    """
    if dtype is None:
        return
    _cast_mismatched_params(model, dtype)
    logger.debug(f"Model cast to dtype (filtered): {dtype}")


def cast_model_to_dtype_if_needed(model, dtype: torch.dtype) -> None:
    """Cast a model to ``dtype`` only when some castable parameter mismatches.

    Replaces the unconditional full-model ``.to(dtype)`` (RC-3). Walks the
    parameters once; if every CASTABLE (floating, non-protected) parameter
    already has the target dtype the function returns without touching the
    model. Otherwise it casts ONLY the mismatched castable parameters in
    place, then re-ties weights if the config marks them as tied.

    Quant-resident parameters are excluded (see :func:`cast_model_to_dtype`).

    Args:
        model: A torch.nn.Module to cast.
        dtype: Target torch.dtype. ``None`` is a no-op.
    """
    if dtype is None:
        return

    protected = _quant_protected_names(model)

    # Fast path: nothing castable to do.
    for name, param in model.named_parameters():
        if name in protected or not param.dtype.is_floating_point:
            continue
        if param.dtype != dtype:
            break
    else:
        return

    _cast_mismatched_params(model, dtype)
    logger.debug(f"Model cast to dtype (mismatched params only): {dtype}")
