"""Dtype utilities for VibeVoice nodes.

Provides dtype resolution and casting helpers using ComfyUI's
model_management functions for consistent precision selection.
"""

import torch
import logging
from typing import List

from .device_utils import should_use_bf16, should_use_fp16, _get_model_management


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


def _torch_dtype_is_deprecated_property(config) -> bool:
    """True when ``type(config).torch_dtype`` is the transformers >= 5.0
    deprecated property alias (which warns on every get/set)."""
    return isinstance(getattr(type(config), "torch_dtype", None), property)


def set_config_dtype(config, dtype) -> None:
    """Record the load dtype on a transformers config without triggering the
    transformers v5 deprecation warning.
    """
    if _torch_dtype_is_deprecated_property(config):
        config.dtype = dtype
    else:
        config.torch_dtype = dtype


def get_config_dtype(config):
    """Return the torch dtype recorded on a transformers config, or None."""
    if _torch_dtype_is_deprecated_property(config):
        dtype = getattr(config, "dtype", None)
    else:
        dtype = getattr(config, "torch_dtype", None)
        if dtype is None:
            dtype = getattr(config, "dtype", None)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype, None)
    return dtype


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
    """Parameter names excluded from bulk dtype casts.

    - Parameters of modules marked ``_quant_resident`` (GGUFLinear / ConvRotInt8Linear / FP8Linear).
    - Any parameter whose leaf name is ``weight_scale``.
    """
    protected = set()
    for name, mod in model.named_modules():
        if getattr(mod, "_quant_resident", False):
            prefix = f"{name}." if name else ""
            protected.add(f"{prefix}weight")
            protected.add(f"{prefix}weight_scale")
    return protected


def representative_dtype(model):
    """First FLOATING parameter dtype (None when the tree has none)."""
    try:
        resident_modules = {
            name for name, mod in model.named_modules()
            if getattr(mod, "_quant_resident", False)
        }
        for name, p in model.named_parameters():
            if not p.dtype.is_floating_point:
                continue
            owner = name.rpartition(".")[0]
            if owner in resident_modules:
                continue
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
            continue
        if param.dtype != dtype:
            param.data = param.data.to(dtype)

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
    fp32 ``weight_scale``) is NEVER touched.
    """
    if dtype is None:
        return
    _cast_mismatched_params(model, dtype)
    logging.debug(f"[VibeVoice TTS] Model cast to dtype (filtered): {dtype}")


def cast_model_to_dtype_if_needed(model, dtype: torch.dtype) -> None:
    """Cast a model to ``dtype`` only when some castable parameter mismatches."""
    if dtype is None:
        return

    protected = _quant_protected_names(model)

    mismatched = []
    for name, param in model.named_parameters():
        if name in protected or not param.dtype.is_floating_point:
            continue
        if param.dtype != dtype:
            mismatched.append((name, param.dtype))
    if not mismatched:
        return

    _cast_mismatched_params(model, dtype)
    sources = ", ".join(sorted({str(src) for _, src in mismatched}))
    logging.debug(
        f"[VibeVoice TTS] Model cast {sources} -> {dtype} ({len(mismatched)} mismatched params)"
    )