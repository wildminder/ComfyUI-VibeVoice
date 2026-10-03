"""Device utilities for VibeVoice nodes.

Provides unified device handling using ComfyUI's model_management
functions for consistent device detection and management across all backends.
"""

import torch
import logging
from typing import List


# Lazy import to avoid circular dependencies
_model_management = None


def _get_model_management():
    """Lazy import of ComfyUI's model_management module."""
    global _model_management
    if _model_management is None:
        import comfy.model_management as mm
        _model_management = mm
    return _model_management


# Device type constants
DEVICE_AUTO = "auto"
DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
DEVICE_MPS = "mps"
DEVICE_XPU = "xpu"
DEVICE_NPU = "npu"
DEVICE_DIRECTML = "directml"
DEVICE_HIP = "hip"


def get_available_devices() -> List[str]:
    """Get list of available devices in order of preference.

    Uses ComfyUI's model_management for consistent device detection.

    Returns:
        List of device type strings (e.g., ["cuda", "cpu"]).
    """
    devices = []
    mm = _get_model_management()

    if mm.is_nvidia():
        devices.append(DEVICE_CUDA)
    elif mm.is_amd():
        # AMD uses HIP backend but we report as "cuda" for compatibility
        devices.append(DEVICE_CUDA)
    elif mm.mps_mode():
        devices.append(DEVICE_MPS)
    else:
        # Check for other GPU backends
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            devices.append(DEVICE_XPU)
        elif hasattr(torch, 'npu') and torch.npu.is_available():
            devices.append(DEVICE_NPU)

    # CPU is always available as fallback
    devices.append(DEVICE_CPU)

    return devices


def get_device_display_name(device_type: str) -> str:
    """Get human-readable display name for a device type.

    Args:
        device_type: Device type string (e.g., "cuda", "cpu").

    Returns:
        Human-readable device name.
    """
    mm = _get_model_management()

    if device_type == DEVICE_CUDA:
        if mm.is_nvidia():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                return f"CUDA ({gpu_name})"
            except Exception:
                return "CUDA (NVIDIA GPU)"
        elif mm.is_amd():
            return "CUDA (AMD ROCm/HIP)"
        return "CUDA"
    elif device_type == DEVICE_MPS:
        return "MPS (Apple Silicon)"
    elif device_type == DEVICE_XPU:
        return "XPU (Intel Arc)"
    elif device_type == DEVICE_NPU:
        return "NPU (Huawei Ascend)"
    elif device_type == DEVICE_DIRECTML:
        return "DirectML (Windows)"
    elif device_type == DEVICE_CPU:
        return "CPU"

    return device_type.upper()


def get_torch_device(device_type: str = None) -> torch.device:
    """Get torch.device for the specified device type.

    Honors MPS / XPU / NPU requests when the corresponding backend is actually
    available on this machine; otherwise logs a warning and falls back to
    ComfyUI's default compute device (previously every non-CPU request silently
    mapped to the default GPU — see IMP-003).

    Args:
        device_type: Device type string. If None or "auto", uses ComfyUI's
            default device.

    Returns:
        torch.device instance.
    """
    mm = _get_model_management()

    # CUDA (and any "cuda:*" index) maps to ComfyUI's default compute device.
    if device_type is None or device_type == "auto" or device_type == DEVICE_CUDA \
            or (isinstance(device_type, str) and device_type.startswith(DEVICE_CUDA)):
        return mm.get_torch_device()

    if device_type == DEVICE_CPU:
        return torch.device(DEVICE_CPU)

    if device_type == DEVICE_MPS:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device(DEVICE_MPS)
        logging.warning("[VibeVoice TTS] Device 'mps' requested but not available; using default device.")
        return mm.get_torch_device()

    if device_type == DEVICE_XPU:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device(DEVICE_XPU)
        logging.warning("[VibeVoice TTS] Device 'xpu' requested but not available; using default device.")
        return mm.get_torch_device()

    if device_type == DEVICE_NPU:
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device(DEVICE_NPU)
        logging.warning("[VibeVoice TTS] Device 'npu' requested but not available; using default device.")
        return mm.get_torch_device()

    # Unknown device type — fall back to ComfyUI's default.
    logging.warning(f"[VibeVoice TTS] Device '{device_type}' not available; using default device.")
    return mm.get_torch_device()


def get_device_type(device: torch.device) -> str:
    """Convert torch.device to our device type string.

    Args:
        device: torch.device instance.

    Returns:
        Device type string (e.g., "cuda", "cpu").
    """
    if device.type == DEVICE_CPU:
        return DEVICE_CPU
    elif device.type == DEVICE_MPS:
        return DEVICE_MPS
    elif device.type == DEVICE_XPU:
        return DEVICE_XPU
    elif device.type == DEVICE_NPU:
        return DEVICE_NPU
    elif device.type == DEVICE_CUDA:
        return DEVICE_CUDA
    else:
        return device.type


def get_offload_device() -> torch.device:
    """Get the offload device (typically CPU) for memory management.

    Uses ComfyUI's intermediate_device() for consistent offload handling.

    Returns:
        torch.device for offloading.
    """
    mm = _get_model_management()
    return mm.intermediate_device()


def should_use_fp16(device: torch.device = None) -> bool:
    """Check if FP16 should be used for the given device.

    Uses ComfyUI's should_use_fp16() for consistent precision selection.

    Args:
        device: torch.device to check. If None, uses default device.

    Returns:
        True if FP16 should be used.
    """
    mm = _get_model_management()
    if device is None:
        device = mm.get_torch_device()
    return mm.should_use_fp16(device)


def should_use_bf16(device: torch.device = None) -> bool:
    """Check if BF16 should be used for the given device.

    Uses ComfyUI's should_use_bf16() for consistent precision selection.

    Args:
        device: torch.device to check. If None, uses default device.

    Returns:
        True if BF16 should be used.
    """
    mm = _get_model_management()
    if device is None:
        device = mm.get_torch_device()
    return mm.should_use_bf16(device)


def supports_fp8_compute(device: torch.device = None) -> bool:
    """Check if FP8 compute is supported on the given device.

    Args:
        device: torch.device to check. If None, uses default device.

    Returns:
        True if FP8 compute is supported.
    """
    mm = _get_model_management()
    if device is None:
        device = mm.get_torch_device()
    return mm.supports_fp8_compute(device)


def get_autocast_device(device: torch.device) -> str:
    """Get the device type string for torch.autocast.

    Args:
        device: torch.device instance.

    Returns:
        Device type string suitable for torch.autocast.
    """
    mm = _get_model_management()
    return mm.get_autocast_device(device)


def is_gpu_device(device_type: str) -> bool:
    """Check if a device type is a GPU device.

    Args:
        device_type: Device type string.

    Returns:
        True if the device is a GPU type.
    """
    gpu_types = {DEVICE_CUDA, DEVICE_MPS, DEVICE_XPU, DEVICE_NPU, DEVICE_DIRECTML, DEVICE_HIP}
    return device_type.lower() in gpu_types


def get_device_memory_info(device_type: str = None) -> dict:
    """Get memory information for a device.

    Args:
        device_type: Device type string. If None, uses default device.

    Returns:
        Dictionary with 'total', 'used', 'free' memory in bytes.
    """
    device = get_torch_device(device_type)

    if device.type == DEVICE_CUDA:
        try:
            total = torch.cuda.get_device_properties(device).total_memory
            reserved = torch.cuda.memory_reserved(device)
            allocated = torch.cuda.memory_allocated(device)
            free = total - reserved + (reserved - allocated)
            return {
                'total': total,
                'used': allocated,
                'free': free
            }
        except Exception:
            pass

    return {}


def get_device_options() -> List[str]:
    """Get list of device options for node dropdown.

    ``"auto"`` is a real, supported value: :func:`get_torch_device` maps it to
    ComfyUI's default compute device. It has to appear in the combo options,
    otherwise ComfyUI's "Value not in list" check rejects workflows saved with
    ``device="auto"`` before ``execute()`` ever runs. It is appended last so the
    existing option indices — and therefore saved ``widgets_values`` — are
    unchanged.

    Returns:
        List of device type strings for UI dropdown.
    """
    return [*get_available_devices(), DEVICE_AUTO]


def get_device_option_labels() -> dict:
    """Get mapping of device options to display labels.

    Returns:
        Dictionary mapping device types to display names.
    """
    return {dev: get_device_display_name(dev) for dev in get_device_options()}
