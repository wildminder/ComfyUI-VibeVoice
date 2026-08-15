"""Tests for modules/device_utils.py - Device detection and management."""

import torch
import pytest
from unittest.mock import patch, MagicMock

from ComfyUI_VibeVoice.modules.device_utils import (
    DEVICE_CPU,
    DEVICE_CUDA,
    DEVICE_MPS,
    DEVICE_XPU,
    DEVICE_NPU,
    get_available_devices,
    get_torch_device,
    get_offload_device,
    is_gpu_device,
    get_device_display_name,
)


class TestGetAvailableDevices:
    """Test get_available_devices function."""

    def test_returns_list(self):
        devices = get_available_devices()
        assert isinstance(devices, list)

    def test_includes_cpu(self):
        devices = get_available_devices()
        assert DEVICE_CPU in devices

    def test_includes_cuda_when_available(self):
        with patch("ComfyUI_VibeVoice.modules.device_utils._get_model_management") as mock_mm:
            mock_mm.return_value.is_nvidia.return_value = True
            mock_mm.return_value.is_amd.return_value = False
            mock_mm.return_value.mps_mode.return_value = False
            devices = get_available_devices()
            assert DEVICE_CUDA in devices

    def test_includes_mps_when_available(self):
        with patch("ComfyUI_VibeVoice.modules.device_utils._get_model_management") as mock_mm:
            mock_mm.return_value.is_nvidia.return_value = False
            mock_mm.return_value.is_amd.return_value = False
            mock_mm.return_value.mps_mode.return_value = True
            devices = get_available_devices()
            assert DEVICE_MPS in devices

    def test_cpu_only_when_no_gpu(self):
        with patch("ComfyUI_VibeVoice.modules.device_utils._get_model_management") as mock_mm:
            mock_mm.return_value.is_nvidia.return_value = False
            mock_mm.return_value.is_amd.return_value = False
            mock_mm.return_value.mps_mode.return_value = False
            devices = get_available_devices()
            assert devices == [DEVICE_CPU]

    def test_excludes_mps_when_unavailable(self):
        with patch("ComfyUI_VibeVoice.modules.device_utils._get_model_management") as mock_mm:
            mock_mm.return_value.is_nvidia.return_value = False
            mock_mm.return_value.is_amd.return_value = False
            mock_mm.return_value.mps_mode.return_value = False
            devices = get_available_devices()
            assert DEVICE_MPS not in devices


class TestGetTorchDevice:
    """Test get_torch_device function."""

    def test_cpu_device(self):
        device = get_torch_device(DEVICE_CPU)
        assert device.type == DEVICE_CPU

    def test_none_uses_default(self):
        with patch("ComfyUI_VibeVoice.modules.device_utils._get_model_management") as mock_mm:
            mock_mm.return_value.get_torch_device.return_value = torch.device("cpu")
            device = get_torch_device(None)
            assert device is not None

    def test_auto_uses_default(self):
        with patch("ComfyUI_VibeVoice.modules.device_utils._get_model_management") as mock_mm:
            mock_mm.return_value.get_torch_device.return_value = torch.device("cuda")
            device = get_torch_device("auto")
            assert device.type == "cuda"

    def test_mps_available(self):
        with patch("ComfyUI_VibeVoice.modules.device_utils._get_model_management") as mock_mm, \
             patch("torch.backends.mps.is_available", return_value=True):
            mock_mm.return_value.get_torch_device.return_value = torch.device("cuda")
            device = get_torch_device(DEVICE_MPS)
            assert device.type == DEVICE_MPS

    def test_xpu_available(self):
        with patch("ComfyUI_VibeVoice.modules.device_utils._get_model_management") as mock_mm, \
             patch("torch.xpu") as mock_xpu:
            mock_xpu.is_available.return_value = True
            mock_mm.return_value.get_torch_device.return_value = torch.device("cuda")
            device = get_torch_device(DEVICE_XPU)
            assert device.type == DEVICE_XPU

    def test_npu_available(self):
        # NPU requires a special torch build that registers the "npu" device type.
        # On standard builds torch.device("npu") raises, so skip there.
        try:
            torch.device(DEVICE_NPU)
        except RuntimeError:
            pytest.skip("torch build does not support the 'npu' device type")

        with patch("ComfyUI_VibeVoice.modules.device_utils._get_model_management") as mock_mm, \
             patch("torch.npu", create=True) as mock_npu:
            mock_npu.is_available.return_value = True
            mock_mm.return_value.get_torch_device.return_value = torch.device("cuda")
            device = get_torch_device(DEVICE_NPU)
            assert device.type == DEVICE_NPU

    def test_mps_unavailable_falls_back_and_warns(self):
        with patch("ComfyUI_VibeVoice.modules.device_utils._get_model_management") as mock_mm, \
             patch("torch.backends.mps.is_available", return_value=False), \
             patch("ComfyUI_VibeVoice.modules.device_utils.logger") as mock_logger:
            mock_mm.return_value.get_torch_device.return_value = torch.device("cuda")
            device = get_torch_device(DEVICE_MPS)
            assert device.type == "cuda"
            mock_logger.warning.assert_called()

    def test_unknown_device_falls_back_and_warns(self):
        with patch("ComfyUI_VibeVoice.modules.device_utils._get_model_management") as mock_mm, \
             patch("ComfyUI_VibeVoice.modules.device_utils.logger") as mock_logger:
            mock_mm.return_value.get_torch_device.return_value = torch.device("cuda")
            device = get_torch_device("something_weird")
            assert device.type == "cuda"
            mock_logger.warning.assert_called()


class TestIsGpuDevice:
    """Test is_gpu_device function."""

    def test_cuda_is_gpu(self):
        assert is_gpu_device(DEVICE_CUDA) is True

    def test_cpu_not_gpu(self):
        assert is_gpu_device(DEVICE_CPU) is False

    def test_mps_is_gpu(self):
        assert is_gpu_device(DEVICE_MPS) is True

    def test_case_insensitive(self):
        assert is_gpu_device("CUDA") is True
        assert is_gpu_device("CPU") is False


class TestGetOffloadDevice:
    """Test get_offload_device function."""

    def test_returns_device(self):
        with patch("ComfyUI_VibeVoice.modules.device_utils._get_model_management") as mock_mm:
            mock_mm.return_value.intermediate_device.return_value = torch.device("cpu")
            device = get_offload_device()
            assert isinstance(device, torch.device)


class TestGetDeviceDisplayName:
    """Test get_device_display_name function."""

    def test_cpu_name(self):
        with patch("ComfyUI_VibeVoice.modules.device_utils._get_model_management"):
            name = get_device_display_name(DEVICE_CPU)
            assert "CPU" in name

    def test_mps_name(self):
        with patch("ComfyUI_VibeVoice.modules.device_utils._get_model_management"):
            name = get_device_display_name(DEVICE_MPS)
            assert "MPS" in name or "Apple" in name
