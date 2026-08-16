"""Tests for GGUF weight support in the external model loader.

Covers:
- ``_load_gguf_state_dict()``: real GGUF round-trip via the ``gguf`` package.
- ``_load_weight_state_dict()``: dispatch between gguf and ComfyUI loaders.
- ``list_external_model_files()``: .gguf files appear in the node dropdown.
- ``resolve_weight_path()``: diffusion_models first, unet_gguf fallback.
"""

import os
import pytest
from unittest.mock import MagicMock, patch

import torch

gguf = pytest.importorskip("gguf", reason="gguf package not installed")

from ComfyUI_VibeVoice.modules import external_loader
from ComfyUI_VibeVoice.modules.external_loader import (
    _load_gguf_state_dict,
    _load_weight_state_dict,
)
from ComfyUI_VibeVoice.nodes.external_loader_node import (
    list_external_model_files,
    resolve_weight_path,
)


# ====================================================================
# Fixture: write a tiny real GGUF file
# ====================================================================

@pytest.fixture
def tiny_gguf_file(tmp_path):
    """Write a minimal valid GGUF file with two F32 tensors; return its path."""
    import numpy as np
    from gguf import GGUFWriter

    path = str(tmp_path / "tiny.gguf")
    writer = GGUFWriter(path, arch="llama")

    t1 = np.arange(6, dtype=np.float32).reshape(2, 3)
    t2 = np.ones((4,), dtype=np.float32) * 7.0
    writer.add_tensor("model.layer.weight", t1)
    writer.add_tensor("model.layer.bias", t2)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return path


# ====================================================================
# _load_gguf_state_dict
# ====================================================================

class TestLoadGgufStateDict:
    """Test _load_gguf_state_dict() with a real GGUF file."""

    def test_loads_all_tensors(self, tiny_gguf_file):
        """All tensors in the GGUF file are present in the state dict."""
        sd = _load_gguf_state_dict(tiny_gguf_file)
        assert "model.layer.weight" in sd
        assert "model.layer.bias" in sd
        assert len(sd) == 2

    def test_tensor_values_correct(self, tiny_gguf_file):
        """Dequantized values match the originals written to the file."""
        sd = _load_gguf_state_dict(tiny_gguf_file)
        weight = sd["model.layer.weight"]
        assert weight.shape == (2, 3)
        assert torch.allclose(weight, torch.arange(6, dtype=torch.float32).reshape(2, 3))

        bias = sd["model.layer.bias"]
        assert bias.shape == (4,)
        assert torch.allclose(bias, torch.full((4,), 7.0))

    def test_tensors_are_torch_tensors(self, tiny_gguf_file):
        """Returned values are torch.Tensor instances."""
        sd = _load_gguf_state_dict(tiny_gguf_file)
        for v in sd.values():
            assert isinstance(v, torch.Tensor)

    def test_default_device_is_cpu(self, tiny_gguf_file):
        """Tensors land on CPU by default."""
        sd = _load_gguf_state_dict(tiny_gguf_file)
        for v in sd.values():
            assert v.device.type == "cpu"

    def test_explicit_cpu_device(self, tiny_gguf_file):
        """Passing device=torch.device('cpu') keeps tensors on CPU."""
        sd = _load_gguf_state_dict(tiny_gguf_file, device=torch.device("cpu"))
        for v in sd.values():
            assert v.device.type == "cpu"

    def test_missing_gguf_package_raises_runtime_error(self, tiny_gguf_file):
        """If the gguf package import fails, a clear RuntimeError is raised."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "gguf":
                raise ImportError("No module named 'gguf'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(RuntimeError, match="gguf"):
                _load_gguf_state_dict(tiny_gguf_file)


# ====================================================================
# _load_weight_state_dict dispatch
# ====================================================================

class TestLoadWeightStateDictDispatch:
    """Test _load_weight_state_dict() routes by extension."""

    def test_gguf_extension_routes_to_gguf_loader(self, tiny_gguf_file):
        """A .gguf path is loaded via the gguf parser, not torch.load."""
        with patch.object(
            external_loader, "_load_gguf_state_dict", wraps=_load_gguf_state_dict
        ) as mock_gguf, patch.object(
            external_loader.comfy.utils, "load_torch_file"
        ) as mock_comfy:
            sd = _load_weight_state_dict(tiny_gguf_file, torch.device("cpu"))

        mock_gguf.assert_called_once()
        mock_comfy.assert_not_called()
        assert "model.layer.weight" in sd

    def test_gguf_extension_case_insensitive(self, tiny_gguf_file):
        """Uppercase .GGUF also routes to the gguf parser."""
        upper_path = tiny_gguf_file[:-5] + ".GGUF"
        os.replace(tiny_gguf_file, upper_path)

        with patch.object(external_loader.comfy.utils, "load_torch_file") as mock_comfy:
            sd = _load_weight_state_dict(upper_path, torch.device("cpu"))

        mock_comfy.assert_not_called()
        assert "model.layer.weight" in sd

    def test_safetensors_routes_to_comfy_loader(self, tmp_path):
        """A .safetensors path is loaded via comfy.utils.load_torch_file."""
        fake_sd = {"w": torch.zeros(2)}
        weight = tmp_path / "model.safetensors"
        weight.write_bytes(b"")

        with patch.object(
            external_loader.comfy.utils, "load_torch_file", return_value=fake_sd
        ) as mock_comfy, patch.object(
            external_loader, "_load_gguf_state_dict"
        ) as mock_gguf:
            sd = _load_weight_state_dict(str(weight), torch.device("cpu"))

        mock_comfy.assert_called_once_with(str(weight), device=torch.device("cpu"))
        mock_gguf.assert_not_called()
        assert sd is fake_sd

    def test_bin_routes_to_comfy_loader(self, tmp_path):
        """A .bin path is loaded via comfy.utils.load_torch_file."""
        fake_sd = {"w": torch.zeros(2)}
        weight = tmp_path / "model.bin"
        weight.write_bytes(b"")

        with patch.object(
            external_loader.comfy.utils, "load_torch_file", return_value=fake_sd
        ) as mock_comfy, patch.object(
            external_loader, "_load_gguf_state_dict"
        ) as mock_gguf:
            sd = _load_weight_state_dict(str(weight), torch.device("cpu"))

        mock_comfy.assert_called_once()
        mock_gguf.assert_not_called()


# ====================================================================
# list_external_model_files (node dropdown)
# ====================================================================

class TestListExternalModelFiles:
    """Test list_external_model_files() includes .gguf files."""

    def test_includes_safetensors_from_filename_list(self):
        """Standard diffusion_models files are included."""
        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_filename_list",
            return_value=["model_a.safetensors", "model_b.bin"],
        ), patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_folder_paths",
            return_value=[],
        ):
            files = list_external_model_files()

        assert "model_a.safetensors" in files
        assert "model_b.bin" in files

    def test_includes_gguf_from_folder_scan(self):
        """.gguf files found by scanning diffusion_models folders are included."""
        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_filename_list",
            return_value=["model_a.safetensors"],
        ), patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_folder_paths",
            return_value=["/fake/models/unet"],
        ), patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.recursive_search",
            return_value=(["vibevoice.gguf", "other.safetensors"], {}),
        ):
            files = list_external_model_files()

        assert "vibevoice.gguf" in files
        assert "model_a.safetensors" in files

    def test_includes_unet_gguf_folder(self):
        """Files from the ComfyUI-GGUF 'unet_gguf' folder are included."""
        def fake_get_filename_list(folder_name):
            if folder_name == "diffusion_models":
                return ["model_a.safetensors"]
            if folder_name == "unet_gguf":
                return ["gguf_model.gguf"]
            return []

        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_filename_list",
            side_effect=fake_get_filename_list,
        ), patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_folder_paths",
            return_value=[],
        ):
            files = list_external_model_files()

        assert "gguf_model.gguf" in files
        assert "model_a.safetensors" in files

    def test_deduplicates_and_sorts(self):
        """Duplicate names across sources are de-duplicated; result is sorted."""
        def fake_get_filename_list(folder_name):
            if folder_name == "diffusion_models":
                return ["b.safetensors", "a.gguf"]
            if folder_name == "unet_gguf":
                return ["a.gguf"]
            return []

        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_filename_list",
            side_effect=fake_get_filename_list,
        ), patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_folder_paths",
            return_value=[],
        ):
            files = list_external_model_files()

        assert files == ["a.gguf", "b.safetensors"]

    def test_unet_gguf_missing_is_tolerated(self):
        """If 'unet_gguf' is not registered, no error is raised."""
        def fake_get_filename_list(folder_name):
            if folder_name == "unet_gguf":
                raise KeyError("unet_gguf")
            return ["model.safetensors"]

        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_filename_list",
            side_effect=fake_get_filename_list,
        ), patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_folder_paths",
            return_value=[],
        ):
            files = list_external_model_files()

        assert files == ["model.safetensors"]

    def test_empty_when_no_files(self):
        """Returns an empty list when no files are found anywhere."""
        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_filename_list",
            return_value=[],
        ), patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_folder_paths",
            return_value=[],
        ):
            files = list_external_model_files()

        assert files == []


# ====================================================================
# resolve_weight_path
# ====================================================================

class TestResolveWeightPath:
    """Test resolve_weight_path() folder fallback ordering."""

    def test_resolves_from_diffusion_models_first(self):
        """A file in diffusion_models is resolved without touching unet_gguf."""
        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_full_path_or_raise",
            return_value="/fake/unet/model.safetensors",
        ) as mock_resolve:
            path = resolve_weight_path("model.safetensors")

        assert path == "/fake/unet/model.safetensors"
        mock_resolve.assert_called_once_with("diffusion_models", "model.safetensors")

    def test_falls_back_to_unet_gguf(self):
        """If diffusion_models misses, unet_gguf is tried."""
        def fake_resolve(folder_name, filename):
            if folder_name == "diffusion_models":
                raise FileNotFoundError("not found")
            return "/fake/unet_gguf/model.gguf"

        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_full_path_or_raise",
            side_effect=fake_resolve,
        ):
            path = resolve_weight_path("model.gguf")

        assert path == "/fake/unet_gguf/model.gguf"

    def test_raises_when_not_found_anywhere(self):
        """FileNotFoundError is raised if the file is in neither folder."""
        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_full_path_or_raise",
            side_effect=FileNotFoundError("not found"),
        ):
            with pytest.raises(FileNotFoundError, match="not found in diffusion_models or unet_gguf"):
                resolve_weight_path("missing.gguf")
