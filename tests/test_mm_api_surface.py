"""D2/D3: ComfyUI API-surface conformance audit.

Asserts that every ``comfy.model_management`` / ``comfy.utils`` symbol the
project relies on actually exists on the real ComfyUI modules in this
checkout. This is a version-drift guard: if ComfyUI renames or removes a
symbol, these tests fail loudly instead of the node crashing at runtime.
"""

import inspect

import comfy.model_management as mm
import comfy.utils as comfy_utils
import folder_paths


class TestModelManagementSurface:
    """Every model_management symbol used by the project must exist."""

    USED_SYMBOLS = [
        "get_torch_device",
        "intermediate_device",
        "load_model_gpu",
        "unload_all_models",
        "soft_empty_cache",
        "should_use_bf16",
        "should_use_fp16",
        "supports_fp8_compute",
        "get_autocast_device",
        "is_nvidia",
        "is_amd",
        "mps_mode",
        "module_size",
        "InterruptProcessingException",
    ]

    def test_all_used_symbols_exist(self):
        missing = [s for s in self.USED_SYMBOLS if not hasattr(mm, s)]
        assert missing == [], f"comfy.model_management missing symbols: {missing}"

    def test_interrupt_is_exception_class(self):
        assert inspect.isclass(mm.InterruptProcessingException)
        # ComfyUI deliberately subclasses BaseException (not Exception) so a
        # bare ``except Exception`` cannot swallow user interrupts. The project's
        # catch order (InterruptProcessingException before Exception) relies on
        # this — lock it here.
        assert issubclass(mm.InterruptProcessingException, BaseException)
        assert not issubclass(mm.InterruptProcessingException, Exception)

    def test_load_model_gpu_is_callable(self):
        assert callable(mm.load_model_gpu)

    def test_should_use_bf16_accepts_device_arg(self):
        sig = inspect.signature(mm.should_use_bf16)
        # Must accept at least one positional (device) argument.
        assert len(sig.parameters) >= 1


class TestComfyUtilsSurface:
    """comfy.utils.load_torch_file must exist with a device parameter."""

    def test_load_torch_file_exists(self):
        assert hasattr(comfy_utils, "load_torch_file")
        assert callable(comfy_utils.load_torch_file)

    def test_load_torch_file_accepts_device_kwarg(self):
        sig = inspect.signature(comfy_utils.load_torch_file)
        assert "device" in sig.parameters

    def test_progressbar_exists(self):
        # generation.py imports ProgressBar from comfy.utils.
        assert hasattr(comfy_utils, "ProgressBar")


class TestFolderPathsSurface:
    """folder_paths symbols used by the project must exist."""

    def test_folder_paths_symbols_exist(self):
        for sym in ("get_folder_paths", "add_model_folder_path",
                    "supported_pt_extensions", "models_dir",
                    "folder_names_and_paths"):
            assert hasattr(folder_paths, sym), f"folder_paths missing: {sym}"

    def test_add_model_folder_path_signature(self):
        sig = inspect.signature(folder_paths.add_model_folder_path)
        assert "folder_name" in sig.parameters
        assert "full_folder_path" in sig.parameters
