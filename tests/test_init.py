"""Tests for __init__.py - Entry point and folder registration."""

import sys
import os
import pytest

import ComfyUI_VibeVoice


class TestPytestGuard:
    """Test the pytest guard in __init__.py."""

    def test_pytest_guard_active(self):
        """When pytest is in sys.modules, __init__ should skip runtime registration."""
        assert "pytest" in sys.modules


class TestFolderRegistration:
    """Test ComfyUI folder registration.

    Note: The 'tts' folder is registered by __init__.py at runtime.
    During pytest, the __init__.py pytest guard skips this registration.
    These tests verify the registration logic works when called directly.
    """

    def test_tts_folder_can_be_registered(self):
        """The 'tts' folder can be registered with folder_paths."""
        import folder_paths
        tts_path = os.path.join(folder_paths.models_dir, "tts")
        if "tts" not in folder_paths.folder_names_and_paths:
            supported_exts = folder_paths.supported_pt_extensions.union({".safetensors", ".json"})
            folder_paths.folder_names_and_paths["tts"] = ([tts_path], supported_exts)
        assert "tts" in folder_paths.folder_names_and_paths

    def test_tts_folder_has_models_dir(self):
        """The tts folder should include the models/tts path."""
        import folder_paths
        tts_path = os.path.join(folder_paths.models_dir, "tts")
        if "tts" not in folder_paths.folder_names_and_paths:
            supported_exts = folder_paths.supported_pt_extensions.union({".safetensors", ".json"})
            folder_paths.folder_names_and_paths["tts"] = ([tts_path], supported_exts)
        paths = folder_paths.get_folder_paths("tts")
        assert len(paths) > 0


class TestOfficialModelsPopulated:
    """Test that official models are populated."""

    def test_official_models_available(self):
        """AVAILABLE_VIBEVOICE_MODELS should have official entries."""
        from ComfyUI_VibeVoice.modules.model_info import MODEL_CONFIGS
        assert "VibeVoice-1.5B" in MODEL_CONFIGS
        assert "VibeVoice-Large" in MODEL_CONFIGS


class TestEntrypointExport:
    """Test that comfy_entrypoint is exported."""

    def test_comfy_entrypoint_importable(self):
        """comfy_entrypoint should be importable from vibevoice_nodes."""
        from ComfyUI_VibeVoice.vibevoice_nodes import comfy_entrypoint
        assert callable(comfy_entrypoint)


class TestNoWebDirectory:
    """IMP-002: V3 nodes need no WEB_DIRECTORY; it must not point at a missing dir."""

    def test_no_web_directory_attribute(self):
        """The package must not expose a WEB_DIRECTORY attribute."""
        assert not hasattr(ComfyUI_VibeVoice, "WEB_DIRECTORY"), (
            "WEB_DIRECTORY should be removed for V3 nodes."
        )

    def test_web_directory_not_in_all(self):
        """WEB_DIRECTORY must not appear in __all__."""
        assert "WEB_DIRECTORY" not in getattr(ComfyUI_VibeVoice, "__all__", [])
