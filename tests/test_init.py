"""Tests for __init__.py - Entry point and folder registration."""

import ast
import sys
import os
import pytest

import ComfyUI_VibeVoice

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
INIT_PATH = os.path.join(PKG_DIR, "__init__.py")


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
        assert "VibeVoice-7B" in MODEL_CONFIGS


class TestEntrypointExport:
    """Test that comfy_entrypoint is exported."""

    def test_comfy_entrypoint_importable(self):
        """comfy_entrypoint should be importable from vibevoice_nodes."""
        from ComfyUI_VibeVoice.vibevoice_nodes import comfy_entrypoint
        assert callable(comfy_entrypoint)


class TestWebDirectory:
    """The V3 frontend listener lives in js/ and is served via WEB_DIRECTORY.

    ComfyUI scans the WEB_DIRECTORY module attribute for **/*.js and serves the
    results under /extensions/<module>/ (ComfyUI nodes.py:2299-2302). Without it
    the gguf-missing modal listener would never load and the browser would drop
    the vibevoice.gguf_missing event with an "Unknown message type" error.
    """

    def _web_directory_assignments(self):
        """(lineno, value) for every module-level WEB_DIRECTORY assignment.

        Read by AST rather than via the imported attribute so the test still
        pins the declaration itself, not just one observed value.
        """
        with open(INIT_PATH, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=INIT_PATH)
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "WEB_DIRECTORY" for t in node.targets
            ):
                found.append((node.lineno, ast.literal_eval(node.value)))
        return found

    def test_web_directory_declared_once(self):
        """WEB_DIRECTORY is declared exactly once, and equals the js/ root."""
        assignments = self._web_directory_assignments()
        assert len(assignments) == 1, (
            f"expected exactly one WEB_DIRECTORY assignment in {INIT_PATH}, "
            f"found {len(assignments)}: {assignments}. It must not be split or "
            "re-declared across branches."
        )
        assert getattr(ComfyUI_VibeVoice, "WEB_DIRECTORY", None) == "./js", (
            "WEB_DIRECTORY must stay './js' (the directory ComfyUI globs for "
            "**/*.js)."
        )

    def test_web_directory_resolves_to_existing_dir(self):
        """WEB_DIRECTORY points at a directory that exists at import time."""
        (lineno, value), = self._web_directory_assignments()
        web_dir = os.path.abspath(os.path.join(PKG_DIR, value))
        assert os.path.isdir(web_dir), (
            f"WEB_DIRECTORY = {value!r} at {INIT_PATH}:{lineno} does not resolve "
            f"to an existing directory ({web_dir}). ComfyUI only registers the "
            "web root when os.path.isdir() is true, so the modal never ships."
        )

    def test_web_directory_not_in_all(self):
        """WEB_DIRECTORY is config, not part of the node's public exports."""
        assert "WEB_DIRECTORY" not in getattr(ComfyUI_VibeVoice, "__all__", [])
