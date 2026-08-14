"""Tests for import paths - verify no old vibevoice/ paths remain."""

import os
import pytest


class TestImportPaths:
    """Verify all imports use the new src/vibevoice/ path."""

    def test_no_old_vibevoice_imports_in_modules(self):
        """modules/ files should import from ..src.vibevoice, not ..vibevoice."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        modules_dir = os.path.join(root, "modules")
        for fname in os.listdir(modules_dir):
            if fname.endswith(".py"):
                fpath = os.path.join(modules_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                # Should NOT have old-style imports
                assert "from ..vibevoice." not in content, \
                    f"Old import path found in vibevoice_modules/{fname}"
                assert "from ..vibevoice " not in content, \
                    f"Old import path found in vibevoice_modules/{fname}"

    def test_src_vibevoice_modular_importable(self):
        """src.vibevoice.modular should be importable (mocked in tests)."""
        import sys
        assert "src.vibevoice.modular" in sys.modules

    def test_src_vibevoice_processor_importable(self):
        """src.vibevoice.processor should be importable (mocked in tests)."""
        import sys
        assert "src.vibevoice.processor" in sys.modules

    def test_src_vibevoice_configs_importable(self):
        """src.vibevoice.configs should be importable (mocked in tests)."""
        import sys
        assert "src.vibevoice.configs" in sys.modules

    def test_all_modules_importable(self):
        """All modules should be importable without errors."""
        import ComfyUI_VibeVoice.modules.model_info
        import ComfyUI_VibeVoice.modules.audio_utils
        import ComfyUI_VibeVoice.modules.device_utils
        import ComfyUI_VibeVoice.modules.dtype_utils
        import ComfyUI_VibeVoice.modules.attention_utils
        import ComfyUI_VibeVoice.modules.utils
        # loader, patcher, generation require heavy deps (transformers, etc.)
        # so we skip them here — they're tested via integration tests

    def test_no_old_vibevoice_directory(self):
        """The old vibevoice/ directory at root should not exist."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        old_path = os.path.join(root, "vibevoice")
        assert not os.path.isdir(old_path), "Old vibevoice/ directory still exists at root"

    def test_src_vibevoice_directory_exists(self):
        """The new src/vibevoice/ directory should exist."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        new_path = os.path.join(root, "src", "vibevoice")
        assert os.path.isdir(new_path), "src/vibevoice/ directory does not exist"
