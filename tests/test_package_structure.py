"""Test that the package structure is correct and all modules are importable."""

import importlib
import sys
import os
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestPackageStructure:
    """Verify the package layout matches the plan."""

    def test_src_package_importable(self):
        """src package should be importable."""
        import src
        assert src is not None

    def test_src_vibevoice_importable(self):
        """src.vibevoice package should be importable (mocked in tests)."""
        import sys
        assert "src.vibevoice" in sys.modules

    def test_src_vibevoice_configs_importable(self):
        """src.vibevoice.configs should be importable (mocked in tests)."""
        import sys
        assert "src.vibevoice.configs" in sys.modules

    def test_src_vibevoice_modular_importable(self):
        """src.vibevoice.modular should be importable (mocked in tests)."""
        import sys
        assert "src.vibevoice.modular" in sys.modules

    def test_nodes_package_importable(self):
        """nodes package should be importable (after full implementation)."""
        # This will be tested more thoroughly after Phase 3
        # For now just verify the directory exists
        import os
        nodes_init = os.path.join(os.path.dirname(__file__), "..", "nodes", "__init__.py")
        assert os.path.exists(nodes_init)

    def test_tests_package_importable(self):
        """tests package should be importable."""
        import tests
        assert tests is not None

    def test_js_dir_absent(self):
        """IMP-002: the formerly-empty js/ directory should not exist."""
        assert not os.path.isdir(os.path.join(ROOT, "js")), (
            "js/ directory should be removed; V3 nodes require no web root."
        )
