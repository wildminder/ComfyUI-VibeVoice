"""Test that conftest fixtures and path configuration work correctly."""

import sys
import os
import pytest


class TestConftestSetup:
    """Verify conftest.py sets up the environment correctly."""

    def test_comfyui_root_in_path(self):
        """ComfyUI root should be in sys.path."""
        conftest_dir = os.path.dirname(os.path.abspath(__file__))
        # The conftest adds COMFYUI_ROOT to sys.path
        # Check that 'comfy' is importable
        try:
            import comfy
            assert comfy is not None
        except ImportError:
            pytest.skip("ComfyUI not available in this environment")

    def test_root_dir_in_path(self):
        """Project root should be in sys.path."""
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assert root_dir in sys.path

    def test_server_mocked(self):
        """server module should be mocked."""
        assert "server" in sys.modules

    def test_mock_prompt_server_fixture(self, mock_prompt_server):
        """mock_prompt_server fixture should provide a MagicMock."""
        assert mock_prompt_server is not None
        assert hasattr(mock_prompt_server, "send_sync")

    def test_temp_model_dir_fixture(self, temp_model_dir):
        """temp_model_dir fixture should create a directory."""
        assert temp_model_dir.exists()
        assert temp_model_dir.is_dir()

    def test_comfyui_env_fixture(self, comfyui_env):
        """comfyui_env fixture should return True."""
        assert comfyui_env is True
