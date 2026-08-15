"""Regression tests for AUD-013: snapshot_download kwarg compatibility.

``local_dir_use_symlinks`` was removed from ``huggingface_hub.snapshot_download``
in newer versions (>= 0.23, absent in 1.x). ``_ensure_downloaded`` must only
pass it when the installed version still accepts it, otherwise the first
official-model download raises ``TypeError``.
"""

import os
from unittest.mock import patch, MagicMock

from ComfyUI_VibeVoice.modules.base_loader import BaseVibeVoiceLoader


class TestEnsureDownloadedSkipsWhenPresent:
    """No download when config.json already exists."""

    def test_skips_download_when_config_exists(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            BaseVibeVoiceLoader._ensure_downloaded(
                repo_id="test/repo", local_dir=str(tmp_path), model_name="Test"
            )
            mock_dl.assert_not_called()

    def test_skips_download_when_repo_id_empty(self, tmp_path):
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            BaseVibeVoiceLoader._ensure_downloaded(
                repo_id="", local_dir=str(tmp_path), model_name="Test"
            )
            mock_dl.assert_not_called()


class TestEnsureDownloadedKwargCompat:
    """The symlinks kwarg must only be passed when supported."""

    def test_no_symlinks_kwarg_on_modern_hub(self, tmp_path):
        """huggingface_hub 1.x: signature lacks local_dir_use_symlinks."""
        import huggingface_hub
        import inspect

        if "local_dir_use_symlinks" in inspect.signature(
            huggingface_hub.snapshot_download
        ).parameters:
            # Old hub installed — this scenario cannot be exercised here.
            return

        with patch("huggingface_hub.snapshot_download") as mock_dl:
            BaseVibeVoiceLoader._ensure_downloaded(
                repo_id="test/repo", local_dir=str(tmp_path), model_name="Test"
            )
            mock_dl.assert_called_once()
            _, kwargs = mock_dl.call_args
            assert "local_dir_use_symlinks" not in kwargs
            assert kwargs["repo_id"] == "test/repo"
            assert kwargs["local_dir"] == str(tmp_path)

    def test_download_triggered_when_config_missing(self, tmp_path):
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            BaseVibeVoiceLoader._ensure_downloaded(
                repo_id="test/repo", local_dir=str(tmp_path), model_name="Test"
            )
            mock_dl.assert_called_once()
