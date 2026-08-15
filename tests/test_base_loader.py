"""Tests for modules/base_loader.py - shared BaseVibeVoiceLoader (IMP-004).

Exercises the GPU-free helpers (path resolution, snapshot download, tokenizer
repo, sharded/single state-dict loading) in isolation.
"""

import json
import torch
import pytest
from unittest.mock import patch, MagicMock

from ComfyUI_VibeVoice.modules.base_loader import BaseVibeVoiceLoader


class TestResolveOfficialModelDir:
    def test_resolve_official_paths_uses_folder_paths(self, tmp_path):
        with patch("ComfyUI_VibeVoice.modules.base_loader.folder_paths") as mock_fp:
            mock_fp.get_folder_paths.return_value = [str(tmp_path)]
            path = BaseVibeVoiceLoader._resolve_official_model_dir("VibeVoice-ASR")
        assert path == str(tmp_path / "VibeVoice" / "VibeVoice-ASR")

    def test_resolve_official_paths_falls_back_to_models_dir(self, tmp_path):
        fake_fp = MagicMock()
        # No "tts" folder registered → falls back to <models_dir>/tts
        fake_fp.get_folder_paths.return_value = []
        fake_fp.models_dir = str(tmp_path)
        with patch("ComfyUI_VibeVoice.modules.base_loader.folder_paths", fake_fp):
            path = BaseVibeVoiceLoader._resolve_official_model_dir("VibeVoice-ASR")
        assert path == str(tmp_path / "tts" / "VibeVoice" / "VibeVoice-ASR")


class TestEnsureDownloaded:
    def test_ensure_downloaded_triggers_snapshot(self, tmp_path):
        local_dir = tmp_path / "model"
        local_dir.mkdir()
        repo_id = "microsoft/VibeVoice-ASR"
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            BaseVibeVoiceLoader._ensure_downloaded(
                repo_id=repo_id, local_dir=str(local_dir), model_name="VibeVoice-ASR"
            )
        mock_dl.assert_called_once()
        _, kwargs = mock_dl.call_args
        assert kwargs.get("repo_id") == repo_id
        assert kwargs.get("local_dir") == str(local_dir)
        # AUD-013: the symlinks kwarg is only passed when the installed
        # huggingface_hub still accepts it (removed in >= 0.23 / 1.x).
        import inspect
        import huggingface_hub
        supports_symlinks = "local_dir_use_symlinks" in inspect.signature(
            huggingface_hub.snapshot_download
        ).parameters
        if supports_symlinks:
            assert kwargs.get("local_dir_use_symlinks") is False
        else:
            assert "local_dir_use_symlinks" not in kwargs

    def test_ensure_downloaded_skips_when_config_present(self, tmp_path):
        local_dir = tmp_path / "model"
        local_dir.mkdir()
        (local_dir / "config.json").write_text("{}")
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            BaseVibeVoiceLoader._ensure_downloaded(repo_id="x/y", local_dir=str(local_dir))
        mock_dl.assert_not_called()

    def test_ensure_downloaded_no_repo_returns_early(self, tmp_path):
        local_dir = tmp_path / "model"
        local_dir.mkdir()
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            BaseVibeVoiceLoader._ensure_downloaded(repo_id="", local_dir=str(local_dir))
        mock_dl.assert_not_called()


class TestTokenizerRepo:
    def test_tokenizer_repo_for_asr_uses_7b(self):
        assert BaseVibeVoiceLoader.tokenizer_repo_for("VibeVoice-ASR") == "Qwen/Qwen2.5-7B"

    def test_tokenizer_repo_for_small_uses_15b(self):
        assert BaseVibeVoiceLoader.tokenizer_repo_for("VibeVoice-1.5B") == "Qwen/Qwen2.5-1.5B"


class TestLoadStateDictSharded:
    def test_load_state_dict_sharded_single_file(self, tmp_path):
        (tmp_path / "model.safetensors").write_text("dummy")
        with patch(
            "ComfyUI_VibeVoice.modules.base_loader.comfy.utils.load_torch_file",
            return_value={"w": "v"},
        ):
            sd = BaseVibeVoiceLoader.load_state_dict_sharded(str(tmp_path), device=torch.device("cpu"))
        assert sd == {"w": "v"}

    def test_load_state_dict_sharded_merges_shards(self, tmp_path):
        (tmp_path / "model-00001-of-00002.safetensors").write_text("d1")
        (tmp_path / "model-00002-of-00002.safetensors").write_text("d2")
        index = {
            "weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        def mock_load(path, device=None):
            if "00001" in path:
                return {"a": 1}
            return {"b": 2}

        with patch(
            "ComfyUI_VibeVoice.modules.base_loader.comfy.utils.load_torch_file",
            side_effect=mock_load,
        ):
            sd = BaseVibeVoiceLoader.load_state_dict_sharded(str(tmp_path), device=torch.device("cpu"))
        assert sd == {"a": 1, "b": 2}

    def test_load_state_dict_sharded_empty_weight_map_raises(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
        with pytest.raises(ValueError, match="empty weight_map"):
            BaseVibeVoiceLoader.load_state_dict_sharded(str(tmp_path), device=torch.device("cpu"))

    def test_load_state_dict_sharded_missing_shard_raises(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors"}})
        )
        with patch(
            "ComfyUI_VibeVoice.modules.base_loader.comfy.utils.load_torch_file",
            return_value={},
        ):
            with pytest.raises(FileNotFoundError, match="Shard file not found"):
                BaseVibeVoiceLoader.load_state_dict_sharded(str(tmp_path), device=torch.device("cpu"))
