"""Tests for modules/asr_loader.py - ASR model loading and caching."""

import torch
import pytest
from unittest.mock import patch, MagicMock

from ComfyUI_VibeVoice.modules.asr_loader import (
    VibeVoiceASRModelHandler,
    VibeVoiceASRLoader,
    LOADED_ASR_MODELS_CACHE,
    cleanup_asr_models,
)
from ComfyUI_VibeVoice.modules.base_loader import BaseVibeVoiceLoader


@pytest.fixture(autouse=True)
def mock_tts_folder():
    """Register a 'tts' folder with folder_paths for tests."""
    import os
    import folder_paths
    tts_path = os.path.join(folder_paths.models_dir, "tts")
    if "tts" not in folder_paths.folder_names_and_paths:
        supported_exts = folder_paths.supported_pt_extensions.union({".safetensors", ".json"})
        folder_paths.folder_names_and_paths["tts"] = ([tts_path], supported_exts)
    yield


class TestVibeVoiceASRModelHandler:
    """Test VibeVoiceASRModelHandler class."""

    def test_handler_init(self):
        handler = VibeVoiceASRModelHandler("VibeVoice-ASR")
        assert handler.model_name == "VibeVoice-ASR"
        assert handler.model is None
        assert handler.processor is None

    def test_handler_size_calculation(self):
        handler = VibeVoiceASRModelHandler("VibeVoice-ASR")
        # size_gb=15.0 → 15.0 * 1024^3 bytes
        assert handler.size == int(15.0 * (1024**3))

    def test_handler_is_torch_module(self):
        handler = VibeVoiceASRModelHandler("VibeVoice-ASR")
        assert isinstance(handler, torch.nn.Module)


class TestCleanupASRModels:
    """Test cleanup_asr_models function."""

    def test_cleanup_clears_all(self):
        LOADED_ASR_MODELS_CACHE.clear()
        LOADED_ASR_MODELS_CACHE["key1"] = "model1"
        LOADED_ASR_MODELS_CACHE["key2"] = "model2"

        with patch("ComfyUI_VibeVoice.modules.asr_loader.model_management"):
            cleanup_asr_models(keep_cache_key=None)

        assert len(LOADED_ASR_MODELS_CACHE) == 0

    def test_cleanup_keeps_specified_key(self):
        LOADED_ASR_MODELS_CACHE.clear()
        LOADED_ASR_MODELS_CACHE["key1"] = "model1"
        LOADED_ASR_MODELS_CACHE["key2"] = "model2"

        with patch("ComfyUI_VibeVoice.modules.asr_loader.model_management"):
            cleanup_asr_models(keep_cache_key="key1")

        assert "key1" in LOADED_ASR_MODELS_CACHE
        assert "key2" not in LOADED_ASR_MODELS_CACHE


class TestASRLoaderBaseInheritance:
    """IMP-004 / CRIT-001: ASR loader shares the base loader + path logic."""

    def test_asr_loader_is_base_loader(self):
        assert issubclass(VibeVoiceASRLoader, BaseVibeVoiceLoader)

    def test_resolve_paths_official_uses_base(self, tmp_path):
        with patch("ComfyUI_VibeVoice.modules.base_loader.folder_paths") as mock_fp, \
             patch.object(BaseVibeVoiceLoader, "_ensure_downloaded"):
            mock_fp.get_folder_paths.return_value = [str(tmp_path)]
            model_path, tokenizer_repo = VibeVoiceASRLoader._resolve_model_paths("VibeVoice-ASR")

        assert model_path == str(tmp_path / "VibeVoice" / "VibeVoice-ASR")
        assert tokenizer_repo == "Qwen/Qwen2.5-7B"

    def test_resolve_paths_local_dir(self, tmp_path):
        from ComfyUI_VibeVoice.modules.model_info import AVAILABLE_VIBEVOICE_MODELS

        with patch.dict(AVAILABLE_VIBEVOICE_MODELS, {"MyASR": {"type": "local_dir", "path": str(tmp_path)}}, clear=True):
            model_path, tokenizer_repo = VibeVoiceASRLoader._resolve_model_paths("MyASR")

        assert model_path == str(tmp_path)
        assert tokenizer_repo == "Qwen/Qwen2.5-7B"
