"""Tests for modules/model_info.py - Model discovery and configuration."""

import os
import json
import pytest
from unittest.mock import patch

from ComfyUI_VibeVoice.modules.model_info import (
    MODEL_CONFIGS,
    AVAILABLE_VIBEVOICE_MODELS,
    scan_vibevoice_models,
    get_tokenizer_repo,
    is_model_type,
)


class TestModelConfigs:
    """Test MODEL_CONFIGS dictionary."""

    def test_model_configs_has_1_5b(self):
        assert "VibeVoice-1.5B" in MODEL_CONFIGS

    def test_model_configs_has_large(self):
        assert "VibeVoice-Large" in MODEL_CONFIGS

    def test_model_configs_has_repo_id(self):
        assert MODEL_CONFIGS["VibeVoice-1.5B"]["repo_id"] == "microsoft/VibeVoice-1.5B"

    def test_model_configs_has_size_gb(self):
        assert "size_gb" in MODEL_CONFIGS["VibeVoice-1.5B"]
        assert isinstance(MODEL_CONFIGS["VibeVoice-1.5B"]["size_gb"], (int, float))

    def test_available_models_is_dict(self):
        assert isinstance(AVAILABLE_VIBEVOICE_MODELS, dict)


class TestGetTokenizerRepo:
    """Test get_tokenizer_repo function."""

    def test_get_tokenizer_repo_large(self):
        assert get_tokenizer_repo("VibeVoice-Large") == "Qwen/Qwen2.5-7B"

    def test_get_tokenizer_repo_small(self):
        assert get_tokenizer_repo("VibeVoice-1.5B") == "Qwen/Qwen2.5-1.5B"

    def test_get_tokenizer_repo_case_insensitive(self):
        assert get_tokenizer_repo("vibevoice-large") == "Qwen/Qwen2.5-7B"

    def test_get_tokenizer_repo_unknown(self):
        assert get_tokenizer_repo("SomeModel") == "Qwen/Qwen2.5-1.5B"


class TestScanVibevoiceModels:
    """Test scan_vibevoice_models function."""

    def test_scan_empty_dir(self, tmp_path):
        results = scan_vibevoice_models(str(tmp_path))
        assert results == []

    def test_scan_nonexistent_dir(self):
        results = scan_vibevoice_models("/nonexistent/path/12345")
        assert results == []

    def test_scan_hf_directory(self, tmp_path):
        """A directory with config.json and .safetensors should be detected."""
        model_dir = tmp_path / "MyModel"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"test": true}')
        (model_dir / "model.safetensors").write_text("dummy")

        results = scan_vibevoice_models(str(tmp_path))
        assert len(results) == 1
        assert results[0]["name"] == "MyModel"
        assert results[0]["type"] == "local_dir"
        assert results[0]["path"] == str(model_dir)

    def test_scan_hf_directory_with_index(self, tmp_path):
        """A directory with config.json and safetensors.index.json should be detected."""
        model_dir = tmp_path / "IndexedModel"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"test": true}')
        (model_dir / "model.safetensors.index.json").write_text('{"test": true}')

        results = scan_vibevoice_models(str(tmp_path))
        assert len(results) == 1
        assert results[0]["name"] == "IndexedModel"

    def test_scan_standalone_file(self, tmp_path):
        """A standalone .safetensors file should be detected."""
        (tmp_path / "standalone_model.safetensors").write_text("dummy")

        results = scan_vibevoice_models(str(tmp_path))
        assert len(results) == 1
        assert results[0]["name"] == "standalone_model"
        assert results[0]["type"] == "standalone"

    def test_scan_no_weights(self, tmp_path):
        """A directory with config but no weights should NOT be detected."""
        model_dir = tmp_path / "NoWeights"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"test": true}')
        # No weight files

        results = scan_vibevoice_models(str(tmp_path))
        assert results == []

    def test_scan_no_config(self, tmp_path):
        """A directory with weights but no config should NOT be detected as local_dir."""
        model_dir = tmp_path / "NoConfig"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_text("dummy")
        # No config.json

        results = scan_vibevoice_models(str(tmp_path))
        assert results == []

    def test_scan_tokenizer_repo_in_results(self, tmp_path):
        """Scanned models should include tokenizer_repo."""
        model_dir = tmp_path / "VibeVoice-Large"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"test": true}')
        (model_dir / "model.safetensors").write_text("dummy")

        results = scan_vibevoice_models(str(tmp_path))
        assert len(results) == 1
        assert results[0]["tokenizer_repo"] == "Qwen/Qwen2.5-7B"

    def test_scan_multiple_models(self, tmp_path):
        """Multiple models should all be detected."""
        # HF dir
        dir1 = tmp_path / "ModelA"
        dir1.mkdir()
        (dir1 / "config.json").write_text('{}')
        (dir1 / "model.safetensors").write_text("dummy")
        # Standalone file
        (tmp_path / "model_b.safetensors").write_text("dummy")

        results = scan_vibevoice_models(str(tmp_path))
        assert len(results) == 2
        names = {r["name"] for r in results}
        assert "ModelA" in names
        assert "model_b" in names


class TestIsModelType:
    """CRIT-002: shared type-check helper."""

    def test_is_model_type_tts(self):
        assert is_model_type("VibeVoice-1.5B", "tts") is True
        assert is_model_type("VibeVoice-Large", "tts") is True

    def test_is_model_type_asr(self):
        assert is_model_type("VibeVoice-ASR", "asr") is True

    def test_is_model_type_streaming(self):
        assert is_model_type("VibeVoice-Realtime-0.5B", "streaming_tts") is True

    def test_is_model_type_variants(self):
        # Multiple types can be passed and any match wins.
        assert is_model_type("VibeVoice-ASR", "tts", "asr") is True
        assert is_model_type("VibeVoice-1.5B", "tts", "streaming_tts") is True

    def test_is_model_type_unknown_defaults_tts(self):
        # Unknown models default to "tts" (consistent with get_models_by_type).
        assert is_model_type("SomeUnknownModel", "tts") is True
        assert is_model_type("SomeUnknownModel", "asr") is False
