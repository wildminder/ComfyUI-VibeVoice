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
    get_tts_family_models,
    MODEL_WEIGHT_EXTENSIONS,
    is_model_type,
    normalize_asr_model_name,
    LEGACY_ASR_MODEL_NAMES,
    LEGACY_ASR_MODEL_TARGET,
)


class TestModelConfigs:
    """Test MODEL_CONFIGS dictionary."""

    def test_model_configs_has_1_5b(self):
        assert "VibeVoice-1.5B" in MODEL_CONFIGS

    def test_model_configs_has_large(self):
        assert "VibeVoice-7B" in MODEL_CONFIGS

    def test_model_configs_has_repo_id(self):
        assert MODEL_CONFIGS["VibeVoice-1.5B"]["repo_id"] == "microsoft/VibeVoice-1.5B"

    def test_model_configs_has_native_asr_model(self):
        """The native HF ASR model is the official downloadable ASR option."""
        assert (
            MODEL_CONFIGS["VibeVoice-ASR-HF"]["repo_id"]
            == "microsoft/VibeVoice-ASR-HF"
        )
        assert MODEL_CONFIGS["VibeVoice-ASR-HF"]["model_type"] == "asr"
        assert isinstance(MODEL_CONFIGS["VibeVoice-ASR-HF"]["size_gb"], (int, float))
        # The streaming family is retired from the dropdown: those checkpoints
        # speak an exclusive chunked protocol and produce garbage under batch
        # transcription (kept loadable only as locally discovered dirs).
        for name in ("VibeVoice-ASR-Streaming-1.5B", "VibeVoice-ASR-Streaming-7B"):
            assert name not in MODEL_CONFIGS
            assert name in LEGACY_ASR_MODEL_NAMES

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

    def test_bare_pt_voice_prompt_is_excluded(self, tmp_path):
        (tmp_path / "voice-prompt.pt").write_bytes(b"")
        (tmp_path / "model.safetensors").write_bytes(b"")
        results = scan_vibevoice_models(str(tmp_path))
        assert [item["name"] for item in results] == ["model"]

    def test_immediate_voices_directory_is_skipped(self, tmp_path):
        voices = tmp_path / "VoIcEs"
        voices.mkdir()
        (voices / "config.json").write_text("{}")
        (voices / "model.safetensors").write_bytes(b"")
        assert scan_vibevoice_models(str(tmp_path)) == []

    @pytest.mark.parametrize("extension", sorted(MODEL_WEIGHT_EXTENSIONS))
    def test_exact_weight_extension_allowlist(self, tmp_path, extension):
        (tmp_path / f"model{extension}").write_bytes(b"")
        results = scan_vibevoice_models(str(tmp_path))
        assert [item["name"] for item in results] == ["model"]

    def test_sibling_safetensors_still_found_beside_voice_directory(self, tmp_path):
        voices = tmp_path / "voices"
        voices.mkdir()
        (voices / "en-Carter_man.pt").write_bytes(b"")
        (tmp_path / "model.safetensors").write_bytes(b"")
        assert [item["name"] for item in scan_vibevoice_models(str(tmp_path))] == ["model"]


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

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("VibeVoice-ASR-Streaming-7B", "asr"),
            ("realtime-0.5b-local", "streaming_tts"),
            ("my-streaming-copy", "streaming_tts"),
            ("ordinary-local-copy", "tts"),
        ],
    )
    def test_name_inference_matrix(self, name, expected):
        assert is_model_type(name, expected) is True


class TestCombinedTTSSelector:
    def test_standard_then_realtime_and_no_asr(self):
        registry = {
            "realtime-local": {"type": "local_dir"},
            "standard-local": {"type": "local_dir"},
            "VibeVoice-ASR-HF": {"type": "official"},
            "VibeVoice-7B": {"type": "official"},
            "VibeVoice-Realtime-0.5B": {"type": "official"},
        }
        with patch(
            "ComfyUI_VibeVoice.modules.model_info.AVAILABLE_VIBEVOICE_MODELS",
            registry,
        ):
            assert tuple(get_tts_family_models()) == (
                "standard-local",
                "VibeVoice-7B",
                "realtime-local",
                "VibeVoice-Realtime-0.5B",
            )


class TestASRNameInference:
    """Name-based family inference: ASR names resolve to the ASR family."""

    def test_asr_names_are_asr(self):
        assert is_model_type("VibeVoice-ASR-HF", "asr") is True
        assert is_model_type("VibeVoice-ASR-Streaming-1.5B", "asr") is True
        assert is_model_type("VibeVoice-ASR-Streaming-7B", "asr") is True

    def test_asr_names_not_tts(self):
        assert is_model_type("VibeVoice-ASR-HF", "tts") is False
        assert is_model_type("VibeVoice-ASR-Streaming-1.5B", "tts") is False
        assert is_model_type("VibeVoice-ASR-Streaming-7B", "streaming_tts") is False

    def test_local_asr_dir_name_classifies_as_asr(self):
        """A locally discovered ASR checkpoint (not in MODEL_CONFIGS under its
        directory name) still lands in the ASR node's dropdown."""
        assert is_model_type("MyASR-Finetune", "asr") is True
        assert is_model_type("VibeVoice-ASR-Streaming-1.5B-fp16", "asr") is True


class TestNormalizeASRModelName:
    """Retired ASR names map onto the native ASR-HF checkpoint."""

    def test_retired_names_map_to_asr_hf(self):
        registry = {name: {"type": "official"} for name in
                    (*LEGACY_ASR_MODEL_NAMES, LEGACY_ASR_MODEL_TARGET)}
        for legacy in LEGACY_ASR_MODEL_NAMES:
            assert (
                normalize_asr_model_name(legacy, available=registry)
                == LEGACY_ASR_MODEL_TARGET
            )

    def test_local_checkpoint_of_retired_name_passes_through(self):
        for legacy in LEGACY_ASR_MODEL_NAMES:
            registry = {legacy: {"type": "local_dir", "path": "x"}}
            assert (
                normalize_asr_model_name(legacy, available=registry)
                == legacy
            )

    def test_target_missing_keeps_retired_name(self):
        for legacy in LEGACY_ASR_MODEL_NAMES:
            registry = {legacy: {"type": "official"}}
            assert (
                normalize_asr_model_name(legacy, available=registry)
                == legacy
            )

    def test_other_names_pass_through(self):
        registry = {LEGACY_ASR_MODEL_TARGET: {"type": "official"}}
        assert (
            normalize_asr_model_name(LEGACY_ASR_MODEL_TARGET, available=registry)
            == LEGACY_ASR_MODEL_TARGET
        )
        assert normalize_asr_model_name("", available=registry) == ""

    def test_populated_registry_resolves_retired_names(self):
        # With the registry populated (as __init__ does from MODEL_CONFIGS),
        # a saved workflow's retired name upgrades to the native ASR-HF.
        registry = {name: {"type": "official"} for name in MODEL_CONFIGS}
        assert LEGACY_ASR_MODEL_TARGET in registry
        for legacy in LEGACY_ASR_MODEL_NAMES:
            assert (
                normalize_asr_model_name(legacy, available=registry)
                == LEGACY_ASR_MODEL_TARGET
            )
