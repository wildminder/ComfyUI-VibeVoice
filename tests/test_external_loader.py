"""Tests for modules/external_loader.py - External model loading.

Covers:
- Path resolution helpers (sidecar config / preprocessor / tokenizer dir)
- load_external_vibevoice_model() core loading function
"""

import os
import pytest
from unittest.mock import MagicMock, patch

import torch

from ComfyUI_VibeVoice.modules import external_loader
from ComfyUI_VibeVoice.modules.external_loader import (
    resolve_sidecar_config,
    resolve_sidecar_preprocessor,
    resolve_sidecar_tokenizer_dir,
    load_external_vibevoice_model,
    EXTERNAL_CONFIG_OPTIONS,
)


# Fake streaming config class for isinstance() checks (MagicMock cannot be
# used with isinstance — see tests/test_loader.py for the same pattern).
class _FakeStreamingCfg:
    pass


# ====================================================================
# Path resolution helpers
# ====================================================================

class TestPathResolution:
    """Test sidecar path resolution helpers."""

    def test_sidecar_config_found(self, tmp_path):
        """<weight>.config.json sidecar is preferred when present."""
        weight = tmp_path / "foo.safetensors"
        weight.write_bytes(b"")
        sidecar = tmp_path / "foo.safetensors.config.json"
        sidecar.write_text("{}")

        result = resolve_sidecar_config(str(weight), "VibeVoice-1.5B")
        assert result == str(sidecar)

    def test_sidecar_config_dir_fallback(self, tmp_path):
        """config.json in the same directory is used when no <weight>.config.json."""
        weight = tmp_path / "foo.safetensors"
        weight.write_bytes(b"")
        dir_config = tmp_path / "config.json"
        dir_config.write_text("{}")

        result = resolve_sidecar_config(str(weight), "VibeVoice-1.5B")
        assert result == str(dir_config)

    def test_sidecar_config_missing_falls_back_to_packaged(self, tmp_path):
        """No sidecar + config_name='VibeVoice-1.5B' → packaged 1.5B default."""
        weight = tmp_path / "foo.safetensors"
        weight.write_bytes(b"")

        result = resolve_sidecar_config(str(weight), "VibeVoice-1.5B")
        assert result.endswith("default_VibeVoice-1.5B_config.json")
        assert os.path.exists(result)

    def test_sidecar_config_missing_falls_back_to_large(self, tmp_path):
        """No sidecar + config_name='VibeVoice-Large' → packaged Large default."""
        weight = tmp_path / "foo.safetensors"
        weight.write_bytes(b"")

        result = resolve_sidecar_config(str(weight), "VibeVoice-Large")
        assert result.endswith("default_VibeVoice-Large_config.json")
        assert os.path.exists(result)

    def test_sidecar_config_no_packaged_default_raises(self, tmp_path):
        """No sidecar + config_name without packaged default → FileNotFoundError."""
        weight = tmp_path / "foo.safetensors"
        weight.write_bytes(b"")

        with pytest.raises(FileNotFoundError):
            resolve_sidecar_config(str(weight), "VibeVoice-Realtime-0.5B")

    def test_sidecar_preprocessor_found(self, tmp_path):
        """<weight>.preprocessor.json sidecar is returned when present."""
        weight = tmp_path / "foo.safetensors"
        weight.write_bytes(b"")
        sidecar = tmp_path / "foo.safetensors.preprocessor.json"
        sidecar.write_text("{}")

        result = resolve_sidecar_preprocessor(str(weight))
        assert result == str(sidecar)

    def test_sidecar_preprocessor_dir_fallback(self, tmp_path):
        """preprocessor_config.json in the same directory is used as fallback."""
        weight = tmp_path / "foo.safetensors"
        weight.write_bytes(b"")
        dir_config = tmp_path / "preprocessor_config.json"
        dir_config.write_text("{}")

        result = resolve_sidecar_preprocessor(str(weight))
        assert result == str(dir_config)

    def test_sidecar_preprocessor_missing(self, tmp_path):
        """No preprocessor sidecar → empty string."""
        weight = tmp_path / "foo.safetensors"
        weight.write_bytes(b"")

        result = resolve_sidecar_preprocessor(str(weight))
        assert result == ""

    def test_sidecar_tokenizer_dir(self, tmp_path):
        """Tokenizer dir is the directory containing the weight file."""
        weight = tmp_path / "foo.safetensors"
        weight.write_bytes(b"")

        result = resolve_sidecar_tokenizer_dir(str(weight))
        assert result == str(tmp_path)

    def test_external_config_options_contains_expected(self):
        """EXTERNAL_CONFIG_OPTIONS contains the documented config names."""
        assert "VibeVoice-1.5B" in EXTERNAL_CONFIG_OPTIONS
        assert "VibeVoice-Large" in EXTERNAL_CONFIG_OPTIONS
        assert "VibeVoice-Realtime-0.5B" in EXTERNAL_CONFIG_OPTIONS
        assert "VibeVoice-ASR" in EXTERNAL_CONFIG_OPTIONS


# ====================================================================
# load_external_vibevoice_model
# ====================================================================

class TestLoadExternalModel:
    """Test the core load_external_vibevoice_model() function."""

    @pytest.fixture
    def weight_file(self, tmp_path):
        """Create a dummy weight file."""
        weight = tmp_path / "model.safetensors"
        weight.write_bytes(b"dummy")
        return str(weight)

    def _run(self, weight_file, streaming=False, **kwargs):
        """Run load_external_vibevoice_model with all deps mocked.

        Returns (result, mocks_dict).
        """
        fake_state_dict = {"model.language_model.weight": torch.zeros(2, 2)}
        if streaming:
            fake_config = _FakeStreamingCfg()
        else:
            fake_config = MagicMock(spec=[])
        fake_tokenizer = MagicMock()
        fake_processor = MagicMock()
        fake_model = MagicMock()
        fake_model.load_state_dict.return_value = ([], [])
        # model = model.to(dtype=...) reassigns model to the .to() return
        # value; return the same mock so subsequent calls (eval) land on it.
        fake_model.to.return_value = fake_model

        mocks = {}
        with patch.object(
            external_loader.comfy.utils, "load_torch_file", return_value=fake_state_dict
        ) as m_load_torch, patch.object(
            external_loader.VibeVoiceLoader, "_load_config", return_value=fake_config
        ) as m_load_config, patch.object(
            external_loader.VibeVoiceLoader, "_load_tokenizer", return_value=fake_tokenizer
        ) as m_load_tokenizer, patch.object(
            external_loader.VibeVoiceLoader, "_load_processor", return_value=fake_processor
        ) as m_load_processor, patch.object(
            external_loader.VibeVoiceLoader, "_instantiate_model", return_value=fake_model
        ) as m_instantiate, patch.object(
            external_loader, "resolve_sidecar_config", return_value="/fake/config.json"
        ), patch.object(
            external_loader, "resolve_sidecar_preprocessor", return_value=""
        ), patch.object(
            external_loader, "resolve_sidecar_tokenizer_dir", return_value="/fake/dir"
        ), patch.object(
            external_loader, "resolve_dtype", return_value=torch.float32
        ), patch.object(
            external_loader, "resolve_attention_mode", side_effect=lambda m, q: m
        ), patch.object(
            external_loader, "get_attn_implementation_for_load", return_value="eager"
        ), patch.object(
            external_loader, "VibeVoiceStreamingConfig", _FakeStreamingCfg
        ):
            mocks.update({
                "state_dict": fake_state_dict,
                "config": fake_config,
                "tokenizer": fake_tokenizer,
                "processor": fake_processor,
                "model": fake_model,
                "load_torch": m_load_torch,
                "load_config": m_load_config,
                "load_tokenizer": m_load_tokenizer,
                "load_processor": m_load_processor,
                "instantiate": m_instantiate,
            })
            result = load_external_vibevoice_model(weight_file, "VibeVoice-1.5B", **kwargs)
        return result, mocks

    def test_load_external_model_returns_dict_with_required_keys(self, weight_file):
        """Returned bundle has all required keys."""
        result, _ = self._run(weight_file)

        assert isinstance(result, dict)
        for key in ("state_dict", "config", "processor", "model", "model_name", "source_path", "is_streaming"):
            assert key in result, f"Missing key: {key}"

    def test_load_external_model_calls_load_torch_file_with_cpu(self, weight_file):
        """load_torch_file is called with device=cpu."""
        _, mocks = self._run(weight_file)

        mocks["load_torch"].assert_called_once()
        call_args, call_kwargs = mocks["load_torch"].call_args
        # device may be positional or keyword
        device = call_kwargs.get("device")
        if device is None and len(call_args) > 1:
            device = call_args[1]
        assert device == torch.device("cpu")

    def test_load_external_model_loads_state_dict_into_model(self, weight_file):
        """model.load_state_dict is called with the loaded state dict, strict=False."""
        _, mocks = self._run(weight_file)

        fake_model = mocks["model"]
        fake_model.load_state_dict.assert_called_once()
        args, kwargs = fake_model.load_state_dict.call_args
        assert args[0] is mocks["state_dict"]
        assert kwargs.get("strict") is False

    def test_load_external_model_applies_dtype(self, weight_file):
        """model.to(dtype=final_dtype) is called."""
        _, mocks = self._run(weight_file, dtype_str="fp32")

        fake_model = mocks["model"]
        fake_model.to.assert_called()
        dtype_calls = [c for c in fake_model.to.call_args_list if "dtype" in c[1]]
        assert len(dtype_calls) > 0

    def test_load_external_model_eval_mode(self, weight_file):
        """model.eval() is called."""
        _, mocks = self._run(weight_file)
        mocks["model"].eval.assert_called_once()

    def test_load_external_model_streaming_detection(self, weight_file):
        """Streaming config → is_streaming=True in the bundle."""
        result, _ = self._run(weight_file, streaming=True)
        assert result["is_streaming"] is True

    def test_load_external_model_non_streaming_detection(self, weight_file):
        """Non-streaming config → is_streaming=False in the bundle."""
        result, _ = self._run(weight_file, streaming=False)
        assert result["is_streaming"] is False

    def test_load_external_model_model_name_in_bundle(self, weight_file):
        """model_name in the bundle matches config_name."""
        result, _ = self._run(weight_file)
        assert result["model_name"] == "VibeVoice-1.5B"

    def test_load_external_model_source_path_in_bundle(self, weight_file):
        """source_path in the bundle matches the weight path."""
        result, _ = self._run(weight_file)
        assert result["source_path"] == weight_file

    def test_load_external_model_missing_weight_raises(self, tmp_path):
        """Non-existent weight file → FileNotFoundError."""
        missing = str(tmp_path / "nonexistent.safetensors")
        with pytest.raises(FileNotFoundError):
            load_external_vibevoice_model(missing, "VibeVoice-1.5B")

    def test_load_external_model_quantize_4bit(self, weight_file):
        """use_llm_4bit=True → replace_with_bnb_linear is called."""
        with patch(
            "transformers.integrations.bitsandbytes.replace_with_bnb_linear"
        ) as mock_bnb:
            self._run(weight_file, use_llm_4bit=True)
            mock_bnb.assert_called_once()

    def test_load_external_model_sage_attention(self, weight_file):
        """attention_mode='sage' + compatible → set_sage_attention called."""
        with patch.object(
            external_loader, "check_sage_attention_compatible", return_value=True
        ), patch.object(
            external_loader, "set_sage_attention", create=True
        ) as mock_sage:
            self._run(weight_file, attention_mode="sage")
            mock_sage.assert_called_once()

    def test_load_external_model_instantiate_called_with_config(self, weight_file):
        """_instantiate_model is called with the resolved config."""
        _, mocks = self._run(weight_file)

        mocks["instantiate"].assert_called_once()
        call_kwargs = mocks["instantiate"].call_args[1]
        assert call_kwargs["config"] is mocks["config"]

    def test_load_external_model_processor_called_with_tokenizer(self, weight_file):
        """_load_processor is called with the loaded tokenizer."""
        _, mocks = self._run(weight_file)

        mocks["load_processor"].assert_called_once()
        call_args = mocks["load_processor"].call_args[0]
        assert call_args[0] is mocks["tokenizer"]

    def test_load_external_model_tts_bundle_is_asr_false(self, weight_file):
        """TTS/streaming bundles are explicitly marked is_asr=False."""
        result, _ = self._run(weight_file)
        assert result["is_asr"] is False


# ====================================================================
# ASR config-name detection
# ====================================================================

class TestIsASRConfigName:
    """Test is_asr_config_name() dispatch helper."""

    def test_asr_config_name_detected(self):
        from ComfyUI_VibeVoice.modules.external_loader import is_asr_config_name
        assert is_asr_config_name("VibeVoice-ASR") is True

    def test_tts_config_names_not_asr(self):
        from ComfyUI_VibeVoice.modules.external_loader import is_asr_config_name
        for name in ("VibeVoice-1.5B", "VibeVoice-Large", "VibeVoice-Realtime-0.5B"):
            assert is_asr_config_name(name) is False


# ====================================================================
# ASR external loading branch
# ====================================================================

class TestLoadExternalASRModel:
    """Test the ASR branch: load_external_vibevoice_asr_model()."""

    @pytest.fixture
    def weight_file(self, tmp_path):
        """Create a dummy weight file."""
        weight = tmp_path / "asr_model.safetensors"
        weight.write_bytes(b"dummy")
        return str(weight)

    def _run(self, weight_file, **kwargs):
        """Run load_external_vibevoice_asr_model with all deps mocked.

        Returns (result, mocks_dict).
        """
        fake_state_dict = {"model.language_model.weight": torch.zeros(2, 2)}
        fake_config = MagicMock(spec=[])
        fake_tokenizer = MagicMock()
        fake_processor = MagicMock()
        fake_model = MagicMock()
        fake_model.load_state_dict.return_value = ([], [])
        fake_model.to.return_value = fake_model

        mocks = {}
        with patch.object(
            external_loader.comfy.utils, "load_torch_file", return_value=fake_state_dict
        ) as m_load_torch, patch.object(
            external_loader, "_load_asr_config", return_value=fake_config
        ) as m_load_config, patch.object(
            external_loader, "_load_asr_tokenizer", return_value=fake_tokenizer
        ) as m_load_tokenizer, patch.object(
            external_loader, "_load_asr_processor", return_value=fake_processor
        ) as m_load_processor, patch.object(
            external_loader, "_instantiate_asr_model", return_value=fake_model
        ) as m_instantiate, patch.object(
            external_loader, "resolve_sidecar_config", return_value="/fake/config.json"
        ), patch.object(
            external_loader, "resolve_sidecar_preprocessor", return_value=""
        ), patch.object(
            external_loader, "resolve_sidecar_tokenizer_dir", return_value="/fake/dir"
        ), patch.object(
            external_loader, "resolve_dtype", return_value=torch.float32
        ), patch.object(
            external_loader, "resolve_attention_mode",
            side_effect=lambda m, quantize_4bit=False: m
        ), patch.object(
            external_loader, "get_attn_implementation_for_load", return_value="sdpa"
        ):
            mocks.update({
                "state_dict": fake_state_dict,
                "config": fake_config,
                "tokenizer": fake_tokenizer,
                "processor": fake_processor,
                "model": fake_model,
                "load_torch": m_load_torch,
                "load_config": m_load_config,
                "load_tokenizer": m_load_tokenizer,
                "load_processor": m_load_processor,
                "instantiate": m_instantiate,
            })
            result = external_loader.load_external_vibevoice_asr_model(
                weight_file, "VibeVoice-ASR", **kwargs
            )
        return result, mocks

    def test_asr_bundle_has_required_keys(self, weight_file):
        """ASR bundle has all required keys plus is_asr=True."""
        result, _ = self._run(weight_file)
        assert isinstance(result, dict)
        for key in ("state_dict", "config", "processor", "model", "model_name", "source_path", "is_streaming", "is_asr"):
            assert key in result, f"Missing key: {key}"
        assert result["is_asr"] is True
        assert result["is_streaming"] is False

    def test_asr_dispatch_from_main_loader(self, weight_file):
        """config_name='VibeVoice-ASR' routes to the ASR branch."""
        with patch.object(
            external_loader, "load_external_vibevoice_asr_model", return_value={"is_asr": True}
        ) as mock_asr:
            result = load_external_vibevoice_model(weight_file, "VibeVoice-ASR")
        mock_asr.assert_called_once()
        assert result["is_asr"] is True

    def test_asr_calls_load_torch_file_with_cpu(self, weight_file):
        """ASR branch loads the state dict onto CPU."""
        _, mocks = self._run(weight_file)
        mocks["load_torch"].assert_called_once()
        call_args, call_kwargs = mocks["load_torch"].call_args
        device = call_kwargs.get("device")
        if device is None and len(call_args) > 1:
            device = call_args[1]
        assert device == torch.device("cpu")

    def test_asr_loads_state_dict_into_model(self, weight_file):
        """ASR model.load_state_dict is called with strict=False."""
        _, mocks = self._run(weight_file)
        fake_model = mocks["model"]
        fake_model.load_state_dict.assert_called_once()
        args, kwargs = fake_model.load_state_dict.call_args
        assert args[0] is mocks["state_dict"]
        assert kwargs.get("strict") is False

    def test_asr_applies_dtype(self, weight_file):
        """ASR model.to(dtype=...) is called."""
        _, mocks = self._run(weight_file, dtype_str="fp32")
        fake_model = mocks["model"]
        dtype_calls = [c for c in fake_model.to.call_args_list if "dtype" in c[1]]
        assert len(dtype_calls) > 0

    def test_asr_eval_mode(self, weight_file):
        """ASR model.eval() is called."""
        _, mocks = self._run(weight_file)
        mocks["model"].eval.assert_called_once()

    def test_asr_instantiate_called_with_config(self, weight_file):
        """_instantiate_asr_model is called with the resolved ASR config."""
        _, mocks = self._run(weight_file)
        mocks["instantiate"].assert_called_once()
        call_kwargs = mocks["instantiate"].call_args[1]
        assert call_kwargs["config"] is mocks["config"]

    def test_asr_processor_called_with_tokenizer(self, weight_file):
        """_load_asr_processor is called with the loaded ASR tokenizer."""
        _, mocks = self._run(weight_file)
        mocks["load_processor"].assert_called_once()
        call_args = mocks["load_processor"].call_args[0]
        assert call_args[0] is mocks["tokenizer"]

    def test_asr_missing_weight_raises(self, tmp_path):
        """Non-existent weight file → FileNotFoundError."""
        missing = str(tmp_path / "nonexistent.safetensors")
        with pytest.raises(FileNotFoundError):
            external_loader.load_external_vibevoice_asr_model(missing, "VibeVoice-ASR")

    def test_asr_no_4bit_quantization(self, weight_file):
        """ASR branch never applies 4-bit quantization (no bnb call)."""
        with patch(
            "transformers.integrations.bitsandbytes.replace_with_bnb_linear"
        ) as mock_bnb:
            self._run(weight_file)
            mock_bnb.assert_not_called()

    def test_asr_model_name_in_bundle(self, weight_file):
        """model_name in the ASR bundle matches config_name."""
        result, _ = self._run(weight_file)
        assert result["model_name"] == "VibeVoice-ASR"

    def test_asr_source_path_in_bundle(self, weight_file):
        """source_path in the ASR bundle matches the weight path."""
        result, _ = self._run(weight_file)
        assert result["source_path"] == weight_file
