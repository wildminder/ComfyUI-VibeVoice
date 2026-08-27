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
# Phase 3 (plan 2026-08-18, D1): external instantiation is meta by default
# ====================================================================
class _TinyCtorModel(torch.nn.Module):
    """Minimal config-ctor stand-in for the vendored model classes."""

    def __init__(self, config):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)


class TestExternalInstantiationIsMeta:
    """Contract: external instantiation builds under a meta context by default."""

    def test_asr_instantiation_defaults_to_meta(self):
        with patch.object(
            external_loader, "VibeVoiceASRForConditionalGeneration", _TinyCtorModel
        ):
            model = external_loader._instantiate_asr_model(
                config=MagicMock(),
                attn_implementation="eager",
                final_load_dtype=torch.bfloat16,
            )
        params = list(model.parameters())
        assert params
        assert all(p.is_meta for p in params), "ASR construction must be meta by default"

    def test_asr_instantiation_use_meta_false_eager(self):
        with patch.object(
            external_loader, "VibeVoiceASRForConditionalGeneration", _TinyCtorModel
        ):
            model = external_loader._instantiate_asr_model(
                config=MagicMock(),
                attn_implementation="eager",
                final_load_dtype=torch.bfloat16,
                use_meta=False,
            )
        params = list(model.parameters())
        assert params
        assert all(not p.is_meta for p in params), "use_meta=False must be eager"

    def test_asr_instantiation_real_config_no_deprecation(self):
        """With a REAL transformers PretrainedConfig (v5: torch_dtype is a
        deprecated property), _instantiate_asr_model must record the dtype on
        the canonical ``dtype`` attribute and emit no deprecation warning.
        A capture handler is attached directly to the emitting transformers
        logger (its warning_once is lru_cache-wrapped, so the cache is
        cleared to keep the absence assertion non-vacuous)."""
        import logging as pylogging
        from transformers import PretrainedConfig

        config = PretrainedConfig()
        config.decoder_config = PretrainedConfig()

        logger = pylogging.getLogger("transformers.configuration_utils")
        records = []

        class _Capture(pylogging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Capture(level=pylogging.WARNING)
        logger.addHandler(handler)
        pylogging.Logger.warning_once.cache_clear()
        try:
            with patch.object(
                external_loader, "VibeVoiceASRForConditionalGeneration", _TinyCtorModel
            ):
                external_loader._instantiate_asr_model(
                    config=config,
                    attn_implementation="eager",
                    final_load_dtype=torch.bfloat16,
                )
        finally:
            logger.removeHandler(handler)
            pylogging.Logger.warning_once.cache_clear()

        assert config.dtype == torch.bfloat16
        assert config.decoder_config.dtype == torch.bfloat16
        assert not any(
            "`torch_dtype` is deprecated" in m for m in records
        ), "torch_dtype deprecation warning was emitted"


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

    def test_sidecar_config_missing_falls_back_to_7b(self, tmp_path):
        """No sidecar + config_name='VibeVoice-7B' → packaged 7B default.

        Plan 2026-08-27: the option 'VibeVoice-Large' was removed (it was an
        alias of 7B); the packaged FILE keeps its historical name.
        """
        weight = tmp_path / "foo.safetensors"
        weight.write_bytes(b"")

        result = resolve_sidecar_config(str(weight), "VibeVoice-7B")
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
        assert "VibeVoice-7B" in EXTERNAL_CONFIG_OPTIONS
        assert "VibeVoice-Realtime-0.5B" in EXTERNAL_CONFIG_OPTIONS
        assert "VibeVoice-ASR" in EXTERNAL_CONFIG_OPTIONS
        # Plan 2026-08-27: ambiguous alias removed from the visible list.
        assert "VibeVoice-Large" not in EXTERNAL_CONFIG_OPTIONS


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
        for key in ("config", "processor", "model", "model_name", "source_path", "is_streaming"):
            assert key in result, f"Missing key: {key}"

    def test_bundle_does_not_retain_state_dict(self, weight_file):
        """Plan 2026-08-18 D7/RC-4: the bundle must NOT retain the state dict."""
        result, _ = self._run(weight_file)
        assert "state_dict" not in result, (
            "bundle must not retain the state dict (dead RAM, RC-4)")

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

    def test_external_load_uses_assign_and_reties(self, weight_file):
        """Plan 2026-08-18 D2: external path delegates to _apply_state_dict."""
        with patch.object(
            external_loader.VibeVoiceLoader, "_apply_state_dict",
            return_value=([], []),
        ) as m_apply:
            self._run(weight_file)

        m_apply.assert_called_once()
        # The in-memory state dict is passed to the shared helper.
        args, _ = m_apply.call_args
        assert args[1] is not None  # state_dict argument present

    def test_load_external_model_applies_dtype(self, weight_file):
        """Plan 2026-08-18 D4/RC-3: conditional cast helper is invoked with the final dtype."""
        with patch.object(
            external_loader, "cast_model_to_dtype_if_needed"
        ) as m_cast:
            self._run(weight_file, dtype_str="fp32")

        m_cast.assert_called_once()
        args, _ = m_cast.call_args
        assert args[1] == torch.float32

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
        """ASR bundle has all required keys plus is_asr=True.

        Plan 2026-08-18 D7/RC-4: the bundle no longer retains ``state_dict``.
        """
        result, _ = self._run(weight_file)
        assert isinstance(result, dict)
        for key in ("config", "processor", "model", "model_name", "source_path", "is_streaming", "is_asr"):
            assert key in result, f"Missing key: {key}"
        assert "state_dict" not in result, "bundle must not retain the state dict (RC-4)"
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
        """Plan 2026-08-18 D4/RC-3: ASR branch uses the conditional cast helper."""
        with patch.object(
            external_loader, "cast_model_to_dtype_if_needed"
        ) as m_cast:
            self._run(weight_file, dtype_str="fp32")

        m_cast.assert_called_once()
        args, _ = m_cast.call_args
        assert args[1] == torch.float32

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


# ====================================================================
# Fix B: low-bit / naive quantization detection
# ====================================================================

import json
import struct
import logging

from ComfyUI_VibeVoice.modules.external_loader import (
    _inspect_safetensors_quantization,
    _inspect_gguf_quantization,
    warn_if_lowbit_quantization,
)


def _write_safetensors_header(path, tensor_specs):
    """Write a minimal safetensors file containing ONLY the header (no tensor data).

    ``_inspect_safetensors_quantization`` reads only the header, so no data bytes
    are needed. ``tensor_specs`` is a list of (name, dtype, shape) tuples.
    """
    header = {}
    offset = 0
    for name, dtype, shape in tensor_specs:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset]}
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
    return str(path)


class TestInspectSafetensorsQuantization:
    """Test _inspect_safetensors_quantization header parsing."""

    def test_naive_int8_cast_detected(self, tmp_path):
        """Many I8 tensors with no scale metadata -> naive cast detected."""
        specs = [(f"model.layers.{i}.weight", "I8", [16, 16]) for i in range(10)]
        specs += [(f"model.layers.{i}.norm", "BF16", [16]) for i in range(2)]
        path = _write_safetensors_header(tmp_path / "naive.safetensors", specs)

        info = _inspect_safetensors_quantization(path)
        assert info is not None
        assert info["total"] == 12
        assert info["int_count"] == 10
        assert info["has_scale_meta"] is False

    def test_proper_quant_with_scales_not_flagged(self, tmp_path):
        """I8 tensors WITH scale tensors -> has_scale_meta True (proper quant)."""
        specs = [(f"model.layers.{i}.weight", "I8", [16, 16]) for i in range(10)]
        specs += [(f"model.layers.{i}.weight_scale", "BF16", [16]) for i in range(10)]
        path = _write_safetensors_header(tmp_path / "proper.safetensors", specs)

        info = _inspect_safetensors_quantization(path)
        assert info["int_count"] == 10
        assert info["has_scale_meta"] is True

    def test_full_precision_no_int(self, tmp_path):
        """All BF16/F32 -> int_count 0."""
        specs = [(f"model.layers.{i}.weight", "BF16", [16, 16]) for i in range(10)]
        path = _write_safetensors_header(tmp_path / "fp.safetensors", specs)

        info = _inspect_safetensors_quantization(path)
        assert info["int_count"] == 0
        assert info["has_scale_meta"] is False

    def test_unparseable_file_returns_none(self, tmp_path):
        """A non-safetensors / corrupt file returns None (no crash)."""
        path = tmp_path / "corrupt.safetensors"
        path.write_bytes(b"not a safetensors file")
        assert _inspect_safetensors_quantization(str(path)) is None


class TestWarnIfLowbitQuantization:
    """Test warn_if_lowbit_quantization end-to-end warning behavior."""

    def test_naive_int8_warns(self, tmp_path, caplog):
        """Naive int8 cast (no scales) must emit a warning."""
        specs = [(f"model.layers.{i}.weight", "I8", [16, 16]) for i in range(10)]
        path = _write_safetensors_header(tmp_path / "naive.safetensors", specs)

        with caplog.at_level(logging.WARNING, logger="ComfyUI_VibeVoice.modules.external_loader"):
            warn_if_lowbit_quantization(path)

        assert any("naive int cast" in r.message for r in caplog.records)

    def test_proper_quant_no_warning(self, tmp_path, caplog):
        """Proper quant with scales must NOT warn."""
        specs = [(f"model.layers.{i}.weight", "I8", [16, 16]) for i in range(10)]
        specs += [(f"model.layers.{i}.weight_scale", "BF16", [16]) for i in range(10)]
        path = _write_safetensors_header(tmp_path / "proper.safetensors", specs)

        with caplog.at_level(logging.WARNING, logger="ComfyUI_VibeVoice.modules.external_loader"):
            warn_if_lowbit_quantization(path)

        assert not any("naive int cast" in r.message for r in caplog.records)

    def test_full_precision_no_warning(self, tmp_path, caplog):
        """Full-precision file must NOT warn."""
        specs = [(f"model.layers.{i}.weight", "BF16", [16, 16]) for i in range(10)]
        path = _write_safetensors_header(tmp_path / "fp.safetensors", specs)

        with caplog.at_level(logging.WARNING, logger="ComfyUI_VibeVoice.modules.external_loader"):
            warn_if_lowbit_quantization(path)

        assert not any("naive int cast" in r.message for r in caplog.records)

    def test_gguf_lowbit_warns(self, tmp_path, caplog):
        """GGUF with many sub-4-bit I-quant tensors must warn."""
        # Build a fake GGUFReader with a tensor table of IQ3_XXS tensors.
        from gguf.constants import GGMLQuantizationType

        fake_tensors = [MagicMock() for _ in range(10)]
        for t in fake_tensors:
            t.tensor_type = GGMLQuantizationType.IQ3_XXS
        fake_reader = MagicMock()
        fake_reader.tensors = fake_tensors

        gguf_path = str(tmp_path / "lowbit.gguf")
        with patch("gguf.GGUFReader", return_value=fake_reader), \
             caplog.at_level(logging.WARNING, logger="ComfyUI_VibeVoice.modules.external_loader"):
            warn_if_lowbit_quantization(gguf_path)

        assert any("sub-4-bit" in r.message for r in caplog.records)

    def test_gguf_highbit_no_warning(self, tmp_path, caplog):
        """GGUF with Q8_0 tensors must NOT warn."""
        from gguf.constants import GGMLQuantizationType

        fake_tensors = [MagicMock() for _ in range(10)]
        for t in fake_tensors:
            t.tensor_type = GGMLQuantizationType.Q8_0
        fake_reader = MagicMock()
        fake_reader.tensors = fake_tensors

        gguf_path = str(tmp_path / "highbit.gguf")
        with patch("gguf.GGUFReader", return_value=fake_reader), \
             caplog.at_level(logging.WARNING, logger="ComfyUI_VibeVoice.modules.external_loader"):
            warn_if_lowbit_quantization(gguf_path)

        assert not any("sub-4-bit" in r.message for r in caplog.records)

    def test_non_weight_extension_no_crash(self, tmp_path, caplog):
        """An unrecognized extension must be a no-op (no crash, no warning)."""
        path = tmp_path / "model.bin"
        path.write_bytes(b"")
        with caplog.at_level(logging.WARNING, logger="ComfyUI_VibeVoice.modules.external_loader"):
            warn_if_lowbit_quantization(str(path))
        assert not any("naive int cast" in r.message for r in caplog.records)


# ====================================================================
# Plan 2026-08-27 (Phase 3): config/weights reconciliation
# ====================================================================

from ComfyUI_VibeVoice.modules.config_detect import WeightsFingerprint
from ComfyUI_VibeVoice.modules.external_loader import reconcile_config


def _cfg_stub(hidden: int, vocab: int):
    """A config object whose decoder_config carries hidden/vocab."""
    decoder = MagicMock(spec=["hidden_size", "vocab_size"])
    decoder.hidden_size = hidden
    decoder.vocab_size = vocab
    cfg = MagicMock(spec=["decoder_config"])
    cfg.decoder_config = decoder
    return cfg


_FP_7B = WeightsFingerprint(
    hidden_size=3584, vocab_size=152064,
    source_key="model.language_model.embed_tokens.weight",
)
_FP_15B = WeightsFingerprint(
    hidden_size=1536, vocab_size=151936,
    source_key="model.language_model.embed_tokens.weight",
)


class TestReconcileConfig:
    """Pure decision helper: fingerprint wins over any config source."""

    def test_no_fingerprint_keeps_selection(self):
        name, changed = reconcile_config("VibeVoice-1.5B", _cfg_stub(1536, 151936), None)
        assert (name, changed) == ("VibeVoice-1.5B", False)

    def test_matching_fingerprint_keeps_selection(self):
        name, changed = reconcile_config(
            "VibeVoice-1.5B", _cfg_stub(1536, 151936), _FP_15B
        )
        assert (name, changed) == ("VibeVoice-1.5B", False)

    def test_mismatch_swaps_to_detected_family(self, caplog):
        with caplog.at_level(
            logging.WARNING, logger="ComfyUI_VibeVoice.modules.external_loader"
        ):
            name, changed = reconcile_config(
                "VibeVoice-1.5B", _cfg_stub(1536, 151936), _FP_7B
            )
        assert (name, changed) == ("VibeVoice-7B", True)
        assert any("Config mismatch" in r.message for r in caplog.records)
        assert any("VibeVoice-7B" in r.message for r in caplog.records)

    def test_mismatch_reverse_direction(self, caplog):
        with caplog.at_level(
            logging.WARNING, logger="ComfyUI_VibeVoice.modules.external_loader"
        ):
            name, changed = reconcile_config(
                "VibeVoice-7B", _cfg_stub(3584, 152064), _FP_15B
            )
        assert (name, changed) == ("VibeVoice-1.5B", True)

    def test_config_without_fingerprint_keeps_selection(self):
        """Unknown config layouts (no decoder_config) are never swapped."""
        cfg = MagicMock(spec=[])
        name, changed = reconcile_config("VibeVoice-Realtime-0.5B", cfg, _FP_7B)
        assert (name, changed) == ("VibeVoice-Realtime-0.5B", False)

    def test_no_warning_when_matching(self, caplog):
        with caplog.at_level(
            logging.WARNING, logger="ComfyUI_VibeVoice.modules.external_loader"
        ):
            reconcile_config("VibeVoice-7B", _cfg_stub(3584, 152064), _FP_7B)
        assert not any("Config mismatch" in r.message for r in caplog.records)


class TestConfigFingerprint:
    """config_fingerprint(): decoder_config extraction.

    The vendored config classes are mocked in the test env (conftest), so
    the packaged JSONs are exercised via the dict form — the same nested
    'decoder_config' layout the loader produces.
    """

    def test_packaged_7b_config(self):
        import json

        from ComfyUI_VibeVoice.modules.config_detect import config_fingerprint

        path = external_loader._get_packaged_config_path("VibeVoice-7B")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        assert config_fingerprint(raw) == (3584, 152064)

    def test_packaged_15b_config(self):
        import json

        from ComfyUI_VibeVoice.modules.config_detect import config_fingerprint

        path = external_loader._get_packaged_config_path("VibeVoice-1.5B")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        assert config_fingerprint(raw) == (1536, 151936)

    def test_missing_decoder_config_returns_none(self):
        from ComfyUI_VibeVoice.modules.config_detect import config_fingerprint

        assert config_fingerprint(MagicMock(spec=[])) is None

    def test_dict_form_supported(self):
        from ComfyUI_VibeVoice.modules.config_detect import config_fingerprint

        raw = {"decoder_config": {"hidden_size": 3584, "vocab_size": 152064}}
        assert config_fingerprint(raw) == (3584, 152064)


class TestLoaderReconciliationWiring:
    """load_external_vibevoice_model swaps a contradicting config (D5)."""

    @pytest.fixture
    def weight_file(self, tmp_path):
        weight = tmp_path / "model.safetensors"
        weight.write_bytes(b"dummy")
        return str(weight)

    def _run(self, weight_file, config_name, weights_fp, caplog=None):
        """Run the loader with heavy deps mocked; fingerprint controlled.

        resolve_sidecar_config records every config_name it receives and
        returns a per-name fake path; _load_config returns a config stub
        whose fingerprint matches the requested family.
        """
        resolve_calls = []

        def fake_resolve(weight_path, name):
            resolve_calls.append(name)
            return f"/fake/{name}.json"

        family_fp = {
            "VibeVoice-7B": (3584, 152064),
            "VibeVoice-1.5B": (1536, 151936),
        }

        def fake_load_config(config_path, name):
            hidden, vocab = family_fp.get(name, (0, 0))
            return _cfg_stub(hidden, vocab)

        fake_model = MagicMock()
        fake_model.load_state_dict.return_value = ([], [])
        fake_model.to.return_value = fake_model

        with patch.object(
            external_loader.comfy.utils, "load_torch_file",
            return_value={"w": torch.zeros(1)},
        ), patch(
            "ComfyUI_VibeVoice.modules.config_detect.fingerprint_weights",
            return_value=weights_fp,
        ), patch.object(
            external_loader, "resolve_sidecar_config", side_effect=fake_resolve
        ), patch.object(
            external_loader.VibeVoiceLoader, "_load_config",
            side_effect=fake_load_config,
        ), patch.object(
            external_loader.VibeVoiceLoader, "_load_tokenizer",
            return_value=MagicMock(),
        ), patch.object(
            external_loader.VibeVoiceLoader, "_load_processor",
            return_value=MagicMock(),
        ), patch.object(
            external_loader.VibeVoiceLoader, "_instantiate_model",
            return_value=fake_model,
        ), patch.object(
            external_loader, "resolve_sidecar_preprocessor", return_value=""
        ), patch.object(
            external_loader, "resolve_sidecar_tokenizer_dir", return_value="/fake"
        ), patch.object(
            external_loader, "resolve_dtype", return_value=torch.float32
        ), patch.object(
            external_loader, "resolve_attention_mode", side_effect=lambda m, q: m
        ), patch.object(
            external_loader, "get_attn_implementation_for_load", return_value="eager"
        ), patch.object(
            external_loader, "VibeVoiceStreamingConfig", _FakeStreamingCfg
        ):
            result = load_external_vibevoice_model(weight_file, config_name)
        return result, resolve_calls

    def test_mismatched_selection_is_swapped(self, weight_file, caplog):
        with caplog.at_level(
            logging.WARNING, logger="ComfyUI_VibeVoice.modules.external_loader"
        ):
            result, resolve_calls = self._run(
                weight_file, "VibeVoice-1.5B", _FP_7B
            )
        # First resolution used the selection, second used the detection.
        assert resolve_calls == ["VibeVoice-1.5B", "VibeVoice-7B"]
        assert result["model_name"] == "VibeVoice-7B"
        assert any("Config mismatch" in r.message for r in caplog.records)

    def test_matching_selection_not_swapped(self, weight_file, caplog):
        with caplog.at_level(
            logging.WARNING, logger="ComfyUI_VibeVoice.modules.external_loader"
        ):
            result, resolve_calls = self._run(
                weight_file, "VibeVoice-1.5B", _FP_15B
            )
        assert resolve_calls == ["VibeVoice-1.5B"]
        assert result["model_name"] == "VibeVoice-1.5B"
        assert not any("Config mismatch" in r.message for r in caplog.records)

    def test_unfingerprintable_weights_honor_selection(self, weight_file):
        result, resolve_calls = self._run(weight_file, "VibeVoice-7B", None)
        assert resolve_calls == ["VibeVoice-7B"]
        assert result["model_name"] == "VibeVoice-7B"

    def test_sidecar_config_mismatch_still_swapped(self, weight_file, caplog):
        """A sidecar-resolved config is not exempt: fingerprint wins (D5)."""
        with caplog.at_level(
            logging.WARNING, logger="ComfyUI_VibeVoice.modules.external_loader"
        ):
            result, resolve_calls = self._run(
                weight_file, "VibeVoice-7B", _FP_15B
            )
        assert resolve_calls == ["VibeVoice-7B", "VibeVoice-1.5B"]
        assert result["model_name"] == "VibeVoice-1.5B"

    def test_asr_branch_never_runs_detection(self, weight_file):
        """ASR dispatch happens before any fingerprinting."""
        with patch(
            "ComfyUI_VibeVoice.modules.config_detect.fingerprint_weights"
        ) as m_fp, patch.object(
            external_loader, "load_external_vibevoice_asr_model",
            return_value={"model_name": "asr"},
        ) as m_asr:
            result = load_external_vibevoice_model(
                weight_file, "VibeVoice-ASR"
            )
        m_asr.assert_called_once()
        m_fp.assert_not_called()
        assert result["model_name"] == "asr"


# ====================================================================
# Phase 4 (plan 2026-08-27, D6): Auto-detect semantics
# ====================================================================

from ComfyUI_VibeVoice.modules.external_loader import (  # noqa: E402
    AUTO_CONFIG_NAME,
    ASR_CONFIG_NAMES,
    is_asr_config_name,
    resolve_auto_config_name,
)


class TestResolveAutoConfigName:
    """resolve_auto_config_name(): detection -> family or actionable error."""

    @pytest.fixture
    def weight_file(self, tmp_path):
        weight = tmp_path / "model.safetensors"
        weight.write_bytes(b"dummy")
        return str(weight)

    def test_conclusive_fingerprint_returns_family(self, weight_file, caplog):
        with caplog.at_level(
            logging.INFO, logger="ComfyUI_VibeVoice.modules.external_loader"
        ):
            name = resolve_auto_config_name(weight_file, weights_fp=_FP_7B)
        assert name == "VibeVoice-7B"
        assert any("Auto-detected" in r.message for r in caplog.records)

    def test_precomputed_fingerprint_skips_recomputation(self, weight_file):
        with patch(
            "ComfyUI_VibeVoice.modules.config_detect.fingerprint_weights"
        ) as m_fp:
            name = resolve_auto_config_name(weight_file, weights_fp=_FP_15B)
        assert name == "VibeVoice-1.5B"
        m_fp.assert_not_called()

    def test_computes_fingerprint_when_not_provided(self, tmp_path):
        weight = tmp_path / "model.gguf"
        weight.write_bytes(b"dummy")
        sentinel_reader = object()
        with patch(
            "ComfyUI_VibeVoice.modules.config_detect.fingerprint_weights",
            return_value=_FP_7B,
        ) as m_fp:
            name = resolve_auto_config_name(
                str(weight), gguf_reader=sentinel_reader
            )
        assert name == "VibeVoice-7B"
        # A caller-provided GGUF reader must be reused (no re-open, D3).
        assert m_fp.call_args[1]["gguf_reader"] is sentinel_reader

    def test_inconclusive_raises_actionable_error(self, weight_file):
        with patch(
            "ComfyUI_VibeVoice.modules.config_detect.fingerprint_weights",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="could not determine"):
                resolve_auto_config_name(weight_file)

    def test_error_message_offers_explicit_choices_only(self, weight_file):
        with patch(
            "ComfyUI_VibeVoice.modules.config_detect.fingerprint_weights",
            return_value=None,
        ):
            with pytest.raises(ValueError) as excinfo:
                resolve_auto_config_name(weight_file)
        message = str(excinfo.value)
        offered = message.split("one of:")[1]
        for option in EXTERNAL_CONFIG_OPTIONS:
            if option != AUTO_CONFIG_NAME:
                assert option in offered
        # The sentinel is never offered as an explicit choice.
        assert AUTO_CONFIG_NAME not in offered

    def test_unknown_family_fingerprint_raises(self, weight_file):
        foreign = WeightsFingerprint(
            hidden_size=999, vocab_size=888, source_key="k"
        )
        with pytest.raises(ValueError, match="could not determine"):
            resolve_auto_config_name(weight_file, weights_fp=foreign)


class TestAutoDetectLoaderSemantics:
    """load_external_vibevoice_model resolves AUTO_CONFIG_NAME (Step 4.2)."""

    @pytest.fixture
    def weight_file(self, tmp_path):
        weight = tmp_path / "model.safetensors"
        weight.write_bytes(b"dummy")
        return str(weight)

    def test_auto_adopts_detected_family(self, weight_file, caplog):
        harness = TestLoaderReconciliationWiring()
        with caplog.at_level(
            logging.INFO, logger="ComfyUI_VibeVoice.modules.external_loader"
        ):
            result, resolve_calls = harness._run(
                weight_file, AUTO_CONFIG_NAME, _FP_7B
            )
        assert resolve_calls == ["VibeVoice-7B"]
        assert result["model_name"] == "VibeVoice-7B"
        assert any("Auto-detected" in r.message for r in caplog.records)

    def test_auto_inconclusive_fails_before_state_dict_load(self, weight_file):
        """Fail-fast: no heavy load work before the actionable error (D6)."""
        with patch.object(
            external_loader.comfy.utils, "load_torch_file",
            return_value={"w": torch.zeros(1)},
        ) as m_load, patch(
            "ComfyUI_VibeVoice.modules.config_detect.fingerprint_weights",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="could not determine"):
                load_external_vibevoice_model(weight_file, AUTO_CONFIG_NAME)
        m_load.assert_not_called()

    def test_auto_never_dispatches_asr(self):
        """Auto is TTS-branch only; ASR remains an explicit selection."""
        assert AUTO_CONFIG_NAME not in ASR_CONFIG_NAMES
        assert is_asr_config_name(AUTO_CONFIG_NAME) is False
