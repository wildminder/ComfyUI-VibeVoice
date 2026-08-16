"""Tests for nodes/tts_node.py - V3 schema validation."""

import pytest
from unittest.mock import patch

from ComfyUI_VibeVoice.nodes.tts_node import VibeVoiceTTSNode
from ComfyUI_VibeVoice.nodes.asr_node import VibeVoiceASRNode


class TestVibeVoiceTTSNodeSchema:
    """Test VibeVoiceTTSNode V3 schema definition."""

    @classmethod
    def _get_schema(cls):
        return VibeVoiceTTSNode.define_schema()

    def _get_input_ids(self):
        schema = self._get_schema()
        return [inp.id for inp in schema.inputs]

    def test_schema_node_id(self):
        schema = self._get_schema()
        assert schema.node_id == "VibeVoiceTTS"

    def test_schema_display_name(self):
        schema = self._get_schema()
        assert schema.display_name == "VibeVoice TTS"

    def test_schema_category(self):
        schema = self._get_schema()
        assert schema.category == "audio/tts"

    def test_schema_has_model_name_input(self):
        assert "model_name" in self._get_input_ids()

    def test_schema_has_text_input(self):
        assert "text" in self._get_input_ids()

    def test_schema_has_cfg_scale(self):
        assert "cfg_scale" in self._get_input_ids()

    def test_schema_has_inference_steps(self):
        assert "inference_steps" in self._get_input_ids()

    def test_schema_has_seed(self):
        assert "seed" in self._get_input_ids()

    def test_schema_has_attention_mode(self):
        assert "attention_mode" in self._get_input_ids()

    def test_schema_has_quantize_llm_4bit(self):
        assert "quantize_llm_4bit" in self._get_input_ids()

    def test_schema_has_device(self):
        assert "device" in self._get_input_ids()

    def test_schema_has_dtype(self):
        assert "dtype" in self._get_input_ids()

    def test_schema_has_speaker_inputs(self):
        input_ids = self._get_input_ids()
        assert "speaker_1_voice" in input_ids
        assert "speaker_2_voice" in input_ids
        assert "speaker_3_voice" in input_ids
        assert "speaker_4_voice" in input_ids

    def test_schema_output_is_audio(self):
        schema = self._get_schema()
        assert len(schema.outputs) >= 1


class TestVibeVoiceTTSNodeValidate:
    """Test VibeVoiceTTSNode.validate_inputs."""

    def test_validate_valid_model(self):
        with patch("ComfyUI_VibeVoice.nodes.tts_node.AVAILABLE_VIBEVOICE_MODELS", {"VibeVoice-1.5B": {}}):
            result = VibeVoiceTTSNode.validate_inputs(model_name="VibeVoice-1.5B")
            assert result is True

    def test_validate_invalid_model(self):
        with patch("ComfyUI_VibeVoice.nodes.tts_node.AVAILABLE_VIBEVOICE_MODELS", {"VibeVoice-1.5B": {}}):
            result = VibeVoiceTTSNode.validate_inputs(model_name="NonExistent")
            assert isinstance(result, str)
            assert "NonExistent" in result

    def test_validate_none_model(self):
        result = VibeVoiceTTSNode.validate_inputs(model_name=None)
        assert result is True


class TestVibeVoiceTTSNodeSchemaModelFiltering:
    """CRIT-002: TTS dropdown must only expose non-streaming TTS models.

    Streaming (realtime) models are excluded: they require the streaming
    generation path (generate_streaming_audio) exposed by the dedicated
    VibeVoice Realtime TTS node. Routing them through this node raised
    ``VibeVoiceStreamingProcessor.__call__() got an unexpected keyword
    argument 'text'``.
    """

    def _get_model_options(self):
        schema = VibeVoiceTTSNode.define_schema()
        for inp in schema.inputs:
            if inp.id == "model_name":
                return inp.options
        return None

    def test_tts_options_exclude_asr(self):
        with patch(
            "ComfyUI_VibeVoice.modules.model_info.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-1.5B": {"type": "official"}, "VibeVoice-ASR": {"type": "official"}},
        ):
            options = self._get_model_options()
        assert options is not None
        assert "VibeVoice-ASR" not in options
        assert "VibeVoice-1.5B" in options

    def test_tts_options_exclude_streaming(self):
        with patch(
            "ComfyUI_VibeVoice.modules.model_info.AVAILABLE_VIBEVOICE_MODELS",
            {
                "VibeVoice-1.5B": {"type": "official"},
                "VibeVoice-ASR": {"type": "official"},
                "VibeVoice-Realtime-0.5B": {"type": "official"},
            },
        ):
            options = self._get_model_options()
        assert "VibeVoice-Realtime-0.5B" not in options
        assert "VibeVoice-ASR" not in options
        assert "VibeVoice-1.5B" in options


class TestVibeVoiceTTSNodeValidateTypeGuard:
    """CRIT-002: validate_inputs must reject non-TTS model types."""

    def test_validate_accepts_tts_model(self):
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-1.5B": {}},
        ):
            result = VibeVoiceTTSNode.validate_inputs(model_name="VibeVoice-1.5B")
        assert result is True

    def test_validate_rejects_asr_model(self):
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-ASR": {}},
        ):
            result = VibeVoiceTTSNode.validate_inputs(model_name="VibeVoice-ASR")
        assert isinstance(result, str)
        assert "ASR" in result

    def test_validate_rejects_streaming_model(self):
        """Streaming models must be rejected with a pointer to the Realtime node."""
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-Realtime-0.5B": {}},
        ):
            result = VibeVoiceTTSNode.validate_inputs(
                model_name="VibeVoice-Realtime-0.5B"
            )
        assert isinstance(result, str)
        assert "Realtime" in result


class TestVibeVoiceASRNodeValidateTypeGuard:
    """CRIT-002: ASR node validate_inputs must reject non-ASR model types."""

    def test_validate_asr_accepts_asr_model(self):
        with patch(
            "ComfyUI_VibeVoice.nodes.asr_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-ASR": {}},
        ):
            result = VibeVoiceASRNode.validate_inputs(model_name="VibeVoice-ASR")
        assert result is True

    def test_validate_asr_rejects_tts_model(self):
        with patch(
            "ComfyUI_VibeVoice.nodes.asr_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-1.5B": {}},
        ):
            result = VibeVoiceASRNode.validate_inputs(model_name="VibeVoice-1.5B")
        assert isinstance(result, str)
        assert "TTS" in result


class TestVibeVoiceTTSNodeAttentionAvailability:
    """IMP-001: attention options must reflect hardware availability."""

    def _get_attention_options(self):
        schema = VibeVoiceTTSNode.define_schema()
        for inp in schema.inputs:
            if inp.id == "attention_mode":
                return inp.options
        return None

    def test_tts_attention_options_exclude_flash_when_unavailable(self):
        with patch(
            "ComfyUI_VibeVoice.modules.attention_utils.check_flash_attention_available",
            return_value=False,
        ):
            options = self._get_attention_options()
        assert options is not None
        assert "flash_attention_2" not in options

    def test_tts_attention_options_include_flash_when_available(self):
        with patch(
            "ComfyUI_VibeVoice.modules.attention_utils.check_flash_attention_available",
            return_value=True,
        ):
            options = self._get_attention_options()
        assert "flash_attention_2" in options
