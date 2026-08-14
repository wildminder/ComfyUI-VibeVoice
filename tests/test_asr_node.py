"""Tests for nodes/asr_node.py - V3 schema validation."""

import pytest
from unittest.mock import patch, MagicMock

from ComfyUI_VibeVoice.nodes.asr_node import VibeVoiceASRNode


class TestVibeVoiceASRNodeSchema:
    """Test VibeVoiceASRNode V3 schema definition."""

    @classmethod
    def _get_schema(cls):
        return VibeVoiceASRNode.define_schema()

    def _get_input_ids(self):
        schema = self._get_schema()
        return [inp.id for inp in schema.inputs]

    def test_schema_node_id(self):
        schema = self._get_schema()
        assert schema.node_id == "VibeVoiceASR"

    def test_schema_display_name(self):
        schema = self._get_schema()
        assert schema.display_name == "VibeVoice ASR"

    def test_schema_category(self):
        schema = self._get_schema()
        assert schema.category == "audio/asr"

    def test_schema_has_model_name_input(self):
        assert "model_name" in self._get_input_ids()

    def test_schema_has_audio_input(self):
        assert "audio" in self._get_input_ids()

    def test_schema_has_context_info(self):
        assert "context_info" in self._get_input_ids()

    def test_schema_has_max_new_tokens(self):
        assert "max_new_tokens" in self._get_input_ids()

    def test_schema_has_temperature(self):
        assert "temperature" in self._get_input_ids()

    def test_schema_has_device(self):
        assert "device" in self._get_input_ids()

    def test_schema_has_dtype(self):
        assert "dtype" in self._get_input_ids()

    def test_schema_has_attention_mode(self):
        assert "attention_mode" in self._get_input_ids()

    def test_schema_has_force_offload(self):
        assert "force_offload" in self._get_input_ids()

    def test_schema_has_two_outputs(self):
        """ASR node should have transcription + segments outputs."""
        schema = self._get_schema()
        assert len(schema.outputs) == 2


class TestVibeVoiceASRNodeValidate:
    """Test VibeVoiceASRNode.validate_inputs."""

    def test_validate_valid_model(self):
        with patch("ComfyUI_VibeVoice.nodes.asr_node.AVAILABLE_VIBEVOICE_MODELS", {"VibeVoice-ASR": {}}):
            result = VibeVoiceASRNode.validate_inputs(model_name="VibeVoice-ASR")
            assert result is True

    def test_validate_invalid_model(self):
        with patch("ComfyUI_VibeVoice.nodes.asr_node.AVAILABLE_VIBEVOICE_MODELS", {"VibeVoice-ASR": {}}):
            result = VibeVoiceASRNode.validate_inputs(model_name="NonExistent")
            assert isinstance(result, str)
            assert "NonExistent" in result

    def test_validate_none_model(self):
        result = VibeVoiceASRNode.validate_inputs(model_name=None)
        assert result is True


class TestVibeVoiceASRNodeExecute:
    """CRIT-001 S4: execute must route through the patcher-based load path."""

    def test_asr_execute_uses_patcher(self):
        stub_patcher = MagicMock()
        stub_patcher.is_loaded = True
        model = MagicMock()
        processor = MagicMock()

        audio = {"waveform": MagicMock(), "sample_rate": 24000}

        with patch(
            "ComfyUI_VibeVoice.nodes.asr_node.load_asr_model_patched",
            return_value=(stub_patcher, model, processor),
        ) as mock_load, patch(
            "ComfyUI_VibeVoice.nodes.asr_node.transcribe_audio",
            return_value=("hello world", []),
        ) as mock_transcribe, patch(
            "ComfyUI_VibeVoice.nodes.asr_node.force_offload_asr_model"
        ) as mock_offload:

            result = VibeVoiceASRNode.execute(
                model_name="VibeVoice-ASR",
                audio=audio,
                context_info="",
                max_new_tokens=32768,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                num_beams=1,
                device="cpu",
                dtype="auto",
                attention_mode="sdpa",
                force_offload=False,
            )

        # The patched loader must be used (not the legacy direct loader).
        mock_load.assert_called_once()
        mock_transcribe.assert_called_once()
        assert mock_transcribe.call_args.kwargs["model"] is model
        assert mock_transcribe.call_args.kwargs["processor"] is processor
        # Two string outputs: transcription + segments JSON.
        assert result is not None
        mock_offload.assert_not_called()

    def test_asr_execute_offloads_when_requested(self):
        stub_patcher = MagicMock()
        stub_patcher.is_loaded = True
        model = MagicMock()
        processor = MagicMock()
        audio = {"waveform": MagicMock(), "sample_rate": 24000}

        with patch(
            "ComfyUI_VibeVoice.nodes.asr_node.load_asr_model_patched",
            return_value=(stub_patcher, model, processor),
        ), patch(
            "ComfyUI_VibeVoice.nodes.asr_node.transcribe_audio",
            return_value=("hello world", []),
        ), patch(
            "ComfyUI_VibeVoice.nodes.asr_node.force_offload_asr_model"
        ) as mock_offload:

            VibeVoiceASRNode.execute(
                model_name="VibeVoice-ASR",
                audio=audio,
                context_info="",
                max_new_tokens=32768,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                num_beams=1,
                device="cpu",
                dtype="auto",
                attention_mode="sdpa",
                force_offload=True,
            )

        # With force_offload=True the patcher must be handed to offload.
        mock_offload.assert_called_once_with("VibeVoice-ASR", stub_patcher)
