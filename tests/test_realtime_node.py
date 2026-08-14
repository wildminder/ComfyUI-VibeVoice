"""Tests for nodes/realtime_node.py - VibeVoice Realtime (streaming TTS) node.

NTH-001: the streaming ``VibeVoice-Realtime-0.5B`` model is configured but was
never reachable through a node. This suite locks the schema (streaming-only
model combo), input validation (rejects non-streaming types), the extension
registration, and that ``execute`` routes through the shared patcher /
``generate_streaming_audio`` path.
"""

import pytest
from unittest.mock import patch, MagicMock

from ComfyUI_VibeVoice.nodes.realtime_node import VibeVoiceRealtimeNode


def _model_options(schema):
    for inp in schema.inputs:
        if inp.id == "model_name":
            return list(inp.options)
    raise AssertionError("model_name input not found in schema")


class TestRealtimeNodeSchema:
    """VibeVoiceRealtimeNode schema exposes streaming models only."""

    def test_schema_node_id(self):
        schema = VibeVoiceRealtimeNode.define_schema()
        assert schema.node_id == "VibeVoiceRealtime"

    def test_schema_display_name(self):
        schema = VibeVoiceRealtimeNode.define_schema()
        assert schema.display_name == "VibeVoice Realtime TTS"

    def test_schema_category(self):
        schema = VibeVoiceRealtimeNode.define_schema()
        assert schema.category == "audio/tts"

    def test_schema_uses_streaming_models_only(self):
        with patch(
            "ComfyUI_VibeVoice.modules.model_info.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-Realtime-0.5B": {"type": "official"}},
        ):
            options = _model_options(VibeVoiceRealtimeNode.define_schema())
        assert options == ["VibeVoice-Realtime-0.5B"]

    def test_schema_excludes_asr_and_tts_models(self):
        with patch(
            "ComfyUI_VibeVoice.modules.model_info.AVAILABLE_VIBEVOICE_MODELS",
            {
                "VibeVoice-Realtime-0.5B": {"type": "official"},
                "VibeVoice-ASR": {"type": "official"},
                "VibeVoice-1.5B": {"type": "official"},
            },
        ):
            options = _model_options(VibeVoiceRealtimeNode.define_schema())
        assert "VibeVoice-Realtime-0.5B" in options
        assert "VibeVoice-ASR" not in options
        assert "VibeVoice-1.5B" not in options

    def test_schema_has_stream_toggle(self):
        schema = VibeVoiceRealtimeNode.define_schema()
        ids = [inp.id for inp in schema.inputs]
        assert "stream" in ids
        assert "force_offload" in ids
        assert "speaker_1_voice" in ids

    def test_schema_has_audio_output(self):
        schema = VibeVoiceRealtimeNode.define_schema()
        assert len(schema.outputs) == 1


class TestRealtimeNodeValidate:
    """VibeVoiceRealtimeNode.validate_inputs accepts only streaming_tts models."""

    def test_validate_accepts_streaming_model(self):
        with patch(
            "ComfyUI_VibeVoice.nodes.realtime_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-Realtime-0.5B": {"type": "official"}},
        ):
            result = VibeVoiceRealtimeNode.validate_inputs(model_name="VibeVoice-Realtime-0.5B")
        assert result is True

    def test_validate_rejects_asr_model(self):
        with patch(
            "ComfyUI_VibeVoice.nodes.realtime_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-Realtime-0.5B": {"type": "official"}, "VibeVoice-ASR": {"type": "official"}},
        ):
            result = VibeVoiceRealtimeNode.validate_inputs(model_name="VibeVoice-ASR")
        assert isinstance(result, str)
        assert "streaming" in result.lower() or "streaming TTS" in result

    def test_validate_rejects_tts_model(self):
        with patch(
            "ComfyUI_VibeVoice.nodes.realtime_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-Realtime-0.5B": {"type": "official"}, "VibeVoice-1.5B": {"type": "official"}},
        ):
            result = VibeVoiceRealtimeNode.validate_inputs(model_name="VibeVoice-1.5B")
        assert isinstance(result, str)
        assert "streaming" in result.lower()

    def test_validate_none_model(self):
        assert VibeVoiceRealtimeNode.validate_inputs(model_name=None) is True


class TestRealtimeNodeExecute:
    """NTH-001: execute must route through the patcher + streaming generate path."""

    def test_execute_calls_streaming_generate(self):
        stub_patcher = MagicMock()
        stub_patcher.is_loaded = True
        model = MagicMock()
        processor = MagicMock()

        with patch(
            "ComfyUI_VibeVoice.nodes.realtime_node.load_vibevoice_model",
            return_value=(stub_patcher, model, processor),
        ) as mock_load, patch(
            "ComfyUI_VibeVoice.nodes.realtime_node.generate_streaming_audio",
            return_value=(MagicMock(), 24000),
        ) as mock_generate, patch(
            "ComfyUI_VibeVoice.nodes.realtime_node.force_offload_model"
        ) as mock_offload, patch(
            "ComfyUI_VibeVoice.nodes.realtime_node.ui"
        ) as mock_ui:
            result = VibeVoiceRealtimeNode.execute(
                model_name="VibeVoice-Realtime-0.5B",
                text="[1] Hello world",
                quantize_llm_4bit=False,
                attention_mode="sdpa",
                cfg_scale=1.3,
                inference_steps=10,
                seed=42,
                do_sample=True,
                temperature=0.95,
                top_p=0.95,
                top_k=0,
                stream=False,
                force_offload=False,
                device="cpu",
                dtype="auto",
            )

        mock_load.assert_called_once()
        # The model/processor from the patcher loader must reach the streaming generator.
        mock_generate.assert_called_once()
        assert mock_generate.call_args.kwargs["model"] is model
        assert mock_generate.call_args.kwargs["processor"] is processor
        assert mock_generate.call_args.kwargs["text"] == "[1] Hello world"
        # Two audio outputs: waveform + (optional) stream — single Audio.Output here.
        assert result is not None
        mock_offload.assert_not_called()
        # ui.PreviewAudio was invoked for the preview.
        mock_ui.PreviewAudio.assert_called_once()

    def test_execute_offloads_when_requested(self):
        stub_patcher = MagicMock()
        stub_patcher.is_loaded = True
        model = MagicMock()
        processor = MagicMock()

        with patch(
            "ComfyUI_VibeVoice.nodes.realtime_node.load_vibevoice_model",
            return_value=(stub_patcher, model, processor),
        ), patch(
            "ComfyUI_VibeVoice.nodes.realtime_node.generate_streaming_audio",
            return_value=(MagicMock(), 24000),
        ), patch(
            "ComfyUI_VibeVoice.nodes.realtime_node.force_offload_model"
        ) as mock_offload, patch(
            "ComfyUI_VibeVoice.nodes.realtime_node.ui"
        ):
            VibeVoiceRealtimeNode.execute(
                model_name="VibeVoice-Realtime-0.5B",
                text="[1] Hello world",
                quantize_llm_4bit=False,
                attention_mode="sdpa",
                cfg_scale=1.3,
                inference_steps=10,
                seed=42,
                do_sample=True,
                temperature=0.95,
                top_p=0.95,
                top_k=0,
                stream=False,
                force_offload=True,
                device="cpu",
                dtype="auto",
            )

        # force_offload=True must hand the patcher to the warm offload path.
        mock_offload.assert_called_once_with(stub_patcher, "VibeVoice-Realtime-0.5B", warm=True)
