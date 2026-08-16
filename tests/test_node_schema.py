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

    def test_schema_has_external_model_input(self):
        """'external_model' is in the input ids."""
        assert "external_model" in self._get_input_ids()

    def test_external_model_input_is_optional(self):
        """The external_model input is marked optional."""
        schema = self._get_schema()
        ext_input = next(inp for inp in schema.inputs if inp.id == "external_model")
        assert ext_input.optional is True

    def test_external_model_input_type(self):
        """The external_model input has the VIBEVOICE_MODEL type."""
        schema = self._get_schema()
        ext_input = next(inp for inp in schema.inputs if inp.id == "external_model")
        assert ext_input.get_io_type() == "VIBEVOICE_MODEL"

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


class TestVibeVoiceTTSNodeValidateExternalModel:
    """External model input bypasses model_name dropdown validation."""

    def test_validate_skips_model_name_when_external_model_provided(self):
        """external_model present → validation passes even with unknown model_name."""
        bundle = {"model": object(), "processor": object(), "model_name": "ext"}
        result = VibeVoiceTTSNode.validate_inputs(
            external_model=bundle, model_name="nonexistent_model"
        )
        assert result is True

    def test_validate_still_validates_model_name_when_no_external(self):
        """Without external_model, an unknown model_name still fails validation."""
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-1.5B": {}},
        ):
            result = VibeVoiceTTSNode.validate_inputs(model_name="nonexistent_model")
        assert isinstance(result, str)
        assert "nonexistent_model" in result

    def test_validate_bypasses_when_external_model_linked_but_none(self):
        """REGRESSION: a *connected* external_model resolves to None during prompt
        validation (ComfyUI has no execution cache yet — see execution.get_input_data
        / mark_missing). The bypass must therefore trigger on the input's *presence*
        in kwargs, not on a non-None value. Previously this fell through and rejected
        the stale model_name widget (e.g. a streaming model on the TTS node)."""
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.is_model_type",
            return_value=True,  # model_name would be flagged as streaming
        ):
            result = VibeVoiceTTSNode.validate_inputs(
                external_model=None, model_name="VibeVoice-Realtime-0.5B"
            )
        assert result is True

    def test_validate_rejects_streaming_model_when_external_absent(self):
        """Without a connected external_model, a streaming model_name is still rejected."""
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-Realtime-0.5B": {}},
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.is_model_type",
            side_effect=lambda name, typ: typ == "streaming_tts",
        ):
            result = VibeVoiceTTSNode.validate_inputs(model_name="VibeVoice-Realtime-0.5B")
        assert isinstance(result, str)
        assert "streaming" in result


class TestVibeVoiceTTSNodeExecuteExternalModel:
    """TTS node execute() must branch on external_model presence."""

    def _make_bundle(self, is_streaming=False, is_asr=False):
        return {
            "model": object(),
            "processor": object(),
            "model_name": "ExtModel",
            "source_path": "/fake/model.safetensors",
            "is_streaming": is_streaming,
            "is_asr": is_asr,
        }

    def test_execute_with_external_model_calls_load_from_external(self):
        """external_model present → load_vibevoice_from_external is called."""
        from unittest.mock import MagicMock

        bundle = self._make_bundle(is_streaming=False)
        mock_patcher = MagicMock()
        mock_model = MagicMock()
        mock_processor = MagicMock()

        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_from_external",
            return_value=(mock_patcher, mock_model, mock_processor),
        ) as mock_ext, patch(
            "ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_model"
        ) as mock_std, patch(
            "ComfyUI_VibeVoice.nodes.tts_node.generate_audio",
            return_value=(MagicMock(), 24000),
        ), patch(
            "ComfyUI_VibeVoice.modules.audio_utils.parse_script_1_based",
            return_value=([], [1]),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.ui.PreviewAudio", MagicMock()
        ):
            VibeVoiceTTSNode.execute(
                model_name="IgnoredModel",
                text="[1] Hello",
                quantize_llm_4bit=False,
                attention_mode="sdpa",
                cfg_scale=1.3,
                inference_steps=10,
                seed=42,
                do_sample=True,
                temperature=0.95,
                top_p=0.95,
                top_k=0,
                force_offload=False,
                device="cpu",
                dtype="fp32",
                external_model=bundle,
            )

        mock_ext.assert_called_once()
        mock_std.assert_not_called()

    def test_execute_without_external_model_calls_load_vibevoice_model(self):
        """external_model=None → the standard dropdown loader is called."""
        from unittest.mock import MagicMock

        mock_patcher = MagicMock()
        mock_model = MagicMock()
        mock_processor = MagicMock()

        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_model",
            return_value=(mock_patcher, mock_model, mock_processor),
        ) as mock_std, patch(
            "ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_from_external"
        ) as mock_ext, patch(
            "ComfyUI_VibeVoice.nodes.tts_node.generate_audio",
            return_value=(MagicMock(), 24000),
        ), patch(
            "ComfyUI_VibeVoice.modules.audio_utils.parse_script_1_based",
            return_value=([], [1]),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.ui.PreviewAudio", MagicMock()
        ):
            VibeVoiceTTSNode.execute(
                model_name="VibeVoice-1.5B",
                text="[1] Hello",
                quantize_llm_4bit=False,
                attention_mode="sdpa",
                cfg_scale=1.3,
                inference_steps=10,
                seed=42,
                do_sample=True,
                temperature=0.95,
                top_p=0.95,
                top_k=0,
                force_offload=False,
                device="cpu",
                dtype="fp32",
                external_model=None,
            )

        mock_std.assert_called_once()
        mock_ext.assert_not_called()

    def test_execute_external_model_skips_dropdown(self):
        """When external_model is provided, model_name is ignored."""
        from unittest.mock import MagicMock

        bundle = self._make_bundle(is_streaming=False)
        mock_patcher = MagicMock()

        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_from_external",
            return_value=(mock_patcher, MagicMock(), MagicMock()),
        ) as mock_ext, patch(
            "ComfyUI_VibeVoice.nodes.tts_node.generate_audio",
            return_value=(MagicMock(), 24000),
        ), patch(
            "ComfyUI_VibeVoice.modules.audio_utils.parse_script_1_based",
            return_value=([], [1]),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.ui.PreviewAudio", MagicMock()
        ):
            VibeVoiceTTSNode.execute(
                model_name="SomeDropdownModel",
                text="[1] Hello",
                quantize_llm_4bit=False,
                attention_mode="sdpa",
                cfg_scale=1.3,
                inference_steps=10,
                seed=42,
                do_sample=True,
                temperature=0.95,
                top_p=0.95,
                top_k=0,
                force_offload=False,
                device="cpu",
                dtype="fp32",
                external_model=bundle,
            )

        # The external loader receives the bundle, not the dropdown name.
        call_args = mock_ext.call_args
        assert call_args[0][0] is bundle

    def test_execute_rejects_streaming_external_model(self):
        """A streaming external model on the TTS node raises ValueError."""
        bundle = self._make_bundle(is_streaming=True)

        with pytest.raises(ValueError, match="Realtime"):
            VibeVoiceTTSNode.execute(
                model_name="VibeVoice-1.5B",
                text="[1] Hello",
                quantize_llm_4bit=False,
                attention_mode="sdpa",
                cfg_scale=1.3,
                inference_steps=10,
                seed=42,
                do_sample=True,
                temperature=0.95,
                top_p=0.95,
                top_k=0,
                force_offload=False,
                device="cpu",
                dtype="fp32",
                external_model=bundle,
            )

    def test_execute_rejects_asr_external_model(self):
        """An ASR external model on the TTS node raises ValueError."""
        bundle = self._make_bundle(is_streaming=False, is_asr=True)

        with pytest.raises(ValueError, match="ASR"):
            VibeVoiceTTSNode.execute(
                model_name="VibeVoice-1.5B",
                text="[1] Hello",
                quantize_llm_4bit=False,
                attention_mode="sdpa",
                cfg_scale=1.3,
                inference_steps=10,
                seed=42,
                do_sample=True,
                temperature=0.95,
                top_p=0.95,
                top_k=0,
                force_offload=False,
                device="cpu",
                dtype="fp32",
                external_model=bundle,
            )


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
