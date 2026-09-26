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
        assert schema.category == "WMNodes/sound/asr"

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

    def test_asr_schema_has_external_model_input(self):
        """Phase 4.1: the ASR node exposes an optional external_model input."""
        assert "external_model" in self._get_input_ids()

    def test_asr_external_model_input_is_optional(self):
        """Phase 4.1: external_model must be optional (dropdown path unchanged)."""
        schema = self._get_schema()
        ext_input = next(inp for inp in schema.inputs if inp.id == "external_model")
        assert getattr(ext_input, "optional", False) is True

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

    def test_validate_external_model_bypasses_dropdown_check(self):
        """Phase 4.1: an external model skips model_name validation entirely."""
        bundle = {"model": MagicMock(), "processor": MagicMock(), "model_name": "ext-asr"}
        # Even with an unknown model_name, validation passes when external_model is set.
        result = VibeVoiceASRNode.validate_inputs(
            model_name="NonExistent", external_model=bundle
        )
        assert result is True

    def test_validate_bypasses_when_external_model_linked_but_none(self):
        """REGRESSION: a *connected* external_model resolves to None during prompt
        validation (ComfyUI has no execution cache yet). The bypass must trigger on
        the input's *presence* in kwargs, not on a non-None value."""
        with patch(
            "ComfyUI_VibeVoice.nodes.asr_node.is_model_type",
            return_value=False,  # model_name would be flagged as non-ASR
        ):
            result = VibeVoiceASRNode.validate_inputs(
                model_name="VibeVoice-1.5B", external_model=None
            )
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


class TestVibeVoiceASRNodeExecuteExternal:
    """Phase 4.1: execute() branches on the optional external_model input."""

    @staticmethod
    def _make_bundle(model_name="ext-asr", is_streaming=False):
        return {
            "model": MagicMock(),
            "processor": MagicMock(),
            "model_name": model_name,
            "is_streaming": is_streaming,
            "source_path": "fake.safetensors",
        }

    def test_asr_execute_with_external_model_calls_load_from_external(self):
        """external_model present → load_asr_from_external is used."""
        stub_patcher = MagicMock()
        model = MagicMock()
        processor = MagicMock()
        bundle = self._make_bundle()
        audio = {"waveform": MagicMock(), "sample_rate": 24000}

        with patch(
            "ComfyUI_VibeVoice.nodes.asr_node.load_asr_from_external",
            return_value=(stub_patcher, model, processor),
        ) as mock_ext, patch(
            "ComfyUI_VibeVoice.nodes.asr_node.load_asr_model_patched"
        ) as mock_std, patch(
            "ComfyUI_VibeVoice.nodes.asr_node.transcribe_audio",
            return_value=("hello world", []),
        ) as mock_transcribe:
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
                external_model=bundle,
            )

        mock_ext.assert_called_once()
        # The bundle itself must be passed through.
        assert mock_ext.call_args.args[0] is bundle
        # The standard dropdown loader must NOT be called.
        mock_std.assert_not_called()
        # Transcription uses the external model/processor.
        assert mock_transcribe.call_args.kwargs["model"] is model
        assert mock_transcribe.call_args.kwargs["processor"] is processor
        assert result is not None

    def test_asr_execute_without_external_model_calls_standard_loader(self):
        """external_model=None → the standard patched loader is used."""
        stub_patcher = MagicMock()
        model = MagicMock()
        processor = MagicMock()
        audio = {"waveform": MagicMock(), "sample_rate": 24000}

        with patch(
            "ComfyUI_VibeVoice.nodes.asr_node.load_asr_model_patched",
            return_value=(stub_patcher, model, processor),
        ) as mock_std, patch(
            "ComfyUI_VibeVoice.nodes.asr_node.load_asr_from_external"
        ) as mock_ext, patch(
            "ComfyUI_VibeVoice.nodes.asr_node.transcribe_audio",
            return_value=("hello world", []),
        ):
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
                force_offload=False,
                external_model=None,
            )

        mock_std.assert_called_once()
        mock_ext.assert_not_called()

    def test_asr_execute_external_streaming_model_raises(self):
        """A streaming external model must be rejected by the ASR node."""
        bundle = self._make_bundle(is_streaming=True)
        audio = {"waveform": MagicMock(), "sample_rate": 24000}

        with patch(
            "ComfyUI_VibeVoice.nodes.asr_node.load_asr_from_external"
        ) as mock_ext, patch(
            "ComfyUI_VibeVoice.nodes.asr_node.transcribe_audio"
        ):
            with pytest.raises(ValueError, match="streaming"):
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
                    force_offload=False,
                    external_model=bundle,
                )

        mock_ext.assert_not_called()

    def test_asr_execute_external_tts_model_raises(self):
        """A TTS external model (is_asr=False) must be rejected by the ASR node."""
        bundle = self._make_bundle(is_streaming=False)
        bundle["is_asr"] = False
        audio = {"waveform": MagicMock(), "sample_rate": 24000}

        with patch(
            "ComfyUI_VibeVoice.nodes.asr_node.load_asr_from_external"
        ) as mock_ext, patch(
            "ComfyUI_VibeVoice.nodes.asr_node.transcribe_audio"
        ):
            with pytest.raises(ValueError, match="TTS"):
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
                    force_offload=False,
                    external_model=bundle,
                )

        mock_ext.assert_not_called()

    def test_asr_execute_external_overrides_model_name_for_offload(self):
        """The bundle's model_name is used for force_offload cache keying."""
        stub_patcher = MagicMock()
        bundle = self._make_bundle(model_name="ext-asr-custom")
        audio = {"waveform": MagicMock(), "sample_rate": 24000}

        with patch(
            "ComfyUI_VibeVoice.nodes.asr_node.load_asr_from_external",
            return_value=(stub_patcher, MagicMock(), MagicMock()),
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
                external_model=bundle,
            )

        mock_offload.assert_called_once_with("ext-asr-custom", stub_patcher)


class TestVibeVoiceASRNodeModelOptions:
    """The ASR node's model_name dropdown exposes the native ASR-HF model,
    and retired saved-workflow names resolve onto it."""

    def _get_model_options(self):
        schema = VibeVoiceASRNode.define_schema()
        for inp in schema.inputs:
            if inp.id == "model_name":
                return inp.options
        return None

    def test_dropdown_lists_native_asr_model(self):
        registry = {
            "VibeVoice-1.5B": {"type": "official"},
            "VibeVoice-ASR-HF": {"type": "official"},
            "VibeVoice-Realtime-0.5B": {"type": "official"},
        }
        with patch(
            "ComfyUI_VibeVoice.modules.model_info.AVAILABLE_VIBEVOICE_MODELS",
            registry,
        ):
            options = self._get_model_options()
        assert options == ["VibeVoice-ASR-HF"]

    def test_dropdown_includes_locally_discovered_streaming_dirs(self):
        """Streaming checkpoints discovered on disk stay selectable (they
        remain transcribable through the streaming protocol)."""
        registry = {
            "VibeVoice-ASR-HF": {"type": "official"},
            "VibeVoice-ASR-Streaming-1.5B": {"type": "local_dir", "path": "x"},
        }
        with patch(
            "ComfyUI_VibeVoice.modules.model_info.AVAILABLE_VIBEVOICE_MODELS",
            registry,
        ):
            options = self._get_model_options()
        assert "VibeVoice-ASR-Streaming-1.5B" in options
        assert "VibeVoice-ASR-HF" in options

    def test_validate_resolves_retired_name_to_asr_hf(self):
        registry = {
            "VibeVoice-ASR-HF": {"type": "official"},
        }
        with patch(
            "ComfyUI_VibeVoice.nodes.asr_node.AVAILABLE_VIBEVOICE_MODELS", registry
        ):
            for retired in ("VibeVoice-ASR", "VibeVoice-ASR-Streaming-1.5B",
                            "VibeVoice-ASR-Streaming-7B"):
                result = VibeVoiceASRNode.validate_inputs(model_name=retired)
                assert result is True

    def test_execute_resolves_retired_name_before_load(self):
        """A saved workflow with model_name='VibeVoice-ASR' loads ASR-HF (and
        its cache keys use the resolved name)."""
        audio = {"waveform": None, "sample_rate": 24000}
        stub_patcher = MagicMock()
        with patch(
            "ComfyUI_VibeVoice.nodes.asr_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-ASR-HF": {"type": "official"}},
        ), patch(
            "ComfyUI_VibeVoice.nodes.asr_node.load_asr_model_patched",
            return_value=(stub_patcher, MagicMock(), MagicMock()),
        ) as mock_load, patch(
            "ComfyUI_VibeVoice.nodes.asr_node.transcribe_audio",
            return_value=("hello world", []),
        ):
            VibeVoiceASRNode.execute(
                model_name="VibeVoice-ASR",
                audio=audio,
                context_info="",
                max_new_tokens=256,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                num_beams=1,
                device="cpu",
                dtype="auto",
                attention_mode="sdpa",
                force_offload=False,
            )

        assert mock_load.call_args.kwargs.get("model_name") == \
            "VibeVoice-ASR-HF"
