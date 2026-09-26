"""Tests for the deprecated VibeVoiceRealtime forwarding shim.

The legacy node keeps its ID, widget order, and one-time deprecation warning,
but owns no generation logic: validation and execution are delegated to the
canonical ``VibeVoiceTTSNode``.
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from ComfyUI_VibeVoice.nodes.realtime_node import VibeVoiceRealtimeNode
from ComfyUI_VibeVoice.nodes.tts_node import VibeVoiceTTSNode
from tests.test_node_schema import LEGACY_REALTIME_INPUT_IDS


LEGACY_WORKFLOW_FIXTURE = (
    Path(__file__).parent / "fixtures" / "legacy_realtime_workflow.json"
)


def _legacy_kwargs(**overrides):
    values = dict(
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
        max_new_tokens=0,
        voice_preset="en-Carter_man",
    )
    values.update(overrides)
    return values


class TestRealtimeShimSchema:
    def test_node_id_and_deprecation(self):
        schema = VibeVoiceRealtimeNode.define_schema()
        assert schema.node_id == "VibeVoiceRealtime"
        assert schema.is_deprecated is True

    def test_display_name_points_to_canonical_node(self):
        schema = VibeVoiceRealtimeNode.define_schema()
        assert "VibeVoice TTS" in schema.display_name
        assert "deprecated" in schema.display_name.lower()

    def test_legacy_prefix_preserved_and_controls_appended(self):
        ids = tuple(inp.id for inp in VibeVoiceRealtimeNode.define_schema().inputs)
        assert ids[: len(LEGACY_REALTIME_INPUT_IDS)] == LEGACY_REALTIME_INPUT_IDS
        assert ids[len(LEGACY_REALTIME_INPUT_IDS):] == ("max_new_tokens", "voice_preset")
        assert "stream" in ids

    def test_single_audio_output(self):
        assert len(VibeVoiceRealtimeNode.define_schema().outputs) == 1

    def test_schema_lists_only_streaming_models(self):
        with patch(
            "ComfyUI_VibeVoice.modules.model_info.AVAILABLE_VIBEVOICE_MODELS",
            {
                "VibeVoice-Realtime-0.5B": {"type": "official"},
                "VibeVoice-ASR": {"type": "official"},
                "VibeVoice-1.5B": {"type": "official"},
            },
        ):
            schema = VibeVoiceRealtimeNode.define_schema()
        options = next(inp for inp in schema.inputs if inp.id == "model_name").options
        assert list(options) == ["VibeVoice-Realtime-0.5B"]


class TestRealtimeShimDelegation:
    def test_validate_inputs_delegates_to_canonical_node(self):
        with patch.object(
            VibeVoiceTTSNode, "validate_inputs", return_value="delegated"
        ) as delegate:
            result = VibeVoiceRealtimeNode.validate_inputs(
                model_name="VibeVoice-Realtime-0.5B"
            )
        delegate.assert_called_once_with(model_name="VibeVoice-Realtime-0.5B")
        assert result == "delegated"

    def test_validate_inputs_matching_error_text(self):
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-Realtime-0.5B": {}},
        ):
            legacy = VibeVoiceRealtimeNode.validate_inputs(
                model_name="VibeVoice-Realtime-0.5B"
            )
            canonical = VibeVoiceTTSNode.validate_inputs(
                model_name="VibeVoice-Realtime-0.5B"
            )
        assert legacy == canonical
        assert isinstance(legacy, str)
        assert "voice_preset" in legacy

    def test_shim_validator_signature_has_no_varkw(self):
        spec = inspect.getfullargspec(VibeVoiceRealtimeNode.validate_inputs)
        assert spec.varkw is None
        assert set(spec.args[1:]) == {"model_name", "voice_preset", "external_model"}
        assert "stream" not in spec.args

    def test_shim_validator_never_receives_the_legacy_stream_key(self):
        with pytest.raises(TypeError):
            VibeVoiceRealtimeNode.validate_inputs(
                model_name="VibeVoice-Realtime-0.5B", stream=False
            )

    def test_shim_forwards_explicitly_set_inputs_only(self):
        with patch.object(
            VibeVoiceTTSNode, "validate_inputs", return_value=True
        ) as delegate:
            VibeVoiceRealtimeNode.validate_inputs(
                model_name="VibeVoice-Realtime-0.5B", voice_preset="en-Carter_man"
            )
        assert delegate.call_args.kwargs == {
            "model_name": "VibeVoice-Realtime-0.5B",
            "voice_preset": "en-Carter_man",
        }

        with patch.object(
            VibeVoiceTTSNode, "validate_inputs", return_value=True
        ) as delegate:
            VibeVoiceRealtimeNode.validate_inputs(
                external_model=None, model_name="ghost"
            )
        assert delegate.call_args.kwargs == {
            "model_name": "ghost",
            "external_model": None,
        }

    def test_execute_delegates_and_strips_stream(self):
        with patch.object(
            VibeVoiceTTSNode, "execute", return_value="canonical-output"
        ) as delegate:
            result = VibeVoiceRealtimeNode.execute(**_legacy_kwargs())
        assert result == "canonical-output"
        delegate.assert_called_once()
        forwarded = delegate.call_args.kwargs
        assert "stream" not in forwarded
        assert forwarded["voice_preset"] == "en-Carter_man"
        assert forwarded["max_new_tokens"] == 0

    def test_stream_toggle_does_not_change_behavior(self):
        outputs = []
        with patch.object(
            VibeVoiceTTSNode, "execute", side_effect=lambda **kwargs: outputs.append(kwargs) or "out"
        ):
            VibeVoiceRealtimeNode.execute(**_legacy_kwargs(stream=False))
            VibeVoiceRealtimeNode.execute(**_legacy_kwargs(stream=True))
        assert outputs[0] == outputs[1]

    def test_deprecation_warning_logged_once_per_process(self, caplog):
        from ComfyUI_VibeVoice.nodes import realtime_node as shim_module

        shim_module._DEPRECATION_LOGGED = False
        with patch.object(VibeVoiceTTSNode, "execute", return_value="out"):
            with caplog.at_level(
                logging.WARNING, logger="ComfyUI_VibeVoice.nodes.realtime_node"
            ):
                VibeVoiceRealtimeNode.execute(**_legacy_kwargs())
                VibeVoiceRealtimeNode.execute(**_legacy_kwargs())
                VibeVoiceRealtimeNode.execute(**_legacy_kwargs())
        messages = [
            record.message
            for record in caplog.records
            if "deprecated" in record.message.lower()
        ]
        assert len(messages) == 1

    def test_shim_does_not_reference_generation_or_loading(self):
        source_path = Path(__file__).parent.parent / "nodes" / "realtime_node.py"
        text = source_path.read_text(encoding="utf-8")
        assert "generate_realtime_audio" not in text
        assert "generate_audio" not in text
        assert "load_vibevoice_model" not in text
        assert "load_vibevoice_from_external" not in text
        assert "get_cached_voice_preset" not in text


class TestRealtimeShimRegistrationAndWorkflow:
    def test_extension_registers_one_canonical_node_and_one_deprecated_delegate(self):
        import asyncio

        from ComfyUI_VibeVoice.vibevoice_nodes import VibeVoiceExtension

        ext = VibeVoiceExtension()
        node_list = asyncio.get_event_loop().run_until_complete(ext.get_node_list())
        node_ids = [node.define_schema().node_id for node in node_list]
        assert node_ids.count("VibeVoiceTTS") == 1
        assert node_ids.count("VibeVoiceRealtime") == 1
        assert len(node_ids) == len(set(node_ids))

    def test_legacy_workflow_fixture_resolves_to_deprecated_node(self):
        import asyncio

        from ComfyUI_VibeVoice.vibevoice_nodes import VibeVoiceExtension

        assert LEGACY_WORKFLOW_FIXTURE.is_file()
        workflow = json.loads(LEGACY_WORKFLOW_FIXTURE.read_text(encoding="utf-8"))
        legacy_types = [
            node["type"] for node in workflow["nodes"] if node["type"] == "VibeVoiceRealtime"
        ]
        assert len(legacy_types) == 1

        ext = VibeVoiceExtension()
        node_list = asyncio.get_event_loop().run_until_complete(ext.get_node_list())
        registered = {
            node.define_schema().node_id: node.define_schema() for node in node_list
        }
        assert "VibeVoiceRealtime" in registered
        assert registered["VibeVoiceRealtime"].is_deprecated is True
