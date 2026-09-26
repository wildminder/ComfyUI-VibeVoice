"""Routing and behavior tests for the canonical dual-family VibeVoice TTS node.

The canonical node owns both generation families: standard reference-audio TTS
and realtime cached-voice-prompt TTS. These tests pin the routing, control
mapping, safety nets, warnings, and shared lifecycle behavior.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from conftest import COMFYUI_ROOT
from ComfyUI_VibeVoice.nodes.tts_node import (
    VibeVoiceTTSNode,
    _MISSING_PRESET_MESSAGE,
)
from ComfyUI_VibeVoice.modules.voice_presets import PRESET_CACHE_KEYS, PRESET_NONE

_PROMPT_VALIDATION_ERROR: str | None = None
try:
    # ComfyUI's own top-level ``nodes`` module must win over this package's
    # ``nodes`` subpackage (conftest puts the repo root ahead of COMFYUI_ROOT),
    # so push the ComfyUI root to the front before importing either module.
    while COMFYUI_ROOT in sys.path:
        sys.path.remove(COMFYUI_ROOT)
    sys.path.insert(0, COMFYUI_ROOT)
    import nodes as comfy_nodes
    import execution as comfy_execution
except Exception as exc:  # noqa: BLE001 - reported, never silently skipped
    _PROMPT_VALIDATION_ERROR = f"{type(exc).__name__}: {exc}"

# This module already imports ``ComfyUI_VibeVoice.nodes.tts_node`` at module
# scope, and that module imports ``comfy.model_management`` and
# ``comfy_api.latest`` at module scope too. Reaching this point therefore
# means ComfyUI itself imported fine, so a failure below is a broken checkout
# or a bad COMFYUI_ROOT -- never "ComfyUI is not installed here". Skipping
# would leave the fan-out contract lock silently unexecuted, so the flag is
# turned into a hard failure by ``_require_prompt_validation``.
_PROMPT_VALIDATION_AVAILABLE = _PROMPT_VALIDATION_ERROR is None


def _require_prompt_validation() -> None:
    """Fail loudly (never skip) when ComfyUI prompt validation is unavailable.

    The ``TestTTSNodePromptValidationFanOut`` tests are the regression lock for
    the reported per-widget error fan-out. A ``pytest.skip`` here would let the
    lock go unexecuted on a misconfigured machine while the suite still reports
    green, which is exactly the failure mode this guards against.
    """
    if not _PROMPT_VALIDATION_AVAILABLE:
        pytest.fail(
            "ComfyUI core modules (nodes/execution) could not be imported from "
            f"COMFYUI_ROOT={COMFYUI_ROOT!r}, so the prompt-validation fan-out "
            "contract cannot be verified: " + str(_PROMPT_VALIDATION_ERROR),
            pytrace=False,
        )


REALTIME_MODEL_CLASS = "VibeVoiceStreamingForConditionalGenerationInference"
REALTIME_PROCESSOR_CLASS = "VibeVoiceStreamingProcessor"


def _fake_class(name: str):
    """Build a stand-in class whose ``__name__`` matches the realtime classes."""
    return type(name, (), {})


def _realtime_pair():
    return _fake_class(REALTIME_MODEL_CLASS)(), _fake_class(REALTIME_PROCESSOR_CLASS)()


def _standard_pair():
    return MagicMock(), MagicMock()


def _preset():
    return {
        key: MagicMock(last_hidden_state=torch.zeros(1, 3, 8))
        for key in PRESET_CACHE_KEYS
    }


def _kwargs(**overrides):
    values = dict(
        model_name="VibeVoice-1.5B",
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
        force_offload=False,
        device="cpu",
        dtype="auto",
        max_new_tokens=0,
        voice_preset=PRESET_NONE,
    )
    values.update(overrides)
    return values


def _audio_dict():
    return {"waveform": torch.zeros(1024), "sample_rate": 24000}


class TestUnifiedTTSNodeSchema:
    def test_voice_preset_is_appended_last(self):
        ids = [inp.id for inp in VibeVoiceTTSNode.define_schema().inputs]
        assert ids[-1] == "voice_preset"

    def test_single_audio_output(self):
        schema = VibeVoiceTTSNode.define_schema()
        assert len(schema.outputs) == 1
        assert schema.outputs[0].display_name == "Audio"

    def test_preset_discovery_failure_falls_back_to_none(self, caplog):
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.list_voice_presets",
            side_effect=RuntimeError("scan exploded"),
        ), caplog.at_level(logging.WARNING):
            schema = VibeVoiceTTSNode.define_schema()
        preset_input = next(inp for inp in schema.inputs if inp.id == "voice_preset")
        assert list(preset_input.options) == [PRESET_NONE]
        assert "scan exploded" in caplog.text

    def test_discovered_presets_follow_none(self):
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.list_voice_presets",
            return_value={"en-Carter_man": "C:/voices/en-Carter_man.pt"},
        ):
            schema = VibeVoiceTTSNode.define_schema()
        preset_input = next(inp for inp in schema.inputs if inp.id == "voice_preset")
        assert list(preset_input.options) == [PRESET_NONE, "en-Carter_man"]


class TestUnifiedTTSNodeValidation:
    def test_validator_signature_has_no_varkw(self):
        # A **kwargs signature makes ComfyUI report one error per prompt input
        # instead of one per inspected input; keep the declaration narrow.
        spec = inspect.getfullargspec(VibeVoiceTTSNode.validate_inputs)
        assert spec.varkw is None
        assert set(spec.args[1:]) == {
            "model_name",
            "voice_preset",
            "external_model",
        }
        assert spec.defaults[0] is None
        assert spec.defaults[1] == PRESET_NONE

    def test_external_model_default_is_a_sentinel_not_none(self):
        import ComfyUI_VibeVoice.nodes.tts_node as m

        spec = inspect.getfullargspec(VibeVoiceTTSNode.validate_inputs)
        assert m._EXTERNAL_UNSET is not None
        assert spec.defaults[2] is m._EXTERNAL_UNSET

    def test_tts_node_submodule_is_a_single_instance(self):
        """``sys.modules`` and the parent package must name the same object.

        The opt-in GPU module (``tests/test_realtime_e2e_gpu.py``) has to
        re-import ``tts_node`` against the un-mocked vendored sources, and an
        import registers a submodule in *two* places: ``sys.modules`` and as an
        attribute of the already-imported parent package. ``import a.b as m``
        prefers the attribute. If a fixture restores only ``sys.modules``, a
        second copy of this module survives the fixture and the assertion above
        compares one copy's sentinel against the other copy's default — a
        failure that shows up only in a ``RUN_VIBEVOICE_E2E=1`` full-suite run.
        """
        import ComfyUI_VibeVoice.nodes as nodes_pkg
        import ComfyUI_VibeVoice.nodes.tts_node as m

        assert sys.modules["ComfyUI_VibeVoice.nodes.tts_node"] is m
        assert nodes_pkg.tts_node is m
        assert m.VibeVoiceTTSNode is VibeVoiceTTSNode

    def test_missing_preset_key_on_named_realtime_model_is_missing(self):
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-Realtime-0.5B": {}},
        ):
            result = VibeVoiceTTSNode.validate_inputs(
                model_name="VibeVoice-Realtime-0.5B"
            )
        assert isinstance(result, str)
        assert "voice_preset" in result

    def test_realtime_model_with_preset_is_plain_true(self):
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-Realtime-0.5B": {}},
        ):
            result = VibeVoiceTTSNode.validate_inputs(
                model_name="VibeVoice-Realtime-0.5B",
                voice_preset="en-Carter_man",
            )
        assert result is True

    def test_missing_preset_message_is_a_single_string(self):
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-Realtime-0.5B": {}},
        ):
            explicit = VibeVoiceTTSNode.validate_inputs(
                model_name="VibeVoice-Realtime-0.5B", voice_preset=PRESET_NONE
            )
            omitted = VibeVoiceTTSNode.validate_inputs(
                model_name="VibeVoice-Realtime-0.5B"
            )
        for result in (explicit, omitted):
            assert isinstance(result, str)
            assert result == _MISSING_PRESET_MESSAGE
            assert "voice_preset" in result
            assert "models/tts/VibeVoice/voices" in result
            assert len(result.splitlines()) == 1

    def test_standard_model_does_not_require_preset(self):
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-1.5B": {}},
        ):
            assert VibeVoiceTTSNode.validate_inputs(
                model_name="VibeVoice-1.5B", voice_preset=PRESET_NONE
            ) is True

    def test_unknown_model_is_rejected(self):
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-1.5B": {}},
        ):
            result = VibeVoiceTTSNode.validate_inputs(model_name="ghost")
        assert isinstance(result, str)
        assert "ghost" in result

    def test_renamed_local_realtime_model_requires_preset(self):
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.AVAILABLE_VIBEVOICE_MODELS",
            {"my-realtime-copy": {}},
        ):
            result = VibeVoiceTTSNode.validate_inputs(model_name="my-realtime-copy")
        assert isinstance(result, str)
        assert "voice_preset" in result

    def test_connected_external_input_bypasses_queue_validation(self):
        assert VibeVoiceTTSNode.validate_inputs(
            external_model=None, model_name="ghost"
        ) is True

    def test_supplied_external_bundle_bypasses_queue_validation(self):
        assert VibeVoiceTTSNode.validate_inputs(
            external_model={"model_name": "x"}, model_name="ghost"
        ) is True


class TestUnifiedTTSNodeRouting:
    def test_standard_family_calls_only_generate_audio(self):
        model, processor = _standard_pair()
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_model",
            return_value=(MagicMock(), model, processor),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.generate_audio",
            return_value=(torch.zeros(1, 1, 8), 24000),
        ) as standard, patch(
            "ComfyUI_VibeVoice.nodes.tts_node.generate_realtime_audio"
        ) as realtime, patch(
            "ComfyUI_VibeVoice.nodes.tts_node.ui.PreviewAudio", MagicMock()
        ):
            VibeVoiceTTSNode.execute(**_kwargs(max_new_tokens=64))

        standard.assert_called_once()
        realtime.assert_not_called()
        assert standard.call_args.kwargs["max_new_tokens"] == 64
        assert standard.call_args.kwargs["inference_steps"] == 10

    def test_realtime_family_calls_only_realtime_adapter(self):
        model, processor = _realtime_pair()
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_model",
            return_value=(MagicMock(), model, processor),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.get_cached_voice_preset",
            return_value=_preset(),
        ) as cached, patch(
            "ComfyUI_VibeVoice.nodes.tts_node.generate_realtime_audio",
            return_value=(torch.zeros(1, 1, 8), 24000),
        ) as realtime, patch(
            "ComfyUI_VibeVoice.nodes.tts_node.generate_audio"
        ) as standard, patch(
            "ComfyUI_VibeVoice.nodes.tts_node.ui.PreviewAudio", MagicMock()
        ):
            VibeVoiceTTSNode.execute(
                **_kwargs(
                    model_name="VibeVoice-Realtime-0.5B",
                    voice_preset="en-Carter_man",
                )
            )

        realtime.assert_called_once()
        standard.assert_not_called()
        cached.assert_called_once()
        assert realtime.call_args.kwargs["diffusion_steps"] == 10
        assert realtime.call_args.kwargs["max_new_tokens"] == 0

    def test_steps_and_length_are_forwarded_independently(self):
        model, processor = _realtime_pair()
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_model",
            return_value=(MagicMock(), model, processor),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.get_cached_voice_preset",
            return_value=_preset(),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.generate_realtime_audio",
            return_value=(torch.zeros(1, 1, 8), 24000),
        ) as realtime, patch(
            "ComfyUI_VibeVoice.nodes.tts_node.ui.PreviewAudio", MagicMock()
        ):
            VibeVoiceTTSNode.execute(
                **_kwargs(
                    model_name="VibeVoice-Realtime-0.5B",
                    voice_preset="en-Carter_man",
                    inference_steps=3,
                    max_new_tokens=120,
                )
            )

        call = realtime.call_args.kwargs
        assert call["diffusion_steps"] == 3
        assert call["max_new_tokens"] == 120

    def test_external_realtime_bundle_uses_external_loader(self):
        model, processor = _realtime_pair()
        bundle = {
            "model_name": "ExtRealtime",
            "is_streaming": True,
            "is_asr": False,
        }
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_from_external",
            return_value=(MagicMock(), model, processor),
        ) as external, patch(
            "ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_model"
        ) as named, patch(
            "ComfyUI_VibeVoice.nodes.tts_node.get_cached_voice_preset",
            return_value=_preset(),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.generate_realtime_audio",
            return_value=(torch.zeros(1, 1, 8), 24000),
        ) as realtime, patch(
            "ComfyUI_VibeVoice.nodes.tts_node.ui.PreviewAudio", MagicMock()
        ):
            VibeVoiceTTSNode.execute(
                **_kwargs(external_model=bundle, voice_preset="en-Carter_man")
            )

        external.assert_called_once()
        named.assert_not_called()
        realtime.assert_called_once()

    def test_external_realtime_bundle_without_preset_fails_before_generation(self):
        model, processor = _realtime_pair()
        bundle = {
            "model_name": "ExtRealtime",
            "is_streaming": True,
            "is_asr": False,
        }
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_from_external",
            return_value=(MagicMock(), model, processor),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.get_cached_voice_preset"
        ) as cached, patch(
            "ComfyUI_VibeVoice.nodes.tts_node.generate_realtime_audio"
        ) as realtime:
            with pytest.raises(ValueError, match="voice_preset"):
                VibeVoiceTTSNode.execute(**_kwargs(external_model=bundle))

        cached.assert_not_called()
        realtime.assert_not_called()

    def test_external_asr_bundle_is_rejected_before_loading(self):
        bundle = {
            "model_name": "ExtASR",
            "is_streaming": False,
            "is_asr": True,
        }
        with patch("ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_from_external") as external:
            with pytest.raises(ValueError, match="ASR"):
                VibeVoiceTTSNode.execute(**_kwargs(external_model=bundle))
        external.assert_not_called()

    def test_named_asr_model_is_rejected_before_loading(self):
        with patch("ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_model") as named:
            with pytest.raises(ValueError, match="ASR"):
                VibeVoiceTTSNode.execute(**_kwargs(model_name="VibeVoice-ASR"))
        named.assert_not_called()


class TestUnifiedTTSNodeLoadedPairSafety:
    def test_renamed_realtime_checkpoint_is_rerouted(self, caplog):
        model, processor = _realtime_pair()
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_model",
            return_value=(MagicMock(), model, processor),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.get_cached_voice_preset",
            return_value=_preset(),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.generate_realtime_audio",
            return_value=(torch.zeros(1, 1, 8), 24000),
        ) as realtime, patch(
            "ComfyUI_VibeVoice.nodes.tts_node.generate_audio"
        ) as standard, patch(
            "ComfyUI_VibeVoice.nodes.tts_node.ui.PreviewAudio", MagicMock()
        ), caplog.at_level(
            logging.WARNING, logger="ComfyUI_VibeVoice.nodes.tts_node"
        ):
            VibeVoiceTTSNode.execute(
                **_kwargs(
                    model_name="my-local-copy",
                    voice_preset="en-Carter_man",
                )
            )

        realtime.assert_called_once()
        standard.assert_not_called()
        assert "realtime" in caplog.text.lower()

    def test_realtime_name_with_standard_classes_is_an_error(self):
        model, processor = _standard_pair()
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_model",
            return_value=(MagicMock(), model, processor),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.generate_realtime_audio"
        ) as realtime, patch(
            "ComfyUI_VibeVoice.nodes.tts_node.generate_audio"
        ) as standard:
            with pytest.raises(ValueError, match="realtime model"):
                VibeVoiceTTSNode.execute(
                    **_kwargs(
                        model_name="VibeVoice-Realtime-0.5B",
                        voice_preset="en-Carter_man",
                    )
                )
        realtime.assert_not_called()
        standard.assert_not_called()

    def test_mismatched_loaded_pair_is_rejected(self):
        model = _fake_class(REALTIME_MODEL_CLASS)()
        processor = MagicMock()
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_model",
            return_value=(MagicMock(), model, processor),
        ), patch("ComfyUI_VibeVoice.nodes.tts_node.generate_audio"):
            with pytest.raises(ValueError, match="inconsistent"):
                VibeVoiceTTSNode.execute(
                    **_kwargs(
                        model_name="VibeVoice-Realtime-0.5B",
                        voice_preset="en-Carter_man",
                    )
                )


class TestUnifiedTTSNodeWarnings:
    def test_connected_speaker_audio_warns_once_for_realtime(self, caplog):
        model, processor = _realtime_pair()
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_model",
            return_value=(MagicMock(), model, processor),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.get_cached_voice_preset",
            return_value=_preset(),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.generate_realtime_audio",
            return_value=(torch.zeros(1, 1, 8), 24000),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.ui.PreviewAudio", MagicMock()
        ), caplog.at_level(logging.WARNING, logger="ComfyUI_VibeVoice.nodes.tts_node"):
            VibeVoiceTTSNode.execute(
                **_kwargs(
                    model_name="VibeVoice-Realtime-0.5B",
                    voice_preset="en-Carter_man",
                    speaker_1_voice=_audio_dict(),
                    speaker_2_voice=_audio_dict(),
                )
            )

        speaker_warnings = [
            record.message
            for record in caplog.records
            if "Speaker reference audio is ignored" in record.message
        ]
        assert len(speaker_warnings) == 1

    def test_sampling_controls_warn_for_realtime(self, caplog):
        model, processor = _realtime_pair()
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_model",
            return_value=(MagicMock(), model, processor),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.get_cached_voice_preset",
            return_value=_preset(),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.generate_realtime_audio",
            return_value=(torch.zeros(1, 1, 8), 24000),
        ) as realtime, patch(
            "ComfyUI_VibeVoice.nodes.tts_node.ui.PreviewAudio", MagicMock()
        ), caplog.at_level(logging.WARNING, logger="ComfyUI_VibeVoice.nodes.tts_node"):
            VibeVoiceTTSNode.execute(
                **_kwargs(
                    model_name="VibeVoice-Realtime-0.5B",
                    voice_preset="en-Carter_man",
                    temperature=1.4,
                )
            )

        assert "Sampling controls" in caplog.text
        for control in ("do_sample", "temperature", "top_p", "top_k"):
            assert control not in realtime.call_args.kwargs


class TestUnifiedTTSNodeSharedLifecycle:
    def test_force_offload_is_shared(self):
        model, processor = _realtime_pair()
        patcher = MagicMock()
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_model",
            return_value=(patcher, model, processor),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.get_cached_voice_preset",
            return_value=_preset(),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.generate_realtime_audio",
            return_value=(torch.zeros(1, 1, 8), 24000),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.force_offload_model"
        ) as offload, patch(
            "ComfyUI_VibeVoice.nodes.tts_node.ui.PreviewAudio", MagicMock()
        ):
            VibeVoiceTTSNode.execute(
                **_kwargs(
                    model_name="VibeVoice-Realtime-0.5B",
                    voice_preset="en-Carter_man",
                    force_offload=True,
                )
            )
        offload.assert_called_once_with(patcher, "VibeVoice-Realtime-0.5B", warm=True)

    def test_output_dictionary_and_preview_are_shared(self):
        model, processor = _realtime_pair()
        waveform = torch.zeros(1, 1, 8)
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_model",
            return_value=(MagicMock(), model, processor),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.get_cached_voice_preset",
            return_value=_preset(),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.generate_realtime_audio",
            return_value=(waveform, 24000),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.ui.PreviewAudio", MagicMock()
        ) as preview:
            result = VibeVoiceTTSNode.execute(
                **_kwargs(
                    model_name="VibeVoice-Realtime-0.5B",
                    voice_preset="en-Carter_man",
                )
            )

        assert result[0]["sample_rate"] == 24000
        assert result[0]["waveform"] is waveform
        preview.assert_called_once()

    def test_interruption_returns_silent_fallback(self):
        import comfy.model_management as model_management

        model, processor = _realtime_pair()
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_model",
            return_value=(MagicMock(), model, processor),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.get_cached_voice_preset",
            return_value=_preset(),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.generate_realtime_audio",
            side_effect=model_management.InterruptProcessingException(),
        ):
            result = VibeVoiceTTSNode.execute(
                **_kwargs(
                    model_name="VibeVoice-Realtime-0.5B",
                    voice_preset="en-Carter_man",
                )
            )

        assert result[0]["sample_rate"] == 24000
        assert result[0]["waveform"].abs().sum().item() == 0.0

    def test_cancellation_named_exception_returns_silent_fallback(self):
        model, processor = _standard_pair()
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_model",
            return_value=(MagicMock(), model, processor),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.generate_audio",
            side_effect=RuntimeError("user cancelled the prompt"),
        ):
            result = VibeVoiceTTSNode.execute(**_kwargs())

        assert result[0]["waveform"].abs().sum().item() == 0.0

    def test_other_errors_propagate(self):
        model, processor = _standard_pair()
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_model",
            return_value=(MagicMock(), model, processor),
        ), patch(
            "ComfyUI_VibeVoice.nodes.tts_node.generate_audio",
            side_effect=RuntimeError("cuda exploded"),
        ):
            with pytest.raises(RuntimeError, match="cuda exploded"):
                VibeVoiceTTSNode.execute(**_kwargs())


class TestUnifiedTTSNodeCustomTypes:
    def test_custom_type_documents_both_tts_families(self):
        from ComfyUI_VibeVoice.modules import custom_types

        source = inspect.getsource(custom_types)
        assert "standard and realtime" in source.lower()


# Widgets named in the bug report: a fanned-out error must never land on one of
# these, only on the inputs the validator actually declares.
REPORTED_PROMPT_WIDGETS = {
    "seed", "text", "cfg_scale", "inference_steps", "do_sample", "temperature",
    "top_p", "top_k", "max_new_tokens", "force_offload", "device", "dtype",
    "quantize_llm_4bit", "attention_mode", "speaker_1_voice", "speaker_2_voice",
}


def _widget_prompt(node_id, **overrides):
    """Build a prompt carrying every declared input but ``external_model``.

    This mirrors the reported workflow: all 20 remaining inputs are present, so
    an un-narrowed ``**kwargs`` validator fans its single failure out once per
    input (measured: 20 ``custom_validation_failed`` entries). ``external_model``
    is deliberately excluded so the externally-loaded-model bypass is not taken.
    """
    values = {}
    for inp in VibeVoiceTTSNode.define_schema().inputs:
        if inp.id == "external_model":
            continue
        if inp.id in overrides:
            values[inp.id] = overrides[inp.id]
        elif getattr(inp, "default", None) is not None:
            # Socket-only inputs (io.Audio) have no ``default`` at all.
            values[inp.id] = inp.default
        else:
            # Optional AUDIO widgets; a literal dict is accepted by core's
            # type check and keeps the input present in the prompt.
            values[inp.id] = {"waveform": [0.0], "sample_rate": 24000}
    return {"1": {"class_type": node_id, "inputs": values}}


def _validate(prompt):
    """Run ComfyUI's own prompt validator over ``prompt`` and unwrap it.

    A dedicated loop is driven directly instead of ``asyncio.run``:
    ``asyncio.run`` installs the loop it creates and then clears the thread's
    current loop on exit, and an explicitly-cleared loop makes a later
    ``asyncio.get_event_loop()`` raise instead of auto-creating one. That
    would break any subsequent test in the same process that reads the current
    loop (e.g. tests/test_realtime_node.py). Never installing the loop leaves
    the thread's event-loop state exactly as the caller left it.
    """
    loop = asyncio.new_event_loop()
    try:
        valid, errors, _ = loop.run_until_complete(
            comfy_execution.validate_inputs("pid", prompt, "1", {})
        )
    finally:
        loop.close()
    return valid, errors


class TestTTSNodePromptValidationFanOut:
    """Prompt-level regression lock for the missing-preset error fan-out.

    ComfyUI applies a V3 ``validate_inputs`` failure once per input name
    present in the prompt, so a ``kwargs`` validator turns one rule into one
    error per input. These tests drive the real ``execution.validate_inputs``
    rather than calling the classmethod, because the fan-out lives in core.

    Nothing here skips: an unavailable ComfyUI core is a hard failure
    (``_require_prompt_validation``), because a skip would silently disable the
    only lock on the reported bug.
    """

    def test_lock_itself_is_active_rather_than_skipped(self):
        """This class is the regression lock; a silent skip would be a green lie.

        The fan-out contract lives in ComfyUI core, so the lock can only be
        exercised where ``nodes``/``execution`` import. Assert that up front
        instead of letting a misconfigured COMFYUI_ROOT turn the whole class
        into skips that still report a passing suite.
        """
        assert _PROMPT_VALIDATION_AVAILABLE, _PROMPT_VALIDATION_ERROR
        assert callable(comfy_execution.validate_inputs)
        assert "validate_inputs" in vars(comfy_execution)

    def test_guard_fails_instead_of_skipping_when_core_is_unavailable(self, monkeypatch):
        """The unavailability path must fail, never skip."""
        this_module = sys.modules[__name__]
        monkeypatch.setattr(this_module, "_PROMPT_VALIDATION_AVAILABLE", False)
        monkeypatch.setattr(
            this_module, "_PROMPT_VALIDATION_ERROR", "ImportError: simulated"
        )
        with pytest.raises(pytest.fail.Exception, match="cannot be verified"):
            _require_prompt_validation()

    @pytest.fixture
    def registered_tts_probe(self):
        _require_prompt_validation()

        class _ProbeNode(VibeVoiceTTSNode):
            @classmethod
            def define_schema(cls):
                schema = super().define_schema()
                schema.node_id = "VibeVoiceTTSFanOutProbe"
                return schema

        comfy_nodes.NODE_CLASS_MAPPINGS["VibeVoiceTTSFanOutProbe"] = _ProbeNode
        try:
            yield "VibeVoiceTTSFanOutProbe"
        finally:
            comfy_nodes.NODE_CLASS_MAPPINGS.pop("VibeVoiceTTSFanOutProbe", None)

    def test_standard_model_produces_no_errors(self, registered_tts_probe):
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-1.5B": {}, "VibeVoice-Realtime-0.5B": {}},
        ):
            valid, errors = _validate(
                _widget_prompt(
                    registered_tts_probe,
                    model_name="VibeVoice-1.5B",
                    voice_preset=PRESET_NONE,
                )
            )

        assert valid is True
        assert errors == []

    def test_realtime_model_with_preset_produces_no_errors(self, registered_tts_probe):
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-1.5B": {}, "VibeVoice-Realtime-0.5B": {}},
        ):
            valid, errors = _validate(
                _widget_prompt(
                    registered_tts_probe,
                    model_name="VibeVoice-Realtime-0.5B",
                    voice_preset="en-Carter_man",
                )
            )

        assert valid is True
        assert errors == []

    def test_realtime_model_with_preset_and_device_auto_produces_no_errors(
        self, registered_tts_probe
    ):
        """The exact reported workflow: realtime model, a real voice preset,
        and device="auto".

        "auto" is a value get_torch_device() accepts, so the device combo must
        offer it; when it did not, core's Combo range-check rejected the
        prompt with "Value not in list: device: 'auto'" before the node ran.
        """
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-1.5B": {}, "VibeVoice-Realtime-0.5B": {}},
        ):
            valid, errors = _validate(
                _widget_prompt(
                    registered_tts_probe,
                    model_name="VibeVoice-Realtime-0.5B",
                    voice_preset="en-Carter_man",
                    device="auto",
                )
            )

        assert valid is True
        assert errors == []

    def test_realtime_model_without_preset_is_not_fanned_out_per_widget(
        self, registered_tts_probe
    ):
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-1.5B": {}, "VibeVoice-Realtime-0.5B": {}},
        ):
            prompt = _widget_prompt(
                registered_tts_probe,
                model_name="VibeVoice-Realtime-0.5B",
                voice_preset=PRESET_NONE,
            )
            valid, errors = _validate(prompt)

        assert valid is False
        failed = [e for e in errors if e["type"] == "custom_validation_failed"]
        declared_params = inspect.getfullargspec(
            VibeVoiceTTSNode.validate_inputs
        ).args[1:]
        # ``execution.py`` fans a non-True validator result out over
        # ``input_filtered`` = declared validator params that are present in the
        # prompt. The missing-preset rule is cross-input (it needs both
        # ``model_name`` and ``voice_preset``), so the floor is two attributed
        # errors carrying the identical message -- not one per widget.
        fanned_inputs = set(declared_params) & set(prompt["1"]["inputs"])

        # The rule still fires at queue time...
        assert failed
        # ...once per declared param, not once per prompt widget.
        assert len(failed) == len({e["extra_info"]["input_name"] for e in failed})
        # Exactly core's fan-out width -- measured 2, down from 20 in the report.
        assert {e["extra_info"]["input_name"] for e in failed} == fanned_inputs
        assert len(failed) <= 3
        assert len(failed) <= len(declared_params)
        assert all(e["message"] == "Custom validation failed for node" for e in failed)
        assert all(
            e["details"].endswith(f" - {_MISSING_PRESET_MESSAGE}") for e in failed
        )
        assert all(e["extra_info"]["input_name"] in declared_params for e in failed)
        assert REPORTED_PROMPT_WIDGETS.isdisjoint(
            e["extra_info"]["input_name"] for e in failed
        )

    def test_core_range_checks_are_active_for_uninspected_widgets(
        self, registered_tts_probe
    ):
        """Narrowing the signature re-enables core's own min/max checks.

        ``execution.py`` only range-checks inputs the validator neither
        declares nor covers with ``**kwargs``, so ``seed`` is checked again
        now that the validator no longer swallows it.
        """
        with patch(
            "ComfyUI_VibeVoice.nodes.tts_node.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-1.5B": {}, "VibeVoice-Realtime-0.5B": {}},
        ):
            valid, errors = _validate(
                _widget_prompt(
                    registered_tts_probe,
                    model_name="VibeVoice-1.5B",
                    voice_preset=PRESET_NONE,
                    seed=2**70,
                )
            )

        assert valid is False
        assert not [e for e in errors if e["type"] == "custom_validation_failed"]
        too_big = [e for e in errors if e["type"] == "value_bigger_than_max"]
        assert len(too_big) == 1
        assert too_big[0]["extra_info"]["input_name"] == "seed"
