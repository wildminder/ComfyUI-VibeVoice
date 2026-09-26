"""Progress and cancellation coverage for the official realtime adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from ComfyUI_VibeVoice.modules import realtime_generation
from ComfyUI_VibeVoice.modules.realtime_generation import generate_realtime_audio
from ComfyUI_VibeVoice.modules.voice_presets import PRESET_CACHE_KEYS


class _Model:
    def __init__(self, side_effect=None):
        self.set_ddpm_inference_steps = MagicMock()
        self.generate = MagicMock(side_effect=side_effect)
        if side_effect is None:
            self.generate.return_value = SimpleNamespace(
                speech_outputs=[torch.ones(24000)]
            )


class _Processor:
    def __init__(self):
        self.tokenizer = object()
        self.process_input_with_cached_prompt = MagicMock(
            return_value={
                "tts_text_ids": torch.ones(1, 3, dtype=torch.long),
                # The adapter budgets the auto length from the script and the
                # TTS-LM prompt length, so both keys must be present.
                "tts_lm_input_ids": torch.ones(1, 3, dtype=torch.long),
            }
        )

    def __call__(self, *args, **kwargs):
        raise AssertionError("streaming processor __call__ must not be used")


class _ProcessorByName(_Processor):
    pass


# The adapter intentionally uses class names to identify the official pair.
_ProcessorByName.__name__ = "VibeVoiceStreamingProcessor"
_Model.__name__ = "VibeVoiceStreamingForConditionalGenerationInference"


def _preset():
    return {
        key: SimpleNamespace(last_hidden_state=torch.ones(1, index + 2, 4))
        for index, key in enumerate(PRESET_CACHE_KEYS)
    }


def _run(model):
    pbar = MagicMock()
    pbar.total = 1
    with patch.object(realtime_generation, "ProgressBarWithConsole", return_value=pbar), patch.object(
        realtime_generation.model_management,
        "get_torch_device",
        return_value=torch.device("cpu"),
    ), patch.object(
        realtime_generation.model_management,
        "throw_exception_if_processing_interrupted",
    ):
        result = generate_realtime_audio(
            model,
            _ProcessorByName(),
            "[1] Hello",
            _preset(),
        )
    return result, pbar


def test_generate_receives_progress_callback():
    model = _Model()
    _run(model)
    assert callable(model.generate.call_args.kwargs["progress_callback"])


def test_callback_updates_total_and_finalizes():
    model = _Model()
    _result, pbar = _run(model)
    callback = model.generate.call_args.kwargs["progress_callback"]
    assert pbar.update_absolute.call_args_list[-1].args == (1,)
    callback(4, 12)
    pbar.update_absolute.assert_any_call(4, total=12)


def test_callback_checks_interruption():
    model = _Model()
    _run(model)
    callback = model.generate.call_args.kwargs["progress_callback"]
    with patch.object(
        realtime_generation.model_management,
        "throw_exception_if_processing_interrupted",
        side_effect=realtime_generation.model_management.InterruptProcessingException(),
    ):
        with pytest.raises(realtime_generation.model_management.InterruptProcessingException):
            callback(1, 2)


def test_generation_failure_still_finalizes_progress():
    model = _Model(side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        _run(model)
