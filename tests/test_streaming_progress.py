"""Tests for ComfyUI progress reporting in ``generate_streaming_audio()``
(Phase 4 of the 2026-08-15 inference-progress-reporting plan).

The streaming wrapper must:
* pass a callable ``progress_callback`` into the vendored streaming
  ``model.generate``,
* map ``callback(current, total)`` -> ``ProgressBar.update_absolute(current,
  total=total)`` (the bar's placeholder total is corrected dynamically),
* check for user interruption inside the callback,
* always send the final 100% event (success and exception paths).
"""

import numpy as np
import torch
import pytest
from unittest.mock import patch, MagicMock

from ComfyUI_VibeVoice.modules.generation import generate_streaming_audio


def _mock_voice_sample(length: int = 24000) -> np.ndarray:
    return np.random.randn(length).astype(np.float32)


def _run_streaming(generate_side_effect=None, expect_exception=None):
    """Run generate_streaming_audio with fully mocked model/processor/prefill.

    When ``expect_exception`` is given, the call is wrapped in
    ``pytest.raises``. Returns (mock_model, mock_pbar_cls, mock_pbar,
    mock_interrupt).
    """
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_output = MagicMock()
    mock_output.speech_outputs = [torch.randn(24000)]
    if generate_side_effect is not None:
        mock_model.generate.side_effect = generate_side_effect
    else:
        mock_model.generate.return_value = mock_output

    mock_processor = MagicMock()
    mock_processor.tokenizer = MagicMock()
    mock_processor.tokenizer.encode = MagicMock(return_value=[10, 11, 12])
    mock_processor.prepare_speech_inputs = MagicMock(
        return_value={
            "padded_speeches": torch.randn(1, 100),
            "speech_masks": torch.ones(1, 100, dtype=torch.bool),
        }
    )

    with patch("ComfyUI_VibeVoice.modules.generation.ProgressBar") as mock_pbar_cls, \
         patch("ComfyUI_VibeVoice.modules.generation.model_management.throw_exception_if_processing_interrupted") as mock_interrupt, \
         patch("ComfyUI_VibeVoice.modules.generation.prefill_voice_prompt", return_value={"lm": MagicMock()}), \
         patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio", return_value=_mock_voice_sample()):
        mock_pbar = MagicMock()
        mock_pbar.total = 1
        mock_pbar_cls.return_value = mock_pbar

        call = lambda: generate_streaming_audio(
            model=mock_model,
            processor=mock_processor,
            text="[1] Hello world",
            voice_samples=[{"waveform": torch.randn(1, 1, 24000), "sample_rate": 24000}],
            speaker_ids=[1],
            inference_steps=10,
        )
        if expect_exception is not None:
            with pytest.raises(expect_exception):
                call()
        else:
            call()
        return mock_model, mock_pbar_cls, mock_pbar, mock_interrupt


class TestStreamingProgressReporting:
    def test_generate_receives_callable_progress_callback(self):
        """T4.1: streaming model.generate must receive a callable progress_callback."""
        mock_model, _, _, _ = _run_streaming()
        gen_kwargs = mock_model.generate.call_args.kwargs
        cb = gen_kwargs.get("progress_callback")
        assert cb is not None and callable(cb)

    def test_callback_maps_to_update_absolute_with_total(self):
        """T4.2: callback(current, total) -> pbar.update_absolute(current, total=total)."""
        mock_model, _, mock_pbar, _ = _run_streaming()
        cb = mock_model.generate.call_args.kwargs["progress_callback"]
        mock_pbar.update_absolute.reset_mock()

        cb(42, 512)
        mock_pbar.update_absolute.assert_called_once_with(42, total=512)

    def test_callback_checks_interrupt(self):
        """T4.3a: the callback must call throw_exception_if_processing_interrupted;
        a raised interrupt propagates to the caller."""
        import comfy.model_management as mm

        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_output = MagicMock()
        mock_output.speech_outputs = [torch.randn(24000)]
        mock_model.generate.return_value = mock_output

        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.encode = MagicMock(return_value=[10, 11, 12])
        mock_processor.prepare_speech_inputs = MagicMock(
            return_value={
                "padded_speeches": torch.randn(1, 100),
                "speech_masks": torch.ones(1, 100, dtype=torch.bool),
            }
        )

        with patch("ComfyUI_VibeVoice.modules.generation.ProgressBar") as mock_pbar_cls, \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.throw_exception_if_processing_interrupted") as mock_interrupt, \
             patch("ComfyUI_VibeVoice.modules.generation.prefill_voice_prompt", return_value={"lm": MagicMock()}), \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio", return_value=_mock_voice_sample()):
            mock_pbar = MagicMock()
            mock_pbar.total = 1
            mock_pbar_cls.return_value = mock_pbar
            mock_interrupt.side_effect = mm.InterruptProcessingException()

            generate_streaming_audio(
                model=mock_model,
                processor=mock_processor,
                text="[1] Hello world",
                voice_samples=[{"waveform": torch.randn(1, 1, 24000), "sample_rate": 24000}],
                speaker_ids=[1],
                inference_steps=10,
            )

            cb = mock_model.generate.call_args.kwargs["progress_callback"]
            with pytest.raises(mm.InterruptProcessingException):
                cb(1, 512)
            mock_interrupt.assert_called()

    def test_final_update_sent_on_success_and_exception(self):
        """T4.3b: the finally-block final update fires on success and on raise."""
        # Success path.
        _, mock_pbar_cls, mock_pbar, _ = _run_streaming()
        mock_pbar_cls.assert_called_once_with(1)  # placeholder total
        final_call = mock_pbar.update_absolute.call_args_list[-1]
        assert final_call.args == (1,), f"final update must be (total,), got {final_call}"

        # Exception path.
        mock_model, mock_pbar_cls2, mock_pbar2, _ = _run_streaming(
            generate_side_effect=RuntimeError("boom"),
            expect_exception=RuntimeError,
        )
        final_call2 = mock_pbar2.update_absolute.call_args_list[-1]
        assert final_call2.args == (1,)
