"""Mocked adapter contract tests for official realtime generation."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from ComfyUI_VibeVoice.modules import realtime_generation
from ComfyUI_VibeVoice.modules.realtime_generation import (
    classify_loaded_tts_pair,
    generate_realtime_audio,
    is_realtime_model,
    to_single_speaker_text,
)
from ComfyUI_VibeVoice.modules.voice_presets import PRESET_CACHE_KEYS


class VibeVoiceStreamingForConditionalGenerationInference:
    pass


class VibeVoiceStreamingProcessor:
    def __init__(self):
        self.call_mock = MagicMock()

    def __call__(self, *args, **kwargs):
        return self.call_mock(*args, **kwargs)


class StandardModel:
    pass


class StandardProcessor:
    pass


def _branch(length=3):
    return SimpleNamespace(last_hidden_state=torch.ones(1, length, 4))


def _preset():
    return {key: _branch(index + 2) for index, key in enumerate(PRESET_CACHE_KEYS)}


def _pair(output=None, generate_error=None):
    model = VibeVoiceStreamingForConditionalGenerationInference()
    model.set_ddpm_inference_steps = MagicMock()
    model.generate = MagicMock()
    processor = VibeVoiceStreamingProcessor()
    processor.process_input_with_cached_prompt = MagicMock()
    processor.prepare_speech_inputs = MagicMock()
    processor.tokenizer = object()
    if generate_error is not None:
        model.generate.side_effect = generate_error
    else:
        if output is None:
            output = torch.arange(24, dtype=torch.float32).reshape(24)
        model.generate.return_value = SimpleNamespace(speech_outputs=[output])
    processor.process_input_with_cached_prompt.return_value = {
        "tts_text_ids": torch.ones(1, 5, dtype=torch.long),
        "tts_lm_input_ids": torch.ones(1, 3, dtype=torch.long),
        "metadata": "unchanged",
    }
    return model, processor


def _run(model, processor, preset=None, **kwargs):
    pbar = MagicMock()
    pbar.total = 1
    with patch.object(realtime_generation, "ProgressBarWithConsole", return_value=pbar), patch.object(
        realtime_generation.model_management,
        "get_torch_device",
        return_value=torch.device("cpu"),
    ), patch.object(realtime_generation.model_management, "throw_exception_if_processing_interrupted"):
        result = generate_realtime_audio(
            model,
            processor,
            text=kwargs.pop("text", "[1] Hello\n[2] world"),
            voice_preset=preset if preset is not None else _preset(),
            **kwargs,
        )
    return result, pbar


class TestRealtimeClassification:
    def test_official_pair_is_realtime(self):
        pair = (VibeVoiceStreamingForConditionalGenerationInference(), VibeVoiceStreamingProcessor())
        assert is_realtime_model(*pair) is True
        assert classify_loaded_tts_pair(*pair) == "realtime"

    def test_standard_pair_is_standard(self):
        assert classify_loaded_tts_pair(StandardModel(), StandardProcessor()) == "standard"

    @pytest.mark.parametrize(
        ("model", "processor"),
        [
            (VibeVoiceStreamingForConditionalGenerationInference(), StandardProcessor()),
            (StandardModel(), VibeVoiceStreamingProcessor()),
        ],
    )
    def test_mixed_pair_is_mismatch(self, model, processor):
        assert classify_loaded_tts_pair(model, processor) == "mismatch"


class TestSingleSpeakerText:
    def test_strips_labels_and_joins_lines(self):
        assert to_single_speaker_text("[1] Hello\nSpeaker 2: world") == "Hello\nworld"

    @pytest.mark.parametrize("text", ["", "   \n\t", "[1]   "])
    def test_empty_normalized_text_rejected(self, text):
        with pytest.raises(ValueError, match="empty or invalid"):
            to_single_speaker_text(text)


class TestLengthBudgetPlanning:
    """The automatic length budget, which is what keeps a short prompt from
    decoding the model's entire 8192-token context."""

    def test_budget_is_a_whole_number_of_windows(self):
        for tokens in range(0, 400, 7):
            budget = realtime_generation.plan_auto_length_budget(tokens)
            if budget > realtime_generation.REALTIME_MIN_BUDGET_UNITS:
                assert budget % realtime_generation.REALTIME_BUDGET_UNITS_PER_WINDOW == 0

    def test_budget_covers_every_text_token(self):
        """The regression that truncated scripts.

        ``max_new_tokens`` pays for text AND speech: each window charges 5
        text tokens plus 6 latents. A 62-token script budgeted at 78 units
        prefilled only 35 tokens and spoke one sentence. The budget must always
        leave room for the whole script.
        """
        for tokens in (7, 19, 35, 62, 88, 200):
            budget = realtime_generation.plan_auto_length_budget(tokens)
            # Units needed just to prefill the text, at 5 tokens per window.
            needed = (
                math.ceil(tokens / realtime_generation.REALTIME_TEXT_WINDOW_SIZE)
                * realtime_generation.REALTIME_TEXT_WINDOW_SIZE
            )
            assert budget >= needed, f"{tokens} tokens: {budget} < {needed}"

    def test_measured_truncation_case_now_fits(self):
        # The exact failure from the user's run: 62 text tokens, 78 units,
        # 35 prefilled. The budget must now clear the prefill cost with margin.
        assert realtime_generation.plan_auto_length_budget(62) > 78

    def test_window_accounting_matches_the_vendored_loop(self):
        assert realtime_generation.REALTIME_TEXT_WINDOW_SIZE == 5
        assert realtime_generation.REALTIME_SPEECH_WINDOW_SIZE == 6
        assert realtime_generation.REALTIME_BUDGET_UNITS_PER_WINDOW == 11

    def test_short_script_gets_a_few_seconds_not_minutes(self):
        # ~7 tokens: two text windows plus five tail windows, 11 units each.
        assert realtime_generation.plan_auto_length_budget(7) == 77

    @pytest.mark.parametrize(
        "text_tokens, latents_measured",
        [
            # VibeVoice-Realtime-0.5B, transformers 5.3.0, node load path,
            # cfg 1.5, 10 diffusion steps, en-Carter_man. The latents are how
            # many the model actually sampled before signalling end of speech.
            (10, 30),    # "The quick brown fox jumps over the lazy dog."
            (50, 108),   # four sentences
            (265, 534),  # the 239-word acceptance script
        ],
    )
    def test_budget_covers_what_the_model_measured_itself_needing(
        self, text_tokens, latents_measured
    ):
        """The regression that truncated every acceptance clip.

        A budget below the latents the model needs to reach its own EOS does
        not merely run long: the clip ends mid-utterance, because generation is
        cut off while the model is still speaking. These three lengths were
        measured end to end, so the budget has to cover them.
        """
        budget = realtime_generation.plan_auto_length_budget(text_tokens)
        latents_the_budget_buys = (
            budget * realtime_generation.REALTIME_SPEECH_WINDOW_SIZE
            // realtime_generation.REALTIME_BUDGET_UNITS_PER_WINDOW
        )
        assert latents_the_budget_buys >= latents_measured, (
            f"{text_tokens} text tokens: budget {budget} buys about "
            f"{latents_the_budget_buys} latents but the model needed "
            f"{latents_measured} to signal end of speech, so the clip would be "
            f"truncated on the length budget."
        )

    def test_budget_scales_with_script_length(self):
        short = realtime_generation.plan_auto_length_budget(20)
        long = realtime_generation.plan_auto_length_budget(200)
        assert long > short

    def test_empty_script_falls_back_to_the_floor(self):
        assert realtime_generation.plan_auto_length_budget(0) == (
            realtime_generation.REALTIME_MIN_BUDGET_UNITS
        )

    def test_tiny_script_still_covers_its_text(self):
        for tokens in (1, 2, 3):
            budget = realtime_generation.plan_auto_length_budget(tokens)
            assert budget >= tokens

    def test_very_long_script_is_capped(self):
        # The reported OOM: 8192 latents is ~18 minutes and exhausts the
        # acoustic-decoder cache. Auto must never reach that.
        assert realtime_generation.plan_auto_length_budget(100000) == (
            realtime_generation.REALTIME_MAX_AUTO_BUDGET_UNITS
        )
        assert realtime_generation.REALTIME_MAX_AUTO_BUDGET_UNITS < 8192

    def test_explicit_request_is_honoured_verbatim(self):
        for requested in (1, 64, 500, 8192):
            assert realtime_generation.resolve_length_budget(
                requested, text_token_count=5, context_room=8000
            ) == requested

    @pytest.mark.parametrize("requested", (None, 0))
    def test_auto_never_exceeds_the_context_room(self, requested):
        resolved = realtime_generation.resolve_length_budget(
            requested, text_token_count=5, context_room=3
        )
        assert resolved == 3
        assert resolved >= 1

    def test_auto_respects_context_room_zero_without_going_negative(self):
        assert realtime_generation.resolve_length_budget(
            None, text_token_count=5, context_room=0
        ) == 1


class TestGenerateRealtimeAdapter:
    def test_exact_processor_contract_and_generation_keywords(self):
        model, processor = _pair()
        preset = _preset()
        _run(model, processor, preset, diffusion_steps=7, max_new_tokens=123, cfg_scale=1.7)
        processor.process_input_with_cached_prompt.assert_called_once_with(
            text="Hello\nworld",
            cached_prompt=preset,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        model.set_ddpm_inference_steps.assert_called_once_with(num_steps=7)
        kwargs = model.generate.call_args.kwargs
        assert kwargs["max_new_tokens"] == 123
        assert kwargs["cfg_scale"] == 1.7
        assert kwargs["tokenizer"] is processor.tokenizer
        assert kwargs["metadata"] == "unchanged"
        assert "do_sample" not in kwargs
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs
        assert "top_k" not in kwargs

    def test_zero_length_budget_is_planned_from_the_script(self):
        """0 means "auto", and auto must NOT fall through to the model's full
        8192-token context (~18 minutes of audio, which exhausts VRAM in the
        acoustic-decoder cache). The fixture script is 5 text tokens."""
        model, processor = _pair()
        _run(model, processor, max_new_tokens=0)
        budget = model.generate.call_args.kwargs["max_new_tokens"]
        assert budget == realtime_generation.plan_auto_length_budget(5)
        assert budget < realtime_generation.REALTIME_MAX_AUTO_BUDGET_UNITS

    def test_none_length_budget_is_planned_from_the_script(self):
        model, processor = _pair()
        _run(model, processor, max_new_tokens=None)
        assert model.generate.call_args.kwargs["max_new_tokens"] == (
            realtime_generation.plan_auto_length_budget(5)
        )

    def test_cached_prompt_is_deep_copied_and_source_unchanged(self):
        model, processor = _pair()
        preset = _preset()
        original = copy.deepcopy(preset)

        def _mutate_generated_prompt(**kwargs):
            for value in kwargs["all_prefilled_outputs"].values():
                value.last_hidden_state.zero_()
            return SimpleNamespace(speech_outputs=[torch.ones(10)])

        model.generate.side_effect = _mutate_generated_prompt
        _run(model, processor, preset)
        for key in PRESET_CACHE_KEYS:
            assert torch.equal(preset[key].last_hidden_state, original[key].last_hidden_state)
            assert model.generate.call_args.kwargs["all_prefilled_outputs"] is not preset

    def test_seed_is_set_before_generation(self):
        model, processor = _pair()
        with patch.object(realtime_generation, "set_seed") as set_seed:
            _run(model, processor, seed=99)
        set_seed.assert_called_once_with(99)
        assert set_seed.call_count == 1

    def test_standard_processor_call_and_speech_prep_never_used(self):
        model, processor = _pair()
        _run(model, processor)
        processor.call_mock.assert_not_called()
        processor.prepare_speech_inputs.assert_not_called()

    @pytest.mark.parametrize("shape", [(24,), (1, 24), (1, 1, 24)])
    def test_output_normalization(self, shape):
        model, processor = _pair(output=torch.ones(shape, dtype=torch.float64))
        waveform, sample_rate = _run(model, processor)[0]
        assert waveform.shape == (1, 1, 24)
        assert waveform.device.type == "cpu"
        assert waveform.dtype == torch.float32
        assert sample_rate == 24000

    def test_no_speech_output_rejected(self):
        model, processor = _pair(output=None)
        model.generate.return_value.speech_outputs = []
        with pytest.raises(RuntimeError, match="no audio"):
            _run(model, processor)

    @pytest.mark.parametrize("missing_key", PRESET_CACHE_KEYS)
    def test_missing_cached_branch_rejected_before_processor(self, missing_key):
        model, processor = _pair()
        preset = _preset()
        del preset[missing_key]
        with pytest.raises(ValueError, match=missing_key):
            _run(model, processor, preset)
        processor.process_input_with_cached_prompt.assert_not_called()

    def test_processor_exception_propagates_and_progress_finalizes(self):
        model, processor = _pair()
        processor.process_input_with_cached_prompt.side_effect = RuntimeError("processor boom")
        with pytest.raises(RuntimeError, match="processor boom"):
            _run(model, processor)

    def test_generation_exception_propagates_and_progress_finalizes(self):
        model, processor = _pair(generate_error=RuntimeError("generation boom"))
        with pytest.raises(RuntimeError, match="generation boom"):
            _run(model, processor)

    def test_user_interruption_propagates(self):
        model, processor = _pair()
        model.generate.side_effect = realtime_generation.model_management.InterruptProcessingException()
        with pytest.raises(realtime_generation.model_management.InterruptProcessingException):
            _run(model, processor)

    def test_progress_callback_maps_and_checks_interruption(self):
        model, processor = _pair()
        _result, pbar = _run(model, processor)
        callback = model.generate.call_args.kwargs["progress_callback"]
        with patch.object(
            realtime_generation.model_management,
            "throw_exception_if_processing_interrupted",
        ) as interrupt:
            callback(4, 9)
        pbar.update_absolute.assert_called_with(4, total=9)
        interrupt.assert_called_once()


class TestRealtimeCfgScaleFloor:
    """The realtime checkpoint degenerates into syllable repetition below ~1.5.

    Measured on VibeVoice-Realtime-0.5B with the local ASR checkpoint scoring
    the output: 1.1/1.3/1.4 all loop ("it's a, it's a, ..."), 1.5-1.8 speak the
    script correctly. The node's shared widget default is 1.3 — right for the
    standard models, gibberish here — so the adapter raises it.
    """

    def test_below_floor_is_raised(self):
        assert realtime_generation.resolve_cfg_scale(1.3) == (
            realtime_generation.REALTIME_MIN_CFG_SCALE,
            True,
        )

    def test_floor_boundary_is_not_adjusted(self):
        scale, adjusted = realtime_generation.resolve_cfg_scale(
            realtime_generation.REALTIME_MIN_CFG_SCALE
        )
        assert (scale, adjusted) == (realtime_generation.REALTIME_MIN_CFG_SCALE, False)

    def test_usable_band_is_passed_through(self):
        for value in (1.5, 1.6, 1.8, 2.0, 5.0):
            assert realtime_generation.resolve_cfg_scale(value) == (value, False)

    @pytest.mark.parametrize("bad", [None, "nonsense", object()])
    def test_unusable_value_falls_back_to_default(self, bad):
        scale, adjusted = realtime_generation.resolve_cfg_scale(bad)
        assert scale == realtime_generation.REALTIME_DEFAULT_CFG_SCALE
        assert adjusted is True

    def test_adapter_raises_the_node_default_before_sampling(self):
        model, processor = _pair()
        _run(model, processor, cfg_scale=1.3)
        assert model.generate.call_args.kwargs["cfg_scale"] == (
            realtime_generation.REALTIME_MIN_CFG_SCALE
        )

    def test_adapter_honours_an_explicit_usable_value(self):
        model, processor = _pair()
        _run(model, processor, cfg_scale=1.7)
        assert model.generate.call_args.kwargs["cfg_scale"] == 1.7

    def test_adapter_warns_only_when_it_actually_adjusts(self, caplog):
        model, processor = _pair()
        with caplog.at_level("WARNING", logger=realtime_generation.logger.name):
            _run(model, processor, cfg_scale=1.0)
        assert any(
            "cfg_scale" in r.message and "raised" in r.message
            for r in caplog.records
        ), caplog.records

        caplog.clear()
        model, processor = _pair()
        with caplog.at_level("WARNING", logger=realtime_generation.logger.name):
            _run(model, processor, cfg_scale=1.6)
        assert not any(
            "has been raised" in r.message for r in caplog.records
        ), caplog.records


class TestTruncationIsReported:
    """Running out of budget leaves post-script filler, which is audible.

    The loop reports this itself via ``reach_max_step_sample``; the adapter has
    to surface it, otherwise the user gets a clip whose tail is babble and no
    explanation.
    """

    def _output(self, reached):
        return SimpleNamespace(
            speech_outputs=[torch.arange(24, dtype=torch.float32).reshape(24)],
            reach_max_step_sample=torch.tensor([reached]),
        )

    def _pair_with(self, output):
        # ``_pair`` wraps its ``output`` argument *inside* speech_outputs, so a
        # full generate() result object has to be installed afterwards.
        model, processor = _pair()
        model.generate.return_value = output
        return model, processor

    def test_warns_when_the_budget_ran_out(self, caplog):
        model, processor = self._pair_with(self._output(True))
        with caplog.at_level("WARNING", logger=realtime_generation.logger.name):
            _run(model, processor)
        assert any(
            "without the model signalling end of speech" in r.message
            for r in caplog.records
        ), caplog.records

    def test_quiet_when_the_model_finished_on_its_own(self, caplog):
        model, processor = self._pair_with(self._output(False))
        with caplog.at_level("WARNING", logger=realtime_generation.logger.name):
            _run(model, processor)
        assert not any(
            "without the model signalling end of speech" in r.message
            for r in caplog.records
        ), caplog.records

    def test_tolerates_outputs_without_the_flag(self):
        # The default _pair() result carries no reach_max_step_sample at all,
        # which is what a non-vendored model would hand back.
        model, processor = _pair()
        assert not hasattr(model.generate.return_value, "reach_max_step_sample")
        (waveform, _sample_rate), _pbar = _run(model, processor)
        assert waveform.shape[-1] == 24
