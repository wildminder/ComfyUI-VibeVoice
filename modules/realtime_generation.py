"""Official cached-voice-prompt adapter for VibeVoice-Realtime generation.

The realtime model is a distinct architecture. This adapter calls the official
streaming processor's cached-prompt method and the vendored windowed generation
loop, then returns a completed ComfyUI-compatible waveform. It does not provide
live PCM transport.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any

import torch

import comfy.model_management as model_management

from .audio_utils import parse_script_1_based, set_seed
from .progress_utils import ProgressBarWithConsole
from .voice_presets import validate_voice_preset
from .gguf_quant import log_gguf_forward_counters



REALTIME_SAMPLE_RATE = 24000

# --- Realtime length budget -------------------------------------------------
# One speech latent decodes to speech_tok_compress_ratio (3200) samples at
# 24 kHz, i.e. 0.133 s, so the model emits ~7.5 latents per second of audio.
#
# ``max_new_tokens`` is a COMBINED budget, and this is the whole point. In
# ``modeling_vibevoice_streaming_inference.generate`` every outer iteration
# grows ``tts_lm_input_ids`` by a whole TTS_TEXT_WINDOW_SIZE (5) *text* tokens
# and then by TTS_SPEECH_WINDOW_SIZE (6) *speech* latents, and the loop exits
# as soon as ``tts_lm_input_ids`` passes ``max_length``. Text and speech are
# therefore charged against the same allowance, so one window of speech costs
# 5 + 6 = 11 units, not 6.
#
# Budgeting on speech latents alone truncates the script. Measured on a
# 62-text-token script with a 78-unit budget: the loop consumed 35 text tokens
# and 42 latents, hit the cap, and spoke only the first sentence.
REALTIME_TEXT_WINDOW_SIZE = 5
REALTIME_SPEECH_WINDOW_SIZE = 6
# Per window, in max_new_tokens units: 5 prefilled text tokens + 6 latents.
REALTIME_BUDGET_UNITS_PER_WINDOW = REALTIME_TEXT_WINDOW_SIZE + REALTIME_SPEECH_WINDOW_SIZE
# Windows of headroom AFTER the script's own text windows, so the model can
# finish the last clause and reach its own end-of-speech instead of being cut
# off the moment the last text window has been fed.
#
# The loop only feeds text for the first ceil(tokens/5) windows; every window
# after that still produces 6 latents, and that tail is what pays for finishing
# the utterance. It is not a fixed overhead - it scales with the script.
# Measured end to end on the real checkpoint (VibeVoice-Realtime-0.5B,
# transformers 5.3.0, node load path, cfg 1.5, 10 diffusion steps) through the
# production adapter, so these are the automatic budgets' own numbers:
#
#   text tokens -> latents before the model's own end of speech
#     10 ->  24 (en-Carter_man) /  24 (en-Emma_woman)
#     50 -> 102                  / 114
#    265 -> 540                  / 600
#
# A single margin window (the previous policy) under-budgeted all of them, which
# is what made every acceptance clip stop on the length budget and be truncated
# mid-utterance. Two tail windows per text window leaves roughly 2x headroom
# over the measured need at every length, so an unusually slow utterance still
# reaches its own end of speech. Over-shooting is harmless in principle - the
# loop stops on the model's EOS, not on the budget - but headroom is what makes
# that true in practice rather than by luck.
REALTIME_TAIL_WINDOWS_PER_TEXT_WINDOW = 2.0
REALTIME_TAIL_WINDOWS_MIN = 1
# Floor: enough for a very short utterance to start and finish (~1.6 s).
REALTIME_MIN_BUDGET_UNITS = 12
# Ceiling for an automatic budget, in the same units. Without a ceiling the
# model falls back to its full 8192-token context, which decodes ~18 minutes of
# audio and exhausts VRAM in the acoustic-tokenizer cache. This guards against
# pathological input, it is not a cap on normal scripts: the longest script
# measured here (265 text tokens) plans 1760 units, and the ceiling only starts
# to bind past roughly 430 text tokens (~380 words). Callers can still request
# more explicitly via max_new_tokens.
REALTIME_MAX_AUTO_BUDGET_UNITS = 3072

# --- Realtime guidance floor ------------------------------------------------
# The realtime architecture does not sample discrete tokens: the TTS-LM emits a
# continuous conditioning vector and ``sample_speech_tokens`` denoises a
# continuous acoustic latent from it. So cfg_scale is guidance on a continuous
# diffusion, and it behaves nothing like the standard model's cfg.
#
# Measured on the real checkpoint (VibeVoice-Realtime-0.5B, en-Carter_man,
# transcription scored with the local VibeVoice-ASR checkpoint):
#
#   cfg 1.1 -> "c'est un peu plus, c'est un peu plus, ..."   (repetition)
#   cfg 1.3 -> "it's a, it's a, it's a, ..."                (repetition)
#   cfg 1.4 -> "you're, you're, you're, ..."                (repetition)
#   cfg 1.5 -> "The quick brown fox jumps over the lazy dog." (exact)
#   cfg 1.6 -> exact    cfg 1.8 -> exact    cfg 2.0 -> drifts
#
# Below ~1.5 the model reliably degenerates into syllable-level repetition
# loops; the official realtime demo's default is 1.5. The node's shared widget
# default is 1.3, which is correct for the standard models but produces
# gibberish here, so the realtime path raises it to the measured floor.
REALTIME_MIN_CFG_SCALE = 1.5
REALTIME_DEFAULT_CFG_SCALE = 1.5

_REALTIME_PROCESSOR_NAMES = {"VibeVoiceStreamingProcessor"}
_REALTIME_MODEL_NAMES = {
    "VibeVoiceStreamingForConditionalGenerationInference",
}


def plan_auto_length_budget(text_token_count: int) -> int:
    """Pick a ``max_new_tokens`` budget for a script of ``text_token_count`` tokens.

    The loop charges text and speech against one allowance, so the budget is
    counted in the loop's own units: a window of 5 text tokens plus 6 speech
    latents costs 11. The script is covered window by window, plus the tail
    windows the model needs to reach its own end of speech, then clamped to
    :data:`REALTIME_MIN_BUDGET_UNITS`/:data:`REALTIME_MAX_AUTO_BUDGET_UNITS`.

    Pure function, so the accounting is unit-testable without a model.
    """
    tokens = max(0, int(text_token_count))
    text_windows = math.ceil(tokens / REALTIME_TEXT_WINDOW_SIZE) if tokens else 0
    tail_windows = (
        math.ceil(text_windows * REALTIME_TAIL_WINDOWS_PER_TEXT_WINDOW)
        + REALTIME_TAIL_WINDOWS_MIN
    )
    budget = (text_windows + tail_windows) * REALTIME_BUDGET_UNITS_PER_WINDOW
    return max(REALTIME_MIN_BUDGET_UNITS, min(budget, REALTIME_MAX_AUTO_BUDGET_UNITS))


def resolve_length_budget(
    requested: int | None,
    text_token_count: int,
    context_room: int,
) -> int:
    """Return the ``max_new_tokens`` value to hand to the streaming loop.

    An explicit positive ``requested`` value is honoured verbatim. ``None``/0
    means automatic, which is budgeted from the script length and then clipped
    to the context room the prompt left in the TTS-LM.
    """
    if requested is not None and int(requested) > 0:
        return int(requested)
    auto = plan_auto_length_budget(text_token_count)
    return max(1, min(auto, int(context_room)))


def resolve_cfg_scale(requested: float) -> tuple[float, bool]:
    """Clamp ``cfg_scale`` up to the realtime family's measured floor.

    Returns ``(scale, adjusted)``. ``adjusted`` is True when the caller's value
    was raised — the caller is expected to log that, because it means a saved
    workflow is about to produce syllable-level gibberish at the value the node
    offers by default.

    Only the floor is enforced. Higher values are left alone: 1.5-1.8 all speak
    the script correctly and 2.0 merely drifts, which is the normal
    CFG-vs-fidelity trade-off rather than a broken state.
    """
    try:
        scale = float(requested)
    except (TypeError, ValueError):
        return REALTIME_DEFAULT_CFG_SCALE, True
    if scale < REALTIME_MIN_CFG_SCALE:
        return REALTIME_MIN_CFG_SCALE, True
    return scale, False


def classify_loaded_tts_pair(model: object, processor: object) -> str:
    """Classify a loaded pair as ``standard``, ``realtime``, or ``mismatch``."""
    model_realtime = type(model).__name__ in _REALTIME_MODEL_NAMES
    processor_realtime = type(processor).__name__ in _REALTIME_PROCESSOR_NAMES
    if model_realtime != processor_realtime:
        return "mismatch"
    return "realtime" if model_realtime else "standard"


def is_realtime_model(model: object, processor: object) -> bool:
    """Return whether the loaded model/processor pair is the official pair."""
    return classify_loaded_tts_pair(model, processor) == "realtime"


def to_single_speaker_text(text: str) -> str:
    """Strip standard speaker labels and join lines into one realtime script."""
    parsed_lines, _speaker_ids = parse_script_1_based(text or "")
    script = "\n".join(
        speaker_text.strip() for _speaker_id, speaker_text in parsed_lines
    ).strip()
    if not script:
        raise ValueError("Script is empty or invalid. Please provide text to generate.")
    return script


def _normalize_waveform(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise RuntimeError("Realtime generation produced a non-tensor waveform.")
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] != 1:
        raise RuntimeError(
            "Realtime generation produced an invalid waveform shape "
            f"{tuple(value.shape)}; expected [1, 1, T]."
        )
    if value.shape[2] <= 0:
        raise RuntimeError("Realtime generation produced no audio samples.")
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous()


def generate_realtime_audio(
    model: Any,
    processor: Any,
    text: str,
    voice_preset: dict[str, Any],
    cfg_scale: float = REALTIME_DEFAULT_CFG_SCALE,
    diffusion_steps: int = 5,
    max_new_tokens: int | None = None,
    seed: int = 42,
) -> tuple[torch.Tensor, int]:
    """Generate one completed 24 kHz waveform from an official cached prompt."""
    pair_classification = classify_loaded_tts_pair(model, processor)
    if pair_classification == "mismatch":
        raise ValueError(
            "Loaded realtime model/processor pair is inconsistent: the model "
            "and processor classes must both be the VibeVoice realtime classes."
        )
    if pair_classification != "realtime":
        raise ValueError(
            "generate_realtime_audio() requires the official VibeVoice "
            "realtime model and processor classes."
        )

    script = to_single_speaker_text(text)
    validate_voice_preset(voice_preset, source="<voice_preset>")
    set_seed(seed)
    device = model_management.get_torch_device()
    model.set_ddpm_inference_steps(num_steps=diffusion_steps)

    # The node shares one cfg_scale widget between both families, and its 1.3
    # default — correct for the standard models — sits below the realtime
    # family's usable band, where the output degenerates into syllable
    # repetition. Raise it before it reaches the sampler.
    cfg_scale, cfg_adjusted = resolve_cfg_scale(cfg_scale)
    if cfg_adjusted:
        logging.warning(
            "[VibeVoice TTS] cfg_scale was below the realtime model's usable guidance floor and "
            "has been raised to %.2f. Values below %.2f make this checkpoint "
            "repeat syllables instead of speaking the script. Raise the "
            "'cfg_scale' widget to silence this warning.",
            cfg_scale,
            REALTIME_MIN_CFG_SCALE,
        )

    inputs = processor.process_input_with_cached_prompt(
        text=script,
        cached_prompt=voice_preset,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    if not hasattr(inputs, "items"):
        raise TypeError("Realtime processor returned a non-mapping result.")
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }

    # Budget the generation. ``None``/0 means "automatic": the streaming loop
    # would otherwise fall back to its full max_position_embeddings budget
    # (~8192 latents, ~18 minutes of audio) and exhaust VRAM decoding it.
    # Budget from the script length instead, clipped to the context the prompt
    # left in the TTS-LM.
    text_token_count = int(inputs["tts_text_ids"].shape[-1])
    tts_lm_prompt_len = int(inputs["tts_lm_input_ids"].shape[-1])
    try:
        context_room = int(model.config.decoder_config.max_position_embeddings) - tts_lm_prompt_len
    except AttributeError:
        logging.warning(
            "[VibeVoice TTS] Realtime model does not expose decoder_config.max_position_embeddings; "
            "falling back to the automatic length budget without a context cap."
        )
        context_room = REALTIME_MAX_AUTO_BUDGET_UNITS
    length_budget = resolve_length_budget(
        max_new_tokens, text_token_count=text_token_count, context_room=context_room
    )
    logging.info(
        "[VibeVoice TTS] Realtime length budget: %d max_new_tokens units (~%.1f s of speech) "
        "for %d text tokens.",
        length_budget,
        # The budget also pays for the text, so the speech it buys is only the
        # latent share of it.
        length_budget
        * REALTIME_SPEECH_WINDOW_SIZE
        / REALTIME_BUDGET_UNITS_PER_WINDOW
        * 3200
        / REALTIME_SAMPLE_RATE,
        text_token_count,
    )
    pbar = ProgressBarWithConsole(1)

    def _progress(current: int, total: int) -> None:
        model_management.throw_exception_if_processing_interrupted()
        pbar.update_absolute(current, total=total)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                all_prefilled_outputs=copy.deepcopy(voice_preset),
                max_new_tokens=length_budget,
                cfg_scale=cfg_scale,
                tokenizer=processor.tokenizer,
                progress_callback=_progress,
            )
    except model_management.InterruptProcessingException:
        logging.info("[VibeVoice TTS] VibeVoice realtime generation interrupted by user")
        raise
    finally:
        pbar.update_absolute(pbar.total)
        pbar.close()

    # After the forwards, not at load time: this is the only point where the
    # fast/streamed split says anything about this run. No-op for non-GGUF.
    log_gguf_forward_counters("realtime_generate")

    speech_outputs = getattr(outputs, "speech_outputs", None)
    if not speech_outputs or speech_outputs[0] is None:
        raise RuntimeError(
            "Realtime generation produced no audio. Check the selected voice "
            "preset and model."
        )

    # ``reach_max_step_sample`` is set when the loop exited on the length budget
    # instead of the model's own end-of-speech signal. That means the budget ran
    # out mid-utterance, so the tail of the clip is the model sampling past the
    # end of the script — the "undistinctive syllables" symptom. Say so, rather
    # than returning audio the user has to diagnose by ear.
    reach_max = getattr(outputs, "reach_max_step_sample", None)
    if reach_max is not None and bool(reach_max.any()):
        logging.warning(
            "[VibeVoice TTS] Realtime generation stopped on the length budget (%d units) "
            "without the model signalling end of speech, so the clip is "
            "truncated. Lower 'max_new_tokens' will not help; raise it, or "
            "shorten the script.",
            length_budget,
        )
    return _normalize_waveform(speech_outputs[0]), REALTIME_SAMPLE_RATE
