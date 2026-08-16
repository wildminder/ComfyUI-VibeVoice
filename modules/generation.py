"""Shared generation utilities for VibeVoice nodes.

Contains common functions for model loading, audio processing,
and generation to avoid code duplication across nodes.
"""

import torch
import gc
import logging
import numpy as np
from typing import Optional, Tuple, Any

import comfy.model_management as model_management
from comfy.utils import ProgressBar

from .loader import VibeVoiceModelHandler, VibeVoiceLoader, cleanup_old_models, LOADED_MODELS_CACHE
from .patcher import VibeVoicePatcher
from .utils import VIBEVOICE_PATCHER_CACHE
from .audio_utils import parse_script_1_based, preprocess_comfy_audio, set_seed, check_for_interrupt
from .device_utils import get_torch_device, get_offload_device, DEVICE_CPU
from .dtype_utils import resolve_dtype, DTYPE_AUTO
from .attention_utils import resolve_attention_mode

logger = logging.getLogger(__name__)


def load_vibevoice_model(
    model_name: str,
    device: str = "auto",
    dtype: str = DTYPE_AUTO,
    attention_mode: str = "sdpa",
    quantize_4bit: bool = False,
    force_reload: bool = False,
) -> Tuple[VibeVoicePatcher, Any, Any]:
    """Load or retrieve cached VibeVoice model.

    Args:
        model_name: Name of the model to load.
        device: Device to load on ("auto", "cuda", "cpu", "mps", etc.).
        dtype: Data type for model ("auto", "bf16", "fp16", "fp32").
        attention_mode: Attention implementation ("eager", "sdpa", "flash_attention_2", "sage").
        quantize_4bit: Whether to quantize the LLM to 4-bit NF4.
        force_reload: Force reload even if cached.

    Returns:
        Tuple of (patcher, model, processor).

    Raises:
        RuntimeError: If model fails to load.
    """
    # Resolve attention mode with fallback logic
    actual_attention_mode = resolve_attention_mode(attention_mode, quantize_4bit)

    # Setup device
    if device == DEVICE_CPU:
        load_device = torch.device(DEVICE_CPU)
        offload_device = torch.device(DEVICE_CPU)
    else:
        load_device = get_torch_device(device)
        offload_device = get_offload_device()

    # Resolve dtype
    target_dtype = resolve_dtype(dtype, load_device)

    # Build cache key
    cache_key = f"{model_name}_attn_{actual_attention_mode}_q4_{int(quantize_4bit)}"

    if cache_key not in VIBEVOICE_PATCHER_CACHE or force_reload:
        if force_reload:
            cleanup_old_models(keep_cache_key=cache_key)

        model_handler = VibeVoiceModelHandler(
            model_name,
            attention_mode=actual_attention_mode,
            use_llm_4bit=quantize_4bit,
            dtype_str=dtype,
        )

        patcher = VibeVoicePatcher(
            model_handler,
            attention_mode=actual_attention_mode,
            load_device=load_device,
            offload_device=offload_device,
            size=model_handler.size,
            dtype=target_dtype,
        )
        VIBEVOICE_PATCHER_CACHE[cache_key] = patcher
        logger.debug(f"Created new patcher for {model_name} with attn={actual_attention_mode}, q4={quantize_4bit}")

    patcher = VIBEVOICE_PATCHER_CACHE[cache_key]
    model_management.load_model_gpu(patcher)
    model = patcher.model.model
    processor = patcher.model.processor

    if model is None or processor is None:
        raise RuntimeError(
            f"VibeVoice model and processor could not be loaded for '{model_name}'. Check logs for errors."
        )

    return patcher, model, processor


def generate_audio(
    model: Any,
    processor: Any,
    text: str,
    voice_samples: list,
    speaker_ids: list,
    cfg_scale: float = 1.3,
    inference_steps: int = 10,
    seed: int = 42,
    do_sample: bool = True,
    temperature: float = 0.95,
    top_p: float = 0.95,
    top_k: int = 0,
    max_new_tokens: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """Generate audio using the VibeVoice model.

    Args:
        model: VibeVoice model instance.
        processor: VibeVoice processor instance.
        text: Multi-speaker script text.
        voice_samples: List of numpy audio arrays for each speaker.
        speaker_ids: List of 1-based speaker IDs.
        cfg_scale: Classifier-Free Guidance scale.
        inference_steps: Number of diffusion steps.
        seed: Random seed (0 for random).
        do_sample: Whether to use sampling methods.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        top_k: Top-K sampling (0 to disable).
        max_new_tokens: Hard cap on generated speech tokens (utterance length budget).
            None = auto (~30x prompt length). Passed through to ``model.generate``
            so the non-streaming AR loop terminates; when the processor tokenizer
            exposes ``speech_end_id`` it also stops on the EOS speech token.

    Returns:
        Tuple of (output audio tensor [1, 1, T], sample_rate).

    Raises:
        ValueError: If script is empty or invalid, or if a streaming
            (realtime) model/processor is passed (those require
            ``generate_streaming_audio`` / the VibeVoice Realtime TTS node).
        RuntimeError: If generation fails.
    """
    # Guard: streaming (realtime) models/processors require the streaming
    # generation path (generate_streaming_audio) — voice-prompt prefill +
    # windowed AR loop. The non-streaming path below calls
    # processor(text=..., voice_samples=...), which VibeVoiceStreamingProcessor
    # does not support (its __call__ takes no arguments by design). Detect by
    # class name to avoid importing the vendored chain (diffusers-dependent).
    _streaming_class_names = {
        "VibeVoiceStreamingProcessor",
        "VibeVoiceStreamingForConditionalGenerationInference",
    }
    if (
        type(processor).__name__ in _streaming_class_names
        or type(model).__name__ in _streaming_class_names
    ):
        raise ValueError(
            "A streaming (realtime) VibeVoice model was passed to generate_audio(). "
            "Streaming models require the 'VibeVoice Realtime TTS' node "
            "(generate_streaming_audio). Please use that node for this model."
        )

    # Parse script using our parser (supports both "[N] text" and "Speaker N: text" formats)
    parsed_lines_0_based, speaker_ids_1_based = parse_script_1_based(text)
    if not parsed_lines_0_based:
        raise ValueError("Script is empty or invalid. Please provide text to generate.")

    # Preprocess voice samples — filter out None values (speakers without voice input)
    # and ensure each sample is a valid 1-D numpy array.
    voice_samples_np = []
    for vs in voice_samples:
        processed = preprocess_comfy_audio(vs)
        if processed is not None:
            # Ensure 1-D array (squeeze extra dimensions but keep the sample axis)
            processed = np.asarray(processed, dtype=np.float32)
            if processed.ndim == 0:
                logger.warning("Voice sample is a scalar (0-d array), skipping")
                continue
            if processed.ndim > 1:
                processed = np.squeeze(processed)
            if processed.ndim != 1:
                logger.warning(f"Voice sample has unexpected shape {processed.shape}, skipping")
                continue
            voice_samples_np.append(processed)

    if not voice_samples_np:
        raise ValueError(
            "No valid voice samples provided. Please connect at least one audio input "
            "as a voice sample for the speaker(s)."
        )

    # Set seed
    set_seed(seed)

    # Convert parsed lines to "Speaker N: text" format for the processor.
    # The vendored processor's _parse_script() only recognizes the "Speaker N: text"
    # format, but our parse_script_1_based() also supports "[N] text". We normalize
    # to the processor's expected format to support both input styles.
    # parsed_lines_0_based contains (0-based speaker_id, text) tuples; convert back
    # to 1-based for the "Speaker N:" prefix.
    normalized_script = "\n".join(
        f"Speaker {speaker_id + 1}:{speaker_text}"
        for speaker_id, speaker_text in parsed_lines_0_based
    )

    # Build model inputs — pass the normalized script to the processor
    inputs = processor(
        text=[normalized_script],
        voice_samples=[voice_samples_np],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # Validate input tensors
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if torch.any(torch.isnan(value)) or torch.any(torch.isinf(value)):
                logger.error(f"Input tensor '{key}' contains NaN or Inf values")
                raise ValueError(f"Invalid values in input tensor: {key}")

    # Move inputs to model device
    inputs = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    # Set inference steps (non-streaming model stores this on the instance)
    model.set_ddpm_inference_steps(num_steps=inference_steps)

    # Map processor output keys to the model's generate() signature.
    # The processor returns `speech_input_mask`; the non-streaming model's
    # generate() expects `acoustic_input_mask`.
    gen_inputs = {
        "input_ids": inputs.get("input_ids"),
        "attention_mask": inputs.get("attention_mask"),
        "speech_tensors": inputs.get("speech_tensors"),
        "speech_masks": inputs.get("speech_masks"),
        "acoustic_input_mask": inputs.get("speech_input_mask"),
        "cfg_scale": cfg_scale,
        "inference_steps": inference_steps,
        "return_speech": True,
        # Option C: pass the processor tokenizer so generate() can resolve
        # `speech_end_id` and terminate the AR loop on the EOS speech token.
        # Also forward the sampling controls + utterance-length budget.
        "tokenizer": processor.tokenizer,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
    }
    # Drop None values so the model uses its own defaults where appropriate.
    gen_inputs = {k: v for k, v in gen_inputs.items() if v is not None}

    # Generate
    with torch.no_grad():
        # Standard ComfyUI progress bar. The initial total is only an estimate
        # (diffusion steps); the vendored AR loop reports its real budget
        # (max_steps) through `progress_callback`, and `update_absolute(value,
        # total=...)` re-sets the bar's total dynamically on the first callback.
        pbar = ProgressBar(inference_steps)

        def _progress(current: int, total: int) -> None:
            # Responsive cancellation: raises InterruptProcessingException when
            # the user pressed cancel (checked once per AR step).
            model_management.throw_exception_if_processing_interrupted()
            pbar.update_absolute(current, total=total)

        try:
            outputs = model.generate(**gen_inputs, progress_callback=_progress)

        except model_management.InterruptProcessingException:
            logger.info("VibeVoice generation interrupted by user")
            raise
        finally:
            # Guarantee the final 100% event even when the AR loop stopped
            # early (EOS before max_steps) or generation raised.
            pbar.update_absolute(pbar.total)

    # Post-process output
    output_waveform = outputs.speech_outputs[0]
    if output_waveform.ndim == 1:
        output_waveform = output_waveform.unsqueeze(0)
    if output_waveform.ndim == 2:
        output_waveform = output_waveform.unsqueeze(0)

    sample_rate = 24000
    return output_waveform.detach().cpu(), sample_rate


def prefill_voice_prompt(
    model: Any,
    processor: Any,
    voice_samples_np: list,
    tts_text_ids: torch.Tensor,
    device: torch.device,
) -> dict:
    """Build ``all_prefilled_outputs`` consumed by the streaming ``model.generate``.

    The vendored streaming ``model.generate`` reads ``all_prefilled_outputs`` with
    keys ``lm``, ``tts_lm``, ``neg_lm``, ``neg_tts_lm`` — each a forward output
    exposing ``.last_hidden_state`` and ``.past_key_values``. This helper runs the
    model's ``forward_lm`` / ``forward_tts_lm`` on the voice-prompt tokens to
    produce those outputs.

    NOTE: The reference-voice -> speech-token encoding
    (``model.set_speech_tokenizers`` + acoustic tokenizer) and the construction of
    the voice-prompt ``input_ids`` / ``tts_lm_input_ids`` from reference audio are
    model/processor-internal steps. This helper builds padded placeholder prompt
    ids of the text-window length and runs the LM prefill, mirroring
    ``generate``'s first-window handling. Validate against the real model on GPU
    before production use.
    """
    tokenizer = processor.tokenizer
    pad_id = getattr(tokenizer, "pad_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "pad_token_id", 0)

    prompt_len = int(tts_text_ids.shape[1])
    lm_input_ids = torch.full((1, prompt_len), int(pad_id), dtype=torch.long, device=device)
    tts_lm_input_ids = torch.full((1, prompt_len), int(pad_id), dtype=torch.long, device=device)
    attn = torch.ones((1, prompt_len), dtype=torch.long, device=device)
    text_masks = torch.ones_like(tts_lm_input_ids)

    with torch.no_grad():
        lm_out = model.forward_lm(
            input_ids=lm_input_ids, attention_mask=attn, use_cache=True,
            return_dict=True, output_attentions=False, output_hidden_states=False,
        )
        tts_lm_out = model.forward_tts_lm(
            input_ids=tts_lm_input_ids, attention_mask=attn,
            lm_last_hidden_state=lm_out.last_hidden_state, tts_text_masks=text_masks,
            use_cache=True, return_dict=True, output_attentions=False, output_hidden_states=False,
        )
        neg_lm_out = model.forward_lm(
            input_ids=lm_input_ids, attention_mask=attn, use_cache=True,
            return_dict=True, output_attentions=False, output_hidden_states=False,
        )
        neg_tts_lm_out = model.forward_tts_lm(
            input_ids=tts_lm_input_ids, attention_mask=attn,
            lm_last_hidden_state=neg_lm_out.last_hidden_state, tts_text_masks=text_masks,
            use_cache=True, return_dict=True, output_attentions=False, output_hidden_states=False,
        )

    return {
        "lm": lm_out,
        "tts_lm": tts_lm_out,
        "neg_lm": neg_lm_out,
        "neg_tts_lm": neg_tts_lm_out,
    }


def generate_streaming_audio(
    model: Any,
    processor: Any,
    text: str,
    voice_samples: list,
    speaker_ids: list,
    cfg_scale: float = 1.3,
    inference_steps: int = 10,
    seed: int = 42,
    do_sample: bool = True,
    temperature: float = 0.95,
    top_p: float = 0.95,
    top_k: int = 0,
    stream: bool = False,
) -> Tuple[torch.Tensor, int]:
    """Streaming TTS generation via the VibeVoice streaming model.

    Routes to ``model.generate`` using the streaming kwarg contract
    (``tts_text_ids``, ``speech_tensors``, ``all_prefilled_outputs``, ...).

    The streaming model requires a *prefill* of the reference voice before
    generation. That prefill (``all_prefilled_outputs`` consumed by
    ``model.generate``) is produced by :func:`prefill_voice_prompt`. The
    voice -> speech-token encoding (``model.set_speech_tokenizers`` + acoustic
    tokenizer) and the construction of the voice-prompt token ids are
    model-internal steps performed by the streaming processor/model; this
    function focuses on input preparation + the LM-level prefill that
    ``generate`` expects. Validate end-to-end on GPU before production use.
    """
    parsed_lines_0_based, _ = parse_script_1_based(text)
    if not parsed_lines_0_based:
        raise ValueError("Script is empty or invalid. Please provide text to generate.")

    # Preprocess voice samples -> 1-D numpy arrays.
    voice_samples_np = []
    for vs in voice_samples:
        processed = preprocess_comfy_audio(vs)
        if processed is not None:
            processed = np.asarray(processed, dtype=np.float32)
            if processed.ndim == 0:
                continue
            if processed.ndim > 1:
                processed = np.squeeze(processed)
            if processed.ndim == 1:
                voice_samples_np.append(processed)

    if not voice_samples_np:
        raise ValueError(
            "No valid voice samples provided. Please connect at least one audio input "
            "as a voice sample for the speaker(s)."
        )

    set_seed(seed)

    device = model.device
    tokenizer = processor.tokenizer
    # Tokenize the script text into tts_text_ids (the windowed-streaming input).
    tts_text_ids = torch.tensor(
        [tokenizer.encode(text.strip() + "\n", add_special_tokens=False)],
        dtype=torch.long,
        device=device,
    )

    # Prepare reference speech tensors via the streaming processor.
    speech = processor.prepare_speech_inputs(
        voice_samples_np, return_tensors="pt", device=device
    )
    speech_tensors = speech.get("padded_speeches")
    speech_masks = speech.get("speech_masks")
    # speech_input_mask marks speech-token positions in the TTS-LM input; the
    # streaming processor builds this from the cached voice prompt. For a pure
    # text window we default to an all-false mask of the right shape, matching
    # process_input_with_cached_prompt's output for text-only windows.
    speech_input_mask = torch.zeros_like(tts_text_ids, dtype=torch.bool)

    # Prefill the reference voice to build all_prefilled_outputs.
    all_prefilled_outputs = prefill_voice_prompt(
        model, processor, voice_samples_np, tts_text_ids, device
    )

    gen_kwargs = dict(
        tts_text_ids=tts_text_ids,
        speech_tensors=speech_tensors,
        speech_masks=speech_masks,
        speech_input_mask=speech_input_mask,
        tokenizer=tokenizer,
        all_prefilled_outputs=all_prefilled_outputs,
        cfg_scale=cfg_scale,
        max_new_tokens=int(inference_steps),
        return_speech=True,
    )
    # `stream` is reserved for incremental output; the assembled waveform is
    # returned either way. If a streamer were wired in, pass audio_streamer here.
    if stream:
        gen_kwargs["return_speech"] = True

    with torch.no_grad():
        # Standard ComfyUI progress bar. The streaming loop's real total
        # (tts_lm max_length) is only known inside the vendored generate();
        # the initial total=1 placeholder is corrected on the first callback
        # via update_absolute(value, total=...).
        pbar = ProgressBar(1)

        def _progress(current: int, total: int) -> None:
            # Responsive cancellation: raises InterruptProcessingException when
            # the user pressed cancel (checked once per loop step).
            model_management.throw_exception_if_processing_interrupted()
            pbar.update_absolute(current, total=total)

        try:
            outputs = model.generate(**gen_kwargs, progress_callback=_progress)
        except model_management.InterruptProcessingException:
            logger.info("VibeVoice streaming generation interrupted by user")
            raise
        finally:
            # Guarantee the final 100% event even when the loop stopped early
            # (EOS classifier) or generation raised.
            pbar.update_absolute(pbar.total)

    speech_outputs = outputs.speech_outputs
    if not speech_outputs or speech_outputs[0] is None:
        raise RuntimeError(
            "Streaming generation produced no audio. Check the model, processor, "
            "and prefill outputs."
        )

    waveform = speech_outputs[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)

    sample_rate = 24000
    return waveform.detach().cpu(), sample_rate


def force_offload_model(patcher: VibeVoicePatcher, model_name: str, warm: bool = False) -> None:
    """Force offload a VibeVoice model from VRAM.

    Args:
        patcher: The VibeVoicePatcher instance.
        model_name: Name of the model (for logging).
        warm: When True, retain the model/processor tensors on the intermediate
            device (NTH-004 warm re-attach) instead of fully freeing them, so a
            subsequent load re-attaches from memory rather than reloading from disk.
    """
    logger.info(f"Force offloading VibeVoice model '{model_name}' from VRAM...")
    if patcher.is_loaded:
        if warm:
            patcher.unpatch_model(unpatch_weights=True, warm=True)
        else:
            patcher.unpatch_model(unpatch_weights=True)
    model_management.unload_all_models()
    gc.collect()
    model_management.soft_empty_cache()
    logger.info("Model force offload completed")
