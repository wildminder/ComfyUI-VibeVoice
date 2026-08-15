"""ASR (Speech-to-Text) generation utilities for VibeVoice.

Handles audio transcription using the VibeVoice ASR model, including:
- Audio preprocessing
- Model inference
- Output decoding and structured transcription parsing
"""

import torch
import gc
import logging
from typing import Optional, Tuple, List, Dict, Any

import comfy.model_management as model_management

from .asr_loader import VibeVoiceASRLoader, VibeVoiceASRModelHandler, LOADED_ASR_MODELS_CACHE, cleanup_asr_models
from .patcher import VibeVoiceASRPatcher
from .utils import VIBEVOICE_ASR_PATCHER_CACHE
from .device_utils import get_torch_device, get_offload_device, DEVICE_CPU
from .dtype_utils import resolve_dtype, DTYPE_AUTO
from .audio_utils import extract_audio_tensor
from .attention_utils import resolve_attention_mode

logger = logging.getLogger(__name__)


def load_asr_model(
    model_name: str,
    device: str = "auto",
    dtype: str = DTYPE_AUTO,
    attention_mode: str = "sdpa",
    force_reload: bool = False,
) -> Tuple[Any, Any]:
    """Load or retrieve cached VibeVoice ASR model.

    Args:
        model_name: Name of the ASR model to load.
        device: Device to load on ("auto", "cuda", "cpu", "mps").
        dtype: Data type for model ("auto", "bf16", "fp16", "fp32").
        attention_mode: Attention implementation ("eager", "sdpa", "flash_attention_2").
        force_reload: Force reload even if cached.

    Returns:
        Tuple of (model, processor).

    Raises:
        RuntimeError: If model fails to load.
    """
    cache_key = f"asr_{model_name}_{dtype}_{attention_mode}"

    if cache_key not in LOADED_ASR_MODELS_CACHE or force_reload:
        if force_reload:
            cleanup_asr_models(keep_cache_key=cache_key)

        model, processor = VibeVoiceASRLoader.load_model(
            model_name=model_name,
            device=device,
            dtype_str=dtype,
            attention_mode=attention_mode,
        )
        logger.debug(f"Loaded ASR model {model_name} with dtype={dtype}, attn={attention_mode}")
    else:
        model, processor = LOADED_ASR_MODELS_CACHE[cache_key]

    return model, processor


def load_asr_model_patched(
    model_name: str,
    device: str = "auto",
    dtype: str = DTYPE_AUTO,
    attention_mode: str = "sdpa",
    force_reload: bool = False,
) -> Tuple[Any, Any, Any]:
    """Load or retrieve a cached VibeVoice ASR model under the patcher/VRAM system.

    This is the ASR counterpart of ``generation.load_vibevoice_model``. It wraps
    the model in a :class:`VibeVoiceASRPatcher` and moves it onto the GPU via
    ComfyUI's ``model_management.load_model_gpu`` so ASR participates in the
    same unified VRAM orchestration (partial offload, cross-model arbitration)
    as TTS. Resolves CRIT-001.

    Args:
        model_name: Name of the ASR model to load.
        device: Device to load on ("auto", "cuda", "cpu", "mps", "xpu", "npu").
        dtype: Data type for model ("auto", "bf16", "fp16", "fp32").
        attention_mode: Attention implementation ("eager", "sdpa",
            "flash_attention_2", "sage").
        force_reload: Force reload even if cached.

    Returns:
        Tuple of (patcher, model, processor).
    """
    # Resolve attention mode with internal fallback (no 4-bit for ASR).
    actual_attn = resolve_attention_mode(attention_mode, quantize_4bit=False)

    # Device placement (mirrors load_vibevoice_model).
    if device == DEVICE_CPU:
        load_device = torch.device(DEVICE_CPU)
        offload_device = torch.device(DEVICE_CPU)
    else:
        load_device = get_torch_device(device)
        offload_device = get_offload_device()

    target_dtype = resolve_dtype(dtype, load_device)

    cache_key = f"asr_{model_name}_attn_{actual_attn}"

    if cache_key not in VIBEVOICE_ASR_PATCHER_CACHE or force_reload:
        if force_reload:
            VIBEVOICE_ASR_PATCHER_CACHE.pop(cache_key, None)

        handler = VibeVoiceASRModelHandler(model_name)
        # Keep the patcher cache key and the ASR model cache key in sync so
        # VibeVoiceASRPatcher.unpatch_model clears the correct entry.
        handler.cache_key = cache_key

        patcher = VibeVoiceASRPatcher(
            handler,
            attention_mode=actual_attn,
            load_device=load_device,
            offload_device=offload_device,
            size=handler.size,
            dtype=target_dtype,
        )
        VIBEVOICE_ASR_PATCHER_CACHE[cache_key] = patcher
        logger.debug(f"Created ASR patcher for {model_name} with attn={actual_attn}")

    patcher = VIBEVOICE_ASR_PATCHER_CACHE[cache_key]
    model_management.load_model_gpu(patcher)
    model = patcher.model.model
    processor = patcher.model.processor

    # Register under the patcher key so the ASR cache reflects the live model
    # and the patcher's unpatch_model() cleanup removes it.
    LOADED_ASR_MODELS_CACHE[cache_key] = (model, processor)

    if model is None or processor is None:
        raise RuntimeError(
            f"Failed to load ASR model '{model_name}': model or processor is None after load."
        )

    return patcher, model, processor


def transcribe_audio(
    model: Any,
    processor: Any,
    audio_input: Dict,
    context_info: Optional[str] = None,
    max_new_tokens: int = 32768,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = True,
    num_beams: int = 1,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Transcribe audio using the VibeVoice ASR model.

    Args:
        model: VibeVoiceASRForConditionalGeneration instance.
        processor: VibeVoiceASRProcessor instance.
        audio_input: ComfyUI audio dict with 'waveform' and 'sample_rate'.
        context_info: Optional hotwords/context info to improve accuracy.
        max_new_tokens: Maximum tokens to generate.
        temperature: Temperature for sampling (0 = greedy).
        top_p: Top-p for nucleus sampling.
        do_sample: Whether to use sampling.
        num_beams: Number of beams for beam search (1 = no beam search).

    Returns:
        Tuple of (raw_text, segments) where segments is a list of dicts with
        keys: speaker, text, start, end.

    Raises:
        ValueError: If audio input is invalid.
        RuntimeError: If transcription fails.
    """
    # Extract audio tensor
    waveform, sample_rate = extract_audio_tensor(audio_input, name="audio_input")
    if waveform is None:
        raise ValueError("Audio input is required for ASR transcription")

    # Convert to numpy array for the processor
    import numpy as np
    if waveform.dim() > 1:
        # Take first channel and convert to 1D
        audio_array = waveform[0].cpu().numpy() if waveform.dim() == 2 else waveform[0, 0].cpu().numpy()
    else:
        audio_array = waveform.cpu().numpy()

    audio_array = audio_array.astype(np.float32)

    # Process audio through the processor
    inputs = processor(
        audio=audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        context_info=context_info,
    )

    # Move inputs to model device
    device = next(model.parameters()).device
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    # Prepare generation config
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": getattr(processor, "pad_id", None),
        "eos_token_id": processor.tokenizer.eos_token_id if hasattr(processor, "tokenizer") else None,
    }

    if num_beams > 1:
        generation_config["num_beams"] = num_beams
        generation_config["do_sample"] = False
    else:
        generation_config["do_sample"] = do_sample
        if do_sample:
            generation_config["temperature"] = temperature
            generation_config["top_p"] = top_p

    # Remove None values
    generation_config = {k: v for k, v in generation_config.items() if v is not None}

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                **generation_config,
            )

        # Decode output (exclude input tokens)
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_length:]

        # Remove padding/eos tokens from the end
        if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "eos_token_id"):
            eos_id = processor.tokenizer.eos_token_id
            if eos_id is not None:
                eos_positions = (generated_ids == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    generated_ids = generated_ids[:eos_positions[0] + 1]

        # Decode to text
        raw_text = processor.decode(generated_ids, skip_special_tokens=True)

        # Parse structured output
        try:
            segments = processor.post_process_transcription(raw_text)
        except Exception as e:
            logger.warning(f"Failed to parse structured transcription: {e}")
            segments = []

        logger.info(f"ASR transcription complete. {len(segments)} segments found.")
        return raw_text, segments

    except model_management.InterruptProcessingException:
        logger.info("ASR transcription interrupted by user")
        raise
    except Exception as e:
        logger.error(f"ASR transcription failed: {e}")
        raise RuntimeError(f"Transcription failed: {e}")


def force_offload_asr_model(model_name: str, patcher=None) -> None:
    """Force offload ASR model from VRAM.

    When ``patcher`` is provided (the patched path introduced for CRIT-001), the
    patcher's ``unpatch_model`` is used so the model leaves the unified VRAM
    system cleanly and the ASR cache entry is cleared. When omitted, the legacy
    direct-load cache is cleared (backwards-compatible path).

    Args:
        model_name: Name of the model (for logging).
        patcher: Optional :class:`VibeVoiceASRPatcher` instance to offload.
    """
    logger.info(f"Force offloading VibeVoice ASR model '{model_name}' from VRAM...")
    if patcher is not None:
        if patcher.is_loaded:
            patcher.unpatch_model(unpatch_weights=True)
        model_management.unload_all_models()
    else:
        # Legacy path (no patcher): clear the direct ASR cache.
        cleanup_asr_models()
    gc.collect()
    model_management.soft_empty_cache()
    logger.info("ASR model force offload completed")
