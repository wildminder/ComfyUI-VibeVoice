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

from .progress_utils import ProgressBarWithConsole

from .loader import VibeVoiceModelHandler, VibeVoiceLoader, cleanup_old_models, LOADED_MODELS_CACHE
from .patcher import VibeVoicePatcher, load_to_device, select_patcher_class
from .model_registry import (
    FAMILY_TTS,
    evict_if_changed,
    identity_for_external,
    register_model_bundle,
)
from .model_info import is_model_type
from .utils import VIBEVOICE_PATCHER_CACHE
from .audio_utils import parse_script_1_based, preprocess_comfy_audio, set_seed, check_for_interrupt
from .device_utils import get_torch_device, get_offload_device, DEVICE_CPU
from .dtype_utils import resolve_dtype, DTYPE_AUTO
from .attention_utils import resolve_attention_mode, resolve_realtime_attention_mode
from .gguf_quant import log_gguf_forward_counters
from .memory_census import measured_load, report_census

logger = logging.getLogger(__name__)


def resolve_generation_family(
    model_name: str,
    external_model: dict | None = None,
) -> str:
    """Return exactly ``tts`` or ``streaming_tts`` for generation routing."""
    if external_model is not None:
        if external_model.get("is_asr"):
            raise ValueError(
                "ASR models cannot generate speech. Use the 'VibeVoice ASR' node."
            )
        if external_model.get("is_streaming"):
            return "streaming_tts"
        return "tts"

    if not model_name:
        raise ValueError(
            "No VibeVoice model was selected. Select a TTS model or use the "
            "'VibeVoice ASR' node for an ASR model."
        )
    if is_model_type(model_name, "streaming_tts"):
        return "streaming_tts"
    if is_model_type(model_name, "tts"):
        return "tts"

    raise ValueError(
        f"Model '{model_name}' has unsupported type 'asr'. Use the "
        "'VibeVoice ASR' node for ASR models."
    )


class ExternalVibeVoiceModelHandler(torch.nn.Module):
    """Handler for an externally-loaded (pre-instantiated) VibeVoice model."""

    def __init__(
        self,
        model,
        processor,
        model_pack_name: str,
        attention_mode: str = "sdpa",
        source_path: str = "",
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.model_pack_name = model_pack_name
        self.attention_mode = attention_mode
        self.source_path = source_path
        self.cache_key = f"external_{model_pack_name}_attn_{attention_mode}"
        self.device = None
        self.size = self._estimate_size(model)

    @staticmethod
    def _estimate_size(model) -> int:
        """Estimate the model's VRAM footprint in bytes from its parameters."""
        try:
            total = 0
            for p in model.parameters():
                total += p.numel() * p.element_size()
            if total > 0:
                return total
        except Exception:
            pass
        return int(4.0 * (1024**3))

    def load_model(self, device, attention_mode: str = "sdpa"):
        """No-op: the model is already loaded."""
        logger.debug(
            f"ExternalVibeVoiceModelHandler.load_model called but model is "
            f"already loaded for '{self.model_pack_name}'"
        )


def load_vibevoice_from_external(
    model_bundle: dict,
    device: str = "auto",
    dtype: str = DTYPE_AUTO,
    attention_mode: str = "sdpa",
) -> Tuple[VibeVoicePatcher, Any, Any]:
    """Wrap an externally-loaded VibeVoice model bundle in a patcher and load to VRAM."""
    if model_bundle.get("model") is None and model_bundle.get("source_path"):
        from .external_loader import load_external_vibevoice_model

        model_bundle = load_external_vibevoice_model(
            model_bundle["source_path"],
            model_bundle.get("model_name") or "",
            attention_mode=model_bundle.get("attention_mode") or "eager",
            use_llm_4bit=bool(model_bundle.get("use_llm_4bit", False)),
            dtype_str=model_bundle.get("dtype_str") or "auto",
        )

    for required_key in ("model", "processor", "model_name"):
        if model_bundle.get(required_key) is None:
            raise ValueError(
                f"External VibeVoice model bundle is missing required key "
                f"'{required_key}'. Got keys: {list(model_bundle.keys())}"
            )

    model = model_bundle["model"]
    processor = model_bundle["processor"]
    model_name = model_bundle["model_name"]
    source_path = model_bundle.get("source_path", "")

    bundle_attention = model_bundle.get("attention_mode")
    if isinstance(bundle_attention, str) and bundle_attention:
        actual_attention_mode = bundle_attention
    else:
        actual_attention_mode = resolve_attention_mode(attention_mode, False)

    if device == DEVICE_CPU:
        load_device = torch.device(DEVICE_CPU)
        offload_device = torch.device(DEVICE_CPU)
    else:
        load_device = get_torch_device(device)
        offload_device = get_offload_device()

    target_dtype = resolve_dtype(dtype, load_device)

    bundle_use_llm_4bit = bool(model_bundle.get("use_llm_4bit", False))
    bundle_dtype_str = model_bundle.get("dtype_str") or dtype
    cache_key = identity_for_external(
        source_path,
        model_name,
        actual_attention_mode,
        use_llm_4bit=bundle_use_llm_4bit,
        dtype_str=bundle_dtype_str,
    )

    register_model_bundle(cache_key, model_bundle)
    evict_if_changed(FAMILY_TTS, cache_key, (VIBEVOICE_PATCHER_CACHE,))

    if cache_key not in VIBEVOICE_PATCHER_CACHE:
        model_handler = ExternalVibeVoiceModelHandler(
            model=model,
            processor=processor,
            model_pack_name=model_name,
            attention_mode=actual_attention_mode,
            source_path=source_path,
        )
        model_handler.cache_key = cache_key

        patcher_cls = select_patcher_class(
            model_bundle.get("weight_family"), load_device, legacy_cls=VibeVoicePatcher
        )
        patcher = patcher_cls(
            model_handler,
            attention_mode=actual_attention_mode,
            load_device=load_device,
            offload_device=offload_device,
            size=model_handler.size,
            dtype=target_dtype,
        )
        VIBEVOICE_PATCHER_CACHE[cache_key] = patcher
        logger.debug(
            f"Created new external patcher for {model_name} with "
            f"attn={actual_attention_mode}"
        )

    patcher = VIBEVOICE_PATCHER_CACHE[cache_key]

    with measured_load("load-to-device"):
        load_to_device(patcher)

    report_census(patcher.model.model, patcher, phase=f"post-h2d:{model_name}")
    loaded_model = patcher.model.model
    loaded_processor = patcher.model.processor

    if loaded_model is None or loaded_processor is None:
        raise RuntimeError(
            f"External VibeVoice model and processor could not be loaded for "
            f"'{model_name}'. Check logs for errors."
        )

    return patcher, loaded_model, loaded_processor


def load_vibevoice_model(
    model_name: str,
    device: str = "auto",
    dtype: str = DTYPE_AUTO,
    attention_mode: str = "sdpa",
    quantize_4bit: bool = False,
    force_reload: bool = False,
) -> Tuple[VibeVoicePatcher, Any, Any]:
    """Load or retrieve cached VibeVoice model."""
    actual_attention_mode = resolve_attention_mode(attention_mode, quantize_4bit)
    if is_model_type(model_name, "streaming_tts"):
        actual_attention_mode = resolve_realtime_attention_mode(actual_attention_mode)

    if device == DEVICE_CPU:
        load_device = torch.device(DEVICE_CPU)
        offload_device = torch.device(DEVICE_CPU)
    else:
        load_device = get_torch_device(device)
        offload_device = get_offload_device()

    target_dtype = resolve_dtype(dtype, load_device)
    cache_key = f"{model_name}_attn_{actual_attention_mode}_q4_{int(quantize_4bit)}"

    evict_if_changed(FAMILY_TTS, cache_key, (VIBEVOICE_PATCHER_CACHE,))

    if cache_key not in VIBEVOICE_PATCHER_CACHE or force_reload:
        if force_reload:
            cleanup_old_models(keep_cache_key=cache_key)

        model_handler = VibeVoiceModelHandler(
            model_name,
            attention_mode=actual_attention_mode,
            use_llm_4bit=quantize_4bit,
            dtype_str=dtype,
        )

        patcher_cls = select_patcher_class(None, load_device, legacy_cls=VibeVoicePatcher)
        patcher = patcher_cls(
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

    with measured_load("load-to-device"):
        load_to_device(patcher)

    report_census(patcher.model.model, patcher, phase=f"post-h2d:{model_name}")
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
    """Generate audio using the VibeVoice model."""
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
            "Use the canonical VibeVoice TTS realtime family path for this model."
        )

    parsed_lines_0_based, speaker_ids_1_based = parse_script_1_based(text)
    if not parsed_lines_0_based:
        raise ValueError("Script is empty or invalid. Please provide text to generate.")

    voice_samples_np = []
    for vs in voice_samples:
        processed = preprocess_comfy_audio(vs)
        if processed is not None:
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

    set_seed(seed)

    normalized_script = "\n".join(
        f"Speaker {speaker_id + 1}:{speaker_text}"
        for speaker_id, speaker_text in parsed_lines_0_based
    )

    inputs = processor(
        text=[normalized_script],
        voice_samples=[voice_samples_np],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if torch.any(torch.isnan(value)) or torch.any(torch.isinf(value)):
                logger.error(f"Input tensor '{key}' contains NaN or Inf values")
                raise ValueError(f"Invalid values in input tensor: {key}")

    compute_device = model_management.get_torch_device()
    inputs = {
        k: v.to(compute_device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    model.set_ddpm_inference_steps(num_steps=inference_steps)

    gen_inputs = {
        "input_ids": inputs.get("input_ids"),
        "attention_mask": inputs.get("attention_mask"),
        "speech_tensors": inputs.get("speech_tensors"),
        "speech_masks": inputs.get("speech_masks"),
        "acoustic_input_mask": inputs.get("speech_input_mask"),
        "cfg_scale": cfg_scale,
        "inference_steps": inference_steps,
        "return_speech": True,
        "tokenizer": processor.tokenizer,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
    }
    gen_inputs = {k: v for k, v in gen_inputs.items() if v is not None}

    with torch.no_grad():
        pbar = ProgressBarWithConsole(inference_steps)

        def _progress(current: int, total: int) -> None:
            model_management.throw_exception_if_processing_interrupted()
            pbar.update_absolute(current, total=total)

        try:
            from .comfy_stream import pull_stats_line, reset_pull_stats

            reset_pull_stats()
            with measured_load("tts-generate") as _rss:
                _rss.mark("gen-enter")
                outputs = model.generate(**gen_inputs, progress_callback=_progress)
                _rss.mark("gen-return")
            logger.info(pull_stats_line())

        except model_management.InterruptProcessingException:
            logger.info("VibeVoice generation interrupted by user")
            raise
        finally:
            pbar.update_absolute(pbar.total)
            pbar.close()

    log_gguf_forward_counters("tts_generate")

    speech_outputs = outputs.speech_outputs
    if not speech_outputs or speech_outputs[0] is None:
        raise RuntimeError(
            "VibeVoice generation produced no audio. The model emitted no speech "
            "tokens during the autoregressive loop. This usually means the loaded "
            "weights are corrupt or over-quantized (e.g. a naive int8 cast without "
            "dequantization scales, or an extremely low-bit GGUF quant). Try a "
            "higher-quality checkpoint (BF16 / FP16 / Q8_0 / Q4_K_M)."
        )

    output_waveform = speech_outputs[0]
    if output_waveform.ndim == 1:
        output_waveform = output_waveform.unsqueeze(0)
    if output_waveform.ndim == 2:
        output_waveform = output_waveform.unsqueeze(0)

    sample_rate = 24000
    return output_waveform.detach().to(device="cpu", dtype=torch.float32), sample_rate


def force_offload_model(patcher: VibeVoicePatcher, model_name: str, warm: bool = False) -> None:
    """Force offload a VibeVoice model from VRAM."""
    logger.info(f"Force offloading VibeVoice model '{model_name}' from VRAM...")
    if patcher.is_loaded:
        if warm:
            patcher.unpatch_model(unpatch_weights=True, warm=True)
        else:
            patcher.unpatch_model(unpatch_weights=True, destroy=True)
    model_management.unload_all_models()
    gc.collect()
    model_management.soft_empty_cache()
    logger.info("Model force offload completed")