"""ASR (Speech-to-Text) generation utilities for VibeVoice.

Handles audio transcription using the VibeVoice ASR model, including:
- Audio preprocessing
- Model inference
- Output decoding and structured transcription parsing
"""

import torch
import gc
import math
import logging
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

import comfy.model_management as model_management
from transformers.generation import BaseStreamer

from .progress_utils import ProgressBarWithConsole

from .asr_loader import VibeVoiceASRLoader, VibeVoiceASRModelHandler, LOADED_ASR_MODELS_CACHE, cleanup_asr_models
from .patcher import VibeVoiceASRPatcher, load_to_device, select_patcher_class
from .model_registry import (
    FAMILY_ASR,
    evict_if_changed,
    identity_for_external,
    register_model_bundle,
)
from .utils import VIBEVOICE_ASR_PATCHER_CACHE
from .device_utils import get_torch_device, get_offload_device, DEVICE_CPU
from .dtype_utils import resolve_dtype, DTYPE_AUTO
from .audio_utils import extract_audio_tensor, resample_audio
from .attention_utils import resolve_attention_mode, resolve_asr_attention_mode
from .gguf_quant import log_gguf_forward_counters
from .memory_census import measured_load, report_census



class _ASRProgressStreamer(BaseStreamer):
    """Token streamer that drives the standard ComfyUI progress bar during ASR.

    HF ``GenerationMixin`` calls ``put(next_tokens)`` once per generated token
    (prompt tokens are NOT streamed). Each call advances the bar by the number
    of tokens in the batch and checks for user interruption, so cancelling an
    ASR transcription becomes responsive. ``end()`` sends the final 100% event.

    Only attached for greedy/sampling decoding (``num_beams <= 1``); beam
    search is not compatible with a plain token streamer and falls back to a
    single 0->100% bar.
    """

    def __init__(self, pbar: "ProgressBarWithConsole", total: int):
        self.pbar = pbar
        self.total = max(1, int(total))
        self.count = 0

    def put(self, value):
        # Responsive cancellation (raises InterruptProcessingException).
        model_management.throw_exception_if_processing_interrupted()
        if isinstance(value, torch.Tensor):
            n = int(value.numel())
        elif isinstance(value, (list, tuple)):
            n = len(value)
        else:
            n = 1
        self.count = min(self.count + n, self.total)
        self.pbar.update_absolute(self.count, total=self.total)

    def end(self):
        self.pbar.update_absolute(self.total)


class ExternalVibeVoiceASRModelHandler(torch.nn.Module):
    """Handler for an externally-loaded (pre-instantiated) VibeVoice ASR model.

    ASR counterpart of
    :class:`~modules.generation.ExternalVibeVoiceModelHandler`. Unlike
    :class:`~modules.asr_loader.VibeVoiceASRModelHandler`, whose ``load_model``
    loads weights from disk on demand, this handler already holds the loaded
    model and processor. The patcher's ``patch_model`` sees
    ``self.model.model is not None`` and skips the lazy-load branch, proceeding
    directly to the single host-to-device transfer.
    """

    def __init__(
        self,
        model,
        processor,
        model_pack_name: str,
        model_bundle: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.model_name = model_pack_name
        self.model_pack_name = model_pack_name
        self.source_path = (model_bundle or {}).get("source_path", "")
        # Default cache key; load_asr_from_external overrides this with the
        # attention-aware key so the patcher and LOADED_ASR_MODELS_CACHE agree.
        self.cache_key = f"asr_external_{model_pack_name}"
        self.size = self._estimate_size(model, model_bundle)

    @staticmethod
    def _estimate_size(model, model_bundle: Optional[Dict[str, Any]] = None) -> int:
        """Estimate the model's VRAM footprint in bytes.

        Prefers an explicit ``size_gb`` hint from the bundle, then the sum of
        parameter bytes, then a 15 GB fallback (matching the ASR default in
        :class:`~modules.asr_loader.VibeVoiceASRModelHandler`).
        """
        try:
            size_gb = (model_bundle or {}).get("size_gb")
            if size_gb:
                return int(float(size_gb) * (1024**3))
        except (TypeError, ValueError):
            pass
        try:
            total = 0
            for p in model.parameters():
                total += p.numel() * p.element_size()
            if total > 0:
                return total
        except Exception:
            pass
        # Fallback: assume ~15 GB (ASR default) if the size cannot be determined.
        return int(15.0 * (1024**3))

    def load_model(self, device, dtype_str: str = "auto", attention_mode: str = "sdpa"):
        """No-op: the model is already loaded.

        The patcher only calls this when ``self.model is None``; for an external
        handler the model is pre-set, so this branch is never reached. Kept for
        interface compatibility with :class:`VibeVoiceASRModelHandler`.
        """
        logging.debug(
            f"[VibeVoice TTS] ExternalVibeVoiceASRModelHandler.load_model called but model is "
            f"already loaded for '{self.model_pack_name}'"
        )


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
        logging.debug(f"[VibeVoice TTS] Loaded ASR model {model_name} with dtype={dtype}, attn={attention_mode}")
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
    # Resolve attention mode with internal fallback (no 4-bit for ASR), then
    # drop the ASR exclusions. This must happen BEFORE the cache key and the
    # patcher are built, or the exclusion would never reach the weights.
    actual_attn = resolve_asr_attention_mode(
        resolve_attention_mode(attention_mode, quantize_4bit=False)
    )

    # Device placement (mirrors load_vibevoice_model).
    if device == DEVICE_CPU:
        load_device = torch.device(DEVICE_CPU)
        offload_device = torch.device(DEVICE_CPU)
    else:
        load_device = get_torch_device(device)
        offload_device = get_offload_device()

    target_dtype = resolve_dtype(dtype, load_device)

    cache_key = f"asr_{model_name}_attn_{actual_attn}"

    # Unload-before-load gate (plan 2026-08-20, C4/RC-1): a different active
    # ASR model is fully released before the new one is built/loaded.
    evict_if_changed(FAMILY_ASR, cache_key, (VIBEVOICE_ASR_PATCHER_CACHE,))

    if cache_key not in VIBEVOICE_ASR_PATCHER_CACHE or force_reload:
        if force_reload:
            VIBEVOICE_ASR_PATCHER_CACHE.pop(cache_key, None)

        handler = VibeVoiceASRModelHandler(model_name)
        handler.cache_key = cache_key

        patcher_cls = select_patcher_class(None, load_device, legacy_cls=VibeVoiceASRPatcher)
        patcher = patcher_cls(
            handler,
            attention_mode=actual_attn,
            load_device=load_device,
            offload_device=offload_device,
            size=handler.size,
            dtype=target_dtype,
        )
        VIBEVOICE_ASR_PATCHER_CACHE[cache_key] = patcher
        logging.debug(f"[VibeVoice TTS] Created ASR patcher for {model_name} with attn={actual_attn}")

    patcher = VIBEVOICE_ASR_PATCHER_CACHE[cache_key]

    with measured_load("load-to-device"):
        load_to_device(patcher)
    report_census(patcher.model.model, patcher, phase=f"post-h2d:{model_name}")
    model = patcher.model.model
    processor = patcher.model.processor

    LOADED_ASR_MODELS_CACHE[cache_key] = (model, processor)

    if model is None or processor is None:
        raise RuntimeError(
            f"Failed to load ASR model '{model_name}': model or processor is None after load."
        )

    return patcher, model, processor


def load_asr_from_external(
    model_bundle: Dict[str, Any],
    device: str = "auto",
    dtype: str = "auto",
    attention_mode: str = "auto",
) -> Tuple[Any, Any, Any]:
    """Load an externally-provided VibeVoice ASR model bundle onto the GPU.

    Mirrors :func:`load_asr_model_patched` but skips the download/instantiation
    step: the model and processor arrive pre-loaded (on CPU) inside
    ``model_bundle`` (produced by ``modules.external_loader``). The bundle is
    wrapped in an ``ExternalVibeVoiceASRModelHandler`` so ComfyUI's model
    manager can schedule it through ``VibeVoiceASRPatcher`` exactly like a
    standard ASR model.

    Args:
        model_bundle: Dict with keys ``model``, ``processor``, ``model_name``
            (required) and optional ``size_gb`` / ``is_streaming``.
        device: Target device string ("auto", "cuda", "cpu", ...).
        dtype: Dtype string ("auto", "bf16", "fp16", "fp32").
        attention_mode: Attention implementation ("auto", "sdpa", ...).

    Returns:
        Tuple of (patcher, model, processor).

    Raises:
        ValueError: If required bundle keys are missing or None.
    """
    for required_key in ("model", "processor", "model_name"):
        if model_bundle.get(required_key) is None:
            raise ValueError(
                f"External ASR model bundle is missing required key '{required_key}'. "
                "Ensure the bundle was produced by the VibeVoice external loader node."
            )

    model = model_bundle["model"]
    processor = model_bundle["processor"]
    model_name = model_bundle["model_name"]

    bundle_attention = model_bundle.get("attention_mode")
    if isinstance(bundle_attention, str) and bundle_attention:
        actual_attn = resolve_asr_attention_mode(bundle_attention)
    else:
        actual_attn = resolve_asr_attention_mode(
            resolve_attention_mode(attention_mode, quantize_4bit=False)
        )

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
        model_bundle.get("source_path", ""),
        model_name,
        actual_attn,
        use_llm_4bit=bundle_use_llm_4bit,
        dtype_str=bundle_dtype_str,
        prefix="asr_external",
    )

    register_model_bundle(cache_key, model_bundle)
    evict_if_changed(FAMILY_ASR, cache_key, (VIBEVOICE_ASR_PATCHER_CACHE,))

    if cache_key not in VIBEVOICE_ASR_PATCHER_CACHE:
        handler = ExternalVibeVoiceASRModelHandler(model, processor, model_name, model_bundle)
        handler.cache_key = cache_key

        patcher_cls = select_patcher_class(
            model_bundle.get("weight_family"), load_device, legacy_cls=VibeVoiceASRPatcher
        )
        patcher = patcher_cls(
            handler,
            attention_mode=actual_attn,
            load_device=load_device,
            offload_device=offload_device,
            size=handler.size,
            dtype=target_dtype,
        )
        VIBEVOICE_ASR_PATCHER_CACHE[cache_key] = patcher
        logging.debug(f"[VibeVoice TTS] Created ASR patcher for external model {model_name} with attn={actual_attn}")

    patcher = VIBEVOICE_ASR_PATCHER_CACHE[cache_key]

    with measured_load("load-to-device"):
        load_to_device(patcher)
    report_census(patcher.model.model, patcher, phase=f"post-h2d:{model_name}")
    model = patcher.model.model
    processor = patcher.model.processor

    LOADED_ASR_MODELS_CACHE[cache_key] = (model, processor)

    return patcher, model, processor


def _parse_streaming_chunk(chunk_idx: int, chunk_text: str, frame_config: dict,
                           num_chunks: int, chunk_duration: float) -> Optional[Dict[str, Any]]:
    """Parse one streaming chunk's 'speaker, content' text into a segment."""
    import re
    text = chunk_text.strip()
    if not text:
        return None
    m = re.match(r"(?:speaker|Speaker)\s*(\d+)\s*[:：]\s*(.*)", text, re.DOTALL)
    if m:
        speaker = int(m.group(1))
        content = m.group(2).strip()
    else:
        speaker = 0
        content = text
    if not content:
        return None
    start = chunk_idx * chunk_duration
    end = min(start + chunk_duration + frame_config["text_audio_delay"],
              num_chunks * chunk_duration) if num_chunks > 0 else start + chunk_duration
    return {"speaker": speaker, "text": content, "start": round(start, 2), "end": round(end, 2)}


def _transcribe_streaming(
    model: Any,
    processor: Any,
    audio_array,
    sample_rate: int,
    frame_config: dict,
    context_info: Optional[str],
    max_new_tokens: int,
    temperature: float,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Transcribe via the chunked streaming protocol (ASR-Streaming checkpoints)."""
    target_sr = frame_config["sample_rate"]
    if sample_rate != target_sr:
        audio_array = resample_audio(audio_array, orig_sr=sample_rate, target_sr=target_sr)

    tokenizer = processor.tokenizer
    if getattr(tokenizer, "text_chunk_end_id", None) is None:
        raise ValueError(
            "This checkpoint is marked as streaming (chunk_frames/lookahead_frames "
            "in preprocessor_config.json) but its tokenizer has no "
            "<|text_chunk_end|> token; cannot transcribe it with the streaming "
            "protocol."
        )

    audio_tensor = torch.from_numpy(np.ascontiguousarray(audio_array))
    duration = audio_tensor.numel() / target_sr if target_sr else 0.0
    chunk_duration = frame_config["chunk_duration"]
    total_chunks_est = max(1, math.ceil(duration / chunk_duration)) if chunk_duration else 1

    pbar = ProgressBarWithConsole(total_chunks_est)
    segments: List[Dict[str, Any]] = []
    texts: List[str] = []

    gen = model.streaming_generate(
        audio_tensor=audio_tensor,
        tokenizer=tokenizer,
        chunk_duration=frame_config["chunk_duration"],
        text_audio_delay=frame_config["text_audio_delay"],
        sample_rate=target_sr,
        max_new_tokens_per_chunk=max(64, max_new_tokens // max(1, total_chunks_est)),
        temperature=temperature if temperature > 0 else 0.0,
        context_info=context_info,
    )
    try:
        total_chunks_seen = 0
        for chunk_idx, num_chunks, chunk_text in gen:
            model_management.throw_exception_if_processing_interrupted()
            total_chunks_seen = num_chunks
            if num_chunks != pbar.total and num_chunks > 0:
                pbar.update_absolute(0, total=num_chunks)
            seg = _parse_streaming_chunk(
                chunk_idx, chunk_text, frame_config, num_chunks, chunk_duration
            )
            if seg is not None:
                segments.append(seg)
            if chunk_text.strip():
                texts.append(chunk_text.strip())
            pbar.update_absolute(chunk_idx + 1)
    finally:
        pbar.update_absolute(pbar.total)
        pbar.close()

    log_gguf_forward_counters("asr_streaming")

    raw_text = "\n".join(texts)
    logging.info(f"[VibeVoice TTS] ASR streaming transcription complete. {len(segments)} segments, "
                f"{total_chunks_seen} chunks.")
    return raw_text, segments


def _asr_processor_kind(processor: Any) -> str:
    """Classify a processor for transcription-branch selection."""
    mod = getattr(type(processor), "__module__", "")
    if mod.startswith("transformers."):
        return "native"
    if "vibevoice" in mod:
        return "vendored"
    return "unknown"


def _transcribe_native(
    model: Any,
    processor: Any,
    audio_array: "np.ndarray",
    sample_rate: int,
    context_info: Optional[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    num_beams: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Transcribe via the HF-native protocol (microsoft/VibeVoice-ASR-HF)."""
    feature_extractor = getattr(processor, "feature_extractor", None)
    target_sr = getattr(feature_extractor, "sampling_rate", 24000)
    if sample_rate != target_sr:
        audio_array = resample_audio(audio_array, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr

    inputs = processor.apply_transcription_request(
        audio=np.ascontiguousarray(audio_array),
        prompt=context_info,
    )

    first_param = next(model.parameters())
    device, model_dtype = first_param.device, first_param.dtype
    inputs = inputs.to(device, model_dtype)

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": getattr(processor.tokenizer, "pad_token_id", None),
        "eos_token_id": processor.tokenizer.eos_token_id,
    }

    if num_beams > 1:
        generation_config["num_beams"] = num_beams
        generation_config["do_sample"] = False
    else:
        generation_config["do_sample"] = do_sample
        if do_sample:
            generation_config["temperature"] = temperature
            generation_config["top_p"] = top_p

    generation_config = {k: v for k, v in generation_config.items() if v is not None}

    pbar = ProgressBarWithConsole(max_new_tokens)
    use_streamer = generation_config.get("num_beams", 1) <= 1
    streamer = _ASRProgressStreamer(pbar, total=max_new_tokens) if use_streamer else None

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                **generation_config,
                **({"streamer": streamer} if streamer is not None else {}),
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        raw_text = processor.decode(generated_ids, skip_special_tokens=True)[0]

        segments: List[Dict[str, Any]] = []
        parsed = processor.decode(generated_ids, return_format="parsed")[0]
        if isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                try:
                    segments.append({
                        "speaker": int(item.get("Speaker", 0)),
                        "text": str(item.get("Content", "")).strip(),
                        "start": round(float(item.get("Start", 0.0)), 2),
                        "end": round(float(item.get("End", 0.0)), 2),
                    })
                except (TypeError, ValueError):
                    continue

        logging.info(f"[VibeVoice TTS] ASR transcription complete. {len(segments)} segments found.")
        return raw_text, segments

    except model_management.InterruptProcessingException:
        logging.info("[VibeVoice TTS] ASR transcription interrupted by user")
        raise
    except Exception as e:
        logging.error(f"[VibeVoice TTS] ASR transcription failed: {e}")
        raise RuntimeError(f"Transcription failed: {e}")
    finally:
        pbar.update_absolute(pbar.total)
        pbar.close()

    log_gguf_forward_counters("asr_transcribe_native")


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
    """Transcribe audio using the VibeVoice ASR model."""
    waveform, sample_rate = extract_audio_tensor(audio_input, name="audio_input")
    if waveform is None:
        raise ValueError("Audio input is required for ASR transcription")

    if waveform.dim() > 1:
        audio_array = waveform[0].cpu().numpy() if waveform.dim() == 2 else waveform[0, 0].cpu().numpy()
    else:
        audio_array = waveform.cpu().numpy()

    audio_array = audio_array.astype(np.float32)

    if _asr_processor_kind(processor) == "native":
        return _transcribe_native(
            model=model,
            processor=processor,
            audio_array=audio_array,
            sample_rate=sample_rate,
            context_info=context_info,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_beams=num_beams,
        )

    if _asr_processor_kind(processor) == "vendored":
        frame_config = getattr(processor, "streaming_frame_config", None)
        if frame_config is not None:
            return _transcribe_streaming(
                model=model,
                processor=processor,
                audio_array=audio_array,
                sample_rate=sample_rate,
                frame_config=frame_config,
                context_info=context_info,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

    inputs = processor(
        audio=audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        context_info=context_info,
    )

    device = next(model.parameters()).device
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

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

    generation_config = {k: v for k, v in generation_config.items() if v is not None}

    pbar = ProgressBarWithConsole(max_new_tokens)
    use_streamer = generation_config.get("num_beams", 1) <= 1
    streamer = _ASRProgressStreamer(pbar, total=max_new_tokens) if use_streamer else None

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                **generation_config,
                **({"streamer": streamer} if streamer is not None else {}),
            )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_length:]

        if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "eos_token_id"):
            eos_id = processor.tokenizer.eos_token_id
            if eos_id is not None:
                eos_positions = (generated_ids == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    generated_ids = generated_ids[:eos_positions[0] + 1]

        raw_text = processor.decode(generated_ids, skip_special_tokens=True)

        try:
            segments = processor.post_process_transcription(raw_text)
        except Exception as e:
            logging.warning(f"[VibeVoice TTS] Failed to parse structured transcription: {e}")
            segments = []

        logging.info(f"[VibeVoice TTS] ASR transcription complete. {len(segments)} segments found.")
        return raw_text, segments

    except model_management.InterruptProcessingException:
        logging.info("[VibeVoice TTS] ASR transcription interrupted by user")
        raise
    except Exception as e:
        logging.error(f"[VibeVoice TTS] ASR transcription failed: {e}")
        raise RuntimeError(f"Transcription failed: {e}")
    finally:
        pbar.update_absolute(pbar.total)
        pbar.close()

    log_gguf_forward_counters("asr_transcribe_audio")


def force_offload_asr_model(model_name: str, patcher=None) -> None:
    """Force offload ASR model from VRAM."""
    logging.info(f"[VibeVoice TTS] Force offloading VibeVoice ASR model '{model_name}' from VRAM...")
    if patcher is not None:
        if patcher.is_loaded:
            patcher.unpatch_model(unpatch_weights=True, destroy=True)
        model_management.unload_all_models()
    else:
        cleanup_asr_models()
    gc.collect()
    model_management.soft_empty_cache()
    logging.info("[VibeVoice TTS] ASR model force offload completed")