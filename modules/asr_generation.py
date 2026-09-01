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
from transformers.generation import BaseStreamer

from .progress_utils import ProgressBarWithConsole

from .asr_loader import VibeVoiceASRLoader, VibeVoiceASRModelHandler, LOADED_ASR_MODELS_CACHE, cleanup_asr_models
from .patcher import VibeVoiceASRPatcher
from .model_registry import FAMILY_ASR, evict_if_changed, identity_for_external
from .utils import VIBEVOICE_ASR_PATCHER_CACHE
from .device_utils import get_torch_device, get_offload_device, DEVICE_CPU
from .dtype_utils import resolve_dtype, DTYPE_AUTO
from .audio_utils import extract_audio_tensor
from .attention_utils import resolve_attention_mode

logger = logging.getLogger(__name__)


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
        logger.debug(
            f"ExternalVibeVoiceASRModelHandler.load_model called but model is "
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

    # Unload-before-load gate (plan 2026-08-20, C4/RC-1): a different active
    # ASR model is fully released before the new one is built/loaded.
    evict_if_changed(FAMILY_ASR, cache_key, (VIBEVOICE_ASR_PATCHER_CACHE,))

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

    # Resolve attention mode with internal fallback (no 4-bit for ASR).
    # Plan 2026-08-20 (P1): prefer the bundle-recorded (loader-resolved) mode
    # so the cache key describes the weights actually built.
    bundle_attention = model_bundle.get("attention_mode")
    if isinstance(bundle_attention, str) and bundle_attention:
        actual_attn = bundle_attention
    else:
        actual_attn = resolve_attention_mode(attention_mode, quantize_4bit=False)

    # Device placement (mirrors load_asr_model_patched).
    if device == DEVICE_CPU:
        load_device = torch.device(DEVICE_CPU)
        offload_device = torch.device(DEVICE_CPU)
    else:
        load_device = get_torch_device(device)
        offload_device = get_offload_device()

    target_dtype = resolve_dtype(dtype, load_device)

    # Plan 2026-08-20 (RC-2/B3): file-identity-aware ASR cache key, derived
    # from the bundle's recorded build fields (fallbacks keep hand-built
    # bundles working).
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

    # Unload-before-load gate (plan 2026-08-20, C4/RC-1).
    evict_if_changed(FAMILY_ASR, cache_key, (VIBEVOICE_ASR_PATCHER_CACHE,))

    if cache_key not in VIBEVOICE_ASR_PATCHER_CACHE:
        handler = ExternalVibeVoiceASRModelHandler(model, processor, model_name, model_bundle)
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
        logger.debug(f"Created ASR patcher for external model {model_name} with attn={actual_attn}")

    patcher = VIBEVOICE_ASR_PATCHER_CACHE[cache_key]
    model_management.load_model_gpu(patcher)
    model = patcher.model.model
    processor = patcher.model.processor

    # Register under the patcher key so the ASR cache reflects the live model
    # and the patcher's unpatch_model() cleanup removes it.
    LOADED_ASR_MODELS_CACHE[cache_key] = (model, processor)

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

    # Standard ComfyUI progress bar. HF generate() reports per-token progress
    # through a streamer (greedy/sampling only; beam search falls back to a
    # single 0->100% bar because plain token streamers are beam-incompatible).
    # Drives both the frontend bar and the standard tqdm console bar.
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
    finally:
        # Guarantee the final 100% event even when generation stopped early
        # (EOS before max_new_tokens) or raised.
        pbar.update_absolute(pbar.total)
        pbar.close()


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
            # Plan 2026-08-18 D5: user-requested force offload keeps its
            # destructive semantics via the explicit flag (the default
            # unpatch_model is now non-destructive, RC-6).
            patcher.unpatch_model(unpatch_weights=True, destroy=True)
        model_management.unload_all_models()
    else:
        # Legacy path (no patcher): clear the direct ASR cache.
        cleanup_asr_models()
    gc.collect()
    model_management.soft_empty_cache()
    logger.info("ASR model force offload completed")
