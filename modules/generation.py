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
from .patcher import VibeVoicePatcher
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

logger = logging.getLogger(__name__)


def resolve_generation_family(
    model_name: str,
    external_model: dict | None = None,
) -> str:
    """Return exactly ``tts`` or ``streaming_tts`` for generation routing.

    This policy function performs no model loading, filesystem access, tensor
    creation, or class inspection. External bundle flags are authoritative;
    named models use the shared model-family inference rules.
    """
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
    """Handler for an externally-loaded (pre-instantiated) VibeVoice model.

    Unlike :class:`~modules.loader.VibeVoiceModelHandler`, whose ``load_model``
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
        # Fallback: assume ~4 GB if the size cannot be determined.
        return int(4.0 * (1024**3))

    def load_model(self, device, attention_mode: str = "sdpa"):
        """No-op: the model is already loaded.

        The patcher only calls this when ``self.model is None``; for an external
        handler the model is pre-set, so this branch is never reached. Kept for
        interface compatibility with :class:`VibeVoiceModelHandler`.
        """
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
    """Wrap an externally-loaded VibeVoice model bundle in a patcher.

    The bundle (produced by
    :func:`~modules.external_loader.load_external_vibevoice_model`) already
    contains an instantiated model + processor on CPU. This function wraps them
    in an :class:`ExternalVibeVoiceModelHandler` + :class:`VibeVoicePatcher` and
    loads the model to GPU via ComfyUI's memory management.

    Args:
        model_bundle: The ``VIBEVOICE_MODEL`` dict. Required keys:
            ``model``, ``processor``, ``model_name``. Optional:
            ``is_streaming``, ``source_path``, ``config``, ``state_dict``.
        device: Device to load on ("auto", "cuda", "cpu", "mps", etc.).
        dtype: Data type for model ("auto", "bf16", "fp16", "fp32").
        attention_mode: Attention implementation.

    Returns:
        Tuple of (patcher, model, processor).

    Raises:
        ValueError: If the bundle is missing required keys.
        RuntimeError: If the model fails to load to GPU.
    """
    # A bundle whose heavy fields were released still carries everything
    # needed to rebuild it, so recover instead of failing. The loader node's
    # output is cached by ComfyUI and can outlive the registry entry that
    # backed it.
    if model_bundle.get("model") is None and model_bundle.get("source_path"):
        from .external_loader import load_external_vibevoice_model

        model_bundle = load_external_vibevoice_model(
            model_bundle["source_path"],
            model_bundle.get("model_name") or "",
            attention_mode=model_bundle.get("attention_mode") or "eager",
            use_llm_4bit=bool(model_bundle.get("use_llm_4bit", False)),
            dtype_str=model_bundle.get("dtype_str") or "auto",
        )

    # Validate required bundle keys (Phase 5.3 guard).
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

    # Resolve attention mode with fallback logic (no quantization on the
    # external path — quantization is applied at load time by the loader node).
    #
    # Plan 2026-08-20 (P1): when the bundle records the attention mode the
    # weights were actually BUILT with (loader-resolved), prefer it — the TTS
    # node's own widget must not fork a second patcher for the same weights.
    bundle_attention = model_bundle.get("attention_mode")
    if isinstance(bundle_attention, str) and bundle_attention:
        actual_attention_mode = bundle_attention
    else:
        actual_attention_mode = resolve_attention_mode(attention_mode, False)

    # Setup device
    if device == DEVICE_CPU:
        load_device = torch.device(DEVICE_CPU)
        offload_device = torch.device(DEVICE_CPU)
    else:
        load_device = get_torch_device(device)
        offload_device = get_offload_device()

    # Resolve dtype
    target_dtype = resolve_dtype(dtype, load_device)

    # Build cache key (namespaced so it never collides with dropdown loaders).
    # Plan 2026-08-20 (RC-2/B2): the key carries full file identity — weight
    # file name + mtime_ns + size, config selector, RESOLVED attention mode,
    # 4-bit flag, and dtype — from the bundle's recorded build fields. Hand-
    # built bundles without those fields fall back to stat'ing source_path /
    # widget values, keeping a valid (if less specific) key.
    bundle_use_llm_4bit = bool(model_bundle.get("use_llm_4bit", False))
    bundle_dtype_str = model_bundle.get("dtype_str") or dtype
    cache_key = identity_for_external(
        source_path,
        model_name,
        actual_attention_mode,
        use_llm_4bit=bundle_use_llm_4bit,
        dtype_str=bundle_dtype_str,
    )

    # Register the bundle under its patcher key so eviction can NEUTRALIZE it.
    # ComfyUI's output cache keeps the node-output bundle strongly alive, so
    # dropping our own patcher entry would not free the weights otherwise.
    # A reused-patcher run re-registers the same live dict (no-op replace).
    register_model_bundle(cache_key, model_bundle)

    # Unload-before-load gate (plan 2026-08-20, C2/RC-1): if a DIFFERENT model
    # is active for this family, fully release it before touching the caches.
    evict_if_changed(FAMILY_TTS, cache_key, (VIBEVOICE_PATCHER_CACHE,))

    if cache_key not in VIBEVOICE_PATCHER_CACHE:
        model_handler = ExternalVibeVoiceModelHandler(
            model=model,
            processor=processor,
            model_pack_name=model_name,
            attention_mode=actual_attention_mode,
            source_path=source_path,
        )
        # Keep the handler's key in sync with the patcher-cache key so the
        # destroy path evicts the right entry (plan 2026-08-20, §4.2).
        model_handler.cache_key = cache_key

        patcher = VibeVoicePatcher(
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
    model_management.load_model_gpu(patcher)
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
    # Backends measured to diverge on the realtime path are dropped here, where
    # the model family is known, so the exclusion reaches the cache key and the
    # patcher as well as the loader (plan 2026-09-26, step S3.2).
    if is_model_type(model_name, "streaming_tts"):
        actual_attention_mode = resolve_realtime_attention_mode(actual_attention_mode)

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

    # Unload-before-load gate (plan 2026-08-20, C3/RC-1/RC-5): switching
    # models (dropdown <-> dropdown or dropdown <-> external — both share the
    # "tts" family) fully releases the previous model BEFORE the new weights
    # are loaded. Same-key calls are a strict no-op.
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
            None = auto (2x prompt length — ``max_length_times=2``, the
            value ``model.generate`` uses when the caller passes nothing).
            Passed through to ``model.generate`` so the non-streaming AR loop
            terminates; when the processor tokenizer exposes ``speech_end_id``
            it also stops on the EOS speech token.

    Returns:
        Tuple of (output audio tensor [1, 1, T], sample_rate).

    Raises:
        ValueError: If script is empty or invalid, or if a streaming
            (realtime) model/processor is passed (those require the
            canonical VibeVoice TTS realtime family path).
        RuntimeError: If generation fails.
    """
    # Guard: streaming (realtime) models/processors require the canonical
    # VibeVoice TTS realtime family path and cached-prompt adapter. The
    # non-streaming path below calls processor(text=..., voice_samples=...),
    # which VibeVoiceStreamingProcessor does not support by design. Detect by
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
            "Use the canonical VibeVoice TTS realtime family path for this model."
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

    # Move inputs to the compute device. NOT `model.device`: that property is
    # derived from the first parameter's residency, which under partial
    # offload is CPU even while compute happens on the load device (BUG-012).
    inputs = {
        k: v.to(model_management.get_torch_device()) if isinstance(v, torch.Tensor) else v
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
        # Drives both the frontend bar and the standard tqdm console bar.
        pbar = ProgressBarWithConsole(inference_steps)

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
            pbar.close()

    # After the forwards, not at load time: this is the only point where the
    # fast/streamed split says anything about this run. No-op for non-GGUF.
    log_gguf_forward_counters("tts_generate")

    # Post-process output.
    # Guard: the vendored AR loop appends None for any sample that never emitted
    # a speech_diffusion_id token (e.g. corrupt / over-quantized weights, or an
    # empty generation). Without this guard, `outputs.speech_outputs[0].ndim`
    # raises an opaque AttributeError on None.
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
    # ComfyUI's AUDIO contract is CPU float32. A bf16/fp16 model decodes in its
    # compute dtype, so cast before handing the waveform back.
    return output_waveform.detach().to(device="cpu", dtype=torch.float32), sample_rate


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
            # Plan 2026-08-18 D5: the user-requested cold offload keeps its
            # destructive semantics via the explicit flag (the default
            # unpatch_model is now non-destructive, RC-6).
            patcher.unpatch_model(unpatch_weights=True, destroy=True)
    model_management.unload_all_models()
    gc.collect()
    model_management.soft_empty_cache()
    logger.info("Model force offload completed")
