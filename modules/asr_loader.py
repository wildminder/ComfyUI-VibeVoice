"""ASR model loading and caching for VibeVoice ASR models.

Handles loading of the VibeVoiceASRForConditionalGeneration model and
VibeVoiceASRProcessor, with caching and device/dtype management.
"""

import os
import gc
import logging
import torch

import comfy.model_management as model_management

from .model_info import AVAILABLE_VIBEVOICE_MODELS, MODEL_CONFIGS
from .attention_utils import resolve_attention_mode, resolve_asr_attention_mode, get_attn_implementation_for_load, check_sage_attention_compatible
from .dtype_utils import resolve_dtype
from .base_loader import BaseVibeVoiceLoader

# Support both package-relative imports and direct imports
from ..src.vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from ..src.vibevoice.modular.sage_attention_patch import set_sage_attention
from ..src.vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

logger = logging.getLogger(__name__)

# Separate cache for ASR models
LOADED_ASR_MODELS_CACHE = {}


class VibeVoiceASRModelHandler(torch.nn.Module):
    """A lightweight handler for a VibeVoice ASR model.

    Acts as a container that ComfyUI's ModelPatcher can manage, while the
    actual heavy model is loaded on demand.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        # Mirror VibeVoiceModelHandler so logs/patcher behave consistently.
        self.model_pack_name = model_name
        # Default cache key; load_asr_model_patched overrides this with the
        # attention-aware key so the patcher and LOADED_ASR_MODELS_CACHE agree.
        self.cache_key = f"asr_{model_name}"
        self.model = None
        self.processor = None

        size_gb = MODEL_CONFIGS.get(model_name, {}).get("size_gb", 15.0)
        self.size = int(size_gb * (1024**3))

    def load_model(self, device, dtype_str: str = "auto", attention_mode: str = "sdpa"):
        """Load the ASR model and processor into memory.

        Device contract (DF-003, same as the TTS handler): the loader builds
        the model on CPU and this handler performs NO device move — ``device``
        is only forwarded for dtype-auto resolution. The single host-to-device
        transfer is owned by ``VibeVoiceASRPatcher.patch_model`` (via core's
        ModelPatcher.load) after ComfyUI's VRAM arbitration.

        Args:
            device: Target device (used for dtype-auto resolution inside the
                loader; NOT used for placement here).
            dtype_str: Dtype string ("auto", "bf16", "fp16", "fp32").
            attention_mode: Attention implementation to use.
        """
        self.model, self.processor = VibeVoiceASRLoader.load_model(
            self.model_name, device, dtype_str=dtype_str, attention_mode=attention_mode
        )
        self._refine_size()

    def _refine_size(self) -> None:
        """Refine ``self.size`` from the real parameters after a load."""
        try:
            total = sum(
                p.numel() * p.element_size() for p in self.model.parameters()
            )
            if total > 0:
                self.size = total
        except Exception:
            pass


def _is_native_asr_checkpoint(model_path: str) -> bool:
    """Return True when the checkpoint dir is a HF-native VibeVoice ASR model.

    Native checkpoints (``microsoft/VibeVoice-ASR-HF``) declare
    ``"model_type": "vibevoice_asr"`` in ``config.json`` and are handled by
    the transformers builtins (``VibeVoiceAsrForConditionalGeneration`` /
    ``AutoProcessor``). Vendored checkpoints (the streaming family, the
    original ``microsoft/VibeVoice-ASR``) share ``model_type: "vibevoice"``
    with TTS and keep going through ``src/vibevoice``.
    """
    config_path = os.path.join(model_path, "config.json")
    try:
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f).get("model_type") == "vibevoice_asr"
    except (OSError, ValueError):
        return False


def _apply_sage_attention_if_requested(model, attention_mode: str) -> None:
    """Apply SageAttention post-load (TTS parity: loader.py's block).

    SageAttention is a per-instance monkey-patch of ``Qwen2Attention.forward``
    (real HF instances in both the vendored and native trees), applied AFTER
    model construction.
    """
    if attention_mode != "sage":
        return
    if check_sage_attention_compatible():
        set_sage_attention(model)
    else:
        raise RuntimeError("Incompatible hardware/setup for SageAttention.")


def _neutralize_generation_config_presets(model) -> None:
    """Drop the checkpoint's preset max/min length knobs from generation_config.

    The ASR-HF checkpoint ships ``generation_config.json`` with BOTH
    ``max_length`` and ``max_new_tokens`` (32768). transformers warns on every
    ``generate()`` when a passed ``max_new_tokens`` meets a non-default
    ``max_length``. The node drives generation exclusively via
    ``max_new_tokens`` (node widget + progress-streamer budget), so the
    preset is neutralized at load time — matching the TTS models, which carry
    no preset at all.
    """
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        return
    if getattr(gen_cfg, "max_length", None) is not None:
        gen_cfg.max_length = None
    if getattr(gen_cfg, "min_length", None) is not None:
        gen_cfg.min_length = None


class VibeVoiceASRLoader(BaseVibeVoiceLoader):
    """Static loader class for VibeVoice ASR models."""

    @staticmethod
    def _resolve_model_paths(model_name: str) -> tuple:
        """Resolve model paths for ASR model.

        Delegates directory resolution, lazy download, and tokenizer-repo
        selection to the shared :class:`BaseVibeVoiceLoader` (IMP-004), keeping
        the ASR and TTS loaders free of duplicated discovery logic.

        Returns:
            Tuple of (model_path, tokenizer_repo).
        """
        model_info = AVAILABLE_VIBEVOICE_MODELS.get(model_name, {})
        model_type = model_info.get("type", "official")

        if model_type == "local_dir":
            model_path = model_info["path"]
        elif model_type == "standalone":
            model_path = model_info["path"]
        else:
            model_path = BaseVibeVoiceLoader._resolve_official_model_dir(model_name)
            repo_id = model_info.get("repo_id") or MODEL_CONFIGS.get(model_name, {}).get("repo_id")
            BaseVibeVoiceLoader._ensure_downloaded(
                repo_id=repo_id, local_dir=model_path, model_name=model_name
            )

        tokenizer_repo = BaseVibeVoiceLoader.tokenizer_repo_for(model_name)

        return model_path, tokenizer_repo

    @staticmethod
    def _load_native(model_path: str, device, model_dtype, attn_implementation: str,
                     attention_mode: str = "sdpa"):
        """Load a HF-native VibeVoice-ASR-HF checkpoint via transformers.

        The native classes (transformers >= 5.3.0) match the checkpoint
        exactly — no vendored compat shims involved. The processor comes
        from ``AutoProcessor`` (reads processor_config.json + the shipped
        tokenizer), the model from ``VibeVoiceAsrForConditionalGeneration``.

        Device contract (DF-003, same as the TTS loader): the model is built
        entirely on CPU and NO device move happens here — ``device`` is only
        used for dtype-auto resolution. The single host-to-device transfer is
        owned by ``VibeVoiceASRPatcher.patch_model`` after ComfyUI's VRAM
        arbitration.
        """
        try:
            logger.debug(
                f"Loading native ASR model from '{model_path}' with dtype: {model_dtype} "
                f"and attention: '{attn_implementation}'"
            )
            from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

            processor = AutoProcessor.from_pretrained(model_path)

            _LM_ONLY = ("acoustic_tokenizer_encoder_config", "semantic_tokenizer_encoder_config")
            from_pretrained_kwargs = {"attn_implementation": attn_implementation}
            try:
                from transformers import AutoConfig
                _cfg = AutoConfig.from_pretrained(model_path)
                if any(getattr(_cfg, k, None) is not None for k in _LM_ONLY):
                    from_pretrained_kwargs["attn_implementation"] = {
                        "": attn_implementation,
                        **{k: "eager" for k in _LM_ONLY},
                    }
            except Exception:
                pass

            import transformers
            from packaging import version
            if version.parse(transformers.__version__) >= version.parse("4.56.0"):
                from_pretrained_kwargs['dtype'] = model_dtype
            else:
                from_pretrained_kwargs['torch_dtype'] = model_dtype

            model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
                model_path,
                **from_pretrained_kwargs,
            )

            try:
                from .comfy_stream import convert_tree_for_streaming

                convert_tree_for_streaming(model)
            except Exception as e:
                logger.warning(
                    f"Streaming conversion failed (continuing without it): {e}"
                )

            model.eval()

            _apply_sage_attention_if_requested(model, attention_mode)
            _neutralize_generation_config_presets(model)

            logger.info(f"Successfully loaded native ASR model from '{model_path}'")
            return model, processor

        except ImportError as e:
            raise RuntimeError(
                f"The checkpoint at '{model_path}' requires transformers >= 5.3.0 "
                f"(native VibeVoice-ASR support); installed: {e}"
            )
        except Exception as e:
            logger.error(f"Failed to load native ASR model from '{model_path}': {e}")
            raise RuntimeError(f"Failed to load ASR model '{model_path}': {e}")

    @staticmethod
    def load_model(
        model_name: str,
        device,
        dtype_str: str = "auto",
        attention_mode: str = "sdpa",
    ):
        """Load a VibeVoice ASR model, downloading if necessary.

        This is a PURE BUILDER (TTS parity): it constructs and returns
        ``(model, processor)`` and does NOT register them in any cache.
        Cache ownership belongs to the callers — ``load_asr_model_patched`` /
        ``load_asr_from_external`` register the live model under the PATCHER
        key.

        Args:
            model_name: Name of the model to load.
            device: Target device for the model.
            dtype_str: Dtype string ("auto", "bf16", "fp16", "fp32").
            attention_mode: Attention implementation ("eager", "sdpa",
                "flash_attention_2", "sage").

        Returns:
            Tuple of (model, processor).

        Raises:
            ValueError: If model_name is not found.
            RuntimeError: If model loading fails.
        """
        if model_name not in AVAILABLE_VIBEVOICE_MODELS:
            raise ValueError(
                f"Unknown VibeVoice ASR model: {model_name}. "
                f"Available models: {list(AVAILABLE_VIBEVOICE_MODELS.keys())}"
            )

        model_path, tokenizer_repo = VibeVoiceASRLoader._resolve_model_paths(model_name)

        load_device = model_management.get_torch_device() if not isinstance(device, torch.device) else device
        model_dtype = resolve_dtype(dtype_str, load_device)

        attention_mode = resolve_asr_attention_mode(
            resolve_attention_mode(attention_mode, quantize_4bit=False)
        )
        attn_implementation = get_attn_implementation_for_load(attention_mode)

        if _is_native_asr_checkpoint(model_path):
            return VibeVoiceASRLoader._load_native(
                model_path, device, model_dtype, attn_implementation,
                attention_mode=attention_mode,
            )

        try:
            logger.debug(
                f"Loading ASR model '{model_name}' with dtype: {model_dtype} "
                f"and attention: '{attn_implementation}'"
            )

            processor = VibeVoiceASRProcessor.from_pretrained(
                model_path,
                language_model_pretrained_name=tokenizer_repo,
            )

            from_pretrained_kwargs = {"attn_implementation": attn_implementation}

            import transformers
            from packaging import version
            if version.parse(transformers.__version__) >= version.parse("4.56.0"):
                from_pretrained_kwargs['dtype'] = model_dtype
            else:
                from_pretrained_kwargs['torch_dtype'] = model_dtype

            model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                model_path,
                **from_pretrained_kwargs,
            )

            try:
                from .comfy_stream import convert_tree_for_streaming

                convert_tree_for_streaming(model)
            except Exception as e:
                logger.warning(
                    f"Streaming conversion failed (continuing without it): {e}"
                )

            model.eval()

            _apply_sage_attention_if_requested(model, attention_mode)
            _neutralize_generation_config_presets(model)

            logger.info(f"Successfully loaded ASR model '{model_name}'")
            return model, processor

        except Exception as e:
            logger.error(f"Failed to load ASR model '{model_name}': {e}")
            raise RuntimeError(f"Failed to load ASR model '{model_name}': {e}")


def cleanup_asr_models(keep_cache_key: str = None) -> None:
    """Remove all cached ASR models except the one matching keep_cache_key.

    Args:
        keep_cache_key: Cache key to preserve. If None, all are cleared.
    """
    keys_to_remove = []
    for key in list(LOADED_ASR_MODELS_CACHE.keys()):
        if key != keep_cache_key:
            keys_to_remove.append(key)
            del LOADED_ASR_MODELS_CACHE[key]

    if keys_to_remove:
        logger.debug(f"Cleaned up cached ASR models: {keys_to_remove}")
        gc.collect()
        model_management.soft_empty_cache()