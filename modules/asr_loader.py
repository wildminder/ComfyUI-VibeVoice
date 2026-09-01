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
from .attention_utils import resolve_attention_mode, get_attn_implementation_for_load
from .dtype_utils import resolve_dtype
from .base_loader import BaseVibeVoiceLoader

# Support both package-relative imports and direct imports
from ..src.vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
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

        Args:
            device: Target device for the model.
            dtype_str: Dtype string ("auto", "bf16", "fp16", "fp32").
            attention_mode: Attention implementation to use.
        """
        self.model, self.processor = VibeVoiceASRLoader.load_model(
            self.model_name, device, dtype_str=dtype_str, attention_mode=attention_mode
        )
        if hasattr(self.model, 'device') and self.model.device != device:
            self.model.to(device)
        # Plan 2026-08-18, Phase 6 (D7/RC-7): refine the size estimate from
        # the real parameters now that the model is loaded (config size_gb
        # can be inaccurate for quantized / merged checkpoints).
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
            # Official model — resolve dir + download if needed (shared logic).
            model_path = BaseVibeVoiceLoader._resolve_official_model_dir(model_name)
            repo_id = model_info.get("repo_id") or MODEL_CONFIGS.get(model_name, {}).get("repo_id")
            BaseVibeVoiceLoader._ensure_downloaded(
                repo_id=repo_id, local_dir=model_path, model_name=model_name
            )

        # Determine tokenizer repo (shared helper; ASR uses the 7B tokenizer).
        tokenizer_repo = BaseVibeVoiceLoader.tokenizer_repo_for(model_name)

        return model_path, tokenizer_repo

    @staticmethod
    def load_model(
        model_name: str,
        device,
        dtype_str: str = "auto",
        attention_mode: str = "sdpa",
    ):
        """Load a VibeVoice ASR model, downloading if necessary. Caches the loaded model.

        Args:
            model_name: Name of the model to load.
            device: Target device for the model.
            dtype_str: Dtype string ("auto", "bf16", "fp16", "fp32").
            attention_mode: Attention implementation ("eager", "sdpa", "flash_attention_2").

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

        cache_key = f"asr_{model_name}_{dtype_str}_{attention_mode}"
        if cache_key in LOADED_ASR_MODELS_CACHE:
            logger.debug(f"Using cached ASR model: {model_name}")
            return LOADED_ASR_MODELS_CACHE[cache_key]

        # Resolve paths
        model_path, tokenizer_repo = VibeVoiceASRLoader._resolve_model_paths(model_name)

        # Resolve dtype
        load_device = model_management.get_torch_device() if not isinstance(device, torch.device) else device
        model_dtype = resolve_dtype(dtype_str, load_device)

        # Resolve attention mode
        attention_mode = resolve_attention_mode(attention_mode, quantize_4bit=False)
        attn_implementation = get_attn_implementation_for_load(attention_mode)

        try:
            logger.debug(
                f"Loading ASR model '{model_name}' with dtype: {model_dtype} "
                f"and attention: '{attn_implementation}'"
            )

            # Load processor
            processor = VibeVoiceASRProcessor.from_pretrained(
                model_path,
                language_model_pretrained_name=tokenizer_repo,
            )

            # Load model
            from_pretrained_kwargs = {
                "attn_implementation": attn_implementation,
                "device_map": device if isinstance(device, str) else None,
            }

            # Handle dtype kwarg based on transformers version
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

            if isinstance(device, torch.device) or (isinstance(device, str) and device != "auto"):
                model = model.to(device)

            model.eval()
            LOADED_ASR_MODELS_CACHE[cache_key] = (model, processor)
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
