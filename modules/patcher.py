"""VibeVoicePatcher: Custom ModelPatcher for VibeVoice models.

Manages VRAM by loading/offloading the VibeVoice model through ComfyUI's
ModelPatcher infrastructure. Handles lazy model loading, device placement,
and dtype casting.
"""

import torch
import gc
import logging

import comfy.model_patcher
import comfy.model_management as model_management

from .loader import VibeVoiceLoader, LOADED_MODELS_CACHE
from .asr_loader import LOADED_ASR_MODELS_CACHE
from .dtype_utils import cast_model_to_dtype, get_dtype_str

logger = logging.getLogger(__name__)


class VibeVoicePatcher(comfy.model_patcher.ModelPatcher):
    """Custom ModelPatcher for managing VibeVoice models in ComfyUI.

    Handles moving the model to the correct device (GPU) for inference
    and offloading it to free VRAM.
    """

    def __init__(self, model, attention_mode: str = "eager", dtype=None, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.attention_mode = attention_mode
        self.cache_key = getattr(model, 'cache_key', 'VibeVoice_Unknown')
        self.target_dtype = dtype
        # NTH-004: tracks whether the model was warm-offloaded (tensors retained
        # on the intermediate device) so `is_loaded` reflects offload state.
        self._warm_offloaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if the model's core components are loaded."""
        return (
            hasattr(self, 'model')
            and self.model is not None
            and hasattr(self.model, 'model')
            and self.model.model is not None
            and not getattr(self, '_warm_offloaded', False)
        )

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False, *args, **kwargs):
        """Called by ComfyUI's model manager to load the model onto the GPU.

        Lazily loads the VibeVoice model if not already loaded, then delegates
        to ComfyUI's standard patch_model() for device placement and weight
        tracking. This ensures ComfyUI's model_management properly tracks
        model_loaded_weight_memory and can manage VRAM offloading.

        Args:
            device_to: Target device for the model.
            lowvram_model_memory: Memory limit for low-VRAM mode (0 = full load).
            load_weights: Whether to load weights (always True for VibeVoice).
            force_patch_weights: Force re-patching of weights.
        """
        # NTH-004: a warm re-attach means the tensors are already present (just
        # offloaded), so mark this patcher as loaded-on-device again.
        self._warm_offloaded = False

        target_device = self.load_device if device_to is None else device_to

        if self.model.model is None:
            logger.info(
                f"Loading VibeVoice models for '{self.model.model_pack_name}' to {target_device}..."
            )
            mode_names = {
                "eager": "Eager (Most Compatible)",
                "sdpa": "SDPA (Balanced Speed/Compatibility)",
                "flash_attention_2": "Flash Attention 2 (Fastest)",
                "sage": "SageAttention (Quantized High-Performance)",
            }
            logger.info(f"Attention Mode: {mode_names.get(self.attention_mode, self.attention_mode)}")
            self.model.load_model(target_device, self.attention_mode)

        # Move model to target device before super().patch_model() so that
        # ComfyUI's load() can properly track model_loaded_weight_memory
        self.model.model.to(target_device)

        # Apply dtype casting if specified
        if self.target_dtype is not None:
            logger.debug(f"Casting model to dtype: {self.target_dtype}")
            cast_model_to_dtype(self.model.model, self.target_dtype)

        # Delegate to ComfyUI's standard patch_model() with load_weights=True
        # so that ComfyUI's load() properly tracks model_loaded_weight_memory
        # and can manage VRAM offloading. The model is already loaded and on
        # the correct device; ComfyUI's load() will iterate over the handler's
        # parameters (which include the VibeVoice model's parameters) and
        # track the loaded weight memory.
        return super().patch_model(
            device_to=target_device,
            lowvram_model_memory=lowvram_model_memory,
            load_weights=load_weights,
            force_patch_weights=force_patch_weights,
            *args, **kwargs
        )

    def unpatch_model(self, device_to=None, unpatch_weights=True, warm: bool = False, *args, **kwargs):
        """Called by ComfyUI's model manager to offload the model.

        Clears the model reference and cache to allow garbage collection.

        When ``warm`` is True, the model/processor tensors are *retained* on the
        intermediate device instead of being nulled and the cache dropped. A
        subsequent ``patch_model`` then re-attaches them without re-instantiating
        or reloading weights from disk — the NTH-004 "warm re-attach" optimization
        that makes repeated force-offload + re-run cycles fast.
        """
        if unpatch_weights:
            if warm and self.model is not None and self.model.model is not None:
                # Warm offload: keep tensors on the intermediate device for fast re-attach.
                try:
                    offload_target = model_management.intermediate_device()
                except Exception:
                    offload_target = self.offload_device
                self.model.model = self.model.model.to(offload_target)
                self._warm_offloaded = True
                logger.info(
                    f"Warm offloading VibeVoice models for '{self.model.model_pack_name}' "
                    f"({self.attention_mode}) to {offload_target} (tensors retained)..."
                )
                # Keep model/processor references and the cache intact. Tell ComfyUI to
                # release the GPU slot without freeing the retained weights.
                return super().unpatch_model(device_to, unpatch_weights=False, *args, **kwargs)

            # Cold offload (default): null references and clear cache.
            logger.info(
                f"Offloading VibeVoice models for '{self.model.model_pack_name}' "
                f"({self.attention_mode}) to {device_to}..."
            )
            self.model.model = None
            self.model.processor = None

            if self.cache_key in LOADED_MODELS_CACHE:
                del LOADED_MODELS_CACHE[self.cache_key]
                logger.info(f"Cleared LOADED_MODELS_CACHE for: {self.cache_key}")

            gc.collect()
            model_management.soft_empty_cache()

        return super().unpatch_model(device_to, unpatch_weights, *args, **kwargs)


class VibeVoiceASRPatcher(VibeVoicePatcher):
    """ASR-specific ModelPatcher.

    Behaves identically to the TTS patcher but clears the dedicated
    ASR model cache (``LOADED_ASR_MODELS_CACHE``) on unload instead of the
    TTS cache. This brings the ASR path under the same ComfyUI memory
    orchestration (``model_management.load_model_gpu`` / partial offload /
    cross-model VRAM arbitration) that the TTS path already uses, resolving
    CRIT-001.
    """

    def unpatch_model(self, device_to=None, unpatch_weights=True, *args, **kwargs):
        """Offload the ASR model and clear the ASR cache entry.

        ``self.model.model`` / ``self.model.processor`` are nulled by the
        base ``unpatch_model`` (via ``super()``); here we additionally drop
        the ASR-specific cache key so the next run re-instantiates cleanly.
        """
        if unpatch_weights and self.cache_key in LOADED_ASR_MODELS_CACHE:
            del LOADED_ASR_MODELS_CACHE[self.cache_key]
            logger.info(f"Cleared LOADED_ASR_MODELS_CACHE for: {self.cache_key}")
        return super().unpatch_model(device_to, unpatch_weights, *args, **kwargs)
