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
from .dtype_utils import cast_model_to_dtype, get_dtype_str, representative_dtype

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
    def _model_cache(self) -> dict:
        """The (model, processor) cache dict this patcher's cold offload clears.

        AUD-012: TTS and ASR patchers own *separate* caches. The cold-offload
        path must only evict from this patcher's own registry so an ASR offload
        can never drop a TTS cache entry (and vice versa).
        """
        return LOADED_MODELS_CACHE

    @property
    def is_loaded(self) -> bool:
        """Check if the model's core components are loaded AND on the load device.

        Plan 2026-08-18, Step 5.3: a CPU-offloaded model is "loaded in RAM"
        but not "loaded for inference". The device check ensures is_loaded
        reflects actual inference-readiness.
        """
        if not (
            hasattr(self, 'model')
            and self.model is not None
            and hasattr(self.model, 'model')
            and self.model.model is not None
        ):
            return False
        if getattr(self, '_warm_offloaded', False):
            return False
        # Device awareness: the heavy model must be on the load device.
        try:
            model_device = next(self.model.model.parameters()).device
            if not isinstance(model_device, torch.device):
                # Can't determine device (e.g., MagicMock in tests) —
                # fall back to pre-5.3 behavior (loaded if refs exist).
                return True
            return model_device == self.load_device
        except (StopIteration, AttributeError, TypeError):
            return True

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

        # Plan 2026-08-18 D6/RC-5 + 2026-08-26: the single H2D transfer is
        # owned by super().patch_model(); the tree itself is made fluent in
        # core's lowvram protocol by modules/comfy_stream.py (applied at load
        # time), so partial loading/offloading streams correctly instead of
        # stranding foreign modules on CPU.

        # Apply dtype casting ONLY if the model's dtype differs from the
        # target (DF-004 fix). The loader now applies the final dtype on CPU
        # before the H2D transfer, so this cast is normally a no-op guard;
        # it still fires for models loaded by other paths (e.g. ASR) or when
        # the dtype cannot be determined up front.
        if self.target_dtype is not None:
            # Representative dtype = first FLOATING param (quant residents
            # hold raw uint8/int8 params that must not influence this check).
            current_dtype = representative_dtype(self.model.model)
            if current_dtype != self.target_dtype:
                logger.debug(
                    f"Casting model to dtype: {self.target_dtype} "
                    f"(current: {current_dtype})"
                )
                cast_model_to_dtype(self.model.model, self.target_dtype)

        # Delegate to ComfyUI's standard patch_model() with load_weights=True
        # so that ComfyUI's load() properly tracks model_loaded_weight_memory.
        # Every parameter is already on target_device at this point, so
        # core's per-module movement skips entirely; its bookkeeping still
        # records full residency for VRAM arbitration.
        result = super().patch_model(
            device_to=target_device,
            lowvram_model_memory=lowvram_model_memory,
            load_weights=load_weights,
            force_patch_weights=force_patch_weights,
            *args, **kwargs
        )

        return result

    def unpatch_model(self, device_to=None, unpatch_weights=True, warm: bool = False,
                      destroy: bool = False, *args, **kwargs):
        """Called by ComfyUI's model manager to offload the model.

        Plan 2026-08-18, D5/RC-6 — the offload contract is now NON-DESTRUCTIVE
        by default, matching ComfyUI's own ``ModelPatcher.unpatch_model`` which
        *moves* weights to the offload device and never destroys them:

        | warm  | destroy | behavior |
        |-------|---------|----------|
        | False | False   | **Default (ComfyUI-initiated):** keep ``handler.model`` + caches; ``super().unpatch_model(device_to, unpatch_weights=True)`` moves the whole handler tree to ``device_to`` (CPU). A later ``patch_model`` is a pure H2D transfer — no disk reload. |
        | True  | False   | Warm path (NTH-004): move to the intermediate device, keep refs, ``super(..., unpatch_weights=False)``. |
        | False | True    | Destructive path: ``handler.model=None``, ``processor=None``, evict cache, ``gc.collect()``, ``soft_empty_cache()``. Only for explicit user-requested full free. |
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

            if destroy:
                # Destructive offload (explicit): null references and clear cache.
                logger.info(
                    f"Destroying VibeVoice models for '{self.model.model_pack_name}' "
                    f"({self.attention_mode}) (weights freed)..."
                )
                # Plan 2026-08-20 (D1/RC-4): drop this patcher from ComfyUI's
                # loaded-model registry FIRST so nothing downstream (free_memory,
                # cleanup passes) later touches a destroyed model. Lazy import
                # avoids an import cycle with model_registry.
                try:
                    from .model_registry import unregister_from_comfy

                    unregister_from_comfy(self)
                except Exception as e:
                    logger.warning(f"Could not unregister patcher from ComfyUI: {e}")

                self.model.model = None
                self.model.processor = None

                cache = self._model_cache
                if self.cache_key in cache:
                    del cache[self.cache_key]
                    logger.info(f"Cleared model cache for: {self.cache_key}")

                gc.collect()
                model_management.soft_empty_cache()
                return super().unpatch_model(device_to, unpatch_weights, *args, **kwargs)

            # Routine offload (default, RC-6 fix): keep the model in CPU RAM.
            # super().unpatch_model(device_to, unpatch_weights=True) moves the
            # whole handler tree (including the heavy model submodule) to
            # device_to and resets model_loaded_weight_memory. The next
            # patch_model() is then a pure host-to-device transfer.
            self._warm_offloaded = False
            logger.info(
                f"Offloading VibeVoice models for '{self.model.model_pack_name}' "
                f"({self.attention_mode}) to {device_to} (weights kept in RAM)..."
            )

        return super().unpatch_model(device_to, unpatch_weights, *args, **kwargs)


class VibeVoiceASRPatcher(VibeVoicePatcher):
    """ASR-specific ModelPatcher.

    Behaves identically to the TTS patcher but clears the dedicated
    ASR model cache (``LOADED_ASR_MODELS_CACHE``) on unload instead of the
    TTS cache. This brings the ASR path under the same ComfyUI memory
    orchestration (``model_management.load_model_gpu`` / partial offload /
    cross-model VRAM arbitration) that the TTS path already uses, resolving
    CRIT-001.

    AUD-012: cache isolation is achieved by overriding ``_model_cache`` (the
    base cold-offload path evicts only from this dict) instead of a separate
    ``unpatch_model`` override, which previously also ran the base cold path
    and wrongly evicted the TTS cache for a colliding key.
    """

    @property
    def _model_cache(self) -> dict:
        """ASR patchers evict from the ASR cache only (never the TTS cache)."""
        return LOADED_ASR_MODELS_CACHE
