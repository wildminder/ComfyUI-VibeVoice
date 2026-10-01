"""VibeVoicePatcher: Custom ModelPatcher for VibeVoice models.

Manages VRAM by loading/offloading the VibeVoice model through ComfyUI's
ModelPatcher infrastructure. Handles lazy model loading, device placement,
and dtype casting.
"""

import torch
import gc
import logging

from itertools import chain

import comfy.model_patcher
import comfy.model_management as model_management

from .loader import VibeVoiceLoader, LOADED_MODELS_CACHE
from .asr_loader import LOADED_ASR_MODELS_CACHE
from .dtype_utils import cast_model_to_dtype, get_dtype_str, representative_dtype

logger = logging.getLogger(__name__)


def _init_legacy_attributes(patcher, model, attention_mode, dtype) -> None:
    """Install the attribute vocabulary every VibeVoice patcher shares.

    Split out of :class:`VibeVoicePatcher.__init__` because the minted dynamic
    subclass (``make_dynamic_patcher_class``) builds its MRO as
    ``(ModelPatcherDynamic, <legacy>)`` — core's dynamic protocol must come
    first so ``patch_model``/``unpatch_model`` dispatch to core rather than to
    the legacy overrides. With that MRO, ``super()`` from the dynamic class
    resolves to ``ModelPatcherDynamic.__init__`` and the legacy ``__init__``
    never runs, so these four attributes are applied here instead. Sharing one
    definition is what keeps the two classes from drifting apart.
    """
    patcher.attention_mode = attention_mode
    patcher.cache_key = getattr(model, 'cache_key', 'VibeVoice_Unknown')
    patcher.target_dtype = dtype
    # NTH-004: tracks whether the model was warm-offloaded (tensors retained
    # on the intermediate device) so `is_loaded` reflects offload state. The
    # dynamic class never sets it (its warm path releases pins, it does not move
    # tensors), and its `is_loaded` does not read it — see the design note in
    # `make_dynamic_patcher_class`.
    patcher._warm_offloaded = False


def _adopt_resident_weights(patcher, model) -> None:
    """Report to core the weight bytes this patcher already has in VRAM.

    Our loader places every tensor on the load device *before* the patcher is
    built, so core's own counter still reads zero. ``partially_load`` then
    computes ``0 + extra_memory > model_size()`` as false, decides the load is
    not full, and plans a partial load into whatever VRAM happens to be left.
    That is how a fully resident 9.33 GB 7B model gets reported as
    ``loaded partially; 5511.92 MB offloaded`` on an otherwise idle 16 GB card:
    the weights never move, only the bookkeeping is wrong — and it is wrong in
    the direction that makes a second model look like there is 5.5 GB free.

    Counting the bytes that are genuinely on an accelerator lets core's own
    guard do the right thing: ``partially_load`` returns 0 and leaves the
    resident model alone.
    """
    inner = getattr(model, "model", None)
    if inner is None:
        return
    resident = 0
    on_accelerator = False
    for tensor in chain(inner.parameters(), inner.buffers()):
        if tensor.is_meta:
            return  # nothing is loaded yet; leave core's counter at zero
        if tensor.device.type == "cuda":
            resident += tensor.numel() * tensor.element_size()
            on_accelerator = True
    if on_accelerator and resident > 0:
        patcher.model.model_loaded_weight_memory = resident


def _init_patch_state(patcher, model, attention_mode, dtype) -> None:
    """The one place a patcher finishes coming up."""
    _init_legacy_attributes(patcher, model, attention_mode, dtype)
    _adopt_resident_weights(patcher, model)


class VibeVoicePatcher(comfy.model_patcher.ModelPatcher):
    """Custom ModelPatcher for managing VibeVoice models in ComfyUI.

    Handles moving the model to the correct device (GPU) for inference
    and offloading it to free VRAM.

    Subclasses ComfyUI's standard ModelPatcher, allowing models that fit into
    VRAM to be fully loaded directly onto the GPU with native CUDA operations
    and zero paging/streaming overhead.
    """

    def __init__(self, model, *args, attention_mode: str = "eager", dtype=None, **kwargs):
        # attention_mode/dtype are keyword-only: core's ModelPatcher.clone()
        # reconstructs the class with POSITIONAL args
        # (model, load_device, offload_device, size, ...) at
        # comfy/model_patcher.py:446, so any positional parameter declared
        # before *args silently binds load_device to attention_mode and then
        # blows up in ModelPatcher.__init__ with "missing 1 required
        # positional argument: 'offload_device'".
        super().__init__(model, *args, **kwargs)
        _init_patch_state(self, model, attention_mode, dtype)

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
                # fall back to loaded if refs exist.
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
            logger.debug(f"Attention Mode: {mode_names.get(self.attention_mode, self.attention_mode)}")
            self.model.load_model(target_device, attention_mode=self.attention_mode)

        # Apply dtype casting ONLY if the model's dtype differs from the target.
        if self.target_dtype is not None and self.model.model is not None:
            current_dtype = representative_dtype(self.model.model)
            if current_dtype != self.target_dtype:
                logger.debug(
                    f"Casting model to dtype: {self.target_dtype} "
                    f"(current: {current_dtype})"
                )
                cast_model_to_dtype(self.model.model, self.target_dtype)

        # Delegate to ComfyUI's standard patch_model() with load_weights=True.
        # When lowvram_model_memory is 0, full_load is True and the model moves
        # completely to the target device (GPU VRAM).
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

        Plan 2026-08-18, D5/RC-6 — the offload contract is non-destructive
        by default, matching ComfyUI's own ``ModelPatcher.unpatch_model`` which
        moves weights to the offload device and never destroys them:

        | warm  | destroy | behavior |
        |-------|---------|----------|
        | False | False   | Default (ComfyUI-initiated): keep handler.model + caches; move to device_to (CPU). |
        | True  | False   | Warm path: move to intermediate device, keep refs. |
        | False | True    | Destructive path: handler.model=None, processor=None, evict cache, gc.collect(). |
        """
        # A cached GGUF dequant weight is real VRAM core does not know about.
        # Offloading must not strand it, or a second load in the same session
        # fails for a reason nothing in the accounting explains.
        try:
            from .gguf_quant import clear_dequant_cache

            clear_dequant_cache()
        except Exception:
            pass

        if unpatch_weights:
            if warm and self.model is not None and self.model.model is not None:
                # Warm offload: keep tensors on the intermediate device for fast re-attach.
                try:
                    offload_target = model_management.intermediate_device()
                except Exception:
                    offload_target = self.offload_device
                self.model.model = self.model.model.to(offload_target)
                self._warm_offloaded = True
                logger.debug(
                    f"Warm offloading VibeVoice models for '{self.model.model_pack_name}' "
                    f"({self.attention_mode}) to {offload_target} (tensors retained)..."
                )
                return super().unpatch_model(device_to, unpatch_weights=False, *args, **kwargs)

            if destroy:
                # Destructive offload (explicit): null references and clear cache.
                logger.debug(
                    f"Destroying VibeVoice models for '{self.model.model_pack_name}' "
                    f"({self.attention_mode}) (weights freed)..."
                )
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
                    logger.debug(f"Cleared model cache for: {self.cache_key}")

                gc.collect()
                model_management.soft_empty_cache()
                return super().unpatch_model(device_to, unpatch_weights, *args, **kwargs)

            # Routine offload: keep the model in CPU RAM.
            self._warm_offloaded = False
            logger.debug(
                f"Offloading VibeVoice models for '{self.model.model_pack_name}' "
                f"({self.attention_mode}) to {device_to} (weights kept in RAM)..."
            )

        return super().unpatch_model(device_to, unpatch_weights, *args, **kwargs)


class VibeVoiceASRPatcher(VibeVoicePatcher):
    """ASR-specific ModelPatcher.

    Behaves identically to the TTS patcher but clears the dedicated
    ASR model cache (``LOADED_ASR_MODELS_CACHE``) on unload instead of the
    TTS cache. This brings the ASR path under the same ComfyUI memory
    orchestration that the TTS path uses.
    """

    @property
    def _model_cache(self) -> dict:
        """ASR patchers evict from the ASR cache only (never the TTS cache)."""
        return LOADED_ASR_MODELS_CACHE


# ====================================================================
# Dynamic-VRAM (aimdo / VBAR) compatibility layer
# ====================================================================

_DYNAMIC_PATCHER_CLASS_CACHE: dict = {}


def resolve_core_patcher_class() -> type:
    """Return the class ``comfy.model_patcher.CoreModelPatcher`` currently names."""
    return getattr(
        comfy.model_patcher, "CoreModelPatcher", comfy.model_patcher.ModelPatcher
    )


_PROBE_SENTINEL = object()


def dynamic_vram_available(patcher_cls, load_device) -> bool:
    """True when ``patcher_cls`` is a real dynamic patcher AND the target is CUDA."""
    try:
        device = torch.device(load_device)
    except (TypeError, ValueError, RuntimeError) as e:
        logger.debug("Dynamic VRAM probe: unusable load_device %r (%s)", load_device, e)
        return False

    if device.type != "cuda":
        return False

    is_dynamic = getattr(patcher_cls, "is_dynamic", None)
    if not callable(is_dynamic):
        return False

    try:
        return bool(is_dynamic(_PROBE_SENTINEL))
    except Exception as e:
        logger.debug("Dynamic VRAM probe: is_dynamic() failed on %s (%s)", patcher_cls, e)
        return False


def make_dynamic_patcher_class(legacy_cls: type = VibeVoicePatcher) -> type:
    """Build (and cache) the Dynamic-VRAM sibling of ``legacy_cls``.

    Retained for compatibility with test suites and external hooks that
    probe for dynamic patcher construction.
    """
    base = resolve_core_patcher_class()
    dynamic_base = getattr(comfy.model_patcher, "ModelPatcherDynamic", None)
    if dynamic_base is None or not (isinstance(base, type) and issubclass(base, dynamic_base)):
        return base

    key = (base, legacy_cls)
    cached = _DYNAMIC_PATCHER_CLASS_CACHE.get(key)
    if cached is not None:
        return cached

    def rebuild_dynamic_patcher(spec, disable_dynamic=False):
        model, args, kwargs = spec
        target = legacy_cls if disable_dynamic else make_dynamic_patcher_class(legacy_cls)
        return target(model, *args, **kwargs)

    class VibeVoiceDynamicPatcher(base, legacy_cls):
        """Demand-paged (aimdo/VBAR) sibling of the legacy VibeVoice patcher."""

        def __new__(cls, model=None, load_device=None, offload_device=None, size=0,
                    weight_inplace_update=False, fast_disk=False, **kwargs):
            return base.__new__(
                cls, model, load_device, offload_device, size,
                weight_inplace_update, fast_disk,
            )

        def __init__(self, model, *args, attention_mode: str = "eager", dtype=None, **kwargs):
            base.__init__(self, model, *args, **kwargs)
            _init_legacy_attributes(self, model, attention_mode, dtype)
            spec = (model, tuple(args), dict(kwargs, attention_mode=attention_mode, dtype=dtype))
            self.cached_patcher_init = (rebuild_dynamic_patcher, (spec,))

        @property
        def is_loaded(self) -> bool:
            if not (
                hasattr(self, 'model')
                and self.model is not None
                and getattr(self.model, 'model', None) is not None
            ):
                return False
            try:
                return bool(self.loaded_size())
            except Exception as e:
                logger.debug("Dynamic patcher is_loaded probe failed: %s", e)
                return False

        def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=False,
                        force_patch_weights=False, *args, **kwargs):
            if load_weights:
                raise ValueError(
                    f"{type(self).__name__}.patch_model(load_weights=True) is not supported: "
                    "ModelPatcherDynamic.patch_model asserts `not load_weights`. "
                    "Use comfy.model_management.load_models_gpu([patcher]) instead."
                )

            target_device = self.load_device if device_to is None else device_to

            if self.model.model is None:
                logger.info(
                    f"Loading VibeVoice models for '{self.model.model_pack_name}' to {target_device}..."
                )
                self.model.load_model(target_device, attention_mode=self.attention_mode)

            if self.target_dtype is not None and self.model.model is not None:
                current_dtype = representative_dtype(self.model.model)
                if current_dtype != self.target_dtype:
                    logger.debug(
                        f"Casting model to dtype: {self.target_dtype} (current: {current_dtype})"
                    )
                    cast_model_to_dtype(self.model.model, self.target_dtype)

            return super().patch_model(
                device_to=self.load_device,
                lowvram_model_memory=lowvram_model_memory,
                load_weights=False,
                force_patch_weights=force_patch_weights,
            )

        def unpatch_model(self, device_to=None, unpatch_weights=True, warm: bool = False,
                          destroy: bool = False, *args, **kwargs):
            if unpatch_weights and warm:
                unpin_all = getattr(self, 'unpin_all_weights', None)
                if callable(unpin_all):
                    logger.debug(
                        f"Warm offloading VibeVoice models for '{self.model.model_pack_name}' "
                        f"({self.attention_mode}) (pins released)..."
                    )
                    unpin_all()

            if unpatch_weights and destroy:
                logger.debug(
                    f"Destroying VibeVoice models for '{self.model.model_pack_name}' "
                    f"({self.attention_mode}) (weights freed)..."
                )
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
                    logger.debug(f"Cleared model cache for: {self.cache_key}")

                gc.collect()
                model_management.soft_empty_cache()

            return super().unpatch_model(device_to, unpatch_weights, *args, **kwargs)

    class_name = "VibeVoiceDynamic" + legacy_cls.__name__.removeprefix("VibeVoice")
    VibeVoiceDynamicPatcher.__name__ = class_name
    VibeVoiceDynamicPatcher.__qualname__ = class_name
    _DYNAMIC_PATCHER_CLASS_CACHE[key] = VibeVoiceDynamicPatcher
    return VibeVoiceDynamicPatcher


def select_patcher_class(weight_family, load_device, legacy_cls: type = VibeVoicePatcher) -> type:
    """Pick the patcher class for model lifecycle management.

    Returns the standard ModelPatcher (legacy_cls) so the model can be loaded
    completely into VRAM without being trapped in CPU virtual memory (VBAR)
    or forced into per-step streaming.
    """
    return legacy_cls


def _ensure_device_index(patcher) -> None:
    """Give an accelerator load device the explicit index core needs."""
    device = getattr(patcher, "load_device", None)
    if device is None or getattr(device, "index", 0) is not None:
        return
    if getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        return
    patcher.load_device = torch.device("cuda", torch.cuda.current_device())


def load_to_device(patcher, memory_required: int = 0):
    """Put ``patcher`` on its load device under standard ComfyUI memory orchestration.

    When the model fits into free VRAM, it triggers a direct full load into GPU VRAM
    (force_full_load=True), matching the swift, single-pass loading of native
    ComfyUI models (Krea, SDXL).
    """
    _ensure_device_index(patcher)
    is_dynamic = getattr(patcher, 'is_dynamic', None)
    dynamic = is_dynamic() is True if callable(is_dynamic) else False

    if dynamic:
        model_management.load_models_gpu([patcher], memory_required=memory_required)
    else:
        device = patcher.load_device
        if device is not None and device.type == "cuda":
            free_vram = model_management.get_free_memory(device)
            model_size = patcher.model_size()
            # If the model fits in VRAM with headroom for inference activations:
            if model_size < (free_vram - int(1.5 * 1024 ** 3)):
                # Ensure weights are loaded and placed completely on GPU VRAM
                patcher.patch_model(device_to=device, lowvram_model_memory=0, load_weights=True)
                model_management.load_models_gpu([patcher], force_full_load=True)
                return patcher

        model_management.load_model_gpu(patcher)

    return patcher