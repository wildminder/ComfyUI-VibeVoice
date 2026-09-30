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


class VibeVoicePatcher(comfy.model_patcher.ModelPatcher):
    """Custom ModelPatcher for managing VibeVoice models in ComfyUI.

    Handles moving the model to the correct device (GPU) for inference
    and offloading it to free VRAM.

    NOTE: the base class is spelled out literally as
    ``comfy.model_patcher.ModelPatcher``, i.e. the LEGACY patcher, not
    ``comfy.model_patcher.CoreModelPatcher``. ``main.py`` rebinds the module
    global ``CoreModelPatcher = ModelPatcherDynamic`` (aimdo), so a pack that
    names ``ModelPatcher`` here gets no VBAR, no CUDA graph capture, and the
    legacy ``partially_unload`` eviction order.

    2026-09-30: this class used to be the base for ALL routes except the
    dense external one, on a "quant families must stay legacy" stop
    condition. That is reverted — see :func:`select_patcher_class`, which now
    hands every family the dynamic sibling when core resolved one. This class
    remains the LEGACY base (and the fallback when aimdo is unavailable), so
    the override below still calls ``super().patch_model(load_weights=True)``
    — which ``ModelPatcherDynamic.patch_model`` asserts against
    (comfy/model_patcher.py:2132-2137). The dynamic sibling minted by
    :func:`make_dynamic_patcher_class` is the class that adapts to that
    protocol: it forwards ``load_weights=False``, forces ``device_to`` to
    ``self.load_device``, maps the pack's ``warm`` flag onto
    ``unpin_all_weights()``, and registers the ``cached_patcher_init`` factory
    core requires at comfy/model_patcher.py:438-441 and :509-516.
    """

    def __init__(self, model, *args, attention_mode: str = "eager", dtype=None, **kwargs):
        # attention_mode/dtype are keyword-only: core's ModelPatcher.clone()
        # reconstructs the class with POSITIONAL args
        # (model, load_device, offload_device, size, ...) at
        # comfy/model_patcher.py:446, so any positional parameter declared
        # before *args silently binds load_device to attention_mode and then
        # blows up in ModelPatcher.__init__ with "missing 1 required
        # positional argument: 'offload_device'". clone eviction runs from
        # comfy/model_management.py, so this is a hard crash on a stock install.
        super().__init__(model, *args, **kwargs)
        _init_legacy_attributes(self, model, attention_mode, dtype)

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
            logger.debug(f"Attention Mode: {mode_names.get(self.attention_mode, self.attention_mode)}")
            # Keyword, not positional: the TTS handler takes
            # (device, attention_mode) but the ASR handler takes
            # (device, dtype_str, attention_mode) — a positional call landed
            # the attention mode in the ASR handler's dtype slot.
            self.model.load_model(target_device, attention_mode=self.attention_mode)

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
                logger.debug(
                    f"Warm offloading VibeVoice models for '{self.model.model_pack_name}' "
                    f"({self.attention_mode}) to {offload_target} (tensors retained)..."
                )
                # Keep model/processor references and the cache intact. Tell ComfyUI to
                # release the GPU slot without freeing the retained weights.
                return super().unpatch_model(device_to, unpatch_weights=False, *args, **kwargs)

            if destroy:
                # Destructive offload (explicit): null references and clear cache.
                logger.debug(
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
                    logger.debug(f"Cleared model cache for: {self.cache_key}")

                gc.collect()
                model_management.soft_empty_cache()
                return super().unpatch_model(device_to, unpatch_weights, *args, **kwargs)

            # Routine offload (default, RC-6 fix): keep the model in CPU RAM.
            # super().unpatch_model(device_to, unpatch_weights=True) moves the
            # whole handler tree (including the heavy model submodule) to
            # device_to and resets model_loaded_weight_memory. The next
            # patch_model() is then a pure host-to-device transfer.
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


# ====================================================================
# Dynamic-VRAM (aimdo / VBAR) route selection
# ====================================================================
#
# The dense external single-file route is the ONLY path that may be handed to
# ``comfy.model_patcher.ModelPatcherDynamic``: it is the only one that keeps a
# full float state dict in host RAM and is therefore the only one that gains
# from demand paging. GGUF / convrot-INT8 / fp8-resident install quant
# residents that stream natively, so they stay on the legacy patcher, byte
# identical.
#
# ``_DYNAMIC_PATCHER_CLASS_CACHE`` holds the minted subclasses, keyed on
# (resolved dynamic base, legacy class) so repeated selections return the SAME
# class object (stable ``type(...)`` repr for logs, stable identity for
# isinstance-style assertions in the suite). The base is re-resolved on EVERY
# call, so rebinding ``comfy.model_patcher.CoreModelPatcher`` (as ``main.py``
# does once aimdo initialises) still mints a fresh class instead of handing
# back a stale one built on the legacy alias.

_DYNAMIC_PATCHER_CLASS_CACHE: dict = {}


def resolve_core_patcher_class() -> type:
    """Return the class ``comfy.model_patcher.CoreModelPatcher`` currently names.

    Resolved by ATTRIBUTE LOOKUP AT CALL TIME, never captured at import time.
    ``comfy/model_patcher.py:2186`` defines ``CoreModelPatcher = ModelPatcher``
    (the legacy class) and ``main.py:300`` rebinds the module global to
    ``ModelPatcherDynamic`` once the aimdo probe succeeds. A module-level
    ``from comfy.model_patcher import CoreModelPatcher`` would therefore freeze
    the legacy choice forever.
    """
    return getattr(
        comfy.model_patcher, "CoreModelPatcher", comfy.model_patcher.ModelPatcher
    )


# Zero-allocation receiver for the feature probe below. ``is_dynamic`` takes no
# part of ``self`` in either core class (comfy/model_patcher.py:403-404 returns
# a literal False, :1803-1804 a literal True), so dispatching the unbound
# function against this sentinel is genuine method dispatch through the
# resolved class without constructing a patcher.
_PROBE_SENTINEL = object()


def dynamic_vram_available(patcher_cls, load_device) -> bool:
    """True when ``patcher_cls`` is a real dynamic patcher AND the target is CUDA.

    Allocation-free and construction-free by design: ``ModelPatcherDynamic.
    __init__`` (comfy/model_patcher.py:1760-1772) calls ``register_load_device``
    which builds six real ``comfy_aimdo.host_buffer.HostBuffer`` objects, and
    ``__del__`` (:1800-1801) then dereferences pin state that a stub or a
    mocked ``__init__`` never established. Probing must therefore never build a
    throwaway patcher.

    Every gate is failure-open to ``False``: an unknown device string, a
    missing ``is_dynamic``, or any exception from core must leave the caller on
    the legacy patcher, which is known-good.
    """
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

    Returns ``resolve_core_patcher_class()`` unchanged when aimdo is not
    enabled: the alias is then the legacy ``ModelPatcher`` and minting a
    subclass would add a class name to the unraisable-warning surface for zero
    behavioural gain.
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
        """core calls this as ``fn(*args, disable_dynamic=True)``.

        Required by ``ModelPatcher.clone`` (comfy/model_patcher.py:438-441) and
        ``deepclone_multigpu`` (:509-516), both of which raise RuntimeError when
        ``cached_patcher_init`` is unset.
        """
        model, args, kwargs = spec
        target = legacy_cls if disable_dynamic else make_dynamic_patcher_class(legacy_cls)
        return target(model, *args, **kwargs)

    class VibeVoiceDynamicPatcher(base, legacy_cls):
        """Demand-paged (aimdo/VBAR) sibling of the legacy VibeVoice patcher.

        MRO is ``(ModelPatcherDynamic, <legacy>)`` so core's dynamic protocol
        wins and the legacy class still supplies ``_model_cache`` (TTS vs ASR
        cache isolation) and the shared attribute vocabulary.
        """

        def __new__(cls, model=None, load_device=None, offload_device=None, size=0,
                    weight_inplace_update=False, fast_disk=False, **kwargs):
            """Drop the pack's extra keywords before core's fixed-signature __new__.

            ``ModelPatcherDynamic.__new__`` (comfy/model_patcher.py:1753) declares
            exactly six named parameters and NO ``**kwargs``. Because
            ``VibeVoiceDynamicPatcher.__init__`` accepts ``attention_mode``/``dtype``
            (and callers pass them as keywords), Python dispatches construction
            through this class's ``__new__`` FIRST and the extra keywords raise
            ``TypeError: ModelPatcherDynamic.__new__() got an unexpected keyword
            argument 'attention_mode'`` before ``__init__`` ever runs — i.e. the
            dynamic route was unconstructible and every T6 selection would have
            crashed. Verified live in this ask.

            Delegating to ``base.__new__`` also preserves core's CPU reroute
            (:1754-1757): when it returns a plain ``ModelPatcher``, that object is
            not an instance of ``cls``, so Python skips our ``__init__`` exactly
            as it does for the class core defines.
            """
            return base.__new__(
                cls, model, load_device, offload_device, size,
                weight_inplace_update, fast_disk,
            )

        def __init__(self, model, *args, attention_mode: str = "eager", dtype=None, **kwargs):
            # Same keyword-only hazard as the legacy class: core reconstructs
            # the class POSITIONALLY at comfy/model_patcher.py:446.
            #
            # `base.__init__` (NOT `super().__init__`): the MRO puts
            # ModelPatcherDynamic ahead of the legacy class, so `super()` here
            # would land on `ModelPatcherDynamic.__init__` — which takes exactly
            # (model, load_device, offload_device, size, weight_inplace_update,
            # fast_disk) and no **kwargs (:1760), and, worse, would bypass
            # VibeVoicePatcher.__init__ entirely so the shared attribute
            # vocabulary would never be installed. Calling the base directly and
            # then `_init_legacy_attributes` reproduces the legacy construction
            # exactly while keeping the dynamic protocol first in the MRO.
            base.__init__(self, model, *args, **kwargs)
            _init_legacy_attributes(self, model, attention_mode, dtype)
            spec = (model, tuple(args), dict(kwargs, attention_mode=attention_mode, dtype=dtype))
            self.cached_patcher_init = (rebuild_dynamic_patcher, (spec,))

        @property
        def is_loaded(self) -> bool:
            """True when the handler holds a model AND the patcher holds weight.

            Core defines no ``is_loaded`` on either patcher class (it is a
            nodepack concept), so ``super().is_loaded`` would raise
            AttributeError on the property access. The load-bearing expression
            is the reference check plus ``loaded_size()``, which core defines
            per class: legacy returns ``model_loaded_weight_memory``
            (comfy/model_patcher.py:412-413) and dynamic returns
            ``vbar.loaded_size() + model_loaded_weight_memory`` (:1815-1817).
            A never-loaded dynamic patcher has no vbar, so this is 0.
            """
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
            """Dynamic load entry point. Weights are paged by the vbar, not moved.

            ``ModelPatcherDynamic.patch_model`` asserts ``not load_weights``
            (comfy/model_patcher.py:2132-2137); its own docstring tells custom
            node authors to call ``load_models_gpu()`` instead. We raise a
            named error rather than a bare assert so the failure is actionable.
            """
            if load_weights:
                raise ValueError(
                    f"{type(self).__name__}.patch_model(load_weights=True) is not supported: "
                    "ModelPatcherDynamic.patch_model asserts `not load_weights` "
                    "(comfy/model_patcher.py:2132-2137). Use "
                    "comfy.model_management.load_models_gpu([patcher]) instead."
                )

            target_device = self.load_device if device_to is None else device_to

            # Lazy build, verbatim from VibeVoicePatcher.patch_model (:119-134).
            if self.model.model is None:
                logger.info(
                    f"Loading VibeVoice models for '{self.model.model_pack_name}' to {target_device}..."
                )
                self.model.load_model(target_device, attention_mode=self.attention_mode)

            # dtype guard, verbatim from VibeVoicePatcher.patch_model (:147-156).
            if self.target_dtype is not None:
                current_dtype = representative_dtype(self.model.model)
                if current_dtype != self.target_dtype:
                    logger.debug(
                        f"Casting model to dtype: {self.target_dtype} (current: {current_dtype})"
                    )
                    cast_model_to_dtype(self.model.model, self.target_dtype)

            # device_to is forced to self.load_device: ModelPatcherDynamic.load
            # asserts ``device_to == self.load_device`` (comfy/model_patcher.py:1870)
            # and core reaches it through load_models_gpu -> model_load ->
            # partially_load with its own device.
            return super().patch_model(
                device_to=self.load_device,
                lowvram_model_memory=lowvram_model_memory,
                load_weights=False,
                force_patch_weights=force_patch_weights,
            )

        def unpatch_model(self, device_to=None, unpatch_weights=True, warm: bool = False,
                          destroy: bool = False, *args, **kwargs):
            """Consume the pack's ``warm``/``destroy`` flags, then defer to core.

            ``ModelPatcherDynamic.unpatch_model`` (comfy/model_patcher.py:2140)
            has no ``warm``/``destroy`` parameters at all, so the pack's five
            call sites would raise TypeError without this adapter.
            """
            if unpatch_weights and warm:
                # There is no "move the tensors to intermediate_device()" notion
                # under vbar: the vbar owns placement. Releasing the pins IS the
                # warm release (comfy/model_patcher.py:1827-1828).
                unpin_all = getattr(self, 'unpin_all_weights', None)
                if callable(unpin_all):
                    logger.debug(
                        f"Warm offloading VibeVoice models for '{self.model.model_pack_name}' "
                        f"({self.attention_mode}) (pins released)..."
                    )
                    unpin_all()

            if unpatch_weights and destroy:
                # Destructive sequence, verbatim from
                # VibeVoicePatcher.unpatch_model (:204-231).
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
    """Pick the patcher protocol from core's own availability, nothing else.

    2026-09-30: the string-equality whitelist this function used to carry is
    GONE. It read ``weight_family == "dense"`` and excluded ``gguf_block`` /
    ``convrot_int8`` / ``fp8_resident`` as a "stop condition", on the premise
    that those families "stream natively" and gain nothing from paging. The
    premise was wrong and the exclusion is what produced the measured
    behaviour: the quant routes CLONE every tensor into host RAM and then
    fully H2D it, because a legacy patcher has nowhere to page from. The
    live 7B fp8 load staged ~9.3 GB of private clones (20->40 GB with the
    read layer) while core's own Krea node stages nothing.

    What actually decides the protocol upstream is core, not this pack:
    ``main.py:300-301`` rebinds ``CoreModelPatcher = ModelPatcherDynamic``
    and sets ``aimdo_enabled`` when the native stack initialises, and
    ``comfy/sd.py:2403-2407`` then loads every model with
    ``assign=is_dynamic()``. This function now does exactly what ``sd.py``
    does — take the dynamic class iff core resolved one for a CUDA target,
    for EVERY family — and the loaders keep the file views when (and only
    when) that class is dynamic, so the parameters are the file mapping and
    the dynamic patcher pages them disk->VRAM at forward.

    ``weight_family`` is consulted for exactly ONE family, and for a measured
    reason rather than a policy: ``gguf_block`` still installs its raw blocks
    through a copy (``GGUFTensor.from_reader_tensor`` clones the reader's
    numpy view into private memory, then the route unmaps the file), so the
    dynamic patcher would page from host RAM — no RAM win, changed offload
    order. It stays legacy until that install path keeps views; fp8-resident
    and convrot-int8, whose safetensors streams DO keep aimdo views, are on
    the dynamic route.

    ``legacy_cls`` is a parameter so the TTS and ASR call sites keep their own
    patcher classes (and therefore their own model caches).
    """
    if weight_family == "gguf_block":
        return legacy_cls
    if dynamic_vram_available(resolve_core_patcher_class(), load_device):
        return make_dynamic_patcher_class(legacy_cls)
    return legacy_cls


def _ensure_device_index(patcher) -> None:
    """Give an accelerator load device the explicit index core's vbar needs."""
    device = getattr(patcher, "load_device", None)
    if device is None or getattr(device, "index", 0) is not None:
        return
    if getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        return
    patcher.load_device = torch.device("cuda", torch.cuda.current_device())


def load_to_device(patcher, memory_required: int = 0):
    """Put ``patcher`` on its load device under whichever protocol it speaks.

    Dynamic patchers must go through ``load_models_gpu``: that is the path core
    documents for DynamicVRAM, it is the path that pages weights through the
    vbar, and it is the path that keeps DynamicVRAM's all-or-nothing free
    accounting (comfy/model_management.py:962-965) intact. Legacy patchers keep
    the exact call shape they have always had.

    Dispatch is on the INSTANCE METHOD ``patcher.is_dynamic()`` — the same
    predicate core itself uses at comfy/model_management.py:963 — and never on
    ``isinstance``: ``ModelPatcherDynamic.__new__``
    (comfy/model_patcher.py:1754-1757) reroutes a CPU load_device to a plain
    ``ModelPatcher``, so a "dynamic" instance is not necessarily an instance of
    the dynamic subclass.

    ``patcher`` is a live local for the whole call: core's ``LoadedModel`` keeps
    only a weakref (comfy/model_management.py:781) plus a finalizer, so the
    caller must keep the object alive across the call. Never pass
    ``force_patch_weights``/``force_full_load`` here — both assert inside the
    dynamic path (:1864, :1868).

    The load device is given an explicit index before the call. ``torch.device
    ("cuda")`` carries ``index is None``, and the dynamic path turns that
    straight into ``ModelVBAR(self.model_size() * 10, self.load_device.index)``
    (comfy/model_patcher.py:1811) → ``int(None)``. Production never trips it
    because ``get_torch_device()`` returns ``cuda:0``, but a caller that names
    the family device by hand does, and the failure then surfaces deep inside
    aimdo rather than at the boundary we own.
    """
    _ensure_device_index(patcher)
    is_dynamic = getattr(patcher, 'is_dynamic', None)
    # `is True`, not `bool(...)`: core's is_dynamic() returns a literal True or
    # False (comfy/model_patcher.py:403-404 and :1803-1804). Anything that is
    # not POSITIVELY declaring itself dynamic -- a MagicMock in a test, a
    # third-party patcher without the method -- stays on the legacy call shape
    # rather than being routed into a protocol it does not implement.
    dynamic = is_dynamic() is True if callable(is_dynamic) else False
    if dynamic:
        model_management.load_models_gpu([patcher], memory_required=memory_required)
    else:
        model_management.load_model_gpu(patcher)
    return patcher
