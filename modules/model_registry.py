"""Active-model registry: unload-before-load eviction on model change.

Single source of truth for tracking which VibeVoice model is currently active
per family (``"tts"`` / ``"asr"``) and for fully releasing a superseded model
(RAM + VRAM + ComfyUI's ``model_management.current_loaded_models`` registry)
BEFORE any allocation for a replacement begins.

Single-active-bundle semantics: at most one live bundle per cache key.

Design notes:

- ``comfy.model_management`` is imported lazily inside the functions so this
  module stays importable under the CPU-only test stubs (conftest) and never
  participates in an import cycle with ``modules.patcher`` / ``modules.loader``
  (patcher caches are passed in by the callers).
- Every destructive sub-step is individually exception-guarded: a failure in
  one step must never block the remaining release steps (a half-released model
  is still better than a leaked one, and warnings surface the failure).
- The BUNDLE registry exists because the patcher cache is NOT the only holder
  of the weights. The external loader node returns the model bundle as a node
  OUTPUT (``nodes/external_loader_node.VibeVoiceModel.Output``), and ComfyUI's
  execution cache keeps that dict STRONGLY (``execution.CacheEntry.outputs``)
  until the node leaves the prompt or re-executes. So popping our own
  patcher-cache entry frees nothing: the cached bundle still references the
  live ``nn.Module`` and its GPU/CPU tensors. Eviction must therefore
  NEUTRALIZE the bundle (null its heavy fields) — see
  :func:`register_model_bundle` / :func:`release_model_bundles`.
"""

import gc
import logging
import os

logger = logging.getLogger(__name__)

# Model families tracked by :func:`set_active` / :func:`evict_if_changed`.
FAMILY_TTS = "tts"
FAMILY_ASR = "asr"

# family -> cache_key of the currently active model.
_ACTIVE_KEYS: dict = {}

# Fields of a model bundle that hold the heavy references (the instantiated
# nn.Module and its processor). Nulling exactly these two drops the weights
# while leaving the plain-string identity fields debuggable.
_BUNDLE_HEAVY_FIELDS = ("model", "processor")

# patcher cache_key -> the ONE live model bundle registered for that key.
# Popping/clearing this dict alone frees nothing (ComfyUI's output cache owns
# a strong ref to the bundle); release_model_bundles() nulls the heavy fields
# of the bundle it removes.
_BUNDLE_REGISTRY: dict = {}


def _neutralize_bundle(bundle) -> None:
    """Null a bundle's heavy fields, per-field guarded, never raising.

    The identity fields (``model_name``, ``source_path``, ``attention_mode``,
    ``use_llm_4bit``, ``dtype_str``, ``is_asr``, ...) are intentionally left
    intact so the dict ComfyUI keeps cached stays small and inspectable — and
    so a still-connected consumer of a DIFFERENT key keeps working.
    """
    if bundle is None:
        return
    for field in _BUNDLE_HEAVY_FIELDS:
        try:
            bundle[field] = None
        except Exception as e:
            logger.warning(
                f"model_registry: could not null bundle field '{field}': {e}"
            )


def register_model_bundle(cache_key: str, bundle) -> None:
    """Record the live model bundle for ``cache_key``.

    Single-slot-per-key semantics: if a DIFFERENT bundle was already registered
    under ``cache_key`` it is neutralized first, then replaced. That self-heals
    the external -> external re-execution case, where the loader node produces
    a brand-new bundle dict for weights that are already cached. Re-registering
    the SAME dict (the reused-patcher path) is a no-op replace — the live
    bundle is never neutralized out from under its own consumer.

    Never raises: registration is bookkeeping and must not break loading.
    """
    try:
        previous = _BUNDLE_REGISTRY.get(cache_key)
        if previous is not None and previous is not bundle:
            _neutralize_bundle(previous)
        _BUNDLE_REGISTRY[cache_key] = bundle
    except Exception as e:
        logger.warning(
            f"model_registry.register_model_bundle({cache_key!r}) failed: {e}"
        )


def release_model_bundles(cache_key: str) -> int:
    """Neutralize the bundle registered under ``cache_key`` and drop it.

    Only the given key is touched, so a consumer still connected to a DIFFERENT
    key is never collateral damage.

    Args:
        cache_key: The patcher-cache key whose bundle is being evicted.

    Returns:
        ``1`` if a bundle was released, ``0`` if nothing was registered.
    """
    try:
        bundle = _BUNDLE_REGISTRY.pop(cache_key, None)
    except Exception as e:
        logger.warning(
            f"model_registry.release_model_bundles({cache_key!r}) failed: {e}"
        )
        return 0
    if bundle is None:
        return 0
    _neutralize_bundle(bundle)
    return 1


def clear_bundle_registry() -> None:
    """Forget all bundle registrations (test isolation helper)."""
    _BUNDLE_REGISTRY.clear()


def set_active(family: str, cache_key: str) -> None:
    """Record ``cache_key`` as the active model for ``family``.

    Args:
        family: One of :data:`FAMILY_TTS` / :data:`FAMILY_ASR`.
        cache_key: The patcher-cache key identifying the active model.
    """
    _ACTIVE_KEYS[family] = cache_key


def get_active(family: str) -> "str | None":
    """Return the active cache key for ``family``, or ``None``."""
    return _ACTIVE_KEYS.get(family)


def clear_active_keys() -> None:
    """Forget all active-key bookkeeping (test isolation helper)."""
    _ACTIVE_KEYS.clear()


def identity_for_external(
    weight_path: str,
    config_name: str,
    attention_mode: str,
    use_llm_4bit: bool = False,
    dtype_str: str = "auto",
    prefix: str = "external",
) -> str:
    """Build the file-identity-aware cache key for an externally-loaded model.

    The key captures everything that changes the built weights: the weight
    file itself (name + mtime_ns + size), the architecture config selector,
    the RESOLVED attention mode, the 4-bit flag, and the requested dtype.
    Different weight files (even sharing a config_name) never collide, and
    re-running the same file + settings yields the identical key so the cached
    patcher is reused without a reload.

    Uses only the file's *basename* so no filesystem separators (:, /, \\)
    leak into the key.

    Args:
        weight_path: Absolute path to the weight file. May be unreadable
            (e.g. hand-built test bundles); the identity then degrades to a
            stable placeholder instead of raising.
        config_name: Architecture config selector (e.g. "VibeVoice-1.5B").
        attention_mode: The RESOLVED attention mode (post-fallback).
        use_llm_4bit: Whether the LLM was quantized to 4-bit.
        dtype_str: The requested dtype string ("auto"/"bf16"/"fp16"/"fp32").
        prefix: Key namespace ("external" for TTS, "asr_external" for ASR).

    Returns:
        Deterministic cache-key string.
    """
    basename = os.path.basename(weight_path) if weight_path else ""
    mtime_ns = "0"
    size = "0"
    try:
        if weight_path and os.path.isfile(weight_path):
            stat = os.stat(weight_path)
            mtime_ns = str(stat.st_mtime_ns)
            size = str(stat.st_size)
    except OSError:
        # Unreadable/missing file: degrade to placeholders (deterministic).
        pass

    return (
        f"{prefix}_{config_name}"
        f"@{basename}@{mtime_ns}@{size}"
        f"_attn_{attention_mode}"
        f"_q4_{int(bool(use_llm_4bit))}"
        f"_dtype_{dtype_str}"
    )


def unregister_from_comfy(patcher) -> list:
    """Remove ``patcher`` from ComfyUI's loaded-model registry.

    Iterates ``comfy.model_management.current_loaded_models`` and drops every
    entry whose wrapped model IS ``patcher``, detaching both finalizers first
    so neither ComfyUI's GC hooks nor the entry itself can resurrect or further
    track the dying patcher.

    Args:
        patcher: The ModelPatcher being evicted.

    Returns:
        The list of removed ``LoadedModel`` entries (for logging/tests).
    """
    removed = []
    try:
        import comfy.model_management as model_management
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"model_registry.unregister_from_comfy: cannot import comfy.model_management: {e}")
        return removed

    loaded_list = getattr(model_management, "current_loaded_models", None)
    if loaded_list is None:
        return removed

    survivors = []
    for loaded in list(loaded_list):
        try:
            if loaded.model is not patcher:
                survivors.append(loaded)
                continue
        except Exception:
            survivors.append(loaded)
            continue

        # Detach both finalizers before dropping the entry so their callbacks
        # never fire against the destroyed patcher/model.
        for attr in ("model_finalizer", "_patcher_finalizer"):
            finalizer = getattr(loaded, attr, None)
            if finalizer is not None:
                try:
                    finalizer.detach()
                except Exception:
                    pass
                try:
                    setattr(loaded, attr, None)
                except Exception:
                    pass
        try:
            loaded.real_model = None
        except Exception:
            pass
        removed.append(loaded)

    if removed:
        loaded_list[:] = survivors
        logger.debug(
            f"Unregistered {len(removed)} ComfyUI loaded-model entr(y/ies) for "
            f"patcher {getattr(patcher, 'cache_key', '<unknown>')}"
        )
    return removed


def evict_patcher(patcher, cache_dict: dict, key: str) -> list:
    """Atomically and destructively evict one patcher.

    Ordered release sequence (each step individually exception-guarded):

        1. :func:`unregister_from_comfy` — drop from ComfyUI's registry.
        2. ``patcher.unpatch_model(unpatch_weights=True, destroy=True)`` — null
           handler refs, evict the model cache entry, free tensors.
        3. ``cache_dict.pop(key)`` — remove the patcher-cache entry itself.
        4. :func:`release_model_bundles` — neutralize the model bundle
           registered under ``key`` (ComfyUI's output cache still holds it
           strongly, so the heavy fields must be nulled explicitly).
        5. ``gc.collect()`` — prompt the release of nulled references.
        6. ``model_management.soft_empty_cache()`` — release cached VRAM.

    Args:
        patcher: The ModelPatcher to evict (may be a dead object reference).
        cache_dict: The patcher cache dict holding ``key`` (or ``None``).
        key: The cache key under which ``patcher`` is stored.

    Returns:
        List of human-readable error strings for steps that failed (empty on
        full success). Eviction never raises.
    """
    errors = []

    try:
        unregister_from_comfy(patcher)
    except Exception as e:
        errors.append(f"unregister_from_comfy failed: {e}")

    try:
        patcher.unpatch_model(unpatch_weights=True, destroy=True)
    except Exception as e:
        errors.append(f"unpatch_model(destroy=True) failed: {e}")

    try:
        if cache_dict is not None:
            cache_dict.pop(key, None)
    except Exception as e:
        errors.append(f"cache pop failed: {e}")

    # ComfyUI's execution cache still holds the loader node's output bundle
    # strongly, so popping the patcher entry alone leaks the weights.
    try:
        release_model_bundles(key)
    except Exception as e:
        errors.append(f"release_model_bundles failed: {e}")

    try:
        gc.collect()
    except Exception as e:
        errors.append(f"gc.collect failed: {e}")

    try:
        import comfy.model_management as model_management

        model_management.soft_empty_cache()
    except Exception as e:
        errors.append(f"soft_empty_cache failed: {e}")

    for err in errors:
        logger.warning(f"model_registry.evict_patcher({key}): {err}")
    return errors


def evict_if_changed(family: str, new_key: str, patcher_caches) -> list:
    """Evict the previous active model of ``family`` when the key changed.

    Single-active-model-per-family gate: call this BEFORE building/loading a
    new model. When the recorded active key differs from ``new_key`` (including
    the no-active-yet case), every patcher in ``patcher_caches`` whose key is
    not ``new_key`` is destroyed via :func:`evict_patcher`, and ``new_key``
    becomes the active key. A same-key call is a strict no-op.

    Args:
        family: :data:`FAMILY_TTS` or :data:`FAMILY_ASR`.
        new_key: Cache key of the model about to be loaded.
        patcher_caches: Iterable of patcher-cache dicts to sweep (only the
            given dicts are touched — TTS eviction can never touch ASR caches
            and vice versa).

    Returns:
        The list of evicted cache keys (empty when nothing changed).
    """
    active = get_active(family)
    if active == new_key:
        return []

    evicted = []
    for cache_dict in patcher_caches:
        if cache_dict is None:
            continue
        for key in [k for k in list(cache_dict.keys()) if k != new_key]:
            patcher = cache_dict.get(key)
            logger.debug(
                f"Model changed for family '{family}' "
                f"(active={active!r} -> {new_key!r}); evicting '{key}'..."
            )
            evict_patcher(patcher, cache_dict, key)
            evicted.append(key)

    set_active(family, new_key)
    return evicted
