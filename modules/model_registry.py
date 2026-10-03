"""Active-model registry: unload-before-load eviction on model change.

Single source of truth for tracking which VibeVoice model is currently active
per family (``"tts"`` / ``"asr"``) and for fully releasing a superseded model
(RAM + VRAM + ComfyUI's ``model_management.current_loaded_models`` registry)
BEFORE any allocation for a replacement begins.

Single-active-bundle semantics: at most one live bundle per cache key.
"""

import gc
import logging
import os


# Model families tracked by :func:`set_active` / :func:`evict_if_changed`.
FAMILY_TTS = "tts"
FAMILY_ASR = "asr"

# family -> cache_key of the currently active model.
_ACTIVE_KEYS: dict = {}

# Fields of a model bundle that hold heavy references.
_BUNDLE_HEAVY_FIELDS = ("model", "processor")

# patcher cache_key -> the live model bundle registered for that key.
_BUNDLE_REGISTRY: dict = {}


def _neutralize_bundle(bundle) -> None:
    """Null a bundle's heavy fields, per-field guarded, never raising."""
    if bundle is None:
        return
    for field in _BUNDLE_HEAVY_FIELDS:
        try:
            bundle[field] = None
        except Exception as e:
            logging.warning(
                f"[VibeVoice TTS] model_registry: could not null bundle field '{field}': {e}"
            )


def register_model_bundle(cache_key: str, bundle) -> None:
    """Record the live model bundle for ``cache_key``."""
    try:
        previous = _BUNDLE_REGISTRY.get(cache_key)
        if previous is not None and previous is not bundle:
            _neutralize_bundle(previous)
        _BUNDLE_REGISTRY[cache_key] = bundle
    except Exception as e:
        logging.warning(
            f"[VibeVoice TTS] model_registry.register_model_bundle({cache_key!r}) failed: {e}"
        )


def release_model_bundles(cache_key: str) -> int:
    """Neutralize the bundle registered under ``cache_key`` and drop it."""
    try:
        bundle = _BUNDLE_REGISTRY.pop(cache_key, None)
    except Exception as e:
        logging.warning(
            f"[VibeVoice TTS] model_registry.release_model_bundles({cache_key!r}) failed: {e}"
        )
        return 0
    if bundle is None:
        return 0
    _neutralize_bundle(bundle)
    return 1


def get_live_bundle(cache_key: str):
    """Return the bundle registered under ``cache_key`` if it is still usable."""
    try:
        bundle = _BUNDLE_REGISTRY.get(cache_key)
    except Exception as e:
        logging.warning(
            f"[VibeVoice TTS] model_registry.get_live_bundle({cache_key!r}) failed: {e}"
        )
        return None
    if bundle is None:
        return None
    try:
        if bundle.get("model") is None:
            return None
    except Exception:
        return None
    return bundle


def clear_bundle_registry() -> None:
    """Forget all bundle registrations."""
    _BUNDLE_REGISTRY.clear()


def set_active(family: str, cache_key: str) -> None:
    """Record ``cache_key`` as the active model for ``family``."""
    _ACTIVE_KEYS[family] = cache_key


def get_active(family: str) -> "str | None":
    """Return the active cache key for ``family``, or ``None``."""
    return _ACTIVE_KEYS.get(family)


def clear_active_keys() -> None:
    """Forget all active-key bookkeeping."""
    _ACTIVE_KEYS.clear()


def identity_for_external(
    weight_path: str,
    config_name: str,
    attention_mode: str,
    use_llm_4bit: bool = False,
    dtype_str: str = "auto",
    prefix: str = "external",
) -> str:
    """Build the file-identity-aware cache key for an externally-loaded model."""
    basename = os.path.basename(weight_path) if weight_path else ""
    mtime_ns = "0"
    size = "0"
    try:
        if weight_path and os.path.isfile(weight_path):
            stat = os.stat(weight_path)
            mtime_ns = str(stat.st_mtime_ns)
            size = str(stat.st_size)
    except OSError:
        pass

    return (
        f"{prefix}_{config_name}"
        f"@{basename}@{mtime_ns}@{size}"
        f"_attn_{attention_mode}"
        f"_q4_{int(bool(use_llm_4bit))}"
        f"_dtype_{dtype_str}"
    )


def unregister_from_comfy(patcher) -> list:
    """Remove ``patcher`` from ComfyUI's loaded-model registry."""
    removed = []
    try:
        import comfy.model_management as model_management
    except Exception as e:
        logging.warning(f"[VibeVoice TTS] model_registry.unregister_from_comfy: cannot import comfy.model_management: {e}")
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
        logging.debug(
            f"[VibeVoice TTS] Unregistered {len(removed)} ComfyUI loaded-model entr(y/ies) for "
            f"patcher {getattr(patcher, 'cache_key', '<unknown>')}"
        )
    return removed


def evict_patcher(patcher, cache_dict: dict, key: str) -> list:
    """Atomically and destructively evict one patcher."""
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
        logging.warning(f"[VibeVoice TTS] model_registry.evict_patcher({key}): {err}")
    return errors


def evict_if_changed(family: str, new_key: str, patcher_caches) -> list:
    """Evict the previous active model of ``family`` when the key changed."""
    active = get_active(family)
    if active == new_key:
        return []

    evicted = []
    for cache_dict in patcher_caches:
        if cache_dict is None:
            continue
        for key in [k for k in list(cache_dict.keys()) if k != new_key]:
            patcher = cache_dict.get(key)
            logging.debug(
                f"[VibeVoice TTS] Model changed for family '{family}' "
                f"(active={active!r} -> {new_key!r}); evicting '{key}'..."
            )
            evict_patcher(patcher, cache_dict, key)
            evicted.append(key)

    set_active(family, new_key)
    return evicted