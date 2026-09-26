"""Discovery, safe loading, validation, and caching for realtime voice prompts.

Official ``VibeVoice-Realtime`` voice prompts are serialized cached LM/TTS-LM
outputs in ``.pt`` files. They are separate from checkpoint weights and are
never discovered or loaded as model files.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from collections.abc import Mapping
from typing import Any

import torch
import torch._weights_only_unpickler as _weights_only_unpickler
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from .folder_registration import VOICE_PRESET_FOLDER_KEY


logger = logging.getLogger(__name__)

VOICE_PRESET_SUBDIR = "voices"
_DEFAULT_VOICE_PRESET_SUBDIR = os.path.join("VibeVoice", VOICE_PRESET_SUBDIR)
PRESET_NONE = "None"
PRESET_CACHE_KEYS = ("lm", "tts_lm", "neg_lm", "neg_tts_lm")
VOICE_PRESET_EXTENSIONS = {".pt"}

# Classes deserialization is allowed to reconstruct. This matches the official
# realtime demo's allowlist: cached HF model outputs plus the dynamic KV cache.
PRESET_SAFE_GLOBALS = (BaseModelOutputWithPast, DynamicCache)

try:
    import folder_paths
except ImportError:  # pragma: no cover - ComfyUI is required in production.
    folder_paths = None

_CACHE: dict[tuple[str, int, int, str], dict[str, Any]] = {}


def _normalized_path(path: str) -> str:
    return os.path.abspath(os.path.normpath(os.fspath(path)))


def _deduplicate_paths(paths: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for path in paths:
        normalized = _normalized_path(path)
        key = os.path.normcase(normalized)
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def voice_preset_search_dirs() -> list[str]:
    """Return voice search roots in deterministic registration order.

    Explicit ``vibevoice_voices`` roots come first, followed by the default
    ``VibeVoice/voices`` directory beneath every registered TTS root. Folder
    registry failures are logged and treated as empty registries so schema
    construction cannot fail because of asset discovery.
    """
    if folder_paths is None:
        return []

    paths: list[str] = []
    try:
        registry = folder_paths.folder_names_and_paths
        for path in registry.get(VOICE_PRESET_FOLDER_KEY, ([], set()))[0]:
            paths.append(os.fspath(path))
    except Exception as exc:
        logger.warning("Cannot inspect registered VibeVoice voice folders: %s", exc)

    try:
        tts_roots = folder_paths.get_folder_paths("tts")
    except Exception as exc:
        logger.warning("Cannot inspect registered TTS folders: %s", exc)
        tts_roots = []

    for tts_root in tts_roots:
        paths.append(os.path.join(os.fspath(tts_root), _DEFAULT_VOICE_PRESET_SUBDIR))
    return _deduplicate_paths(paths)


def list_voice_presets(search_dirs: list[str] | None = None) -> dict[str, str]:
    """Recursively discover ``.pt`` prompts using first-registration collisions.

    File stems are compared case-insensitively. Within each root, paths are
    sorted by stem and absolute path. Across roots, the first matching stem
    wins. The returned mapping preserves the winning display stem.
    """
    roots = _deduplicate_paths(
        voice_preset_search_dirs() if search_dirs is None else list(search_dirs)
    )
    result: dict[str, str] = {}
    owners: dict[str, str] = {}

    for root in roots:
        if not os.path.isdir(root):
            continue
        try:
            candidates = []
            for directory, _subdirs, filenames in os.walk(root):
                for filename in filenames:
                    if os.path.splitext(filename)[1].casefold() != ".pt":
                        continue
                    path = _normalized_path(os.path.join(directory, filename))
                    candidates.append(path)
            candidates.sort(
                key=lambda path: (
                    os.path.splitext(os.path.basename(path))[0].casefold(),
                    os.path.normcase(path),
                )
            )
        except OSError as exc:
            logger.warning("Cannot scan VibeVoice voice folder '%s': %s", root, exc)
            continue

        for path in candidates:
            stem = os.path.splitext(os.path.basename(path))[0]
            collision_key = stem.casefold()
            if collision_key in owners:
                logger.warning(
                    "Ignoring duplicate VibeVoice voice preset '%s'; first "
                    "registration wins: '%s' then '%s'",
                    stem,
                    owners[collision_key],
                    path,
                )
                continue
            owners[collision_key] = path
            result[stem] = path
    return result


def resolve_voice_preset_path(name: str) -> str:
    """Resolve a preset name case-insensitively to an absolute path."""
    if not isinstance(name, str) or not name.strip():
        raise FileNotFoundError(
            f"Voice preset name is empty. Searched directories: "
            f"{voice_preset_search_dirs()}"
        )

    search_dirs = voice_preset_search_dirs()
    presets = list_voice_presets(search_dirs)
    selected_key = name.casefold()
    for preset_name, path in presets.items():
        if preset_name.casefold() == selected_key:
            return path
    raise FileNotFoundError(
        f"Voice preset '{name}' was not found. Searched directories: "
        f"{search_dirs}"
    )


def _validate_branch_value(value: Any, key: str, source: str) -> None:
    hidden_state = getattr(value, "last_hidden_state", None)
    if isinstance(value, Mapping) and hidden_state is None:
        raise ValueError(
            f"Invalid VibeVoice voice preset '{source}': key '{key}' must be a "
            "cached model output, not a plain mapping."
        )
    if hidden_state is None:
        raise ValueError(
            f"Invalid VibeVoice voice preset '{source}': key '{key}' is missing "
            "property 'last_hidden_state'."
        )
    if not isinstance(hidden_state, torch.Tensor):
        raise ValueError(
            f"Invalid VibeVoice voice preset '{source}': key '{key}' property "
            "'last_hidden_state' must be a torch.Tensor."
        )
    if hidden_state.ndim < 2 or hidden_state.shape[0] <= 0 or hidden_state.shape[1] <= 0:
        raise ValueError(
            f"Invalid VibeVoice voice preset '{source}': key '{key}' property "
            f"'last_hidden_state' has invalid sequence dimensions {tuple(hidden_state.shape)}."
        )


def validate_voice_preset(preset: object, source: str) -> None:
    """Validate a mapping containing all four official cached-output branches."""
    if not isinstance(preset, Mapping):
        raise ValueError(
            f"Invalid VibeVoice voice preset '{source}': expected a mapping, got "
            f"{type(preset).__name__}."
        )
    for key in PRESET_CACHE_KEYS:
        if key not in preset:
            raise ValueError(
                f"Invalid VibeVoice voice preset '{source}': missing key '{key}'."
            )
        _validate_branch_value(preset[key], key, source)


_WEIGHTS_ONLY_COMPAT_LOCK = threading.RLock()


@contextmanager
def _weights_only_mapping_compat():
    """Allow the approved HF mapping subclass in torch's weights-only loader.

    ``BaseModelOutputWithPast`` is an ``OrderedDict`` subclass, but the torch
    2.11 restricted unpickler checks mapping targets by exact type. The public
    ``torch.load(..., weights_only=True)`` API does not expose a hook for that
    check. Temporarily widening this one check to the two explicitly approved
    classes preserves the public API and leaves every other restricted opcode
    intact. The lock prevents concurrent preset loads from observing a
    half-patched unpickler.
    """
    unpickler = _weights_only_unpickler.Unpickler
    original = unpickler._check_set_item_target

    def _check_set_item_target(self, opcode: str) -> None:
        target = self.stack[-1]
        if type(target) in PRESET_SAFE_GLOBALS:
            return
        original(self, opcode)

    with _WEIGHTS_ONLY_COMPAT_LOCK:
        unpickler._check_set_item_target = _check_set_item_target
        try:
            yield
        finally:
            unpickler._check_set_item_target = original


def load_voice_preset(path: str, device: torch.device) -> dict[str, Any]:
    """Safely deserialize and validate one cached voice prompt.

    The official public API is used with ``weights_only=True`` inside the
    approved ``safe_globals`` context. The temporary mapping compatibility hook
    is limited to ``BaseModelOutputWithPast`` and ``DynamicCache``; arbitrary
    globals remain rejected by torch's restricted unpickler.
    """
    normalized = _normalized_path(path)
    target_device = torch.device(device)
    with _weights_only_mapping_compat(), torch.serialization.safe_globals(
        list(PRESET_SAFE_GLOBALS)
    ):
        preset = torch.load(
            normalized,
            map_location=target_device,
            weights_only=True,
        )
    validate_voice_preset(preset, normalized)
    return preset


def _cache_identity(
    path: str,
    device: torch.device,
) -> tuple[str, int, int, str]:
    stat = os.stat(path)
    return (
        _normalized_path(path),
        stat.st_mtime_ns,
        stat.st_size,
        str(torch.device(device)),
    )


def get_cached_voice_preset(name: str, device: torch.device) -> dict[str, Any]:
    """Return a cached preset, reloading on path metadata or device changes.

    The returned object is the shared cache object. Realtime generation owns a
    deep copy and must never mutate it in place.
    """
    path = resolve_voice_preset_path(name)
    key = _cache_identity(path, device)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    preset = load_voice_preset(path, device)
    _CACHE[key] = preset
    return preset


def clear_voice_preset_cache() -> None:
    """Clear all loaded voice-prompt objects."""
    _CACHE.clear()
