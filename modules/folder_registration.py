"""ComfyUI folder-registration helpers for VibeVoice model assets.

The helpers accept ``folder_paths`` explicitly so registration behavior can be
unit tested without importing the package startup path or starting ComfyUI.
"""

from __future__ import annotations

import os
from typing import Any


TTS_FOLDER_KEY = "tts"
VOICE_PRESET_FOLDER_KEY = "vibevoice_voices"
VOICE_PRESET_SUBDIR = os.path.join("VibeVoice", "voices")


def _append_unique_path(paths: list[str], candidate: str) -> bool:
    """Append a normalized path once, preserving registration order."""
    normalized = os.path.abspath(os.path.normpath(candidate))
    existing = {os.path.abspath(os.path.normpath(path)) for path in paths}
    if normalized in existing:
        return False
    paths.append(normalized)
    return True


def register_vibevoice_folders(folder_paths_module: Any) -> list[str]:
    """Register the primary TTS root and return all registered TTS roots.

    Existing ComfyUI registrations are preserved. Calling the helper more than
    once does not add duplicate paths.
    """
    models_dir = os.fspath(folder_paths_module.models_dir)
    primary_tts_root = os.path.join(models_dir, "tts")
    folder_names_and_paths = folder_paths_module.folder_names_and_paths

    if TTS_FOLDER_KEY not in folder_names_and_paths:
        supported_exts = set(folder_paths_module.supported_pt_extensions)
        supported_exts.update({".safetensors", ".json"})
        folder_names_and_paths[TTS_FOLDER_KEY] = ([], supported_exts)

    registered_paths = folder_names_and_paths[TTS_FOLDER_KEY][0]
    _append_unique_path(registered_paths, primary_tts_root)
    return list(registered_paths)


def register_voice_preset_folder(
    folder_paths_module: Any,
    primary_tts_root: str,
    additional_tts_roots: list[str] | None = None,
) -> list[str]:
    """Register ``<TTS root>/VibeVoice/voices`` for cached voice prompts.

    Every TTS root ComfyUI knows about gets a candidate ``voices`` directory,
    not just the primary one. A machine can resolve models across several roots
    at once (the primary ``models/tts`` plus whatever ``extra_model_paths.yaml``
    contributes), and the prompts may sit under any of them. Registering only
    the first root made ``vibevoice_voices`` resolve to a directory that does
    not exist, so the node reported every preset as missing even though the
    prompts were installed under a different root.

    Directories that do not exist are still registered: ComfyUI populates the
    folder list at startup, and a root may be populated later. Discovery skips
    non-existent paths, so this costs nothing.
    """
    folder_names_and_paths = folder_paths_module.folder_names_and_paths
    if VOICE_PRESET_FOLDER_KEY not in folder_names_and_paths:
        folder_names_and_paths[VOICE_PRESET_FOLDER_KEY] = ([], {".pt"})

    registered_paths = folder_names_and_paths[VOICE_PRESET_FOLDER_KEY][0]
    roots = [primary_tts_root, *(additional_tts_roots or [])]
    for root in roots:
        if not root:
            continue
        voice_root = os.path.abspath(
            os.path.join(os.fspath(root), VOICE_PRESET_SUBDIR)
        )
        _append_unique_path(registered_paths, voice_root)
    return list(registered_paths)
