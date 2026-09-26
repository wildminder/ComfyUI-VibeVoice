"""ComfyUI-VibeVoice: Text-to-speech custom node integrating the VibeVoice model family.

This is the package entrypoint loaded by ComfyUI. It handles:
- Environment detection (ComfyUI available or not)
- ComfyUI folder_paths registration for tts models
- Model discovery at startup (official + local)

All heavy logic is delegated to focused modules:
- modules/model_info.py: Model configs, registry, and scanning
- modules/loader.py: Model loading and caching
- modules/patcher.py: ModelPatcher subclass for VRAM management
- modules/generation.py: Audio generation orchestration
"""

import os
import sys
import logging

# ── Diffusers Compatibility Patch ─────────────────────────────────────
# Apply before any imports that might trigger the vendored code.
# The vendored dpm_solver.py imports from diffusers, which may fail if
# huggingface_hub is too new (cached_download was removed).
try:
    from huggingface_hub import cached_download
except ImportError:
    import huggingface_hub
    import warnings

    def cached_download(*args, **kwargs):
        """Compatibility shim for removed cached_download function."""
        warnings.warn(
            "cached_download is deprecated and removed from huggingface_hub. "
            "Please update diffusers to a newer version.",
            DeprecationWarning,
            stacklevel=2,
        )
        return huggingface_hub.hf_hub_download(*args, **kwargs)

    huggingface_hub.cached_download = cached_download

# ── Transformers Compatibility ────────────────────────────────────────
# The vendored code uses a compatibility layer (src/vibevoice/modular/transformers_compat.py)
# to handle different transformers versions (4.x vs 5.x).

# ── Pytest Guard ──────────────────────────────────────────────────────
# Pytest forcefully imports __init__ out-of-context during test collection.
# This causes relative imports to crash. Detect and exit early.
if "pytest" in sys.modules:
    __all__ = []

else:
    logger = logging.getLogger(__name__)

    # ── Environment Detection ─────────────────────────────────────────
    try:
        import folder_paths
        _COMFYUI_AVAILABLE = True
    except ImportError:
        folder_paths = None
        _COMFYUI_AVAILABLE = False

    # ── Package Imports ───────────────────────────────────────────────
    from .modules.model_info import (
        AVAILABLE_VIBEVOICE_MODELS, MODEL_CONFIGS, scan_vibevoice_models,
    )
    from .modules.folder_registration import (
        register_vibevoice_folders, register_voice_preset_folder,
    )

    # ── Logger Setup ──────────────────────────────────────────────────
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[ComfyUI-VibeVoice] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # ── sys.path Registration ─────────────────────────────────────────
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    # ── ComfyUI Integration ───────────────────────────────────────────
    if _COMFYUI_AVAILABLE:
        # Register tts folder path with ComfyUI
        VIBEVOICE_SUBDIR_NAME = "VibeVoice"
        primary_vibevoice_models_path = os.path.join(folder_paths.models_dir, "tts", VIBEVOICE_SUBDIR_NAME)
        os.makedirs(primary_vibevoice_models_path, exist_ok=True)

        registered_tts_roots = register_vibevoice_folders(folder_paths)
        # Every TTS root, not just the primary: extra_model_paths.yaml can add
        # roots that hold the voice prompts while the primary one is empty.
        register_voice_preset_folder(
            folder_paths, registered_tts_roots[0], registered_tts_roots[1:]
        )

        # Populate AVAILABLE_VIBEVOICE_MODELS with official models
        for model_name, config in MODEL_CONFIGS.items():
            AVAILABLE_VIBEVOICE_MODELS[model_name] = {
                "type": "official",
                "repo_id": config["repo_id"],
                "tokenizer_repo": ("Qwen/Qwen2.5-7B" if ("Large" in model_name
                                    or "7B" in model_name or "ASR" in model_name)
                   else "Qwen/Qwen2.5-1.5B")
            }

        # Discover local models in tts/VibeVoice/ subdirectories
        vibevoice_search_paths = []
        for tts_folder in folder_paths.get_folder_paths("tts"):
            potential_path = os.path.join(tts_folder, VIBEVOICE_SUBDIR_NAME)
            if os.path.isdir(potential_path) and potential_path not in vibevoice_search_paths:
                vibevoice_search_paths.append(potential_path)

        # Add the primary path just in case it wasn't registered
        if primary_vibevoice_models_path not in vibevoice_search_paths:
            vibevoice_search_paths.insert(0, primary_vibevoice_models_path)

        for search_path in vibevoice_search_paths:
            logger.info(f"Scanning for VibeVoice models in: {search_path}")
            if not os.path.isdir(search_path):
                continue
            for model_info in scan_vibevoice_models(search_path):
                item = model_info["name"]
                if item not in AVAILABLE_VIBEVOICE_MODELS:
                    AVAILABLE_VIBEVOICE_MODELS[item] = {
                        "type": model_info["type"],
                        "path": model_info["path"],
                        "tokenizer_repo": model_info.get("tokenizer_repo", "Qwen/Qwen2.5-1.5B"),
                    }

        logger.info(f"Discovered VibeVoice models: {sorted(list(AVAILABLE_VIBEVOICE_MODELS.keys()))}")

    # ── Exports ───────────────────────────────────────────────────────
    from .vibevoice_nodes import comfy_entrypoint

    __all__ = ['comfy_entrypoint']
