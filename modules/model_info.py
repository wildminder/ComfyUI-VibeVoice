"""Model configuration, registry, and discovery for VibeVoice models.

This module contains:
- MODEL_CONFIGS: Official, downloadable model configurations
- AVAILABLE_VIBEVOICE_MODELS: Runtime registry of all available models
- scan_vibevoice_models(): Discovers local models in a directory
- get_tokenizer_repo(): Determines the tokenizer repo for a model name
"""

import os
import logging

logger = logging.getLogger(__name__)

# Official, downloadable model configurations.
# Each config includes:
# - repo_id: HuggingFace repo ID for download
# - size_gb: Approximate model size in GB (for VRAM estimation)
# - model_type: "tts" (multi-speaker TTS), "streaming_tts" (realtime TTS), "asr" (speech recognition)
MODEL_CONFIGS = {
    "VibeVoice-1.5B": {
        "repo_id": "microsoft/VibeVoice-1.5B",
        "size_gb": 3.0,
        "model_type": "tts",
    },
    "VibeVoice-Large": {
        "repo_id": "aoi-ot/VibeVoice-Large",
        "size_gb": 17.4,
        "model_type": "tts",
    },
    "VibeVoice-Realtime-0.5B": {
        "repo_id": "microsoft/VibeVoice-Realtime-0.5B",
        "size_gb": 1.5,
        "model_type": "streaming_tts",
    },
    "VibeVoice-ASR": {
        "repo_id": "microsoft/VibeVoice-ASR",
        "size_gb": 15.0,
        "model_type": "asr",
    },
}

# Runtime registry of all available models (official + local).
# Populated at startup by __init__.py.
AVAILABLE_VIBEVOICE_MODELS = {}


def get_tokenizer_repo(model_name: str) -> str:
    """Determine the HuggingFace tokenizer repo for a model name.

    Large models use the 7B tokenizer; others use the 1.5B tokenizer.
    ASR models use the 7B tokenizer.

    Args:
        model_name: Name of the VibeVoice model.

    Returns:
        HuggingFace repo ID for the tokenizer.
    """
    name_lower = model_name.lower()
    if "large" in name_lower or "asr" in name_lower:
        return "Qwen/Qwen2.5-7B"
    return "Qwen/Qwen2.5-1.5B"


def get_models_by_type(model_type: str) -> dict:
    """Filter AVAILABLE_VIBEVOICE_MODELS by model type.

    Args:
        model_type: One of "tts", "streaming_tts", "asr".

    Returns:
        Dict of model_name -> model_info for models matching the type.
    """
    result = {}
    for name, info in AVAILABLE_VIBEVOICE_MODELS.items():
        config = MODEL_CONFIGS.get(name, {})
        if config.get("model_type", "tts") == model_type:
            result[name] = info
    return result


def get_tts_models() -> dict:
    """Get all TTS models (multi-speaker, non-streaming)."""
    return get_models_by_type("tts")


def get_streaming_tts_models() -> dict:
    """Get all streaming TTS models (realtime)."""
    return get_models_by_type("streaming_tts")


def get_asr_models() -> dict:
    """Get all ASR models."""
    return get_models_by_type("asr")


def is_model_type(model_name: str, *types: str) -> bool:
    """Return True if model_name's configured model_type is one of `types`.

    Unknown model names default to "tts" (consistent with get_models_by_type).
    """
    cfg = MODEL_CONFIGS.get(model_name, {})
    return cfg.get("model_type", "tts") in types


def scan_vibevoice_models(search_path: str) -> list[dict]:
    """Scan a directory for VibeVoice models.

    Detects two types of models:
    1. HF directory: A subdirectory containing config.json and weight files
       (.safetensors or .bin).
    2. Standalone file: A single weight file (.safetensors, .bin, etc.)

    Args:
        search_path: Directory to scan for models.

    Returns:
        List of dicts with keys: "name", "path", "type", "tokenizer_repo".
        "type" is "local_dir" for HF directories, "standalone" for files.
    """
    results = []

    if not os.path.isdir(search_path):
        return results

    try:
        items = os.listdir(search_path)
    except OSError as e:
        logger.warning(f"Cannot read directory {search_path}: {e}")
        return results

    # Import supported extensions from folder_paths if available,
    # otherwise use a sensible default set.
    try:
        import folder_paths
        supported_exts = folder_paths.supported_pt_extensions
    except ImportError:
        supported_exts = {".safetensors", ".bin", ".pt", ".ckpt", ".gguf"}

    for item in items:
        item_path = os.path.join(search_path, item)

        # Case 1: HF directory with config + weights
        if os.path.isdir(item_path):
            config_exists = os.path.exists(os.path.join(item_path, "config.json"))
            weights_exist = (
                os.path.exists(os.path.join(item_path, "model.safetensors.index.json"))
                or any(
                    f.endswith((".safetensors", ".bin"))
                    for f in os.listdir(item_path)
                )
            )

            if config_exists and weights_exist:
                results.append({
                    "name": item,
                    "path": item_path,
                    "type": "local_dir",
                    "tokenizer_repo": get_tokenizer_repo(item),
                })

        # Case 2: Standalone weight file
        elif os.path.isfile(item_path):
            if any(item.endswith(ext) for ext in supported_exts):
                model_name = os.path.splitext(item)[0]
                results.append({
                    "name": model_name,
                    "path": item_path,
                    "type": "standalone",
                    "tokenizer_repo": get_tokenizer_repo(model_name),
                })

    return results
