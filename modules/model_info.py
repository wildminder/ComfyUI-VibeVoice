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
    "VibeVoice-7B": {
        "repo_id": "vibevoice/VibeVoice-7B",
        "size_gb": 17.4,
        "model_type": "tts",
    },
    "VibeVoice-Realtime-0.5B": {
        "repo_id": "microsoft/VibeVoice-Realtime-0.5B",
        "size_gb": 1.5,
        "model_type": "streaming_tts",
    },
    # Native transformers-5.3 checkpoint (model_type "vibevoice_asr"): the
    # same ASR-7B model as microsoft/VibeVoice-ASR, in HF-native form. The
    # streaming checkpoints (VibeVoice-ASR-Streaming-*) are chunked-protocol
    # models meant for live/streaming use; ComfyUI's batch pipeline semantics
    # fit the single-pass ASR-HF checkpoint instead.
    "VibeVoice-ASR-HF": {
        "repo_id": "microsoft/VibeVoice-ASR-HF",
        "size_gb": 17.4,
        "model_type": "asr",
    },
}

# Retired ASR dropdown options (repos microsoft/VibeVoice-ASR and the
# VibeVoice-ASR-Streaming-* family). Old saved workflows keep working by
# resolving these names onto the native ASR-HF checkpoint, unless a locally
# discovered checkpoint of the exact same name exists.
LEGACY_ASR_MODEL_NAMES = ("VibeVoice-ASR", "VibeVoice-ASR-Streaming-1.5B", "VibeVoice-ASR-Streaming-7B")
LEGACY_ASR_MODEL_TARGET = "VibeVoice-ASR-HF"


def normalize_asr_model_name(model_name: str, available: dict = None) -> str:
    """Map retired ASR model_names onto the canonical ASR-HF option.

    Exact options and locally discovered checkpoints of a retired name pass
    through unchanged; retired names map onto VibeVoice-ASR-HF (the same ASR
    model in its native transformers form).

    Args:
        model_name: Raw model_name value (may come from a saved workflow).
        available: Registry to check against (defaults to the live
            AVAILABLE_VIBEVOICE_MODELS; overridable for tests).

    Returns:
        Canonical model_name when a legacy alias applies, else the input.
    """
    if not model_name or model_name not in LEGACY_ASR_MODEL_NAMES:
        return model_name
    registry = AVAILABLE_VIBEVOICE_MODELS if available is None else available
    info = registry.get(model_name) or {}
    if info.get("type") in ("local_dir", "standalone"):
        return model_name
    if LEGACY_ASR_MODEL_TARGET in registry:
        return LEGACY_ASR_MODEL_TARGET
    return model_name


def _infer_model_type(name: str) -> str:
    """Resolve a model's family: MODEL_CONFIGS entry, else name-based.

    Locally discovered checkpoints are only in MODEL_CONFIGS when their
    directory name matches an official model; anything with "asr" in the
    name belongs to the ASR family (mirrors get_tokenizer_repo's heuristic).
    """
    config_type = MODEL_CONFIGS.get(name, {}).get("model_type")
    if config_type:
        return config_type
    name_lower = name.lower()
    if "asr" in name_lower:
        return "asr"
    if "realtime" in name_lower or "stream" in name_lower:
        return "streaming_tts"
    return "tts"

VOICE_PRESET_DIRECTORY_NAME = "voices"

# Exact extensions accepted for official-model weight files. ``.pt`` is
# intentionally excluded: in the official VibeVoice tree it denotes a realtime
# cached voice prompt, not a checkpoint. External checkpoint loading retains its
# independent ComfyUI-supported extension set, including ``.pt``.
MODEL_WEIGHT_EXTENSIONS = {
    ".safetensors",
    ".bin",
    ".gguf",
    ".ckpt",
    ".pth",
    ".pt2",
    ".pkl",
    ".sft",
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
    if "large" in name_lower or "asr" in name_lower or "7b" in name_lower:
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
        if _infer_model_type(name) == model_type:
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


def get_tts_family_models() -> dict:
    """Return standard TTS models first, then realtime TTS models.

    Registry order is preserved within each family. ASR models are excluded.
    """
    result = get_tts_models()
    result.update(get_streaming_tts_models())
    return result


# Descriptive alias for callers that treat the selector as the complete TTS
# registry rather than one family.
get_all_tts_models = get_tts_family_models


def is_model_type(model_name: str, *types: str) -> bool:
    """Return True if model_name's family is one of `types`.

    Families come from MODEL_CONFIGS when the name matches an official model,
    otherwise from the name itself ("asr" substring → asr, else tts).
    """
    return _infer_model_type(model_name) in types


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

    for item in sorted(items, key=str.casefold):
        item_path = os.path.join(search_path, item)

        # Case 1: HF directory with config + weights. A direct ``voices``
        # directory is an asset tree, never a model directory.
        if os.path.isdir(item_path):
            if item.casefold() == VOICE_PRESET_DIRECTORY_NAME:
                continue
            config_exists = os.path.exists(os.path.join(item_path, "config.json"))
            try:
                child_files = os.listdir(item_path)
            except OSError as exc:
                logger.warning(f"Cannot read model directory {item_path}: {exc}")
                continue
            weights_exist = (
                os.path.exists(os.path.join(item_path, "model.safetensors.index.json"))
                or any(
                    os.path.splitext(filename)[1].casefold() in MODEL_WEIGHT_EXTENSIONS
                    for filename in child_files
                )
            )

            if config_exists and weights_exist:
                results.append({
                    "name": item,
                    "path": item_path,
                    "type": "local_dir",
                    "tokenizer_repo": get_tokenizer_repo(item),
                })

        # Case 2: Standalone official-model weight file. ``.pt`` is reserved
        # for realtime voice prompts in this tree and is not scanned here.
        elif os.path.isfile(item_path):
            extension = os.path.splitext(item)[1].casefold()
            if extension in MODEL_WEIGHT_EXTENSIONS:
                model_name = os.path.splitext(item)[0]
                results.append({
                    "name": model_name,
                    "path": item_path,
                    "type": "standalone",
                    "tokenizer_repo": get_tokenizer_repo(model_name),
                })

    return results
