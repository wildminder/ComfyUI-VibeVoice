import os
import sys
import logging
import folder_paths
import json

try:
    import sageattention
    SAGE_ATTENTION_AVAILABLE = True
except ImportError:
    SAGE_ATTENTION_AVAILABLE = False

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from .modules.model_info import AVAILABLE_VIBEVOICE_MODELS, MODEL_CONFIGS

# Configure a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f"[ComfyUI-VibeVoice] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# This is just the *name* of the subdirectory, not the full path.
VIBEVOICE_SUBDIR_NAME = "VibeVoice"

# This is the *primary* path where official models will be downloaded.
primary_vibevoice_models_path = os.path.join(folder_paths.models_dir, "tts", VIBEVOICE_SUBDIR_NAME)
os.makedirs(primary_vibevoice_models_path, exist_ok=True)

# Register the tts path type with ComfyUI so get_folder_paths works
tts_path = os.path.join(folder_paths.models_dir, "tts")
if "tts" not in folder_paths.folder_names_and_paths:
    supported_exts = folder_paths.supported_pt_extensions.union({".safetensors", ".json"})
    folder_paths.folder_names_and_paths["tts"] = ([tts_path], supported_exts)
else:
    # Ensure the default path is in the list if it's not already
    if tts_path not in folder_paths.folder_names_and_paths["tts"][0]:
        folder_paths.folder_names_and_paths["tts"][0].append(tts_path)

# The logic for dynamic model discovery
# ToDo: optimize finding

# official models that can be auto-downloaded
for model_name, config in MODEL_CONFIGS.items():
    AVAILABLE_VIBEVOICE_MODELS[model_name] = {
        "type": "official",
        "repo_id": config["repo_id"],
        "tokenizer_repo": "Qwen/Qwen2.5-7B" if "Large" in model_name else "Qwen/Qwen2.5-1.5B"
    }

# just workaround, default + custom
vibevoice_search_paths = []
# Use ComfyUI's API to get all registered 'tts' folders
for tts_folder in folder_paths.get_folder_paths("tts"):
    potential_path = os.path.join(tts_folder, VIBEVOICE_SUBDIR_NAME)
    if os.path.isdir(potential_path) and potential_path not in vibevoice_search_paths:
        vibevoice_search_paths.append(potential_path)

# Add the primary path just in case it wasn't registered for some reason
if primary_vibevoice_models_path not in vibevoice_search_paths:
     vibevoice_search_paths.insert(0, primary_vibevoice_models_path)

# Messy... Discover all local models in the search paths
for search_path in vibevoice_search_paths:
    logger.info(f"Scanning for VibeVoice models in: {search_path}")
    if not os.path.exists(search_path): continue
    for item in os.listdir(search_path):
        item_path = os.path.join(search_path, item)
        
        # Case 1: we have a standard HF directory
        if os.path.isdir(item_path):
            model_name = item
            if model_name in AVAILABLE_VIBEVOICE_MODELS: continue
            
            config_exists = os.path.exists(os.path.join(item_path, "config.json"))
            weights_exist = os.path.exists(os.path.join(item_path, "model.safetensors.index.json")) or any(f.endswith(('.safetensors', '.bin')) for f in os.listdir(item_path))
            
            if config_exists and weights_exist:
                tokenizer_repo = "Qwen/Qwen2.5-7B" if "large" in model_name.lower() else "Qwen/Qwen2.5-1.5B"
                AVAILABLE_VIBEVOICE_MODELS[model_name] = {
                    "type": "local_dir",
                    "path": item_path,
                    "tokenizer_repo": tokenizer_repo
                }

        # Case 2: Item is a standalone file
        elif os.path.isfile(item_path) and any(item.endswith(ext) for ext in folder_paths.supported_pt_extensions):
            model_name = os.path.splitext(item)[0]
            if model_name in AVAILABLE_VIBEVOICE_MODELS: continue
            
            tokenizer_repo = "Qwen/Qwen2.5-7B" if "large" in model_name.lower() else "Qwen/Qwen2.5-1.5B"
            AVAILABLE_VIBEVOICE_MODELS[model_name] = {
                "type": "standalone",
                "path": item_path,
                "tokenizer_repo": tokenizer_repo
            }

logger.info(f"Discovered VibeVoice models: {sorted(list(AVAILABLE_VIBEVOICE_MODELS.keys()))}")

from .vibevoice_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']