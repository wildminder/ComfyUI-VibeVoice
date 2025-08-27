import os
import sys
import logging

# allowing absolute imports like 'from vibevoice.modular...' to work.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import folder_paths

from .vibevoice_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(f"[ComfyUI-VibeVoice] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


VIBEVOICE_MODEL_SUBDIR = os.path.join("tts", "VibeVoice")

vibevoice_models_full_path = os.path.join(folder_paths.models_dir, VIBEVOICE_MODEL_SUBDIR)
os.makedirs(vibevoice_models_full_path, exist_ok=True)

# Register the tts/VibeVoice path with ComfyUI
tts_path = os.path.join(folder_paths.models_dir, "tts")
if "tts" not in folder_paths.folder_names_and_paths:
    supported_exts = folder_paths.supported_pt_extensions.union({".safetensors", ".json"})
    folder_paths.folder_names_and_paths["tts"] = ([tts_path], supported_exts)
else:
    if tts_path not in folder_paths.folder_names_and_paths["tts"][0]:
        folder_paths.folder_names_and_paths["tts"][0].append(tts_path)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']