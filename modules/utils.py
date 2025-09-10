import re
import torch
import numpy as np
import random
import logging

from comfy.utils import ProgressBar
from comfy.model_management import throw_exception_if_processing_interrupted

try:
    import librosa
except ImportError:
    print("VibeVoice Node: `librosa` is not installed. Resampling of reference audio will not be available.")
    librosa = None

logger = logging.getLogger(__name__)

def set_vibevoice_seed(seed: int):
    """Sets the seed for torch, numpy, and random, handling large seeds for numpy."""
    if seed == 0:
        seed = random.randint(1, 0xffffffffffffffff)
    
    MAX_NUMPY_SEED = 2**32 - 1
    numpy_seed = seed % MAX_NUMPY_SEED
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(numpy_seed)
    random.seed(seed)

def parse_script_1_based(script: str) -> tuple[list[tuple[int, str]], list[int]]:
    """
    Parses a 1-based speaker script into a list of (speaker_id, text) tuples
    and a list of unique speaker IDs in the order of their first appearance.
    Internally, it converts speaker IDs to 0-based for the model.
    """
    parsed_lines = []
    speaker_ids_in_script = [] # This will store the 1-based IDs from the script
    for line in script.strip().split("\n"):
        if not (line := line.strip()): continue
        match = re.match(r'^Speaker\s+(\d+)\s*:\s*(.*)$', line, re.IGNORECASE)
        if match:
            speaker_id = int(match.group(1))
            if speaker_id < 1:
                logger.warning(f"Speaker ID must be 1 or greater. Skipping line: '{line}'")
                continue
            text = ' ' + match.group(2).strip()
            # Internally, the model expects 0-based indexing for speakers
            internal_speaker_id = speaker_id - 1
            parsed_lines.append((internal_speaker_id, text))
            if speaker_id not in speaker_ids_in_script:
                speaker_ids_in_script.append(speaker_id)
        else:
            logger.warning(f"Could not parse line, skipping: '{line}'")
    return parsed_lines, sorted(list(set(speaker_ids_in_script)))

def preprocess_comfy_audio(audio_dict: dict, target_sr: int = 24000) -> np.ndarray:
    """
    Converts a ComfyUI AUDIO dict to a mono NumPy array, resampling if necessary.
    """
    if not audio_dict: return None
    waveform_tensor = audio_dict.get('waveform')
    if waveform_tensor is None or waveform_tensor.numel() == 0: return None
    
    waveform = waveform_tensor[0].cpu().numpy()
    original_sr = audio_dict['sample_rate']
    
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=0)

    # Check for invalid values
    if np.any(np.isnan(waveform)) or np.any(np.isinf(waveform)):
        logger.error("Audio contains NaN or Inf values, replacing with zeros")
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure audio is not completely silent or has extreme values
    if np.all(waveform == 0):
        logger.warning("Audio waveform is completely silent")
    
    # Normalize extreme values
    max_val = np.abs(waveform).max()
    if max_val > 10.0:
        logger.warning(f"Audio values are very large (max: {max_val}), normalizing")
        waveform = waveform / max_val

    if original_sr != target_sr:
        if librosa is None:
            raise ImportError("`librosa` package is required for audio resampling. Please install it with `pip install librosa`.")
        logger.warning(f"Resampling reference audio from {original_sr}Hz to {target_sr}Hz.")
        waveform = librosa.resample(y=waveform, orig_sr=original_sr, target_sr=target_sr)
    
    # Final check after resampling
    if np.any(np.isnan(waveform)) or np.any(np.isinf(waveform)):
        logger.error("Audio contains NaN or Inf after resampling, replacing with zeros")
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        
    return waveform.astype(np.float32)

def check_for_interrupt():
    try:
        throw_exception_if_processing_interrupted()
        return False
    except:
        return True