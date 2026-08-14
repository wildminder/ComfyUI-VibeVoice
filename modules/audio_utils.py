"""Audio and script utilities for VibeVoice nodes.

Contains:
- parse_script_1_based(): Parse multi-speaker script text
- preprocess_comfy_audio(): Convert ComfyUI AUDIO dict to mono numpy array
- extract_audio_tensor(): Extract waveform + sample_rate from audio dict
- set_seed(): Set random seeds for reproducibility
- check_for_interrupt(): Check if processing was interrupted
"""

import re
import torch
import numpy as np
import random
import logging
from typing import Optional, Tuple

from comfy.model_management import throw_exception_if_processing_interrupted

try:
    import scipy.signal as sp_signal
    _HAS_SCIPY = True
except ImportError:
    sp_signal = None
    _HAS_SCIPY = False
    logger = logging.getLogger(__name__)
    logger.warning("VibeVoice Node: `scipy` is not installed. Resampling of reference audio will not be available.")

logger = logging.getLogger(__name__)


def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a numpy audio array from ``orig_sr`` to ``target_sr``.

    Uses :func:`scipy.signal.resample_poly` with integer up/down factors derived
    from the greatest common divisor of the two sample rates. This avoids
    depending on ``librosa``, which may be a broken stub/namespace package in
    some embedded Python environments (e.g. ComfyUI's portable Python).

    Args:
        waveform: 1-D (or 2-D) float audio array.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled audio array with the same dtype/shape semantics as input.

    Raises:
        ImportError: If scipy is not installed.
        ValueError: If sample rates are invalid.
    """
    if not _HAS_SCIPY:
        raise ImportError(
            "`scipy` is required for audio resampling but is not installed. "
            "Please install it with `pip install scipy`."
        )
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"Invalid sample rates for resampling: orig_sr={orig_sr}, target_sr={target_sr}")
    if int(orig_sr) == int(target_sr):
        return waveform

    orig_sr = int(orig_sr)
    target_sr = int(target_sr)
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    # resample_poly operates along the last axis; works for 1-D and (N, channels).
    return sp_signal.resample_poly(waveform, up, down)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across torch, numpy, and random.

    Args:
        seed: Seed value. Use 0 for a random seed.
    """
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
    """Parse a 1-based speaker script into parsed lines and speaker IDs.

    Supports two formats:
    1. "Speaker 1: Some text..."
    2. "[1] Some text..."

    If no speaker markers are found, the entire script is assigned to Speaker 1.
    Internally, speaker IDs are converted to 0-based for the model.

    Args:
        script: The multi-speaker script text.

    Returns:
        Tuple of:
        - parsed_lines: List of (0-based speaker_id, text) tuples
        - speaker_ids: Sorted list of unique 1-based speaker IDs
    """
    parsed_lines = []
    speaker_ids_in_script = []

    line_format_regex = re.compile(
        r'^(?:Speaker\s+(\d+)\s*:|\[(\d+)\])\s*(.*)$',
        re.IGNORECASE
    )

    for line in script.strip().split("\n"):
        if not (line := line.strip()):
            continue

        match = line_format_regex.match(line)
        if match:
            speaker_id_str = match.group(1) or match.group(2)
            speaker_id = int(speaker_id_str)
            text_content = match.group(3)

            if match.group(1) is None and text_content.lstrip().startswith(':'):
                colon_index = text_content.find(':')
                text_content = text_content[colon_index + 1:]

            if speaker_id < 1:
                logger.warning(f"Speaker ID must be 1 or greater. Skipping line: '{line}'")
                continue

            text = text_content.strip()
            internal_speaker_id = speaker_id - 1
            parsed_lines.append((internal_speaker_id, text))

            if speaker_id not in speaker_ids_in_script:
                speaker_ids_in_script.append(speaker_id)
        else:
            logger.warning(f"Could not parse speaker marker, treating as part of previous line if any, or ignoring: '{line}'")

    if not parsed_lines and script.strip():
        logger.info("No speaker markers found. Treating entire text as a single utterance for Speaker 1.")
        parsed_lines.append((0, ' ' + script.strip()))
        speaker_ids_in_script.append(1)

    return parsed_lines, sorted(list(set(speaker_ids_in_script)))


def preprocess_comfy_audio(audio_dict: dict, target_sr: int = 24000) -> Optional[np.ndarray]:
    """Convert a ComfyUI AUDIO dict to a mono NumPy array, resampling if necessary.

    Args:
        audio_dict: ComfyUI audio dict with 'waveform' and 'sample_rate' keys.
        target_sr: Target sample rate for resampling.

    Returns:
        Mono float32 numpy array, or None if input is empty/invalid.

    Raises:
        ImportError: If scipy is needed for resampling but not installed.
    """
    if not audio_dict:
        return None
    waveform_tensor = audio_dict.get('waveform')
    if waveform_tensor is None or waveform_tensor.numel() == 0:
        return None

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
        logger.warning(f"Resampling reference audio from {original_sr}Hz to {target_sr}Hz.")
        waveform = resample_audio(waveform, original_sr, target_sr)

    # Final check after resampling
    if np.any(np.isnan(waveform)) or np.any(np.isinf(waveform)):
        logger.error("Audio contains NaN or Inf after resampling, replacing with zeros")
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)

    return waveform.astype(np.float32)


def extract_audio_tensor(
    audio_input: Optional[dict],
    name: str = "audio"
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """Extract waveform tensor and sample rate from ComfyUI audio input.

    Args:
        audio_input: ComfyUI audio dictionary with 'waveform' and 'sample_rate'.
        name: Name for error messages.

    Returns:
        Tuple of (waveform tensor, sample rate) or (None, None) if input is None.

    Raises:
        ValueError: If audio format is invalid or empty.
    """
    if audio_input is None:
        return None, None

    if not isinstance(audio_input, dict):
        raise ValueError(f"{name}: Expected dict, got {type(audio_input).__name__}")

    if 'waveform' not in audio_input or 'sample_rate' not in audio_input:
        raise ValueError(f"{name}: Missing 'waveform' or 'sample_rate' keys")

    waveform = audio_input['waveform']
    sample_rate = audio_input['sample_rate']

    # Remove batch dimension if present [1, C, T] -> [C, T]
    if waveform.dim() == 3:
        waveform = waveform[0]

    if waveform.numel() == 0:
        raise ValueError(f"{name}: Audio is empty")

    return waveform, sample_rate


def check_for_interrupt() -> bool:
    """Check if processing was interrupted by the user.

    Returns:
        True if interrupted, False otherwise.
    """
    try:
        throw_exception_if_processing_interrupted()
        return False
    except Exception:
        return True
