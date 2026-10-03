"""Audio and script utilities for VibeVoice nodes.

Contains:
- parse_script_1_based(): Parse multi-speaker script text
- preprocess_comfy_audio(): Convert ComfyUI AUDIO dict to mono numpy array
- extract_audio_tensor(): Extract waveform + sample_rate from audio dict
- set_seed(): Set random seeds for reproducibility
- check_for_interrupt(): Check if processing was interrupted

Resampling is delegated to :mod:`modules.audio_backend`, which uses
``torchaudio.functional.resample`` as the primary backend (the ComfyUI-core
idiom) with scipy/librosa as optional fallbacks. ``librosa`` is never
required: the node degrades gracefully when it (or scipy) is absent.
"""

import re
import torch
import numpy as np
import random
import logging
from typing import Optional, Tuple

from comfy.model_management import throw_exception_if_processing_interrupted

from . import audio_backend


if audio_backend._active_resample_backend() == "none":
    logging.warning(
        "[VibeVoice TTS] VibeVoice Node: no audio resampling backend available "
        "(torchaudio/scipy/librosa all missing). Resampling of reference "
        "audio will fail. Install torchaudio (preferred) or scipy."
    )


def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a numpy audio array from ``orig_sr`` to ``target_sr``.

    Thin facade over :func:`modules.audio_backend.resample_audio`. The
    primary backend is ``torchaudio.functional.resample`` (Kaiser sinc —
    the same family of resampler ComfyUI core uses); scipy and librosa are
    optional fallbacks, so this never hard-requires any single library.

    Args:
        waveform: 1-D (or 2-D) float audio array.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled audio array with the same dtype/shape semantics as input.
        If the rates are equal, the same array object is returned unchanged.

    Raises:
        ValueError: If sample rates are invalid.
        RuntimeError: If no resampling backend is available.
    """
    return audio_backend.resample_audio(waveform, orig_sr, target_sr)


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
                logging.warning(f"[VibeVoice TTS] Speaker ID must be 1 or greater. Skipping line: '{line}'")
                continue

            text = text_content.strip()
            internal_speaker_id = speaker_id - 1
            parsed_lines.append((internal_speaker_id, text))

            if speaker_id not in speaker_ids_in_script:
                speaker_ids_in_script.append(speaker_id)
        else:
            logging.warning(f"[VibeVoice TTS] Could not parse speaker marker, treating as part of previous line if any, or ignoring: '{line}'")

    if not parsed_lines and script.strip():
        logging.debug("[VibeVoice TTS] No speaker markers found. Treating entire text as a single utterance for Speaker 1.")
        parsed_lines.append((0, ' ' + script.strip()))
        speaker_ids_in_script.append(1)

    return parsed_lines, sorted(list(set(speaker_ids_in_script)))


def preprocess_comfy_audio(audio_dict: dict, target_sr: int = 24000) -> Optional[np.ndarray]:
    """Convert a ComfyUI AUDIO dict to a mono NumPy array, resampling if necessary.

    Resampling happens in tensor space via
    :func:`modules.audio_backend.resample_audio_tensor` (torchaudio primary —
    the ComfyUI-core idiom), avoiding a numpy round-trip.

    Args:
        audio_dict: ComfyUI audio dict with 'waveform' and 'sample_rate' keys.
        target_sr: Target sample rate for resampling.

    Returns:
        Mono float32 numpy array, or None if input is empty/invalid.

    Raises:
        RuntimeError: If resampling is needed but no backend is available.
    """
    if not audio_dict:
        return None
    waveform_tensor = audio_dict.get('waveform')
    if waveform_tensor is None or waveform_tensor.numel() == 0:
        return None

    original_sr = int(audio_dict['sample_rate'])

    # Tensor-native path: strip batch dim and mix to mono in tensor space.
    tensor = waveform_tensor[0]
    if tensor.dim() > 1:
        tensor = tensor.mean(dim=0)

    # Scrub invalid values BEFORE resampling (NaN would poison the sinc filter).
    if not torch.isfinite(tensor).all():
        logging.error("[VibeVoice TTS] Audio contains NaN or Inf values, replacing with zeros")
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    if original_sr != int(target_sr):
        tensor = audio_backend.resample_audio_tensor(tensor, original_sr, int(target_sr))

    waveform = tensor.cpu().numpy()

    # Ensure audio is not completely silent or has extreme values
    if np.all(waveform == 0):
        logging.warning("[VibeVoice TTS] Audio waveform is completely silent")

    # Normalize extreme values
    max_val = np.abs(waveform).max()
    if max_val > 10.0:
        logging.warning(f"[VibeVoice TTS] Audio values are very large (max: {max_val}), normalizing")
        waveform = waveform / max_val

    # Final check after resampling
    if np.any(np.isnan(waveform)) or np.any(np.isinf(waveform)):
        logging.error("[VibeVoice TTS] Audio contains NaN or Inf after resampling, replacing with zeros")
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
