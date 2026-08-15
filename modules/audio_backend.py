"""Audio backend abstraction for VibeVoice nodes.

Provides a single, dependency-resilient entry point for the three audio
primitives the node needs:

- **Resampling** — ``resample_audio`` (numpy API) / ``resample_audio_tensor``
  (torch API). Backend order: ``torchaudio.functional.resample`` (primary,
  the ComfyUI-core idiom) → ``scipy.signal.resample_poly`` → ``librosa.resample``
  (optional) → ``RuntimeError``.
- **File decoding** — ``load_audio_file``. Backend order: PyAV ``av`` (the
  ComfyUI-core decoder) → ``soundfile`` → ``torchaudio.load`` (guarded: needs
  torchcodec on torchaudio >= 2.9) → ``librosa.load`` (optional) → ``RuntimeError``.
- **File encoding** — ``save_audio_file``. Backend order: ``soundfile`` →
  ``torchaudio.save`` (guarded) → ``RuntimeError``.

Design rules:

1. Importing this module **never raises**, no matter which optional libraries
   are missing or broken. Capability is detected once at import time via
   ``_HAS_*`` flags. Detection checks *attributes*, not just import success,
   because some embedded Python environments ship a broken ``librosa``
   namespace stub that imports fine but exposes nothing.
2. The public API is numpy-in/numpy-out (plus a tensor-native resample
   variant) so existing callers keep working unchanged.
3. A missing optional library degrades gracefully to the next backend; only
   when *no* backend is available does a clear ``RuntimeError`` raise.
"""

import importlib
import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "resample_audio",
    "resample_audio_tensor",
    "load_audio_file",
    "save_audio_file",
    "f32_pcm",
    "get_backend_info",
]


# ====================================================================
# 1. CAPABILITY DETECTION (import-time, never raises)
# ====================================================================

def _probe(module_name: str, required_attr: Optional[str] = None):
    """Import ``module_name``; return the module or None.

    If ``required_attr`` is given, the module only counts as usable when the
    attribute exists (guards against broken namespace stubs, e.g. an empty
    ``librosa`` package that imports but has no ``resample``/``load``).
    """
    try:
        mod = importlib.import_module(module_name)
    except Exception:  # ImportError or broken-install exceptions
        return None
    if required_attr is not None and not hasattr(mod, required_attr):
        return None
    return mod


_torchaudio = _probe("torchaudio", "functional")
_HAS_TORCHAUDIO = _torchaudio is not None and hasattr(_torchaudio.functional, "resample")

try:
    import scipy.signal as _scipy_signal

    _HAS_SCIPY = hasattr(_scipy_signal, "resample_poly")
except Exception:
    _scipy_signal = None
    _HAS_SCIPY = False

_soundfile = _probe("soundfile", "read")
_HAS_SOUNDFILE = _soundfile is not None and hasattr(_soundfile, "write")

_av = _probe("av", "open")
_HAS_AV = _av is not None

# librosa is strictly optional; the embedded ComfyUI Python often ships a
# broken empty namespace stub, hence the attribute checks.
_librosa = _probe("librosa", "resample")
_HAS_LIBROSA = _librosa is not None


def _probe_torchaudio_io() -> bool:
    """Best-effort check whether ``torchaudio.load``/``save`` are usable.

    torchaudio >= 2.9 consolidated decoding/encoding into TorchCodec; without
    torchcodec installed, ``torchaudio.load`` raises ``ImportError`` at call
    time. Older versions (< 2.9) still carry native backends.
    """
    if not _HAS_TORCHAUDIO:
        return False
    if _probe("torchcodec") is not None:
        return True
    version = getattr(_torchaudio, "__version__", "") or "0.0"
    try:
        major, minor = (int(x) for x in version.split("+")[0].split(".")[:2])
    except Exception:
        return False
    return (major, minor) < (2, 9)


def _active_resample_backend() -> str:
    if _HAS_TORCHAUDIO:
        return "torchaudio"
    if _HAS_SCIPY:
        return "scipy"
    if _HAS_LIBROSA:
        return "librosa"
    return "none"


def get_backend_info() -> dict:
    """Report which audio backends are available and which is active.

    Useful for diagnostics and tests.
    """
    load_order = [
        name
        for name, ok in (
            ("av", _HAS_AV),
            ("soundfile", _HAS_SOUNDFILE),
            ("torchaudio", _probe_torchaudio_io()),
            ("librosa", _HAS_LIBROSA and hasattr(_librosa, "load")),
        )
        if ok
    ]
    save_order = [
        name
        for name, ok in (
            ("soundfile", _HAS_SOUNDFILE),
            ("torchaudio", _probe_torchaudio_io()),
        )
        if ok
    ]
    return {
        "torchaudio": _HAS_TORCHAUDIO,
        "torchaudio_version": getattr(_torchaudio, "__version__", None),
        "torchaudio_io": _probe_torchaudio_io(),
        "scipy": _HAS_SCIPY,
        "soundfile": _HAS_SOUNDFILE,
        "av": _HAS_AV,
        "librosa": _HAS_LIBROSA,
        "resample_backend": _active_resample_backend(),
        "load_backend_order": load_order or ["none"],
        "save_backend_order": save_order or ["none"],
    }


# ====================================================================
# 2. RESAMPLING
# ====================================================================

# Kaiser-windowed sinc parameters equivalent to torchaudio's "kaiser_best"
# preset (≈ librosa kaiser_best) — higher quality than the default hann window.
_KAISER_WIDTH = 64
_KAISER_ROLLOFF = 0.9475937167399596
_KAISER_BETA = 14.769621229636083


def _validate_rates(orig_sr: int, target_sr: int) -> None:
    if int(orig_sr) <= 0 or int(target_sr) <= 0:
        raise ValueError(
            f"Invalid sample rates for resampling: orig_sr={orig_sr}, target_sr={target_sr}"
        )


def _resample_scipy(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resample along the last axis (works for 1-D and (C, T))."""
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return _scipy_signal.resample_poly(waveform, up, down)


def _resample_librosa(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """librosa resample along the last axis (1-D or (C, T))."""
    return _librosa.resample(
        waveform, orig_sr=orig_sr, target_sr=target_sr, res_type="kaiser_best"
    )


def resample_audio_tensor(
    waveform: torch.Tensor, orig_sr: int, target_sr: int
) -> torch.Tensor:
    """Resample a torch waveform from ``orig_sr`` to ``target_sr``.

    Backend order: torchaudio (Kaiser sinc, ComfyUI idiom) → scipy → librosa.
    Falls through to the next backend if the primary one raises at runtime.

    Args:
        waveform: 1-D ``(T,)`` or 2-D ``(C, T)`` audio tensor.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled tensor (same device; float dtype preserved). If the rates
        are equal, the *same* tensor object is returned unchanged.

    Raises:
        ValueError: If sample rates are invalid.
        RuntimeError: If no resampling backend is available.
    """
    _validate_rates(orig_sr, target_sr)
    orig_sr = int(orig_sr)
    target_sr = int(target_sr)
    if orig_sr == target_sr:
        return waveform
    if waveform.numel() == 0:
        return waveform

    work = waveform
    if not work.is_floating_point():
        work = work.float()

    if _HAS_TORCHAUDIO:
        try:
            return _torchaudio.functional.resample(
                work,
                orig_sr,
                target_sr,
                resampling_method="sinc_interp_kaiser",
                lowpass_filter_width=_KAISER_WIDTH,
                rolloff=_KAISER_ROLLOFF,
                beta=_KAISER_BETA,
            )
        except Exception as e:  # pragma: no cover - defensive fallthrough
            logger.warning(
                f"VibeVoice: torchaudio resample failed ({e}); falling back."
            )

    if _HAS_SCIPY:
        np_wav = work.detach().cpu().numpy()
        resampled = _resample_scipy(np_wav, orig_sr, target_sr)
        out = torch.from_numpy(np.ascontiguousarray(resampled))
        return out.to(dtype=work.dtype)

    if _HAS_LIBROSA:
        np_wav = work.detach().cpu().numpy().astype(np.float32, copy=False)
        resampled = _resample_librosa(np_wav, orig_sr, target_sr)
        out = torch.from_numpy(np.ascontiguousarray(resampled))
        return out.to(dtype=work.dtype)

    raise RuntimeError(
        "No audio resampling backend available. Install one of: "
        "`uv pip install torchaudio` (preferred), `uv pip install scipy`, "
        "or `uv pip install librosa`."
    )


def resample_audio(
    waveform: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray:
    """Resample a numpy audio array from ``orig_sr`` to ``target_sr``.

    Numpy facade over :func:`resample_audio_tensor`. torchaudio is the primary
    backend (ComfyUI-core idiom); scipy and librosa are fallbacks.

    Args:
        waveform: 1-D (or 2-D ``(C, T)``) float audio array.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled audio array. Floating-point inputs preserve their dtype;
        non-float inputs return float32. If the rates are equal, the *same*
        array object is returned unchanged.

    Raises:
        ValueError: If sample rates are invalid.
        RuntimeError: If no resampling backend is available.
    """
    _validate_rates(orig_sr, target_sr)
    orig_sr = int(orig_sr)
    target_sr = int(target_sr)
    if orig_sr == target_sr:
        return waveform

    in_dtype = waveform.dtype
    arr = np.ascontiguousarray(waveform)
    tensor = torch.as_tensor(arr)
    if not tensor.is_floating_point():
        tensor = tensor.float()

    resampled = resample_audio_tensor(tensor, orig_sr, target_sr)
    out = resampled.detach().cpu().numpy()

    if np.issubdtype(in_dtype, np.floating):
        return out.astype(in_dtype, copy=False)
    return out.astype(np.float32, copy=False)


# ====================================================================
# 3. FILE DECODING
# ====================================================================

def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio tensor to float32 PCM (mirror of ComfyUI's helper).

    Float tensors pass through unchanged; int16/int32 are scaled to [-1, 1].
    """
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / (2**15)
    elif wav.dtype == torch.int32:
        return wav.float() / (2**31)
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def _load_with_av(filepath: str) -> Tuple[np.ndarray, int]:
    """Decode an audio file with PyAV (ComfyUI-core pattern).

    Returns mono float32 numpy ``(T,)`` and the native sample rate.
    """
    with _av.open(filepath) as af:
        if not af.streams.audio:
            raise ValueError("No audio stream found in the file.")

        stream = af.streams.audio[0]
        sr = stream.codec_context.sample_rate
        try:
            n_channels = stream.codec_context.layout.nb_channels
        except Exception:  # older PyAV
            n_channels = stream.channels

        frames = []
        for frame in af.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != n_channels:
                buf = buf.view(-1, n_channels).t()
            frames.append(buf)

        if not frames:
            raise ValueError("No audio frames decoded.")

        wav = torch.cat(frames, dim=1)
        wav = f32_pcm(wav)
        mono = wav.mean(dim=0) if wav.shape[0] > 1 else wav[0]
        return np.ascontiguousarray(mono.numpy(), dtype=np.float32), int(sr)


def load_audio_file(path: str) -> Tuple[np.ndarray, int]:
    """Load an audio file as mono float32 numpy at its native sample rate.

    Backend order: PyAV → soundfile → torchaudio.load (guarded) →
    librosa.load (optional). No resampling is performed here.

    Args:
        path: Path to the audio file.

    Returns:
        Tuple of (mono float32 numpy array ``(T,)``, native sample rate).

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If no backend can decode the file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    errors = []

    if _HAS_AV:
        try:
            return _load_with_av(path)
        except Exception as e:
            errors.append(f"av: {e}")

    if _HAS_SOUNDFILE:
        try:
            data, sr = _soundfile.read(path, dtype="float32", always_2d=True)
            # soundfile returns (samples, channels)
            mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
            return np.ascontiguousarray(mono, dtype=np.float32), int(sr)
        except Exception as e:
            errors.append(f"soundfile: {e}")

    if _HAS_TORCHAUDIO:
        try:
            wav, sr = _torchaudio.load(path)
            mono = wav.mean(dim=0) if wav.dim() > 1 else wav[0]
            return (
                np.ascontiguousarray(mono.numpy(), dtype=np.float32),
                int(sr),
            )
        except Exception as e:
            errors.append(f"torchaudio: {e}")

    if _HAS_LIBROSA and hasattr(_librosa, "load"):
        try:
            data, sr = _librosa.load(path, sr=None, mono=True)
            return np.asarray(data, dtype=np.float32), int(sr)
        except Exception as e:
            errors.append(f"librosa: {e}")

    raise RuntimeError(
        f"Could not load audio file '{path}'. "
        f"Tried: {'; '.join(errors) if errors else 'no audio backends available'}. "
        "Install `av` (PyAV) or `soundfile` for audio decoding."
    )


# ====================================================================
# 4. FILE ENCODING
# ====================================================================

def save_audio_file(path: str, waveform: np.ndarray, sample_rate: int) -> None:
    """Save a waveform to an audio file (wav/flac via soundfile).

    Accepts 1-D ``(T,)``, 2-D ``(C, T)`` (mixed to mono), or 3-D ``(B, C, T)``
    (first batch item, then mixed to mono). Values are clipped to [-1, 1].

    Args:
        path: Destination file path.
        waveform: Audio array (any numeric dtype; converted to float32).
        sample_rate: Sample rate in Hz.

    Raises:
        ValueError: If the sample rate is invalid.
        RuntimeError: If no encoding backend is available.
    """
    if int(sample_rate) <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate}")

    arr = np.asarray(waveform)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr.mean(axis=0) if arr.shape[0] > 1 else arr[0]
    arr = np.clip(np.asarray(arr, dtype=np.float32), -1.0, 1.0)

    errors = []

    if _HAS_SOUNDFILE:
        try:
            _soundfile.write(path, arr, int(sample_rate))
            return
        except Exception as e:
            errors.append(f"soundfile: {e}")

    if _HAS_TORCHAUDIO:
        try:
            _torchaudio.save(path, torch.from_numpy(arr).unsqueeze(0), int(sample_rate))
            return
        except Exception as e:
            errors.append(f"torchaudio: {e}")

    raise RuntimeError(
        f"Could not save audio file '{path}'. "
        f"Tried: {'; '.join(errors) if errors else 'no audio backends available'}. "
        "Install `soundfile` for audio encoding."
    )
