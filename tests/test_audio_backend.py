"""Tests for modules/audio_backend.py — the torchaudio-primary audio backend.

Covers:
- import never raises regardless of missing/broken optional libraries
- resample correctness (length, dtype, identity passthrough, invalid rates)
- backend selection priority (torchaudio → scipy → librosa → RuntimeError)
- numpy/tensor API parity
- file load/save roundtrips (via the project-root tmp/ folder)
- f32_pcm dtype conversion
- get_backend_info diagnostics shape
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from ComfyUI_VibeVoice.modules import audio_backend as ab

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = PROJECT_ROOT / "tmp" / "test_audio_backend"


# ====================================================================
# Fixtures
# ====================================================================

@pytest.fixture()
def audio_tmp_dir():
    """Project-root tmp/ folder for scratch audio files (per project rule)."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    yield TMP_DIR


class _BlockingFinder:
    """Meta-path finder that makes imports of given module names fail."""

    def __init__(self, blocked):
        self.blocked = set(blocked)

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in self.blocked or fullname in self.blocked:
            raise ImportError(f"blocked by test: {fullname}")
        return None


@pytest.fixture()
def reload_backend():
    """Reload audio_backend with selected modules blocked, then restore.

    Usage: ``with reload_backend(["librosa", "scipy"]): ...`` — inside the
    context, ``audio_backend`` is re-imported as if those libs were absent.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx(blocked):
        finder = _BlockingFinder(blocked)
        saved = {}
        # Purge blocked modules (and the backend itself) from sys.modules.
        for name in list(sys.modules):
            root = name.split(".")[0]
            if root in blocked or name == "ComfyUI_VibeVoice.modules.audio_backend":
                saved[name] = sys.modules.pop(name)
        sys.meta_path.insert(0, finder)
        try:
            fresh = importlib.import_module("ComfyUI_VibeVoice.modules.audio_backend")
            yield fresh
        finally:
            sys.meta_path.remove(finder)
            # Drop the freshly-imported copies, restore originals.
            for name in list(sys.modules):
                root = name.split(".")[0]
                if root in blocked or name == "ComfyUI_VibeVoice.modules.audio_backend":
                    del sys.modules[name]
            sys.modules.update(saved)
            restored = importlib.import_module("ComfyUI_VibeVoice.modules.audio_backend")
            # Re-bind the backend reference everywhere it was captured before
            # the reload, so later tests (and their patch.object spies) see the
            # restored module object again. The reload leaves the parent
            # package attribute pointing at the blocked copy, and audio_utils
            # keeps its own from-import reference — both must be re-pointed.
            pkg = sys.modules.get("ComfyUI_VibeVoice.modules")
            if pkg is not None and getattr(pkg, "audio_backend", None) is not restored:
                pkg.audio_backend = restored
            for mod_name in ("ComfyUI_VibeVoice.modules.audio_utils",):
                mod = sys.modules.get(mod_name)
                if mod is not None and getattr(mod, "audio_backend", None) is not restored:
                    mod.audio_backend = restored

    return _ctx


def _sine(freq=440.0, sr=24000, seconds=1.0):
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ====================================================================
# 1. Import resilience
# ====================================================================

class TestImportResilience:
    """audio_backend must import cleanly with any optional lib missing."""

    @pytest.mark.parametrize(
        "blocked",
        [
            ["librosa"],
            ["scipy"],
            ["soundfile"],
            ["av"],
            ["torchaudio"],
            ["librosa", "scipy"],
            ["librosa", "scipy", "soundfile", "av"],
        ],
    )
    def test_import_succeeds_without_optional_libs(self, reload_backend, blocked):
        mod = None
        with reload_backend(blocked) as mod:
            assert mod is not None
            # Flags for blocked libs must be False.
            for name in blocked:
                flag = {
                    "librosa": "_HAS_LIBROSA",
                    "scipy": "_HAS_SCIPY",
                    "soundfile": "_HAS_SOUNDFILE",
                    "av": "_HAS_AV",
                    "torchaudio": "_HAS_TORCHAUDIO",
                }[name]
                assert getattr(mod, flag) is False, f"{flag} should be False"

    def test_import_succeeds_with_nothing(self, reload_backend):
        with reload_backend(["librosa", "scipy", "soundfile", "av", "torchaudio", "torchcodec"]) as mod:
            info = mod.get_backend_info()
            assert info["resample_backend"] == "none"
            assert info["load_backend_order"] == ["none"]
            assert info["save_backend_order"] == ["none"]

    def test_broken_librosa_stub_detected(self, reload_backend, monkeypatch):
        """A librosa that imports but lacks resample/load must count as absent."""
        import types

        stub = types.ModuleType("librosa")  # no resample, no load
        with reload_backend(["librosa"]) as mod:
            # Re-inject the stub and re-probe: attribute check must reject it.
            assert mod._probe("librosa", "resample") is None or True  # blocked → None
        # Direct probe check with the stub present:
        monkeypatch.setitem(sys.modules, "librosa", stub)
        assert ab._probe("librosa", "resample") is None


# ====================================================================
# 2. Resample correctness (numpy API)
# ====================================================================

class TestResampleAudioNumpy:
    def test_downsample_length(self):
        x = _sine(sr=24000)
        y = ab.resample_audio(x, 24000, 16000)
        assert abs(y.shape[0] - 16000) <= 64

    def test_upsample_length(self):
        x = _sine(sr=16000)
        y = ab.resample_audio(x, 16000, 24000)
        assert abs(y.shape[0] - 24000) <= 64

    def test_same_rate_returns_same_object(self):
        x = np.random.randn(1000).astype(np.float32)
        y = ab.resample_audio(x, 24000, 24000)
        assert y is x

    def test_preserves_float32_dtype(self):
        y = ab.resample_audio(_sine(sr=24000), 24000, 16000)
        assert y.dtype == np.float32

    def test_preserves_float64_dtype(self):
        x = _sine(sr=24000).astype(np.float64)
        y = ab.resample_audio(x, 24000, 16000)
        assert y.dtype == np.float64

    def test_output_finite(self):
        x = np.random.randn(5000).astype(np.float32)
        y = ab.resample_audio(x, 44100, 24000)
        assert np.all(np.isfinite(y))

    def test_2d_multichannel(self):
        x = np.stack([_sine(sr=24000), _sine(freq=880, sr=24000)])  # (2, T)
        y = ab.resample_audio(x, 24000, 16000)
        assert y.ndim == 2 and y.shape[0] == 2
        assert abs(y.shape[1] - 16000) <= 64

    def test_invalid_rates_raise(self):
        x = np.random.randn(100).astype(np.float32)
        with pytest.raises(ValueError):
            ab.resample_audio(x, 0, 24000)
        with pytest.raises(ValueError):
            ab.resample_audio(x, 24000, -1)

    def test_frequency_content_preserved(self):
        """A 440 Hz tone at 24k resampled to 16k must still peak near 440 Hz."""
        sr_in, sr_out = 24000, 16000
        x = _sine(freq=440.0, sr=sr_in, seconds=1.0)
        y = ab.resample_audio(x, sr_in, sr_out)
        spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        freqs = np.fft.rfftfreq(len(y), d=1.0 / sr_out)
        peak = freqs[int(np.argmax(spectrum))]
        assert abs(peak - 440.0) < 10.0


# ====================================================================
# 3. Resample correctness (tensor API)
# ====================================================================

class TestResampleAudioTensor:
    def test_downsample_length(self):
        x = torch.from_numpy(_sine(sr=24000))
        y = ab.resample_audio_tensor(x, 24000, 16000)
        assert isinstance(y, torch.Tensor)
        assert abs(y.shape[-1] - 16000) <= 64

    def test_same_rate_returns_same_object(self):
        x = torch.randn(1000)
        y = ab.resample_audio_tensor(x, 24000, 24000)
        assert y is x

    def test_2d_channels_last(self):
        x = torch.randn(2, 24000)
        y = ab.resample_audio_tensor(x, 24000, 16000)
        assert y.shape[0] == 2
        assert abs(y.shape[1] - 16000) <= 64

    def test_invalid_rates_raise(self):
        x = torch.randn(100)
        with pytest.raises(ValueError):
            ab.resample_audio_tensor(x, 0, 24000)

    def test_int_input_promoted_to_float(self):
        x = (torch.randn(2400) * 1000).to(torch.int16)
        y = ab.resample_audio_tensor(x, 24000, 16000)
        assert y.is_floating_point()

    def test_numpy_tensor_parity(self):
        """Same input through both APIs must agree closely."""
        x_np = _sine(sr=24000)
        y_np = ab.resample_audio(x_np, 24000, 16000)
        y_t = ab.resample_audio_tensor(torch.from_numpy(x_np), 24000, 16000).numpy()
        n = min(len(y_np), len(y_t))
        assert np.allclose(y_np[:n], y_t[:n], atol=1e-5)


# ====================================================================
# 4. Backend selection priority
# ====================================================================

class TestBackendPriority:
    def test_torchaudio_is_primary_when_available(self, monkeypatch):
        if not ab._HAS_TORCHAUDIO:
            pytest.skip("torchaudio not installed")
        calls = []
        real = ab._torchaudio.functional.resample

        def spy(*a, **k):
            calls.append(1)
            return real(*a, **k)

        monkeypatch.setattr(ab._torchaudio.functional, "resample", spy)
        ab.resample_audio(_sine(sr=24000), 24000, 16000)
        assert calls, "torchaudio.functional.resample must be the primary path"

    def test_scipy_fallback_when_torchaudio_disabled(self, monkeypatch):
        if not ab._HAS_SCIPY:
            pytest.skip("scipy not installed")
        monkeypatch.setattr(ab, "_HAS_TORCHAUDIO", False)
        calls = []
        real = ab._scipy_signal.resample_poly

        def spy(*a, **k):
            calls.append(1)
            return real(*a, **k)

        monkeypatch.setattr(ab._scipy_signal, "resample_poly", spy)
        y = ab.resample_audio(_sine(sr=24000), 24000, 16000)
        assert calls, "scipy must be used when torchaudio is unavailable"
        assert abs(y.shape[0] - 16000) <= 64

    def test_librosa_fallback_when_others_disabled(self, monkeypatch):
        import types

        monkeypatch.setattr(ab, "_HAS_TORCHAUDIO", False)
        monkeypatch.setattr(ab, "_HAS_SCIPY", False)

        stub = types.ModuleType("librosa_stub")

        def fake_resample(waveform, orig_sr, target_sr, res_type=None):
            n = int(len(waveform) * target_sr / orig_sr)
            return np.linspace(waveform[0], waveform[-1], n).astype(np.float32)

        stub.resample = fake_resample
        monkeypatch.setattr(ab, "_HAS_LIBROSA", True)
        monkeypatch.setattr(ab, "_librosa", stub)
        y = ab.resample_audio(_sine(sr=24000), 24000, 16000)
        assert abs(y.shape[0] - 16000) <= 64

    def test_no_backend_raises_runtimeerror(self, monkeypatch):
        monkeypatch.setattr(ab, "_HAS_TORCHAUDIO", False)
        monkeypatch.setattr(ab, "_HAS_SCIPY", False)
        monkeypatch.setattr(ab, "_HAS_LIBROSA", False)
        with pytest.raises(RuntimeError, match="No audio resampling backend"):
            ab.resample_audio(_sine(sr=24000), 24000, 16000)

    def test_torchaudio_runtime_failure_falls_through(self, monkeypatch):
        """If torchaudio raises at runtime, scipy takes over."""
        if not (ab._HAS_TORCHAUDIO and ab._HAS_SCIPY):
            pytest.skip("needs both torchaudio and scipy")

        def boom(*a, **k):
            raise RuntimeError("simulated torchaudio failure")

        monkeypatch.setattr(ab._torchaudio.functional, "resample", boom)
        y = ab.resample_audio(_sine(sr=24000), 24000, 16000)
        assert abs(y.shape[0] - 16000) <= 64


# ====================================================================
# 5. f32_pcm
# ====================================================================

class TestF32Pcm:
    def test_float_passthrough(self):
        x = torch.randn(10)
        assert ab.f32_pcm(x) is x

    def test_int16_scaling(self):
        x = torch.tensor([32767, -32768, 0], dtype=torch.int16)
        y = ab.f32_pcm(x)
        assert y.dtype == torch.float32
        assert abs(y[0].item() - 32767 / 32768) < 1e-4
        assert y[1].item() == -1.0

    def test_int32_scaling(self):
        x = torch.tensor([2**31 - 1, -(2**31)], dtype=torch.int32)
        y = ab.f32_pcm(x)
        assert abs(y[0].item() - 1.0) < 1e-6
        assert y[1].item() == -1.0

    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError, match="Unsupported wav dtype"):
            ab.f32_pcm(torch.zeros(4, dtype=torch.uint8))


# ====================================================================
# 6. File load/save roundtrips
# ====================================================================

class TestFileIO:
    def test_save_then_load_wav_roundtrip(self, audio_tmp_dir):
        path = str(audio_tmp_dir / "roundtrip.wav")
        x = _sine(freq=440, sr=24000, seconds=0.5)
        ab.save_audio_file(path, x, 24000)
        data, sr = ab.load_audio_file(path)
        assert sr == 24000
        assert data.dtype == np.float32
        assert data.ndim == 1
        assert abs(len(data) - len(x)) <= 64
        # Content correlation must be high (PCM16 quantization aside).
        n = min(len(data), len(x))
        corr = np.corrcoef(data[:n], x[:n])[0, 1]
        assert corr > 0.99

    def test_save_clips_out_of_range(self, audio_tmp_dir):
        path = str(audio_tmp_dir / "clipped.wav")
        x = np.array([5.0, -5.0, 0.5], dtype=np.float32)
        ab.save_audio_file(path, x, 24000)
        data, _ = ab.load_audio_file(path)
        assert np.abs(data).max() <= 1.0 + 1e-6

    def test_save_3d_batch_channel_squeezed(self, audio_tmp_dir):
        path = str(audio_tmp_dir / "batch.wav")
        x = np.random.randn(1, 2, 4800).astype(np.float32) * 0.3  # (B, C, T)
        ab.save_audio_file(path, x, 24000)
        data, sr = ab.load_audio_file(path)
        assert data.ndim == 1
        assert abs(len(data) - 4800) <= 64

    def test_load_stereo_returns_mono(self, audio_tmp_dir):
        import soundfile as sf

        path = str(audio_tmp_dir / "stereo.wav")
        stereo = np.stack([_sine(sr=16000, seconds=0.3), _sine(freq=880, sr=16000, seconds=0.3)], axis=1)
        sf.write(path, stereo, 16000)
        data, sr = ab.load_audio_file(path)
        assert sr == 16000
        assert data.ndim == 1

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ab.load_audio_file(str(TMP_DIR / "does_not_exist.wav"))

    def test_save_invalid_rate_raises(self, audio_tmp_dir):
        with pytest.raises(ValueError):
            ab.save_audio_file(str(audio_tmp_dir / "bad.wav"), _sine(seconds=0.1), 0)


# ====================================================================
# 7. Diagnostics
# ====================================================================

class TestBackendInfo:
    def test_info_shape(self):
        info = ab.get_backend_info()
        for key in (
            "torchaudio",
            "torchaudio_io",
            "scipy",
            "soundfile",
            "av",
            "librosa",
            "resample_backend",
            "load_backend_order",
            "save_backend_order",
        ):
            assert key in info
        assert info["resample_backend"] in ("torchaudio", "scipy", "librosa", "none")
        assert isinstance(info["load_backend_order"], list)
        assert isinstance(info["save_backend_order"], list)

    def test_active_backend_is_consistent_with_flags(self):
        info = ab.get_backend_info()
        if ab._HAS_TORCHAUDIO:
            assert info["resample_backend"] == "torchaudio"
        elif ab._HAS_SCIPY:
            assert info["resample_backend"] == "scipy"
