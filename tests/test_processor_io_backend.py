"""Phase 3 tests — vendored processor file I/O routed through audio_backend.

The root ``conftest.py`` mocks the entire ``src.vibevoice`` package (to keep
heavy model imports out of the unit suite), so these tests load the REAL
vendored processor modules under a fresh, side-effect-free package alias
(``VV_Real``) built from bare ``types.ModuleType`` packages. This exercises
the actual production code paths:

- ``VibeVoiceTokenizerProcessor._load_audio_from_path`` → audio_backend.load_audio_file
- ``VibeVoiceTokenizerProcessor.save_audio``            → audio_backend.save_audio_file
- ``VibeVoiceASRProcessor._process_single_audio``       → audio_backend.load_audio_file

Scratch audio files go to the project-root ``tmp/`` folder (project rule).
"""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = PROJECT_ROOT / "tmp" / "test_processor_io"
PKG = "VV_Real"


# ====================================================================
# Real-module loader (bypasses the conftest mocks)
# ====================================================================

def _make_pkg(name: str, path: Path):
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    sys.modules[name] = mod
    return mod


@pytest.fixture(scope="module")
def real_vendored():
    """Load the real vendored processors under the VV_Real alias."""
    already = f"{PKG}.src.vibevoice.processor.vibevoice_tokenizer_processor" in sys.modules
    if not already:
        _make_pkg(PKG, PROJECT_ROOT)
        _make_pkg(f"{PKG}.src", PROJECT_ROOT / "src")
        _make_pkg(f"{PKG}.src.vibevoice", PROJECT_ROOT / "src" / "vibevoice")
        _make_pkg(
            f"{PKG}.src.vibevoice.processor",
            PROJECT_ROOT / "src" / "vibevoice" / "processor",
        )
        _make_pkg(f"{PKG}.modules", PROJECT_ROOT / "modules")

    tok_mod = importlib.import_module(
        f"{PKG}.src.vibevoice.processor.vibevoice_tokenizer_processor"
    )
    asr_mod = importlib.import_module(
        f"{PKG}.src.vibevoice.processor.vibevoice_asr_processor"
    )
    backend = sys.modules[f"{PKG}.modules.audio_backend"]
    return {"tokenizer": tok_mod, "asr": asr_mod, "backend": backend}


@pytest.fixture()
def audio_tmp_dir():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    return TMP_DIR


def _write_wav(path, sr=24000, seconds=0.5, freq=440.0, channels=1):
    import soundfile as sf

    t = np.arange(int(sr * seconds)) / sr
    mono = 0.5 * np.sin(2 * np.pi * freq * t)
    data = mono if channels == 1 else np.stack([mono] * channels, axis=1)
    sf.write(str(path), data.astype(np.float32), sr)
    return str(path)


class _StubTokenizer:
    """Minimal tokenizer stand-in satisfying VibeVoiceASRProcessor's needs."""

    speech_start_id = 100
    speech_end_id = 101
    speech_pad_id = 102
    pad_id = 0

    def convert_ids_to_tokens(self, i):
        return f"<tok{i}>"

    def encode(self, text):
        return [1, 2, 3]

    def apply_chat_template(self, msgs, tokenize=False, **kwargs):
        return [4, 5, 6] if tokenize else "system-prompt"


# ====================================================================
# 1. Tokenizer processor: _load_audio_from_path
# ====================================================================

class TestTokenizerLoadAudioFromPath:
    def test_wav_roundtrip_at_model_sr(self, real_vendored, audio_tmp_dir):
        proc = real_vendored["tokenizer"].VibeVoiceTokenizerProcessor(
            sampling_rate=24000, normalize_audio=False
        )
        path = _write_wav(audio_tmp_dir / "tok_load.wav", sr=24000, seconds=0.5)
        out = proc._load_audio_from_path(path)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        assert out.ndim == 1
        assert abs(len(out) - 12000) <= 64  # 0.5 s @ 24 kHz

    def test_resamples_to_model_sr(self, real_vendored, audio_tmp_dir):
        """A 16 kHz file must come back at the processor's 24 kHz."""
        proc = real_vendored["tokenizer"].VibeVoiceTokenizerProcessor(
            sampling_rate=24000, normalize_audio=False
        )
        path = _write_wav(audio_tmp_dir / "tok_16k.wav", sr=16000, seconds=1.0)
        out = proc._load_audio_from_path(path)
        assert abs(len(out) - 24000) <= 96  # resampled 16k → 24k

    def test_stereo_becomes_mono(self, real_vendored, audio_tmp_dir):
        proc = real_vendored["tokenizer"].VibeVoiceTokenizerProcessor(
            sampling_rate=24000, normalize_audio=False
        )
        path = _write_wav(audio_tmp_dir / "tok_stereo.wav", sr=24000, seconds=0.3, channels=2)
        out = proc._load_audio_from_path(path)
        assert out.ndim == 1

    def test_npy_branch_still_works(self, real_vendored, audio_tmp_dir):
        proc = real_vendored["tokenizer"].VibeVoiceTokenizerProcessor(
            sampling_rate=24000, normalize_audio=False
        )
        arr = np.random.randn(4800).astype(np.float32)
        path = audio_tmp_dir / "tok.npy"
        np.save(str(path), arr)
        out = proc._load_audio_from_path(str(path))
        assert out.dtype == np.float32
        assert np.allclose(out, arr)

    def test_pt_branch_still_works(self, real_vendored, audio_tmp_dir):
        import torch

        proc = real_vendored["tokenizer"].VibeVoiceTokenizerProcessor(
            sampling_rate=24000, normalize_audio=False
        )
        tensor = torch.randn(4800)
        path = audio_tmp_dir / "tok.pt"
        torch.save(tensor, str(path))
        out = proc._load_audio_from_path(str(path))
        assert out.dtype == np.float32
        assert len(out) == 4800

    def test_unsupported_extension_raises(self, real_vendored, audio_tmp_dir):
        proc = real_vendored["tokenizer"].VibeVoiceTokenizerProcessor(
            sampling_rate=24000, normalize_audio=False
        )
        bad = audio_tmp_dir / "tok.xyz"
        bad.write_bytes(b"not audio")
        with pytest.raises(ValueError, match="Unsupported file format"):
            proc._load_audio_from_path(str(bad))

    def test_load_routes_through_backend(self, real_vendored, audio_tmp_dir, monkeypatch):
        """The backend's load_audio_file must be the decode entry point."""
        backend = real_vendored["backend"]
        calls = []
        original = backend.load_audio_file

        def spy(path):
            calls.append(path)
            return original(path)

        monkeypatch.setattr(backend, "load_audio_file", spy)
        proc = real_vendored["tokenizer"].VibeVoiceTokenizerProcessor(
            sampling_rate=24000, normalize_audio=False
        )
        path = _write_wav(audio_tmp_dir / "tok_spy.wav", sr=24000, seconds=0.2)
        proc._load_audio_from_path(path)
        assert calls == [path]


# ====================================================================
# 2. Tokenizer processor: save_audio
# ====================================================================

class TestTokenizerSaveAudio:
    def test_save_single_array(self, real_vendored, audio_tmp_dir):
        proc = real_vendored["tokenizer"].VibeVoiceTokenizerProcessor(
            sampling_rate=24000, normalize_audio=False
        )
        path = str(audio_tmp_dir / "save_single.wav")
        audio = (0.4 * np.sin(2 * np.pi * 440 * np.arange(12000) / 24000)).astype(np.float32)
        saved = proc.save_audio(audio, output_path=path, sampling_rate=24000)
        assert saved == [path]
        data, sr = real_vendored["backend"].load_audio_file(path)
        assert sr == 24000
        assert abs(len(data) - 12000) <= 64

    def test_save_torch_tensor(self, real_vendored, audio_tmp_dir):
        import torch

        proc = real_vendored["tokenizer"].VibeVoiceTokenizerProcessor(
            sampling_rate=24000, normalize_audio=False
        )
        path = str(audio_tmp_dir / "save_tensor.wav")
        audio = torch.randn(1, 1, 12000) * 0.3
        saved = proc.save_audio(audio, output_path=path, sampling_rate=24000)
        assert saved == [path]
        assert Path(path).exists()

    def test_save_list_creates_dir_of_files(self, real_vendored, audio_tmp_dir):
        proc = real_vendored["tokenizer"].VibeVoiceTokenizerProcessor(
            sampling_rate=24000, normalize_audio=False
        )
        outdir = str(audio_tmp_dir / "save_list")
        audios = [np.random.randn(4800).astype(np.float32) * 0.3 for _ in range(3)]
        saved = proc.save_audio(audios, output_path=outdir, sampling_rate=24000)
        assert len(saved) == 3
        for p in saved:
            assert Path(p).exists()

    def test_save_batch_3d(self, real_vendored, audio_tmp_dir):
        proc = real_vendored["tokenizer"].VibeVoiceTokenizerProcessor(
            sampling_rate=24000, normalize_audio=False
        )
        outdir = str(audio_tmp_dir / "save_batch")
        audio = np.random.randn(2, 1, 4800).astype(np.float32) * 0.3  # (B, C, T)
        saved = proc.save_audio(audio, output_path=outdir, sampling_rate=24000)
        assert len(saved) == 2

    def test_save_routes_through_backend(self, real_vendored, audio_tmp_dir, monkeypatch):
        backend = real_vendored["backend"]
        calls = []
        original = backend.save_audio_file

        def spy(path, waveform, sample_rate):
            calls.append((path, sample_rate))
            return original(path, waveform, sample_rate)

        monkeypatch.setattr(backend, "save_audio_file", spy)
        proc = real_vendored["tokenizer"].VibeVoiceTokenizerProcessor(
            sampling_rate=24000, normalize_audio=False
        )
        path = str(audio_tmp_dir / "save_spy.wav")
        proc.save_audio(np.random.randn(4800).astype(np.float32) * 0.3,
                        output_path=path, sampling_rate=24000)
        assert len(calls) == 1
        assert calls[0][0] == path
        assert calls[0][1] == 24000


# ====================================================================
# 3. ASR processor: _process_single_audio file path branch
# ====================================================================

class TestAsrProcessorFileLoading:
    def _make_asr_processor(self, real_vendored, target_sr=24000):
        return real_vendored["asr"].VibeVoiceASRProcessor(
            tokenizer=_StubTokenizer(),
            speech_tok_compress_ratio=3200,
            target_sample_rate=target_sr,
            normalize_audio=False,
        )

    def test_file_path_loads_via_backend(self, real_vendored, audio_tmp_dir):
        proc = self._make_asr_processor(real_vendored)
        path = _write_wav(audio_tmp_dir / "asr_load.wav", sr=24000, seconds=1.0)
        out = proc._process_single_audio(path)
        assert "speech" in out
        speech = out["speech"]
        assert speech.dtype == np.float32
        assert abs(len(speech) - 24000) <= 96  # 1 s @ 24 kHz

    def test_file_path_resamples_to_target(self, real_vendored, audio_tmp_dir):
        proc = self._make_asr_processor(real_vendored, target_sr=24000)
        path = _write_wav(audio_tmp_dir / "asr_16k.wav", sr=16000, seconds=1.0)
        out = proc._process_single_audio(path)
        assert abs(len(out["speech"]) - 24000) <= 96

    def test_file_path_calls_backend_load(self, real_vendored, audio_tmp_dir, monkeypatch):
        backend = real_vendored["backend"]
        calls = []
        original = backend.load_audio_file

        def spy(path):
            calls.append(path)
            return original(path)

        monkeypatch.setattr(backend, "load_audio_file", spy)
        proc = self._make_asr_processor(real_vendored)
        path = _write_wav(audio_tmp_dir / "asr_spy.wav", sr=24000, seconds=0.5)
        proc._process_single_audio(path)
        assert calls == [path]

    def test_tensor_input_bypasses_file_loading(self, real_vendored, monkeypatch):
        import torch

        backend = real_vendored["backend"]

        def boom(path):
            raise AssertionError("load_audio_file must not be called for tensor input")

        monkeypatch.setattr(backend, "load_audio_file", boom)
        proc = self._make_asr_processor(real_vendored)
        out = proc._process_single_audio(torch.randn(24000))
        assert len(out["speech"]) == 24000

    def test_output_structure(self, real_vendored, audio_tmp_dir):
        proc = self._make_asr_processor(real_vendored)
        path = _write_wav(audio_tmp_dir / "asr_struct.wav", sr=24000, seconds=1.0)
        out = proc._process_single_audio(path)
        for key in ("input_ids", "acoustic_input_mask", "speech", "vae_tok_len"):
            assert key in out
        assert out["vae_tok_len"] > 0
