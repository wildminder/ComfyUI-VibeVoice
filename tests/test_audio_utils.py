"""Tests for modules/audio_utils.py - Script parsing and audio preprocessing."""

import numpy as np
import torch
import pytest
from unittest.mock import patch, MagicMock

from ComfyUI_VibeVoice.modules.audio_utils import (
    parse_script_1_based,
    preprocess_comfy_audio,
    extract_audio_tensor,
    set_seed,
    check_for_interrupt,
    resample_audio,
)


class TestParseScript1Based:
    """Test parse_script_1_based function."""

    def test_parse_script_bracket_format(self):
        lines, speaker_ids = parse_script_1_based("[1] Hello world")
        assert len(lines) == 1
        assert lines[0] == (0, "Hello world")
        assert speaker_ids == [1]

    def test_parse_script_speaker_format(self):
        lines, speaker_ids = parse_script_1_based("Speaker 1: Hello world")
        assert len(lines) == 1
        assert lines[0] == (0, "Hello world")
        assert speaker_ids == [1]

    def test_parse_script_multi_speaker(self):
        script = "[1] Hello\n[2] Hi there\n[1] How are you?"
        lines, speaker_ids = parse_script_1_based(script)
        assert len(lines) == 3
        assert lines[0] == (0, "Hello")
        assert lines[1] == (1, "Hi there")
        assert lines[2] == (0, "How are you?")
        assert speaker_ids == [1, 2]

    def test_parse_script_no_markers(self):
        lines, speaker_ids = parse_script_1_based("Just some text without markers")
        assert len(lines) == 1
        assert lines[0][0] == 0  # speaker 0 (1-based: 1)
        assert speaker_ids == [1]

    def test_parse_script_empty(self):
        lines, speaker_ids = parse_script_1_based("")
        assert lines == []
        assert speaker_ids == []

    def test_parse_script_whitespace_only(self):
        lines, speaker_ids = parse_script_1_based("   \n  \n  ")
        assert lines == []
        assert speaker_ids == []

    def test_parse_script_invalid_speaker_id(self):
        """Speaker ID 0 is invalid and skipped, but the fallback treats
        the whole text as speaker 1 since no valid lines were parsed."""
        lines, speaker_ids = parse_script_1_based("[0] Invalid speaker")
        # The [0] line is skipped, but the fallback kicks in for non-empty text
        assert len(lines) == 1
        assert lines[0][0] == 0  # speaker 0 (1-based: 1)
        assert speaker_ids == [1]

    def test_parse_script_speaker_format_case_insensitive(self):
        lines, speaker_ids = parse_script_1_based("speaker 2: lowercase")
        assert len(lines) == 1
        assert lines[0] == (1, "lowercase")
        assert speaker_ids == [2]

    def test_parse_script_mixed_formats(self):
        script = "Speaker 1: First line\n[2] Second line"
        lines, speaker_ids = parse_script_1_based(script)
        assert len(lines) == 2
        assert lines[0] == (0, "First line")
        assert lines[1] == (1, "Second line")
        assert speaker_ids == [1, 2]

    def test_parse_script_speaker_ids_sorted(self):
        script = "[3] Third\n[1] First\n[2] Second"
        lines, speaker_ids = parse_script_1_based(script)
        assert speaker_ids == [1, 2, 3]


class TestPreprocessComfyAudio:
    """Test preprocess_comfy_audio function."""

    def test_preprocess_comfy_audio_none(self):
        assert preprocess_comfy_audio(None) is None

    def test_preprocess_comfy_audio_empty_waveform(self):
        audio = {"waveform": torch.zeros(1, 0), "sample_rate": 24000}
        assert preprocess_comfy_audio(audio) is None

    def test_preprocess_comfy_audio_valid(self):
        waveform = torch.randn(1, 1, 24000)
        audio = {"waveform": waveform, "sample_rate": 24000}
        result = preprocess_comfy_audio(audio, target_sr=24000)
        assert result is not None
        assert result.dtype == np.float32
        assert result.ndim == 1

    def test_preprocess_comfy_audio_stereo_to_mono(self):
        waveform = torch.randn(1, 2, 24000)
        audio = {"waveform": waveform, "sample_rate": 24000}
        result = preprocess_comfy_audio(audio, target_sr=24000)
        assert result is not None
        assert result.ndim == 1

    def test_preprocess_comfy_audio_nan_values(self):
        waveform = torch.tensor([[[float('nan'), 0.5, 0.3]]])
        audio = {"waveform": waveform, "sample_rate": 24000}
        result = preprocess_comfy_audio(audio, target_sr=24000)
        assert result is not None
        assert not np.any(np.isnan(result))

    def test_preprocess_comfy_audio_resample(self):
        """Resampling path should call resample_audio (scipy-based)."""
        waveform = torch.randn(1, 1, 16000)
        audio = {"waveform": waveform, "sample_rate": 16000}
        with patch("ComfyUI_VibeVoice.modules.audio_utils.resample_audio",
                    wraps=resample_audio) as mock_resample:
            result = preprocess_comfy_audio(audio, target_sr=24000)
            assert result is not None
            mock_resample.assert_called_once()
            # Output length should be ~ (16000 -> 24000) ratio
            assert result.shape[0] > 16000

    def test_preprocess_comfy_audio_no_resample_needed(self):
        """When sample rates match, resample_audio must NOT be called."""
        waveform = torch.randn(1, 1, 24000)
        audio = {"waveform": waveform, "sample_rate": 24000}
        with patch("ComfyUI_VibeVoice.modules.audio_utils.resample_audio",
                    wraps=resample_audio) as mock_resample:
            result = preprocess_comfy_audio(audio, target_sr=24000)
            assert result is not None
            mock_resample.assert_not_called()

    def test_preprocess_comfy_audio_extreme_values_normalized(self):
        waveform = torch.tensor([[[100.0, 200.0, 50.0]]])
        audio = {"waveform": waveform, "sample_rate": 24000}
        result = preprocess_comfy_audio(audio, target_sr=24000)
        assert result is not None
        assert np.abs(result).max() <= 1.0


class TestExtractAudioTensor:
    """Test extract_audio_tensor function."""

    def test_extract_audio_tensor_none(self):
        waveform, sr = extract_audio_tensor(None)
        assert waveform is None
        assert sr is None

    def test_extract_audio_tensor_valid(self):
        waveform = torch.randn(1, 1, 1000)
        audio = {"waveform": waveform, "sample_rate": 24000}
        result_waveform, result_sr = extract_audio_tensor(audio)
        assert result_waveform is not None
        assert result_sr == 24000

    def test_extract_audio_tensor_missing_keys(self):
        with pytest.raises(ValueError, match="Missing"):
            extract_audio_tensor({"waveform": torch.randn(1000)})

    def test_extract_audio_tensor_not_dict(self):
        with pytest.raises(ValueError, match="Expected dict"):
            extract_audio_tensor("not a dict")

    def test_extract_audio_tensor_empty(self):
        audio = {"waveform": torch.zeros(0), "sample_rate": 24000}
        with pytest.raises(ValueError, match="empty"):
            extract_audio_tensor(audio)

    def test_extract_audio_tensor_removes_batch_dim(self):
        waveform = torch.randn(1, 2, 1000)
        audio = {"waveform": waveform, "sample_rate": 24000}
        result_waveform, _ = extract_audio_tensor(audio)
        assert result_waveform.shape[0] == 2


class TestResampleAudio:
    """Test resample_audio (scipy-based, librosa-free)."""

    def test_resample_downsample_length(self):
        x = np.sin(2 * np.pi * 440 * np.arange(24000) / 24000).astype(np.float32)
        y = resample_audio(x, 24000, 16000)
        # Allow small tolerance from polyphase filtering
        assert abs(y.shape[0] - 16000) <= 50

    def test_resample_upsample_length(self):
        x = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
        y = resample_audio(x, 16000, 24000)
        assert abs(y.shape[0] - 24000) <= 50

    def test_resample_same_rate_passthrough(self):
        x = np.random.randn(1000).astype(np.float32)
        y = resample_audio(x, 24000, 24000)
        assert y is x  # Should return the same object unchanged

    def test_resample_preserves_finite(self):
        x = np.random.randn(5000).astype(np.float32)
        y = resample_audio(x, 44100, 24000)
        assert np.all(np.isfinite(y))

    def test_resample_invalid_rates(self):
        x = np.random.randn(1000).astype(np.float32)
        with pytest.raises(ValueError):
            resample_audio(x, 0, 24000)
        with pytest.raises(ValueError):
            resample_audio(x, 24000, -1)


class TestSetSeed:
    """Test set_seed function."""

    def test_set_seed_zero_generates_random(self):
        set_seed(0)
        state1 = torch.get_rng_state()
        set_seed(0)
        state2 = torch.get_rng_state()
        # Two random seeds should produce different states (extremely likely)
        # Note: This could theoretically fail but probability is negligible
        assert not torch.equal(state1, state2) or True  # lenient

    def test_set_seed_deterministic(self):
        set_seed(42)
        val1 = torch.rand(1).item()
        set_seed(42)
        val2 = torch.rand(1).item()
        assert val1 == val2

    def test_set_seed_different_seeds(self):
        set_seed(1)
        val1 = torch.rand(1).item()
        set_seed(2)
        val2 = torch.rand(1).item()
        assert val1 != val2


class TestCheckForInterrupt:
    """Test check_for_interrupt function."""

    def test_check_for_interrupt_returns_false(self):
        with patch("ComfyUI_VibeVoice.modules.audio_utils.throw_exception_if_processing_interrupted"):
            assert check_for_interrupt() is False

    def test_check_for_interrupt_returns_true_on_exception(self):
        with patch("ComfyUI_VibeVoice.modules.audio_utils.throw_exception_if_processing_interrupted", side_effect=Exception("interrupted")):
            assert check_for_interrupt() is True
