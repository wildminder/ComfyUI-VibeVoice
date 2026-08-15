"""Integration tests for VibeVoice TTS pipeline (mocked)."""

import numpy as np
import torch
import pytest
from unittest.mock import patch, MagicMock

from ComfyUI_VibeVoice.modules.generation import load_vibevoice_model, generate_audio, force_offload_model
from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_PATCHER_CACHE
from ComfyUI_VibeVoice.modules.loader import LOADED_MODELS_CACHE


def _mock_voice_sample(length: int = 24000) -> np.ndarray:
    """Create a mock 1-D voice sample numpy array."""
    return np.random.randn(length).astype(np.float32)


def _mock_audio_dict(length: int = 24000, sr: int = 24000) -> dict:
    """Create a mock ComfyUI AUDIO dict."""
    return {"waveform": torch.randn(1, 1, length), "sample_rate": sr}


class TestFullPipelineMocked:
    """End-to-end pipeline tests with mocked model."""

    def test_full_pipeline_mocked(self):
        """Test the full pipeline with a mocked model."""
        VIBEVOICE_PATCHER_CACHE.clear()
        LOADED_MODELS_CACHE.clear()

        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_output = MagicMock()
        mock_output.speech_outputs = [torch.randn(24000)]
        mock_model.generate.return_value = mock_output

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.randint(0, 100, (1, 10))}
        mock_processor.tokenizer = MagicMock()

        mock_patcher = MagicMock()
        mock_patcher.model.model = mock_model
        mock_patcher.model.processor = mock_processor

        with patch("ComfyUI_VibeVoice.modules.generation.VibeVoiceModelHandler") as mock_handler_cls, \
             patch("ComfyUI_VibeVoice.modules.generation.VibeVoicePatcher", return_value=mock_patcher), \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.load_model_gpu"), \
             patch("ComfyUI_VibeVoice.modules.generation.ProgressBar"), \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio", return_value=_mock_voice_sample()):
            mock_handler = MagicMock()
            mock_handler.size = 1000
            mock_handler_cls.return_value = mock_handler

            patcher, model, processor = load_vibevoice_model(
                model_name="TestModel",
                device="cpu",
                dtype="fp32",
                attention_mode="sdpa",
            )

            assert model is not None
            assert processor is not None

            waveform, sr = generate_audio(
                model=model,
                processor=processor,
                text="[1] Hello world",
                voice_samples=[_mock_audio_dict()],
                speaker_ids=[1],
            )

            assert waveform is not None
            assert sr == 24000
            assert waveform.ndim == 3

        VIBEVOICE_PATCHER_CACHE.clear()
        LOADED_MODELS_CACHE.clear()

    def test_model_loading_flow(self):
        """Test that model loading creates a patcher and loads via ComfyUI."""
        VIBEVOICE_PATCHER_CACHE.clear()
        LOADED_MODELS_CACHE.clear()

        mock_patcher = MagicMock()
        mock_patcher.model.model = MagicMock()
        mock_patcher.model.processor = MagicMock()

        with patch("ComfyUI_VibeVoice.modules.generation.VibeVoiceModelHandler") as mock_handler_cls, \
             patch("ComfyUI_VibeVoice.modules.generation.VibeVoicePatcher", return_value=mock_patcher) as mock_patcher_cls, \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.load_model_gpu") as mock_load_gpu:
            mock_handler = MagicMock()
            mock_handler.size = 1000
            mock_handler_cls.return_value = mock_handler

            patcher, model, processor = load_vibevoice_model(
                model_name="TestModel",
                device="cpu",
                dtype="fp32",
                attention_mode="sdpa",
            )

            mock_handler_cls.assert_called_once()
            mock_patcher_cls.assert_called_once()
            mock_load_gpu.assert_called_once_with(mock_patcher)

        VIBEVOICE_PATCHER_CACHE.clear()
        LOADED_MODELS_CACHE.clear()

    def test_force_offload(self):
        """Test that force_offload calls unpatch_model."""
        mock_patcher = MagicMock()
        mock_patcher.is_loaded = True

        with patch("ComfyUI_VibeVoice.modules.generation.model_management"):
            force_offload_model(mock_patcher, "TestModel")

            mock_patcher.unpatch_model.assert_called_once_with(unpatch_weights=True)

    def test_multi_speaker(self):
        """Test multi-speaker generation with reference audio."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_output = MagicMock()
        mock_output.speech_outputs = [torch.randn(48000)]
        mock_model.generate.return_value = mock_output

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.randint(0, 100, (1, 10))}
        mock_processor.tokenizer = MagicMock()

        with patch("ComfyUI_VibeVoice.modules.generation.ProgressBar"), \
             patch("ComfyUI_VibeVoice.modules.generation.model_management"), \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio", return_value=_mock_voice_sample()):
            waveform, sr = generate_audio(
                model=mock_model,
                processor=mock_processor,
                text="[1] Hello\n[2] Hi there",
                voice_samples=[_mock_audio_dict(), _mock_audio_dict()],
                speaker_ids=[1, 2],
            )

            assert waveform is not None
            assert sr == 24000

    def test_zero_shot(self):
        """Test zero-shot TTS (no reference audio)."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_output = MagicMock()
        mock_output.speech_outputs = [torch.randn(24000)]
        mock_model.generate.return_value = mock_output

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.randint(0, 100, (1, 10))}
        mock_processor.tokenizer = MagicMock()

        with patch("ComfyUI_VibeVoice.modules.generation.ProgressBar"), \
             patch("ComfyUI_VibeVoice.modules.generation.model_management"), \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio", return_value=_mock_voice_sample()):
            waveform, sr = generate_audio(
                model=mock_model,
                processor=mock_processor,
                text="[1] This is zero-shot TTS",
                voice_samples=[_mock_audio_dict()],
                speaker_ids=[1],
            )

            assert waveform is not None
            assert sr == 24000

    def test_interrupt_handling(self):
        """Test that interrupt during generation raises InterruptProcessingException."""
        import comfy.model_management as mm
        InterruptException = mm.InterruptProcessingException

        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.generate.side_effect = InterruptException()

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.randint(0, 100, (1, 10))}
        mock_processor.tokenizer = MagicMock()

        # Patch ProgressBar but keep model_management real for the exception class
        with patch("ComfyUI_VibeVoice.modules.generation.ProgressBar"), \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.interrupt_current_processing", return_value=False), \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio", return_value=_mock_voice_sample()):
            with pytest.raises(InterruptException):
                generate_audio(
                    model=mock_model,
                    processor=mock_processor,
                    text="[1] Test",
                    voice_samples=[_mock_audio_dict()],
                    speaker_ids=[1],
                )


class TestNodeProgressIntegration:
    """Phase 6 (2026-08-15 progress plan): each node's execute path must drive
    the standard ComfyUI ProgressBar with increasing values. The mocked model
    invokes the progress hook 3x, simulating a 3-step loop."""

    def test_tts_node_reports_increasing_progress(self):
        from ComfyUI_VibeVoice.nodes.tts_node import VibeVoiceTTSNode

        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_output = MagicMock()
        mock_output.speech_outputs = [torch.randn(24000)]

        def _generate(**kwargs):
            cb = kwargs.get("progress_callback")
            if cb is not None:
                for i in (1, 2, 3):
                    cb(i, 10)
            return mock_output

        mock_model.generate.side_effect = _generate
        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.randint(0, 100, (1, 10))}
        mock_processor.tokenizer = MagicMock()
        mock_patcher = MagicMock()

        with patch("ComfyUI_VibeVoice.nodes.tts_node.load_vibevoice_model",
                   return_value=(mock_patcher, mock_model, mock_processor)), \
             patch("ComfyUI_VibeVoice.nodes.tts_node.ui.PreviewAudio", MagicMock()), \
             patch("ComfyUI_VibeVoice.modules.generation.ProgressBar") as mock_pbar_cls, \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.throw_exception_if_processing_interrupted"), \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio",
                   return_value=_mock_voice_sample()):
            mock_pbar = MagicMock()
            mock_pbar.total = 10
            mock_pbar_cls.return_value = mock_pbar

            VibeVoiceTTSNode.execute(
                model_name="TestModel",
                text="[1] Hello world",
                quantize_llm_4bit=False,
                attention_mode="sdpa",
                cfg_scale=1.3,
                inference_steps=10,
                seed=42,
                do_sample=True,
                temperature=0.95,
                top_p=0.95,
                top_k=0,
                force_offload=False,
                device="cpu",
                dtype="fp32",
                max_new_tokens=0,
                speaker_1_voice=_mock_audio_dict(),
            )

            values = [c.args[0] for c in mock_pbar.update_absolute.call_args_list if c.args]
            assert 1 in values and 2 in values and 3 in values
            assert values == sorted(values), f"progress must be monotonic, got {values}"

    def test_realtime_node_reports_increasing_progress(self):
        from ComfyUI_VibeVoice.nodes.realtime_node import VibeVoiceRealtimeNode

        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_output = MagicMock()
        mock_output.speech_outputs = [torch.randn(24000)]

        def _generate(**kwargs):
            cb = kwargs.get("progress_callback")
            if cb is not None:
                for i in (1, 2, 3):
                    cb(i, 512)
            return mock_output

        mock_model.generate.side_effect = _generate
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.encode = MagicMock(return_value=[10, 11, 12])
        mock_processor.prepare_speech_inputs = MagicMock(return_value={
            "padded_speeches": torch.randn(1, 100),
            "speech_masks": torch.ones(1, 100, dtype=torch.bool),
        })
        mock_patcher = MagicMock()

        with patch("ComfyUI_VibeVoice.nodes.realtime_node.load_vibevoice_model",
                   return_value=(mock_patcher, mock_model, mock_processor)), \
             patch("ComfyUI_VibeVoice.nodes.realtime_node.ui.PreviewAudio", MagicMock()), \
             patch("ComfyUI_VibeVoice.modules.generation.ProgressBar") as mock_pbar_cls, \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.throw_exception_if_processing_interrupted"), \
             patch("ComfyUI_VibeVoice.modules.generation.prefill_voice_prompt",
                   return_value={"lm": MagicMock()}), \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio",
                   return_value=_mock_voice_sample()):
            mock_pbar = MagicMock()
            mock_pbar.total = 1
            mock_pbar_cls.return_value = mock_pbar

            VibeVoiceRealtimeNode.execute(
                model_name="TestStreamingModel",
                text="[1] Hello world",
                quantize_llm_4bit=False,
                attention_mode="sdpa",
                cfg_scale=1.3,
                inference_steps=10,
                seed=42,
                do_sample=True,
                temperature=0.95,
                top_p=0.95,
                top_k=0,
                stream=False,
                force_offload=False,
                device="cpu",
                dtype="fp32",
                speaker_1_voice=_mock_audio_dict(),
            )

            values = [c.args[0] for c in mock_pbar.update_absolute.call_args_list if c.args]
            # Loop updates (all but the final guaranteed event) must be monotonic.
            loop_values = values[:-1]
            assert 1 in loop_values and 2 in loop_values and 3 in loop_values
            assert loop_values == sorted(loop_values), f"progress must be monotonic, got {loop_values}"
            # The final event must always be present (guaranteed 100%).
            assert values[-1] == mock_pbar.total

    def test_asr_node_reports_increasing_progress(self):
        from ComfyUI_VibeVoice.nodes.asr_node import VibeVoiceASRNode

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        def _generate(**kwargs):
            streamer = kwargs.get("streamer")
            if streamer is not None:
                for tok in (10, 11, 12):
                    streamer.put(torch.tensor([tok]))
                streamer.end()
            return torch.tensor([[1, 2, 3, 10, 11, 12, 0]])

        mock_model.generate.side_effect = _generate
        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_processor.pad_id = 0
        mock_processor.tokenizer.eos_token_id = 0
        mock_processor.decode.return_value = "Hello world"
        mock_processor.post_process_transcription.return_value = []
        mock_patcher = MagicMock()

        with patch("ComfyUI_VibeVoice.nodes.asr_node.load_asr_model_patched",
                   return_value=(mock_patcher, mock_model, mock_processor)), \
             patch("ComfyUI_VibeVoice.modules.asr_generation.ProgressBar") as mock_pbar_cls, \
             patch("ComfyUI_VibeVoice.modules.asr_generation.model_management.throw_exception_if_processing_interrupted"):
            mock_pbar = MagicMock()
            mock_pbar.total = 32768
            mock_pbar_cls.return_value = mock_pbar

            VibeVoiceASRNode.execute(
                model_name="VibeVoice-ASR",
                audio=_mock_audio_dict(),
                context_info="",
                max_new_tokens=32768,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                num_beams=1,
                device="cpu",
                dtype="fp32",
                attention_mode="sdpa",
                force_offload=False,
            )

            values = [c.args[0] for c in mock_pbar.update_absolute.call_args_list if c.args]
            assert 1 in values and 2 in values and 3 in values
            assert values == sorted(values), f"progress must be monotonic, got {values}"
