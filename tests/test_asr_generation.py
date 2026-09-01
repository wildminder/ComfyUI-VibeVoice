"""Tests for modules/asr_generation.py - ASR transcription."""

import torch
import pytest
from unittest.mock import patch, MagicMock

from ComfyUI_VibeVoice.modules.asr_generation import load_asr_model, load_asr_model_patched, transcribe_audio, force_offload_asr_model


class TestLoadASRModel:
    """Test load_asr_model function."""

    def test_load_model_cache_miss(self):
        """When not cached, model should be loaded."""
        from ComfyUI_VibeVoice.modules.asr_loader import LOADED_ASR_MODELS_CACHE
        LOADED_ASR_MODELS_CACHE.clear()

        mock_model = MagicMock()
        mock_processor = MagicMock()

        with patch("ComfyUI_VibeVoice.modules.asr_generation.VibeVoiceASRLoader") as mock_loader_cls:
            mock_loader_cls.load_model.return_value = (mock_model, mock_processor)

            model, processor = load_asr_model(
                model_name="VibeVoice-ASR",
                device="cpu",
                dtype="fp32",
                attention_mode="sdpa",
            )

            assert model == mock_model
            assert processor == mock_processor
            mock_loader_cls.load_model.assert_called_once()

        LOADED_ASR_MODELS_CACHE.clear()

    def test_load_model_cache_hit(self):
        """When cached, existing model should be returned."""
        from ComfyUI_VibeVoice.modules.asr_loader import LOADED_ASR_MODELS_CACHE

        mock_model = MagicMock()
        mock_processor = MagicMock()
        cache_key = "asr_VibeVoice-ASR_fp32_sdpa"
        LOADED_ASR_MODELS_CACHE[cache_key] = (mock_model, mock_processor)

        with patch("ComfyUI_VibeVoice.modules.asr_generation.VibeVoiceASRLoader") as mock_loader_cls:
            model, processor = load_asr_model(
                model_name="VibeVoice-ASR",
                device="cpu",
                dtype="fp32",
                attention_mode="sdpa",
            )

            assert model == mock_model
            mock_loader_cls.load_model.assert_not_called()

        LOADED_ASR_MODELS_CACHE.clear()

    def test_load_model_force_reload(self):
        """force_reload should clear cache and reload."""
        from ComfyUI_VibeVoice.modules.asr_loader import LOADED_ASR_MODELS_CACHE

        old_model = MagicMock()
        cache_key = "asr_VibeVoice-ASR_fp32_sdpa"
        LOADED_ASR_MODELS_CACHE[cache_key] = (old_model, MagicMock())

        new_model = MagicMock()
        new_processor = MagicMock()

        with patch("ComfyUI_VibeVoice.modules.asr_generation.VibeVoiceASRLoader") as mock_loader_cls, \
             patch("ComfyUI_VibeVoice.modules.asr_generation.cleanup_asr_models"):
            mock_loader_cls.load_model.return_value = (new_model, new_processor)

            model, processor = load_asr_model(
                model_name="VibeVoice-ASR",
                device="cpu",
                dtype="fp32",
                attention_mode="sdpa",
                force_reload=True,
            )

            assert model == new_model

        LOADED_ASR_MODELS_CACHE.clear()


class TestTranscribeAudio:
    """Test transcribe_audio function."""

    def test_transcribe_basic(self):
        """Test basic transcription with mocked model."""
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        # Mock generate output
        mock_output = torch.tensor([[1, 2, 3, 4, 5, 0]])  # input + generated + eos
        mock_model.generate.return_value = mock_output

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_processor.pad_id = 0
        mock_processor.tokenizer.eos_token_id = 0
        mock_processor.decode.return_value = "Hello world"
        mock_processor.post_process_transcription.return_value = [
            {"speaker": 0, "text": "Hello world", "start": 0.0, "end": 1.0}
        ]

        audio_input = {
            "waveform": torch.randn(1, 1, 24000),
            "sample_rate": 24000,
        }

        with patch("ComfyUI_VibeVoice.modules.asr_generation.extract_audio_tensor") as mock_extract:
            mock_extract.return_value = (torch.randn(24000), 24000)
            raw_text, segments = transcribe_audio(
                model=mock_model,
                processor=mock_processor,
                audio_input=audio_input,
            )

            assert raw_text == "Hello world"
            assert len(segments) == 1
            assert segments[0]["text"] == "Hello world"

    def test_transcribe_with_context_info(self):
        """Test transcription with hotwords/context info."""
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        mock_output = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = mock_output

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_processor.pad_id = 0
        mock_processor.tokenizer.eos_token_id = 0
        mock_processor.decode.return_value = "Transcribed text"
        mock_processor.post_process_transcription.return_value = []

        audio_input = {
            "waveform": torch.randn(1, 1, 24000),
            "sample_rate": 24000,
        }

        with patch("ComfyUI_VibeVoice.modules.asr_generation.extract_audio_tensor") as mock_extract:
            mock_extract.return_value = (torch.randn(24000), 24000)
            raw_text, segments = transcribe_audio(
                model=mock_model,
                processor=mock_processor,
                audio_input=audio_input,
                context_info="Tea Brew, Aiden Host",
            )

            # Verify context_info was passed to processor
            call_kwargs = mock_processor.call_args[1]
            assert call_kwargs.get("context_info") == "Tea Brew, Aiden Host"

    def test_transcribe_none_audio_raises(self):
        """Should raise ValueError if audio is None."""
        mock_model = MagicMock()
        mock_processor = MagicMock()

        with patch("ComfyUI_VibeVoice.modules.asr_generation.extract_audio_tensor", return_value=(None, None)):
            with pytest.raises(ValueError, match="Audio input is required"):
                transcribe_audio(
                    model=mock_model,
                    processor=mock_processor,
                    audio_input={"waveform": None, "sample_rate": 24000},
                )


class TestForceOffloadASRModel:
    """Test force_offload_asr_model function."""

    def test_force_offload_calls_cleanup(self):
        with patch("ComfyUI_VibeVoice.modules.asr_generation.cleanup_asr_models") as mock_cleanup, \
             patch("ComfyUI_VibeVoice.modules.asr_generation.gc"), \
             patch("ComfyUI_VibeVoice.modules.asr_generation.model_management"):
            force_offload_asr_model("VibeVoice-ASR")
            mock_cleanup.assert_called_once()


class TestLoadASRModelPatched:
    """CRIT-001: load ASR through the ModelPatcher / VRAM system."""

    def _patched_load(self, model_name="VibeVoice-ASR", device="cpu", dtype="fp32", attention_mode="sdpa"):
        """Run load_asr_model_patched with the heavy bits stubbed."""
        from ComfyUI_VibeVoice.modules.asr_generation import load_asr_model_patched
        from ComfyUI_VibeVoice.modules.asr_loader import VibeVoiceASRModelHandler

        def fake_load(self, device, attention_mode="sdpa"):
            self.model = torch.nn.Linear(8, 8)
            self.processor = object()

        with patch.object(VibeVoiceASRModelHandler, "load_model", fake_load), \
             patch("comfy.model_patcher.ModelPatcher.patch_model"), \
             patch("ComfyUI_VibeVoice.modules.asr_generation.model_management.load_model_gpu",
                   side_effect=lambda p: p.patch_model()), \
             patch("ComfyUI_VibeVoice.modules.asr_generation.get_torch_device", return_value=torch.device("cpu")), \
             patch("ComfyUI_VibeVoice.modules.asr_generation.get_offload_device", return_value=torch.device("cpu")):
            return load_asr_model_patched(
                model_name=model_name, device=device, dtype=dtype, attention_mode=attention_mode
            )

    def test_load_asr_model_patched_returns_patcher(self):
        from ComfyUI_VibeVoice.modules.patcher import VibeVoiceASRPatcher
        from ComfyUI_VibeVoice.modules.asr_loader import LOADED_ASR_MODELS_CACHE
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_ASR_PATCHER_CACHE

        LOADED_ASR_MODELS_CACHE.clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()

        patcher, model, processor = self._patched_load()

        assert isinstance(patcher, VibeVoiceASRPatcher)
        assert model is not None
        assert processor is not None
        assert patcher.is_loaded is True
        assert "asr_VibeVoice-ASR_attn_sdpa" in VIBEVOICE_ASR_PATCHER_CACHE

        LOADED_ASR_MODELS_CACHE.clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()

    def test_asr_patcher_cache_key_reuse(self):
        """A second load with the same key returns the same patcher (no reload)."""
        from ComfyUI_VibeVoice.modules.asr_loader import VibeVoiceASRModelHandler, LOADED_ASR_MODELS_CACHE
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_ASR_PATCHER_CACHE
        from ComfyUI_VibeVoice.modules.asr_generation import load_asr_model_patched

        LOADED_ASR_MODELS_CACHE.clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()

        load_calls = {"n": 0}

        def fake_load(self, device, attention_mode="sdpa"):
            load_calls["n"] += 1
            self.model = torch.nn.Linear(8, 8)
            self.processor = object()

        with patch.object(VibeVoiceASRModelHandler, "load_model", fake_load), \
             patch("comfy.model_patcher.ModelPatcher.patch_model"), \
             patch("ComfyUI_VibeVoice.modules.asr_generation.model_management.load_model_gpu",
                   side_effect=lambda p: p.patch_model()), \
             patch("ComfyUI_VibeVoice.modules.asr_generation.get_torch_device", return_value=torch.device("cpu")), \
             patch("ComfyUI_VibeVoice.modules.asr_generation.get_offload_device", return_value=torch.device("cpu")):
            patcher1, _, _ = load_asr_model_patched(
                model_name="VibeVoice-ASR", device="cpu", dtype="fp32", attention_mode="sdpa"
            )
            patcher2, _, _ = load_asr_model_patched(
                model_name="VibeVoice-ASR", device="cpu", dtype="fp32", attention_mode="sdpa"
            )

        assert patcher1 is patcher2
        assert load_calls["n"] == 1

        LOADED_ASR_MODELS_CACHE.clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()


class TestExternalVibeVoiceASRModelHandler:
    """Phase 4: handler for an externally-loaded (pre-instantiated) ASR model."""

    def _make_handler(self, model=None, processor=None, name="ext-asr", bundle=None):
        from ComfyUI_VibeVoice.modules.asr_generation import ExternalVibeVoiceASRModelHandler
        model = model if model is not None else torch.nn.Linear(8, 8)
        processor = processor if processor is not None else object()
        return ExternalVibeVoiceASRModelHandler(model, processor, name, bundle)

    def test_handler_holds_preloaded_model_and_processor(self):
        model = torch.nn.Linear(8, 8)
        processor = object()
        handler = self._make_handler(model=model, processor=processor)
        assert handler.model is model
        assert handler.processor is processor

    def test_handler_model_is_not_none(self):
        """The patcher's skip-reload contract requires model to be pre-set."""
        handler = self._make_handler()
        assert handler.model is not None

    def test_handler_default_cache_key(self):
        handler = self._make_handler(name="ext-asr")
        assert handler.cache_key == "asr_external_ext-asr"

    def test_handler_pack_name_matches(self):
        handler = self._make_handler(name="ext-asr")
        assert handler.model_pack_name == "ext-asr"
        assert handler.model_name == "ext-asr"

    def test_handler_size_from_bundle_hint(self):
        bundle = {"size_gb": 2.0}
        handler = self._make_handler(bundle=bundle)
        assert handler.size == int(2.0 * (1024**3))

    def test_handler_size_from_parameters_when_no_hint(self):
        model = torch.nn.Linear(16, 16)
        handler = self._make_handler(model=model, bundle=None)
        expected = sum(p.numel() * p.element_size() for p in model.parameters())
        assert handler.size == expected

    def test_handler_load_model_is_noop(self):
        """load_model must not replace the pre-loaded model."""
        model = torch.nn.Linear(8, 8)
        handler = self._make_handler(model=model)
        handler.load_model(torch.device("cpu"))
        assert handler.model is model


class TestLoadASRFromExternal:
    """Phase 4: load_asr_from_external wraps a pre-loaded bundle in a patcher."""

    def _make_bundle(self, name="ext-asr"):
        return {
            "model": torch.nn.Linear(8, 8),
            "processor": object(),
            "model_name": name,
            "source_path": "fake.safetensors",
        }

    def _patched_load(self, bundle, device="cpu", dtype="fp32", attention_mode="sdpa"):
        from ComfyUI_VibeVoice.modules.asr_generation import load_asr_from_external

        with patch("comfy.model_patcher.ModelPatcher.patch_model"), \
             patch("ComfyUI_VibeVoice.modules.asr_generation.model_management.load_model_gpu",
                   side_effect=lambda p: p.patch_model()), \
             patch("ComfyUI_VibeVoice.modules.asr_generation.get_torch_device", return_value=torch.device("cpu")), \
             patch("ComfyUI_VibeVoice.modules.asr_generation.get_offload_device", return_value=torch.device("cpu")):
            return load_asr_from_external(
                bundle, device=device, dtype=dtype, attention_mode=attention_mode
            )

    def test_missing_model_key_raises(self):
        from ComfyUI_VibeVoice.modules.asr_generation import load_asr_from_external
        bundle = self._make_bundle()
        del bundle["model"]
        with pytest.raises(ValueError, match="model"):
            load_asr_from_external(bundle)

    def test_missing_processor_key_raises(self):
        from ComfyUI_VibeVoice.modules.asr_generation import load_asr_from_external
        bundle = self._make_bundle()
        del bundle["processor"]
        with pytest.raises(ValueError, match="processor"):
            load_asr_from_external(bundle)

    def test_missing_model_name_key_raises(self):
        from ComfyUI_VibeVoice.modules.asr_generation import load_asr_from_external
        bundle = self._make_bundle()
        del bundle["model_name"]
        with pytest.raises(ValueError, match="model_name"):
            load_asr_from_external(bundle)

    def test_none_model_value_raises(self):
        from ComfyUI_VibeVoice.modules.asr_generation import load_asr_from_external
        bundle = self._make_bundle()
        bundle["model"] = None
        with pytest.raises(ValueError, match="model"):
            load_asr_from_external(bundle)

    def test_returns_patcher_model_processor(self):
        from ComfyUI_VibeVoice.modules.patcher import VibeVoiceASRPatcher
        from ComfyUI_VibeVoice.modules.asr_loader import LOADED_ASR_MODELS_CACHE
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_ASR_PATCHER_CACHE

        LOADED_ASR_MODELS_CACHE.clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()

        bundle = self._make_bundle()
        patcher, model, processor = self._patched_load(bundle)

        assert isinstance(patcher, VibeVoiceASRPatcher)
        assert model is bundle["model"]
        assert processor is bundle["processor"]

        LOADED_ASR_MODELS_CACHE.clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()

    def test_caches_patcher_under_external_key(self):
        from ComfyUI_VibeVoice.modules.asr_loader import LOADED_ASR_MODELS_CACHE
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_ASR_PATCHER_CACHE
        from ComfyUI_VibeVoice.modules.model_registry import identity_for_external

        LOADED_ASR_MODELS_CACHE.clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()

        bundle = self._make_bundle(name="ext-asr")
        self._patched_load(bundle, attention_mode="sdpa")

        # Hand-built bundle: stat fallback (missing file), widget dtype.
        key = identity_for_external(
            "fake.safetensors", "ext-asr", "sdpa",
            use_llm_4bit=False, dtype_str="fp32", prefix="asr_external",
        )
        assert key in VIBEVOICE_ASR_PATCHER_CACHE

        LOADED_ASR_MODELS_CACHE.clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()

    def test_registers_loaded_asr_cache_entry(self):
        from ComfyUI_VibeVoice.modules.asr_loader import LOADED_ASR_MODELS_CACHE
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_ASR_PATCHER_CACHE
        from ComfyUI_VibeVoice.modules.model_registry import identity_for_external

        LOADED_ASR_MODELS_CACHE.clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()

        bundle = self._make_bundle(name="ext-asr")
        patcher, model, processor = self._patched_load(bundle)

        key = identity_for_external(
            "fake.safetensors", "ext-asr", "sdpa",
            use_llm_4bit=False, dtype_str="fp32", prefix="asr_external",
        )
        assert key in LOADED_ASR_MODELS_CACHE
        assert LOADED_ASR_MODELS_CACHE[key] == (model, processor)

        LOADED_ASR_MODELS_CACHE.clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()

    def test_handler_cache_key_synced_with_patcher(self):
        from ComfyUI_VibeVoice.modules.asr_loader import LOADED_ASR_MODELS_CACHE
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_ASR_PATCHER_CACHE

        LOADED_ASR_MODELS_CACHE.clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()

        bundle = self._make_bundle(name="ext-asr")
        patcher, _, _ = self._patched_load(bundle)

        # The handler's cache_key must match the patcher cache key so
        # unpatch_model clears the correct LOADED_ASR_MODELS_CACHE entry.
        assert patcher.model.cache_key == patcher.cache_key

        LOADED_ASR_MODELS_CACHE.clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()

    def test_cache_reuse_returns_same_patcher(self):
        from ComfyUI_VibeVoice.modules.asr_loader import LOADED_ASR_MODELS_CACHE
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_ASR_PATCHER_CACHE

        LOADED_ASR_MODELS_CACHE.clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()

        bundle = self._make_bundle(name="ext-asr")
        patcher1, _, _ = self._patched_load(bundle)
        patcher2, _, _ = self._patched_load(bundle)

        assert patcher1 is patcher2

        LOADED_ASR_MODELS_CACHE.clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()


class TestForceOffloadASRPatcher:
    """CRIT-001 S5: force_offload via patcher nulls the model and clears the cache."""

    def test_force_offload_asr_patcher_nulls_model(self):
        from ComfyUI_VibeVoice.modules.asr_generation import load_asr_model_patched, force_offload_asr_model
        from ComfyUI_VibeVoice.modules.asr_loader import VibeVoiceASRModelHandler, LOADED_ASR_MODELS_CACHE
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_ASR_PATCHER_CACHE

        LOADED_ASR_MODELS_CACHE.clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()

        def fake_load(self, device, attention_mode="sdpa"):
            self.model = torch.nn.Linear(8, 8)
            self.processor = object()

        with patch.object(VibeVoiceASRModelHandler, "load_model", fake_load), \
             patch("comfy.model_patcher.ModelPatcher.patch_model"), \
             patch("ComfyUI_VibeVoice.modules.asr_generation.model_management.load_model_gpu",
                   side_effect=lambda p: p.patch_model()), \
             patch("ComfyUI_VibeVoice.modules.asr_generation.model_management.unload_all_models"), \
             patch("ComfyUI_VibeVoice.modules.asr_generation.model_management.soft_empty_cache"), \
             patch("ComfyUI_VibeVoice.modules.asr_generation.gc"), \
             patch("ComfyUI_VibeVoice.modules.asr_generation.get_torch_device", return_value=torch.device("cpu")), \
             patch("ComfyUI_VibeVoice.modules.asr_generation.get_offload_device", return_value=torch.device("cpu")):

            patcher, model, processor = load_asr_model_patched(
                model_name="VibeVoice-ASR", device="cpu", dtype="fp32", attention_mode="sdpa"
            )
            cache_key = patcher.cache_key
            LOADED_ASR_MODELS_CACHE[cache_key] = (model, processor)

            assert patcher.is_loaded is True
            force_offload_asr_model("VibeVoice-ASR", patcher)

            assert patcher.is_loaded is False
            assert cache_key not in LOADED_ASR_MODELS_CACHE

        LOADED_ASR_MODELS_CACHE.clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()


class TestASRProgressReporting:
    """Phase 5 (2026-08-15 progress plan): ASR transcription must drive the
    standard ComfyUI ProgressBar through an HF token streamer."""

    @staticmethod
    def _make_mocks():
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])
        mock_output = torch.tensor([[1, 2, 3, 4, 5, 0]])
        mock_model.generate.return_value = mock_output

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_processor.pad_id = 0
        mock_processor.tokenizer.eos_token_id = 0
        mock_processor.decode.return_value = "Hello world"
        mock_processor.post_process_transcription.return_value = []
        return mock_model, mock_processor

    def _transcribe(self, mock_model, mock_processor, **kwargs):
        with patch("ComfyUI_VibeVoice.modules.asr_generation.extract_audio_tensor") as mock_extract, \
             patch("ComfyUI_VibeVoice.modules.asr_generation.ProgressBarWithConsole") as mock_pbar_cls, \
             patch("ComfyUI_VibeVoice.modules.asr_generation.model_management.throw_exception_if_processing_interrupted") as mock_interrupt:
            mock_extract.return_value = (torch.randn(24000), 24000)
            mock_pbar = MagicMock()
            mock_pbar.total = kwargs.get("max_new_tokens", 32768)
            mock_pbar_cls.return_value = mock_pbar

            result = transcribe_audio(
                model=mock_model,
                processor=mock_processor,
                audio_input={"waveform": torch.randn(1, 1, 24000), "sample_rate": 24000},
                **kwargs,
            )
            return result, mock_model, mock_pbar_cls, mock_pbar, mock_interrupt

    def test_streamer_put_advances_progress(self):
        """T5.1: streamer.put() of a 2-token tensor advances the bar to 2."""
        from ComfyUI_VibeVoice.modules.asr_generation import _ASRProgressStreamer

        mock_pbar = MagicMock()
        streamer = _ASRProgressStreamer(mock_pbar, total=100)
        streamer.put(torch.tensor([[7, 8]]))

        assert streamer.count == 2
        mock_pbar.update_absolute.assert_called_once_with(2, total=100)

    def test_streamer_end_sends_final_total(self):
        """T5.2: streamer.end() sends the final total update."""
        from ComfyUI_VibeVoice.modules.asr_generation import _ASRProgressStreamer

        mock_pbar = MagicMock()
        streamer = _ASRProgressStreamer(mock_pbar, total=50)
        streamer.end()
        mock_pbar.update_absolute.assert_called_once_with(50)

    def test_generate_receives_streamer_for_sampling(self):
        """T5.3a: model.generate receives streamer= when num_beams == 1."""
        mock_model, mock_processor = self._make_mocks()
        _, mock_model, _, _, _ = self._transcribe(mock_model, mock_processor, num_beams=1)

        gen_kwargs = mock_model.generate.call_args.kwargs
        assert "streamer" in gen_kwargs
        assert gen_kwargs["streamer"] is not None

    def test_generate_no_streamer_for_beam_search(self):
        """T5.3b: model.generate must NOT receive streamer= when num_beams > 1."""
        mock_model, mock_processor = self._make_mocks()
        _, mock_model, _, _, _ = self._transcribe(mock_model, mock_processor, num_beams=4)

        gen_kwargs = mock_model.generate.call_args.kwargs
        assert "streamer" not in gen_kwargs

    def test_streamer_put_checks_interrupt(self):
        """T5.4: put() calls throw_exception_if_processing_interrupted; a raised
        interrupt propagates."""
        import comfy.model_management as mm
        from ComfyUI_VibeVoice.modules.asr_generation import _ASRProgressStreamer

        with patch("ComfyUI_VibeVoice.modules.asr_generation.model_management.throw_exception_if_processing_interrupted") as mock_interrupt:
            mock_interrupt.side_effect = mm.InterruptProcessingException()
            mock_pbar = MagicMock()
            streamer = _ASRProgressStreamer(mock_pbar, total=100)

            with pytest.raises(mm.InterruptProcessingException):
                streamer.put(torch.tensor([7]))
            mock_interrupt.assert_called()

    def test_transcription_output_unchanged_with_progress(self):
        """T5.5: progress plumbing must not change the transcription output."""
        mock_model, mock_processor = self._make_mocks()
        (raw_text, segments), _, mock_pbar_cls, mock_pbar, _ = self._transcribe(
            mock_model, mock_processor, max_new_tokens=100
        )

        assert raw_text == "Hello world"
        assert segments == []
        # Bar created with the max_new_tokens budget and driven to 100% at the end.
        mock_pbar_cls.assert_called_once_with(100)
        final_call = mock_pbar.update_absolute.call_args_list[-1]
        assert final_call.args == (100,)
