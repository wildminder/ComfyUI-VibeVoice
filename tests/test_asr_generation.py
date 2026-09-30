"""Tests for modules/asr_generation.py - ASR transcription."""

import torch
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from ComfyUI_VibeVoice.modules.asr_generation import load_asr_model, load_asr_model_patched, transcribe_audio, force_offload_asr_model

# The dynamic patcher's aimdo-free stand-in, its alias fixture and the
# load_models_gpu stand-in live in test_patcher.py, which owns the
# patcher-level tests. Imported, not redefined, so the suite has exactly one
# definition of each.
from tests.test_patcher import (  # noqa: E402
    dynamic_core_alias,
    load_side_effect,
)


def _asr_generation_module():
    """The asr_generation module OBJECT (monkeypatch cannot resolve the alias
    package ``ComfyUI_VibeVoice.modules.asr_generation`` from a dotted string)."""
    import ComfyUI_VibeVoice.modules.asr_generation as asr_gen

    return asr_gen


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
             patch("ComfyUI_VibeVoice.modules.asr_generation.model_management.load_models_gpu",
                   side_effect=lambda models: models[0].patch_model()), \
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
             patch("ComfyUI_VibeVoice.modules.asr_generation.model_management.load_models_gpu",
                   side_effect=lambda models: models[0].patch_model()), \
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

    def test_sage_request_reaches_the_cache_key_and_the_patcher(self):
        """The ASR sage exclusion must land on BOTH the cache key and the
        patcher, exactly as the realtime exclusion does.

        A downgrade that misses either leaves a mismatch: a sage-built model
        cached under an sdpa key (or vice versa) means the key no longer
        describes the weights, and a stale entry under the old key pins the
        wrong model in VRAM.
        """
        from ComfyUI_VibeVoice.modules.asr_loader import VibeVoiceASRModelHandler, LOADED_ASR_MODELS_CACHE
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_ASR_PATCHER_CACHE
        from ComfyUI_VibeVoice.modules.asr_generation import load_asr_model_patched

        LOADED_ASR_MODELS_CACHE.clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()

        def fake_load(self, device, attention_mode="sdpa"):
            self.model = torch.nn.Linear(8, 8)
            self.processor = object()

        with patch.object(VibeVoiceASRModelHandler, "load_model", fake_load),              patch("comfy.model_patcher.ModelPatcher.patch_model"),              patch("ComfyUI_VibeVoice.modules.asr_generation.model_management.load_models_gpu",
                   side_effect=lambda models: models[0].patch_model()),              patch("ComfyUI_VibeVoice.modules.asr_generation.get_torch_device",
                   return_value=torch.device("cpu")),              patch("ComfyUI_VibeVoice.modules.asr_generation.get_offload_device",
                   return_value=torch.device("cpu")):
            patcher, _, _ = load_asr_model_patched(
                model_name="VibeVoice-ASR", device="cpu", dtype="fp32",
                attention_mode="sage",
            )

        assert patcher.attention_mode == "sdpa", (
            "the patcher must carry the mode the weights were actually built "
            "with, not the one the user asked for"
        )
        assert list(VIBEVOICE_ASR_PATCHER_CACHE) == ["asr_VibeVoice-ASR_attn_sdpa"]
        assert list(LOADED_ASR_MODELS_CACHE) == ["asr_VibeVoice-ASR_attn_sdpa"]
        assert "sage" not in "".join(VIBEVOICE_ASR_PATCHER_CACHE)

        LOADED_ASR_MODELS_CACHE.clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()

    def test_sage_request_collides_with_the_sdpa_entry(self):
        """A sage request must reuse the sdpa entry, not build a second model
        under a different key for the same backend."""
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

        with patch.object(VibeVoiceASRModelHandler, "load_model", fake_load),              patch("comfy.model_patcher.ModelPatcher.patch_model"),              patch("ComfyUI_VibeVoice.modules.asr_generation.model_management.load_models_gpu",
                   side_effect=lambda models: models[0].patch_model()),              patch("ComfyUI_VibeVoice.modules.asr_generation.get_torch_device",
                   return_value=torch.device("cpu")),              patch("ComfyUI_VibeVoice.modules.asr_generation.get_offload_device",
                   return_value=torch.device("cpu")):
            p1, _, _ = load_asr_model_patched(
                model_name="VibeVoice-ASR", device="cpu", dtype="fp32",
                attention_mode="sdpa")
            p2, _, _ = load_asr_model_patched(
                model_name="VibeVoice-ASR", device="cpu", dtype="fp32",
                attention_mode="sage")

        assert p1 is p2
        assert load_calls["n"] == 1
        assert len(VIBEVOICE_ASR_PATCHER_CACHE) == 1

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
             patch("ComfyUI_VibeVoice.modules.asr_generation.model_management.load_models_gpu",
                   side_effect=lambda models: models[0].patch_model()), \
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


class TestLoadASRUnderBothPatcherClasses:
    """T8: the external ASR load path must work on EITHER patcher class.

    The selector (modules/patcher.py:select_patcher_class) is the only thing
    that chooses, so the class is forced here by rebinding the name the
    consuming module imported. Everything downstream — construction, the lazy
    build, ``load_to_device``'s branch, the cache registration — then runs for
    real.

    Assertions are on ``is_dynamic()`` and ``is_loaded``, never on
    ``isinstance``: ``ModelPatcherDynamic.__new__``
    (comfy/model_patcher.py:1754-1757) reroutes a CPU load_device to a plain
    ModelPatcher, so a "dynamic" instance is not necessarily an instance of the
    dynamic subclass. This is a correctness requirement, not a style preference.
    """

    # The aimdo-free dynamic stand-in and its alias fixture live in
    # test_patcher.py, which owns the patcher-level tests. Imported, not
    # redefined, so the suite has exactly one definition of the stand-in.
    def _bundle(self, name="ext-asr"):
        return {
            "model": torch.nn.Linear(8, 8),
            "processor": object(),
            "model_name": name,
            "source_path": "fake.safetensors",
            # The dense family is the ONLY one that may go dynamic; naming it
            # keeps the fixture honest about what the selector would pick.
            "weight_family": "dense",
            "dynamic_vram_route": True,
        }

    @pytest.mark.parametrize("patcher_kind", ["legacy", "dynamic"])
    def test_asr_generation_works_under_both_patcher_classes(
        self, monkeypatch, dynamic_core_alias, patcher_kind
    ):
        from ComfyUI_VibeVoice.modules.asr_generation import load_asr_from_external
        from ComfyUI_VibeVoice.modules.patcher import (
            VibeVoiceASRPatcher,
            make_dynamic_patcher_class,
        )
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_ASR_PATCHER_CACHE
        from ComfyUI_VibeVoice.modules.asr_loader import LOADED_ASR_MODELS_CACHE

        VIBEVOICE_ASR_PATCHER_CACHE.clear()
        LOADED_ASR_MODELS_CACHE.clear()

        patcher_cls = (
            make_dynamic_patcher_class(VibeVoiceASRPatcher)
            if patcher_kind == "dynamic"
            else VibeVoiceASRPatcher
        )
        monkeypatch.setattr(
            _asr_generation_module(), "select_patcher_class",
            lambda *args, **kwargs: patcher_cls,
        )

        bundle = self._bundle()

        with patch("comfy.model_patcher.ModelPatcher.patch_model"),              patch("ComfyUI_VibeVoice.modules.asr_generation.model_management.load_model_gpu"),              patch("ComfyUI_VibeVoice.modules.asr_generation.model_management.load_models_gpu",
                   side_effect=load_side_effect()),              patch("ComfyUI_VibeVoice.modules.asr_generation.get_torch_device", return_value=torch.device("cpu")),              patch("ComfyUI_VibeVoice.modules.asr_generation.get_offload_device", return_value=torch.device("cpu")):
            patcher, model, processor = load_asr_from_external(
                bundle, device="cpu", dtype="fp32", attention_mode="sdpa"
            )

        assert patcher.is_dynamic() is (patcher_kind == "dynamic")
        assert patcher.is_loaded is True
        assert model is not None
        assert processor is not None

        # The identity key is an internal format; the contract under test is
        # "the cache entry is registered", so assert membership, not spelling.
        assert patcher in VIBEVOICE_ASR_PATCHER_CACHE.values()
        assert len(VIBEVOICE_ASR_PATCHER_CACHE) == 1
        assert patcher.attention_mode == "sdpa"
        assert patcher.target_dtype == torch.float32
        assert patcher.model.cache_key == patcher.cache_key

        VIBEVOICE_ASR_PATCHER_CACHE.clear()
        LOADED_ASR_MODELS_CACHE.clear()

    @pytest.mark.parametrize("patcher_kind", ["legacy", "dynamic"])
    def test_asr_family_label_no_longer_decides_the_protocol(
            self, monkeypatch, dynamic_core_alias, patcher_kind):
        """2026-09-30: see the TTS twin — the ASR selector is the same rule
        keyed on ``legacy_cls``. (The previous version asserted the removed
        quant-family exclusion using a CPU device, where every family returns
        legacy anyway — vacuous.)"""
        import torch
        from ComfyUI_VibeVoice.modules.patcher import (
            VibeVoiceASRPatcher,
            select_patcher_class,
        )

        for family in ("gguf_block", "convrot_int8", "fp8_resident", "", None):
            assert select_patcher_class(
                family, torch.device("cpu"), legacy_cls=VibeVoiceASRPatcher
            ) is VibeVoiceASRPatcher

        if torch.cuda.is_available():
            for family in ("convrot_int8", "fp8_resident", "", None):
                selected = select_patcher_class(
                    family, torch.device("cuda"), legacy_cls=VibeVoiceASRPatcher
                )
                assert selected is not VibeVoiceASRPatcher, family
            assert select_patcher_class(
                "gguf_block", torch.device("cuda"), legacy_cls=VibeVoiceASRPatcher
            ) is VibeVoiceASRPatcher

        del patcher_kind  # the parametrisation runs the same body twice



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
             patch("ComfyUI_VibeVoice.modules.asr_generation.model_management.load_models_gpu",
                   side_effect=lambda models: models[0].patch_model()), \
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


class TestASRHandlerReceivesAttentionMode:
    """Regression: the patcher called the handler's load_model POSITIONALLY
    with the attention mode; the ASR handler's signature is
    (device, dtype_str, attention_mode), so "sdpa" landed in dtype_str and
    resolve_dtype raised 'Unknown dtype: sdpa'."""

    def test_real_asr_handler_gets_dtype_and_attention(self):
        from ComfyUI_VibeVoice.modules.asr_loader import VibeVoiceASRModelHandler
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_ASR_PATCHER_CACHE

        VIBEVOICE_ASR_PATCHER_CACHE.clear()
        received = {}

        def fake_loader_load(model_name, device, dtype_str="auto", attention_mode="sdpa"):
            received["dtype_str"] = dtype_str
            received["attention_mode"] = attention_mode
            return torch.nn.Linear(2, 2).eval(), object()

        handler = VibeVoiceASRModelHandler("VibeVoice-ASR-Streaming-1.5B")

        # Exercise the REAL handler.load_model with VibeVoiceASRLoader mocked.
        with patch(
            "ComfyUI_VibeVoice.modules.asr_loader.VibeVoiceASRLoader.load_model",
            side_effect=fake_loader_load,
        ):
            handler.load_model(torch.device("cpu"), "auto", "flash_attention_2")

        assert received["dtype_str"] == "auto"
        assert received["attention_mode"] == "flash_attention_2"

    def test_patcher_calls_handler_with_keyword_attention(self):
        """The shared VibeVoicePatcher.patch_model must pass attention_mode
        as a keyword so the ASR handler's (device, dtype_str, attention_mode)
        signature receives it in the right slot."""
        from ComfyUI_VibeVoice.modules.patcher import VibeVoicePatcher
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_ASR_PATCHER_CACHE

        VIBEVOICE_ASR_PATCHER_CACHE.clear()
        tiny = torch.nn.Linear(8, 8)

        class _ASRStyleHandler(torch.nn.Module):
            """Handler with the ASR signature; records how it was called."""
            def __init__(self):
                super().__init__()
                self.model = None
                self.processor = object()
                self.model_pack_name = "asr-style"
                self.cache_key = "asr-style"
                self.size = 1024
                self.calls = []

            def load_model(self, device, dtype_str="auto", attention_mode="sdpa"):
                self.calls.append({"dtype_str": dtype_str, "attention_mode": attention_mode})
                self.model = tiny

        handler = _ASRStyleHandler()
        with patch("comfy.model_patcher.ModelPatcher.__init__"):
            patcher = VibeVoicePatcher(
                handler,
                attention_mode="flash_attention_2",
                load_device=torch.device("cpu"),
                offload_device=torch.device("cpu"),
                size=1024,
            )
        patcher.load_device = torch.device("cpu")
        patcher.offload_device = torch.device("cpu")
        patcher.model = handler
        patcher.pinned = set()
        patcher.target_dtype = None

        with patch("comfy.model_patcher.ModelPatcher.patch_model"):
            patcher.patch_model()

        assert handler.calls == [
            {"dtype_str": "auto", "attention_mode": "flash_attention_2"}
        ]


class TestTranscribeNative:
    """The HF-native path (microsoft/VibeVoice-ASR-HF): processors from the
    transformers builtins route through apply_transcription_request -> generate
    -> decode(return_format='parsed'), and segments map onto our schema."""

    @staticmethod
    def _make_native_processor(parsed):
        """A processor stub whose class module marks it as transformers-native."""
        class _BatchFeature(dict):
            """Minimal BatchFeature stand-in with .to(device, dtype)."""
            def to(self, *args, **kwargs):
                return self

        class _NativeProcessor:
            def __init__(self):
                self.tokenizer = MagicMock()
                self.tokenizer.pad_token_id = 151655
                self.tokenizer.eos_token_id = 151643
                self._parsed = parsed
                self.requests = []

            def apply_transcription_request(self, audio, prompt=None, **kwargs):
                self.requests.append({"audio": audio, "prompt": prompt, **kwargs})
                return _BatchFeature(input_ids=torch.tensor([[1, 2, 3]]),
                                     input_values=torch.zeros(1, 100),
                                     padding_mask=torch.ones(1, 100, dtype=torch.bool),
                                     attention_mask=torch.ones(1, 3, dtype=torch.bool))

            def decode(self, ids, **kwargs):
                if kwargs.get("return_format") == "parsed":
                    return [self._parsed]
                return ["raw text"]

        _NativeProcessor.__module__ = "transformers.models.vibevoice_asr.processing_vibevoice_asr"
        return _NativeProcessor()

    def _transcribe(self, parsed, **kwargs):
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_param.dtype = torch.float32
        mock_model.parameters.return_value = iter([mock_param])
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 0]])
        processor = self._make_native_processor(parsed)

        with patch("ComfyUI_VibeVoice.modules.asr_generation.extract_audio_tensor") as mock_extract, \
             patch("ComfyUI_VibeVoice.modules.asr_generation.ProgressBarWithConsole") as mock_pbar_cls:
            mock_extract.return_value = (torch.randn(24000), 24000)
            mock_pbar = MagicMock()
            mock_pbar.total = kwargs.get("max_new_tokens", 32768)
            mock_pbar_cls.return_value = mock_pbar
            result = transcribe_audio(
                model=mock_model,
                processor=processor,
                audio_input={"waveform": torch.randn(1, 1, 24000), "sample_rate": 24000},
                **kwargs,
            )
        return result, mock_model, processor

    def test_native_processor_routes_to_native_path(self):
        """A transformers-native processor never hits the legacy JSON-prompt
        processor call: apply_transcription_request builds the inputs."""
        (raw_text, segments), mock_model, processor = self._transcribe(
            parsed=[{"Start": 0.0, "End": 6.27, "Speaker": 0,
                     "Content": "Hello from native."}]
        )
        # generate() received the chat-template inputs, not a legacy dict.
        gen_kwargs = mock_model.generate.call_args.kwargs
        assert "input_ids" in gen_kwargs
        # streamer present for greedy/sampling
        assert gen_kwargs.get("streamer") is not None
        assert raw_text == "raw text"

    def test_parsed_segments_map_to_node_schema(self):
        """Start/End/Speaker/Content map onto speaker/text/start/end."""
        (raw_text, segments), _, _ = self._transcribe(
            parsed=[{"Start": 0.0, "End": 6.27, "Speaker": 1,
                     "Content": "Hello from native."},
                    {"Start": 6.27, "End": 12.5, "Speaker": 0,
                     "Content": "Second segment."}]
        )
        assert segments == [
            {"speaker": 1, "text": "Hello from native.", "start": 0.0, "end": 6.27},
            {"speaker": 0, "text": "Second segment.", "start": 6.27, "end": 12.5},
        ]

    def test_parse_failure_returns_no_segments(self):
        """decode(parsed) falling back to the raw string yields no segments
        (and doesn't raise)."""
        (raw_text, segments), _, _ = self._transcribe(parsed="not a list")
        assert segments == []
        assert raw_text == "raw text"

    def test_non_native_rate_input_is_resampled(self):
        """The native feature extractor refuses non-24k audio ("trained
        using a sampling rate of 24000 ... not 44100") — the node resamples
        to the extractor's declared rate before the request, like the main
        VibeVoice routine."""
        import numpy as np
        from ComfyUI_VibeVoice.modules import asr_generation

        processor = self._make_native_processor([])
        # Real feature_extractor attribute so the target rate is 24000.
        processor.feature_extractor = SimpleNamespace(sampling_rate=24000)

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_param.dtype = torch.float32
        mock_model.parameters.return_value = iter([mock_param])
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])

        resampled = {"called": False}

        def fake_resample(arr, orig_sr, target_sr):
            resampled["called"] = True
            resampled["orig_sr"] = orig_sr
            resampled["target_sr"] = target_sr
            return arr

        # 1 second of 44.1 kHz audio arriving from the ComfyUI audio input.
        with patch("ComfyUI_VibeVoice.modules.asr_generation.extract_audio_tensor") as mock_extract, \
             patch("ComfyUI_VibeVoice.modules.asr_generation.ProgressBarWithConsole"), \
             patch.object(asr_generation, "resample_audio", side_effect=fake_resample):
            mock_extract.return_value = (torch.randn(44100), 44100)
            transcribe_audio(
                model=mock_model,
                processor=processor,
                audio_input={"waveform": torch.randn(1, 1, 44100), "sample_rate": 44100},
            )

        assert resampled["called"] is True
        assert resampled["orig_sr"] == 44100
        assert resampled["target_sr"] == 24000

    def test_native_rate_input_skips_resample(self):
        """24 kHz input goes straight into the request untouched."""
        processor = self._make_native_processor([])
        processor.feature_extractor = SimpleNamespace(sampling_rate=24000)

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_param.dtype = torch.float32
        mock_model.parameters.return_value = iter([mock_param])
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])

        with patch("ComfyUI_VibeVoice.modules.asr_generation.extract_audio_tensor") as mock_extract, \
             patch("ComfyUI_VibeVoice.modules.asr_generation.ProgressBarWithConsole"), \
             patch("ComfyUI_VibeVoice.modules.asr_generation.resample_audio") as mock_resample:
            mock_extract.return_value = (torch.randn(24000), 24000)
            transcribe_audio(
                model=mock_model,
                processor=processor,
                audio_input={"waveform": torch.randn(1, 1, 24000), "sample_rate": 24000},
            )
        mock_resample.assert_not_called()

    def test_context_info_passed_as_prompt(self):
        """Hotword context flows into apply_transcription_request(prompt=)."""
        processor = self._make_native_processor([])
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_param.dtype = torch.float32
        mock_model.parameters.return_value = iter([mock_param])
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])

        with patch("ComfyUI_VibeVoice.modules.asr_generation.extract_audio_tensor") as mock_extract, \
             patch("ComfyUI_VibeVoice.modules.asr_generation.ProgressBarWithConsole"):
            mock_extract.return_value = (torch.randn(24000), 24000)
            transcribe_audio(
                model=mock_model,
                processor=processor,
                audio_input={"waveform": torch.randn(1, 1, 24000), "sample_rate": 24000},
                context_info="Tea Brew, Aiden Host",
            )
        assert len(processor.requests) == 1
        assert processor.requests[0]["prompt"] == "Tea Brew, Aiden Host"


class TestASRAttentionSwitchNoLeak:
    """Deterministic proof of the 2026-09-09 leak fix: switching attention
    mode (sdpa -> sage) must FULLY release the previous model — exactly one
    cache entry per active model, no stale third-format entries, the family
    active key tracks the new model."""

    def _run_patched_load(self, models_by_attn, attention_mode):
        from ComfyUI_VibeVoice.modules.asr_loader import VibeVoiceASRModelHandler
        from ComfyUI_VibeVoice.modules.asr_generation import load_asr_model_patched

        def fake_load(self, device, attention_mode="sdpa"):
            self.model = models_by_attn[attention_mode]
            self.processor = object()

        with patch.object(VibeVoiceASRModelHandler, "load_model", fake_load), \
             patch("comfy.model_patcher.ModelPatcher.patch_model"), \
             patch("ComfyUI_VibeVoice.modules.asr_generation.model_management.load_models_gpu",
                   side_effect=lambda models: models[0].patch_model()), \
             patch("ComfyUI_VibeVoice.modules.asr_generation.get_torch_device",
                   return_value=torch.device("cpu")), \
             patch("ComfyUI_VibeVoice.modules.asr_generation.get_offload_device",
                   return_value=torch.device("cpu")):
            return load_asr_model_patched(
                model_name="VibeVoice-ASR", device="cpu", dtype="fp32",
                attention_mode=attention_mode,
            )

    def test_attention_switch_releases_everything(self):
        from ComfyUI_VibeVoice.modules.asr_loader import LOADED_ASR_MODELS_CACHE
        from ComfyUI_VibeVoice.modules.model_registry import FAMILY_ASR, get_active, clear_active_keys
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_ASR_PATCHER_CACHE

        LOADED_ASR_MODELS_CACHE.clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()
        clear_active_keys()
        # The switch is eager -> sdpa, not sdpa -> sage: sage is excluded from
        # the ASR path (ASR_EXCLUDED_ATTENTION_MODES), so a sage request
        # resolves to sdpa and would collide with run 1 instead of switching.
        eager_model, sdpa_model = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
        models_by_attn = {"eager": eager_model, "sdpa": sdpa_model}

        try:
            # Run 1: eager — one patcher + one model entry, under ONE key format.
            _, model1, _ = self._run_patched_load(models_by_attn, "eager")
            assert model1 is eager_model
            assert list(VIBEVOICE_ASR_PATCHER_CACHE.keys()) == ["asr_VibeVoice-ASR_attn_eager"]
            assert list(LOADED_ASR_MODELS_CACHE.keys()) == ["asr_VibeVoice-ASR_attn_eager"]
            assert get_active(FAMILY_ASR) == "asr_VibeVoice-ASR_attn_eager"

            # Run 2: sdpa — the eager model must be FULLY released before the
            # new one loads (this is where the old code leaked a stale
            # static-format entry pinning the 17 GB tree).
            _, model2, _ = self._run_patched_load(models_by_attn, "sdpa")
            assert model2 is sdpa_model
            assert list(VIBEVOICE_ASR_PATCHER_CACHE.keys()) == ["asr_VibeVoice-ASR_attn_sdpa"]
            assert list(LOADED_ASR_MODELS_CACHE.keys()) == ["asr_VibeVoice-ASR_attn_sdpa"]
            assert get_active(FAMILY_ASR) == "asr_VibeVoice-ASR_attn_sdpa"
        finally:
            LOADED_ASR_MODELS_CACHE.clear()
            VIBEVOICE_ASR_PATCHER_CACHE.clear()
            clear_active_keys()

    def test_dtype_switch_keeps_single_entry(self):
        """dtype is NOT part of the ASR patcher key (matching TTS): a dtype
        change reuses the patcher and the DF-004 cast applies the new dtype —
        no rebuild, no extra entries."""
        from ComfyUI_VibeVoice.modules.asr_loader import LOADED_ASR_MODELS_CACHE
        from ComfyUI_VibeVoice.modules.model_registry import clear_active_keys
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_ASR_PATCHER_CACHE

        LOADED_ASR_MODELS_CACHE.clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()
        clear_active_keys()
        one_model = torch.nn.Linear(4, 4)
        models_by_attn = {"sdpa": one_model}

        try:
            _, model1, _ = self._run_patched_load(models_by_attn, "sdpa")
            # Same key again (dtype not in key) -> same patcher, no rebuild.
            _, model2, _ = self._run_patched_load(models_by_attn, "sdpa")
            assert model1 is one_model and model2 is one_model
            assert list(VIBEVOICE_ASR_PATCHER_CACHE.keys()) == ["asr_VibeVoice-ASR_attn_sdpa"]
            assert list(LOADED_ASR_MODELS_CACHE.keys()) == ["asr_VibeVoice-ASR_attn_sdpa"]
        finally:
            LOADED_ASR_MODELS_CACHE.clear()
            VIBEVOICE_ASR_PATCHER_CACHE.clear()
            clear_active_keys()
