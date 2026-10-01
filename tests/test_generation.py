"""Tests for modules/generation.py - Model loading and audio generation."""

import numpy as np
import torch
import pytest
from unittest.mock import patch, MagicMock

from ComfyUI_VibeVoice.modules.generation import (
    load_vibevoice_model,
    load_vibevoice_from_external,
    ExternalVibeVoiceModelHandler,
    generate_audio,
    force_offload_model,
)


def _mock_voice_sample(length: int = 24000) -> np.ndarray:
    """Create a mock 1-D voice sample numpy array."""
    return np.random.randn(length).astype(np.float32)


def _patch_model_management():
    """Patch generation.model_management with a realistic stand-in.

    BUG-012: generate_audio places inputs on the runtime's compute device
    (``model_management.get_torch_device()``), NOT ``model.device`` (which
    lies under partial offload). A bare MagicMock breaks ``Tensor.to()``,
    so the mock must return a real torch.device.
    """
    mm = MagicMock()
    mm.get_torch_device.return_value = torch.device("cpu")
    return patch("ComfyUI_VibeVoice.modules.generation.model_management", mm)


class TestLoadVibevoiceModel:
    """Test load_vibevoice_model function."""

    def test_load_model_cache_miss(self):
        """When not cached, a new patcher should be created."""
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_PATCHER_CACHE
        VIBEVOICE_PATCHER_CACHE.clear()

        mock_patcher = MagicMock()
        mock_patcher.model.model = MagicMock()
        mock_patcher.model.processor = MagicMock()

        with patch("ComfyUI_VibeVoice.modules.generation.VibeVoiceModelHandler") as mock_handler_cls, \
             patch("ComfyUI_VibeVoice.modules.generation.VibeVoicePatcher", return_value=mock_patcher), \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.load_model_gpu"):
            mock_handler = MagicMock()
            mock_handler.size = 1000
            mock_handler_cls.return_value = mock_handler

            patcher, model, processor = load_vibevoice_model(
                model_name="TestModel",
                device="cpu",
                dtype="fp32",
                attention_mode="sdpa",
                quantize_4bit=False,
            )

            assert patcher == mock_patcher
            assert model is not None
            assert processor is not None

    def test_load_model_cache_hit(self):
        """When cached, the existing patcher should be returned."""
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_PATCHER_CACHE

        mock_patcher = MagicMock()
        mock_patcher.model.model = MagicMock()
        mock_patcher.model.processor = MagicMock()
        cache_key = "TestModel_attn_sdpa_q4_0"
        VIBEVOICE_PATCHER_CACHE[cache_key] = mock_patcher

        with patch("ComfyUI_VibeVoice.modules.generation.model_management.load_model_gpu"):
            patcher, model, processor = load_vibevoice_model(
                model_name="TestModel",
                device="cpu",
                dtype="fp32",
                attention_mode="sdpa",
                quantize_4bit=False,
            )

            assert patcher == mock_patcher

        VIBEVOICE_PATCHER_CACHE.clear()

    def test_load_model_force_reload(self):
        """force_reload should clear cache and create new patcher."""
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_PATCHER_CACHE

        old_patcher = MagicMock()
        cache_key = "TestModel_attn_sdpa_q4_0"
        VIBEVOICE_PATCHER_CACHE[cache_key] = old_patcher

        new_patcher = MagicMock()
        new_patcher.model.model = MagicMock()
        new_patcher.model.processor = MagicMock()

        with patch("ComfyUI_VibeVoice.modules.generation.VibeVoiceModelHandler") as mock_handler_cls, \
             patch("ComfyUI_VibeVoice.modules.generation.VibeVoicePatcher", return_value=new_patcher), \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.load_model_gpu"), \
             patch("ComfyUI_VibeVoice.modules.generation.cleanup_old_models"):
            mock_handler = MagicMock()
            mock_handler.size = 1000
            mock_handler_cls.return_value = mock_handler

            patcher, model, processor = load_vibevoice_model(
                model_name="TestModel",
                device="cpu",
                dtype="fp32",
                attention_mode="sdpa",
                quantize_4bit=False,
                force_reload=True,
            )

            assert patcher == new_patcher

        VIBEVOICE_PATCHER_CACHE.clear()

    def test_load_model_raises_on_none_model(self):
        """Should raise RuntimeError if model is None after loading."""
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_PATCHER_CACHE
        VIBEVOICE_PATCHER_CACHE.clear()

        mock_patcher = MagicMock()
        mock_patcher.model.model = None
        mock_patcher.model.processor = None

        with patch("ComfyUI_VibeVoice.modules.generation.VibeVoiceModelHandler") as mock_handler_cls, \
             patch("ComfyUI_VibeVoice.modules.generation.VibeVoicePatcher", return_value=mock_patcher), \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.load_model_gpu"):
            mock_handler = MagicMock()
            mock_handler.size = 1000
            mock_handler_cls.return_value = mock_handler

            with pytest.raises(RuntimeError, match="could not be loaded"):
                load_vibevoice_model(
                    model_name="TestModel",
                    device="cpu",
                    dtype="fp32",
                    attention_mode="sdpa",
                    quantize_4bit=False,
                )


class TestLoadFromExternal:
    """Test load_vibevoice_from_external function."""

    def _make_bundle(self):
        """Create a minimal valid external model bundle."""
        return {
            "model": MagicMock(),
            "processor": MagicMock(),
            "model_name": "ExtModel",
            "source_path": "/fake/model.safetensors",
            "is_streaming": False,
        }

    def test_load_from_external_returns_patcher_model_processor(self):
        """Returns a 3-tuple of (patcher, model, processor)."""
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_PATCHER_CACHE
        VIBEVOICE_PATCHER_CACHE.clear()

        bundle = self._make_bundle()

        with patch("ComfyUI_VibeVoice.modules.generation.model_management.load_model_gpu"):
            patcher, model, processor = load_vibevoice_from_external(
                bundle, device="cpu", dtype="fp32", attention_mode="sdpa"
            )

        assert patcher is not None
        assert model is not None
        assert processor is not None
        VIBEVOICE_PATCHER_CACHE.clear()

    def test_load_from_external_creates_patcher(self):
        """A VibeVoicePatcher is created wrapping the external handler."""
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_PATCHER_CACHE
        from ComfyUI_VibeVoice.modules.patcher import VibeVoicePatcher
        VIBEVOICE_PATCHER_CACHE.clear()

        bundle = self._make_bundle()

        with patch("ComfyUI_VibeVoice.modules.generation.model_management.load_model_gpu"):
            patcher, _, _ = load_vibevoice_from_external(
                bundle, device="cpu", dtype="fp32", attention_mode="sdpa"
            )

        assert isinstance(patcher, VibeVoicePatcher)
        VIBEVOICE_PATCHER_CACHE.clear()

    def test_load_from_external_calls_load_model_gpu(self):
        """model_management.load_model_gpu is called with the patcher."""
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_PATCHER_CACHE
        VIBEVOICE_PATCHER_CACHE.clear()

        bundle = self._make_bundle()

        with patch("ComfyUI_VibeVoice.modules.generation.model_management.load_model_gpu") as mock_load_gpu:
            patcher, _, _ = load_vibevoice_from_external(
                bundle, device="cpu", dtype="fp32", attention_mode="sdpa"
            )

        mock_load_gpu.assert_called_once_with(patcher)
        VIBEVOICE_PATCHER_CACHE.clear()

    def test_load_from_external_caches_patcher(self):
        """The patcher is stored in VIBEVOICE_PATCHER_CACHE under its identity key."""
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_PATCHER_CACHE
        from ComfyUI_VibeVoice.modules.model_registry import identity_for_external
        VIBEVOICE_PATCHER_CACHE.clear()

        bundle = self._make_bundle()

        with patch("ComfyUI_VibeVoice.modules.generation.model_management.load_model_gpu"):
            patcher, _, _ = load_vibevoice_from_external(
                bundle, device="cpu", dtype="fp32", attention_mode="sdpa"
            )

        # Hand-built bundle without identity fields: stat fallback on
        # source_path (missing file -> mtime/size placeholders), attention
        # resolved from the widget, dtype falls back to the widget value.
        cache_key = identity_for_external(
            "/fake/model.safetensors", "ExtModel", "sdpa",
            use_llm_4bit=False, dtype_str="fp32",
        )
        assert cache_key in VIBEVOICE_PATCHER_CACHE
        assert VIBEVOICE_PATCHER_CACHE[cache_key] is patcher
        VIBEVOICE_PATCHER_CACHE.clear()

    def test_load_from_external_handler_holds_preloaded_model(self):
        """The handler's .model and .processor are the bundle's (not None)."""
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_PATCHER_CACHE
        VIBEVOICE_PATCHER_CACHE.clear()

        bundle = self._make_bundle()

        with patch("ComfyUI_VibeVoice.modules.generation.model_management.load_model_gpu"):
            patcher, _, _ = load_vibevoice_from_external(
                bundle, device="cpu", dtype="fp32", attention_mode="sdpa"
            )

        handler = patcher.model
        assert handler.model is bundle["model"]
        assert handler.processor is bundle["processor"]
        VIBEVOICE_PATCHER_CACHE.clear()

    def test_load_from_external_missing_model_raises(self):
        """Bundle without 'model' and without a source to rebuild → ValueError.

        A released bundle normally recovers via ``source_path``; this covers
        the unrecoverable case.
        """
        bundle = self._make_bundle()
        bundle["model"] = None
        bundle["source_path"] = ""

        with pytest.raises(ValueError, match="model"):
            load_vibevoice_from_external(bundle, device="cpu")

    def test_load_from_external_released_model_rebuilds(self):
        """A released bundle is rebuilt from its recorded source."""
        bundle = self._make_bundle()
        bundle["model"] = None
        rebuilt = {
            "model": MagicMock(), "processor": MagicMock(),
            "model_name": "ExtModel",
        }
        with patch(
            "ComfyUI_VibeVoice.modules.external_loader."
            "load_external_vibevoice_model", return_value=rebuilt
        ) as reload_:
            _patcher, model, processor = load_vibevoice_from_external(
                bundle, device="cpu"
            )
        assert model is rebuilt["model"] and processor is rebuilt["processor"]
        assert reload_.call_args[0][0] == "/fake/model.safetensors"

    def test_load_from_external_missing_processor_raises(self):
        """Bundle without 'processor' → ValueError."""
        bundle = self._make_bundle()
        bundle["processor"] = None

        with pytest.raises(ValueError, match="processor"):
            load_vibevoice_from_external(bundle, device="cpu")

    def test_load_from_external_missing_model_name_raises(self):
        """Bundle without 'model_name' → ValueError."""
        bundle = self._make_bundle()
        bundle["model_name"] = None

        with pytest.raises(ValueError, match="model_name"):
            load_vibevoice_from_external(bundle, device="cpu")

    def test_load_from_external_cache_hit_reuses_patcher(self):
        """A second call with the same bundle reuses the cached patcher."""
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_PATCHER_CACHE
        VIBEVOICE_PATCHER_CACHE.clear()

        bundle = self._make_bundle()

        with patch("ComfyUI_VibeVoice.modules.generation.model_management.load_model_gpu"):
            patcher1, _, _ = load_vibevoice_from_external(
                bundle, device="cpu", dtype="fp32", attention_mode="sdpa"
            )
            patcher2, _, _ = load_vibevoice_from_external(
                bundle, device="cpu", dtype="fp32", attention_mode="sdpa"
            )

        assert patcher1 is patcher2
        VIBEVOICE_PATCHER_CACHE.clear()


# The dynamic-patcher's aimdo-free stand-in and its alias fixture live in
# test_patcher.py, which owns the patcher-level tests. They are imported, not
# redefined, so there is exactly one definition of the stand-in in the suite.
from tests.test_patcher import dynamic_core_alias, load_side_effect  # noqa: E402,F401


def _generation_module():
    """The generation module OBJECT (monkeypatch cannot resolve the alias
    package ``ComfyUI_VibeVoice.modules.generation`` from a dotted string)."""
    import ComfyUI_VibeVoice.modules.generation as gen

    return gen


class TestLoadFromExternalUnderBothPatcherClasses:
    """T8: the external TTS load path, driven through the real selector.

    ``select_patcher_class`` (modules/patcher.py) is the only thing that
    chooses the patcher class, so the test uses its real answer rather than
    forcing one. Everything downstream — construction, the lazy build,
    ``load_to_device``'s branch, the cache registration — then runs for real.

    Assertions are on ``is_dynamic()`` and ``is_loaded``, never on
    ``isinstance``: ``ModelPatcherDynamic.__new__`` (comfy/model_patcher.py)
    reroutes a CPU load_device to a plain ModelPatcher, so a "dynamic"
    instance is not necessarily an instance of the dynamic subclass. This is a
    correctness requirement, not a style preference.
    """

    def _make_bundle(self):
        return {
            "model": MagicMock(),
            "processor": MagicMock(),
            "model_name": "ExtModel",
            "source_path": "/fake/model.safetensors",
            "is_streaming": False,
            # The dense family is the ONLY one that may go dynamic; naming it
            # here keeps the fixture honest about what the selector would pick.
            "weight_family": "dense",
            "dynamic_vram_route": True,
        }

    def test_tts_generation_works_under_the_selected_patcher_class(self, monkeypatch):
        """The external TTS load path, driven through the real selector.

        This used to be parametrized over the legacy and the minted dynamic
        class. ``select_patcher_class`` now always returns the legacy class
        (a model that fits in VRAM must not be trapped in vbar paging), so the
        dynamic arm tested a configuration production cannot produce. What
        remains load-bearing is that the selector's real answer flows through
        construction, the lazy build, ``load_to_device`` and the cache
        registration unchanged.
        """
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_PATCHER_CACHE
        from ComfyUI_VibeVoice.modules.patcher import (
            VibeVoicePatcher,
            select_patcher_class,
        )
        VIBEVOICE_PATCHER_CACHE.clear()

        # Whatever the selector decides for this family/device is what the
        # load path must actually use -- no rebinding, no forcing.
        selected = select_patcher_class("dense", torch.device("cpu"))
        monkeypatch.setattr(
            _generation_module(), "select_patcher_class",
            lambda *args, **kwargs: selected,
        )

        bundle = self._make_bundle()

        with patch("comfy.model_patcher.ModelPatcher.patch_model"), \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.load_model_gpu"), \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.load_models_gpu",
                   side_effect=load_side_effect()):
            patcher, model, processor = load_vibevoice_from_external(
                bundle, device="cpu", dtype="fp32", attention_mode="sdpa"
            )

        # The selector is the single source of truth, and today it is legacy.
        assert selected is VibeVoicePatcher
        assert patcher.is_dynamic() is False
        assert patcher.is_loaded is True
        assert model is not None
        assert processor is not None

        # The identity key is an internal format; the contract under test is
        # "the cache entry is registered", so assert membership, not spelling.
        assert patcher in VIBEVOICE_PATCHER_CACHE.values()
        assert len(VIBEVOICE_PATCHER_CACHE) == 1
        assert patcher.attention_mode == "sdpa"
        assert patcher.target_dtype == torch.float32

        VIBEVOICE_PATCHER_CACHE.clear()

    def test_tts_family_label_no_longer_decides_the_protocol(self):
        """The selector returns the legacy patcher for EVERY family and device.

        This test has been rewritten twice as the selector's contract changed.
        It now pins what is true: ``select_patcher_class`` ignores both its
        arguments and returns ``legacy_cls`` unconditionally, so no weight
        family can be routed into vbar paging by the label it carries.

        The earlier version asserted a whitelist ("quant families on CUDA are
        dynamic, except gguf_block"). That whitelist was what pinned weights
        in CPU virtual memory and made a 5 GB model take 10-12 s to fault in.
        """
        import torch
        from ComfyUI_VibeVoice.modules.patcher import (
            VibeVoicePatcher,
            VibeVoiceASRPatcher,
            select_patcher_class,
        )

        families = ("gguf_block", "convrot_int8", "fp8_resident", "dense", "", None)
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        for family in families:
            for device in devices:
                assert select_patcher_class(family, device) is VibeVoicePatcher, (
                    f"family={family!r} device={device}"
                )
            # The ASR legacy class is honoured when passed explicitly, so a
            # caller can still opt into it without touching the selector.
            assert select_patcher_class(
                family, torch.device("cpu"), legacy_cls=VibeVoiceASRPatcher,
            ) is VibeVoiceASRPatcher, family



class TestExternalVibeVoiceModelHandler:
    """Test the ExternalVibeVoiceModelHandler container."""

    def test_handler_is_nn_module(self):
        """Handler is a torch.nn.Module."""
        handler = ExternalVibeVoiceModelHandler(
            model=MagicMock(), processor=MagicMock(), model_pack_name="ext"
        )
        assert isinstance(handler, torch.nn.Module)

    def test_handler_holds_model_and_processor(self):
        """Handler stores the provided model and processor."""
        model = MagicMock()
        processor = MagicMock()
        handler = ExternalVibeVoiceModelHandler(
            model=model, processor=processor, model_pack_name="ext"
        )
        assert handler.model is model
        assert handler.processor is processor

    def test_handler_cache_key_namespaced(self):
        """Cache key is prefixed with 'external_'."""
        handler = ExternalVibeVoiceModelHandler(
            model=MagicMock(), processor=MagicMock(),
            model_pack_name="ext", attention_mode="sdpa",
        )
        assert handler.cache_key == "external_ext_attn_sdpa"

    def test_handler_load_model_is_noop(self):
        """load_model() does not replace the pre-loaded model."""
        model = MagicMock()
        handler = ExternalVibeVoiceModelHandler(
            model=model, processor=MagicMock(), model_pack_name="ext"
        )
        handler.load_model(torch.device("cpu"))
        assert handler.model is model

    def test_handler_estimates_size_from_parameters(self):
        """Size is estimated from a real nn.Module's parameters."""
        real_model = torch.nn.Linear(8, 8)
        handler = ExternalVibeVoiceModelHandler(
            model=real_model, processor=MagicMock(), model_pack_name="ext"
        )
        expected = sum(p.numel() * p.element_size() for p in real_model.parameters())
        assert handler.size == expected

    def test_handler_size_fallback_for_mock(self):
        """Size falls back to ~4GB when parameters cannot be summed."""
        handler = ExternalVibeVoiceModelHandler(
            model=MagicMock(), processor=MagicMock(), model_pack_name="ext"
        )
        assert handler.size > 0


class TestGenerateAudio:
    """Test generate_audio function."""

    def test_generate_audio_basic(self):
        """Test basic audio generation with mocked model."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")

        # Mock the generate output
        mock_output = MagicMock()
        mock_output.speech_outputs = [torch.randn(24000)]
        mock_model.generate.return_value = mock_output

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.randint(0, 100, (1, 10))}
        mock_processor.tokenizer = MagicMock()

        with patch("ComfyUI_VibeVoice.modules.generation.ProgressBarWithConsole"), \
             _patch_model_management(), \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio", return_value=_mock_voice_sample()):
            waveform, sample_rate = generate_audio(
                model=mock_model,
                processor=mock_processor,
                text="[1] Hello world",
                voice_samples=[{"waveform": torch.randn(1, 1, 24000), "sample_rate": 24000}],
                speaker_ids=[1],
                cfg_scale=1.3,
                inference_steps=10,
                seed=42,
                do_sample=True,
                temperature=0.95,
                top_p=0.95,
                top_k=0,
            )

            assert waveform is not None
            assert sample_rate == 24000
            assert waveform.ndim == 3  # [1, 1, T]

    def test_generate_audio_calls_set_ddpm_and_generate_with_correct_kwargs(self):
        """Verify generate_audio calls model.set_ddpm_inference_steps and
        model.generate with the non-streaming API contract.

        The non-streaming VibeVoiceForConditionalGeneration.generate() expects
        acoustic_input_mask (mapped from the processor's speech_input_mask),
        cfg_scale, inference_steps, return_speech, the processor tokenizer (so
        generate() can resolve speech_end_id and terminate the AR loop), and an
        optional max_new_tokens length budget — NOT the streaming-only kwargs
        (generation_config, stop_check_fn, all_prefilled_outputs, tts_text_ids).
        """
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")

        mock_output = MagicMock()
        mock_output.speech_outputs = [torch.randn(24000)]
        mock_model.generate.return_value = mock_output

        mock_processor = MagicMock()
        # Processor returns speech_input_mask (not acoustic_input_mask)
        mock_processor.return_value = {
            "input_ids": torch.randint(0, 100, (1, 10)),
            "speech_input_mask": torch.zeros(1, 10, dtype=torch.bool),
        }
        mock_processor.tokenizer = MagicMock()

        with patch("ComfyUI_VibeVoice.modules.generation.ProgressBarWithConsole"), \
             _patch_model_management(), \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio", return_value=_mock_voice_sample()):
            generate_audio(
                model=mock_model,
                processor=mock_processor,
                text="[1] Hello world",
                voice_samples=[{"waveform": torch.randn(1, 1, 24000), "sample_rate": 24000}],
                speaker_ids=[1],
                cfg_scale=2.0,
                inference_steps=15,
            )

        # set_ddpm_inference_steps must be called with the requested steps
        mock_model.set_ddpm_inference_steps.assert_called_once_with(num_steps=15)

        # generate() must receive the non-streaming kwargs
        gen_kwargs = mock_model.generate.call_args.kwargs
        assert gen_kwargs.get("cfg_scale") == 2.0
        assert gen_kwargs.get("inference_steps") == 15
        assert gen_kwargs.get("return_speech") is True
        # speech_input_mask from the processor must be remapped to acoustic_input_mask
        assert "acoustic_input_mask" in gen_kwargs
        assert gen_kwargs.get("speech_input_mask") is None
        # The processor tokenizer must be forwarded (needed for speech_end_id EOS).
        assert gen_kwargs.get("tokenizer") is mock_processor.tokenizer
        # max_new_tokens defaults to None and is dropped by the None-filter; when
        # unset it must NOT appear, but it is a legitimate non-streaming kwarg.
        assert "max_new_tokens" not in gen_kwargs
        # Streaming-only kwargs must NOT be passed
        assert "generation_config" not in gen_kwargs
        assert "stop_check_fn" not in gen_kwargs

    def test_generate_audio_processor_called_with_text_param(self):
        """Verify processor is called with 'text=' parameter containing normalized script text.

        The user's input format '[N] text' is converted to 'Speaker N: text' format
        that the vendored processor's _parse_script() expects.
        """
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_output = MagicMock()
        mock_output.speech_outputs = [torch.randn(24000)]
        mock_model.generate.return_value = mock_output

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.randint(0, 100, (1, 10))}
        mock_processor.tokenizer = MagicMock()

        test_script = "[1] Hello world"

        with patch("ComfyUI_VibeVoice.modules.generation.ProgressBarWithConsole"), \
             _patch_model_management(), \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio", return_value=_mock_voice_sample()):
            generate_audio(
                model=mock_model,
                processor=mock_processor,
                text=test_script,
                voice_samples=[{"waveform": torch.randn(1, 1, 24000), "sample_rate": 24000}],
                speaker_ids=[1],
            )

            # Verify processor was called with 'text' keyword (not 'parsed_scripts')
            call_kwargs = mock_processor.call_args.kwargs
            assert "text" in call_kwargs, "Processor must be called with 'text=' parameter"
            assert "parsed_scripts" not in call_kwargs, "Processor must NOT be called with 'parsed_scripts='"
            assert "speaker_ids_for_prompt" not in call_kwargs, "Processor must NOT be called with 'speaker_ids_for_prompt='"
            # Verify the script is normalized to "Speaker N: text" format
            # Input "[1] Hello world" → parsed as (0, "Hello world") → normalized to "Speaker 1:Hello world"
            assert call_kwargs["text"] == ["Speaker 1:Hello world"], \
                "Processor must receive normalized 'Speaker N: text' format, not raw '[N] text' format"

    def test_generate_audio_multi_speaker_bracket_format_normalized(self):
        """Verify multi-speaker '[N] text' format is normalized to 'Speaker N: text'."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_output = MagicMock()
        mock_output.speech_outputs = [torch.randn(24000)]
        mock_model.generate.return_value = mock_output

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.randint(0, 100, (1, 10))}
        mock_processor.tokenizer = MagicMock()

        # Multi-speaker script in [N] format
        test_script = "[1] Hello world\n[2] Hi there"

        with patch("ComfyUI_VibeVoice.modules.generation.ProgressBarWithConsole"), \
             _patch_model_management(), \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio", return_value=_mock_voice_sample()):
            generate_audio(
                model=mock_model,
                processor=mock_processor,
                text=test_script,
                voice_samples=[
                    {"waveform": torch.randn(1, 1, 24000), "sample_rate": 24000},
                    {"waveform": torch.randn(1, 1, 24000), "sample_rate": 24000},
                ],
                speaker_ids=[1, 2],
            )

            call_kwargs = mock_processor.call_args.kwargs
            # Verify both speakers are normalized to "Speaker N: text" format
            expected = "Speaker 1:Hello world\nSpeaker 2:Hi there"
            assert call_kwargs["text"] == [expected], \
                f"Multi-speaker [N] format must be normalized to 'Speaker N: text'. Got: {call_kwargs['text']}"

    def test_generate_audio_speaker_format_passthrough(self):
        """Verify 'Speaker N: text' format passes through correctly (already normalized)."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_output = MagicMock()
        mock_output.speech_outputs = [torch.randn(24000)]
        mock_model.generate.return_value = mock_output

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.randint(0, 100, (1, 10))}
        mock_processor.tokenizer = MagicMock()

        # Script already in "Speaker N: text" format
        test_script = "Speaker 1: Hello world"

        with patch("ComfyUI_VibeVoice.modules.generation.ProgressBarWithConsole"), \
             _patch_model_management(), \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio", return_value=_mock_voice_sample()):
            generate_audio(
                model=mock_model,
                processor=mock_processor,
                text=test_script,
                voice_samples=[{"waveform": torch.randn(1, 1, 24000), "sample_rate": 24000}],
                speaker_ids=[1],
            )

            call_kwargs = mock_processor.call_args.kwargs
            # Should be normalized to the same format
            assert call_kwargs["text"] == ["Speaker 1:Hello world"], \
                f"'Speaker N: text' format should pass through. Got: {call_kwargs['text']}"

    def test_generate_audio_empty_script_raises(self):
        """Empty script should raise ValueError."""
        mock_model = MagicMock()
        mock_processor = MagicMock()

        with pytest.raises(ValueError, match="empty or invalid"):
            generate_audio(
                model=mock_model,
                processor=mock_processor,
                text="",
                voice_samples=[],
                speaker_ids=[],
            )

    def test_generate_audio_no_valid_voice_samples_raises(self):
        """No valid voice samples should raise ValueError."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_processor = MagicMock()

        with patch("ComfyUI_VibeVoice.modules.generation.ProgressBarWithConsole"), \
             _patch_model_management(), \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio", return_value=None):
            with pytest.raises(ValueError, match="No valid voice samples"):
                generate_audio(
                    model=mock_model,
                    processor=mock_processor,
                    text="[1] Hello world",
                    voice_samples=[None],
                    speaker_ids=[1],
                )

    def test_generate_audio_requires_voice_sample(self):
        """White-box lock for CRIT-003: an all-None multi-speaker voice list must
        raise ValueError before any model use (matches README's required-reference rule)."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_processor = MagicMock()

        with patch("ComfyUI_VibeVoice.modules.generation.ProgressBarWithConsole"), \
             _patch_model_management(), \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio", return_value=None):
            with pytest.raises(ValueError, match="voice sample"):
                generate_audio(
                    model=mock_model,
                    processor=mock_processor,
                    text="[1] Hello\n[2] World",
                    voice_samples=[None, None],
                    speaker_ids=[1, 2],
                )
            # The model must never be touched when validation fails early.
            mock_model.generate.assert_not_called()

    def test_generate_audio_output_shape(self):
        """Output should be [1, 1, T] shape."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")

        mock_output = MagicMock()
        mock_output.speech_outputs = [torch.randn(48000)]
        mock_model.generate.return_value = mock_output

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.randint(0, 100, (1, 10))}
        mock_processor.tokenizer = MagicMock()

        with patch("ComfyUI_VibeVoice.modules.generation.ProgressBarWithConsole"), \
             _patch_model_management(), \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio", return_value=_mock_voice_sample()):
            waveform, sr = generate_audio(
                model=mock_model,
                processor=mock_processor,
                text="[1] Test",
                voice_samples=[{"waveform": torch.randn(1, 1, 24000), "sample_rate": 24000}],
                speaker_ids=[1],
            )

            assert waveform.shape[0] == 1
            assert waveform.shape[1] == 1

    def test_generate_audio_sample_rate(self):
        """Sample rate should be 24000."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")

        mock_output = MagicMock()
        mock_output.speech_outputs = [torch.randn(1000)]
        mock_model.generate.return_value = mock_output

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.randint(0, 100, (1, 10))}
        mock_processor.tokenizer = MagicMock()

        with patch("ComfyUI_VibeVoice.modules.generation.ProgressBarWithConsole"), \
             _patch_model_management(), \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio", return_value=_mock_voice_sample()):
            _, sr = generate_audio(
                model=mock_model,
                processor=mock_processor,
                text="[1] Test",
                voice_samples=[{"waveform": torch.randn(1, 1, 24000), "sample_rate": 24000}],
                speaker_ids=[1],
            )

            assert sr == 24000


class TestForceOffloadModel:
    """Test force_offload_model function."""

    def test_force_offload_calls_unpatch(self):
        """Plan 2026-08-18 D5: the user-requested cold offload passes destroy=True."""
        mock_patcher = MagicMock()
        mock_patcher.is_loaded = True

        with _patch_model_management():
            force_offload_model(mock_patcher, "TestModel")

            mock_patcher.unpatch_model.assert_called_once_with(
                unpatch_weights=True, destroy=True
            )

    def test_force_offload_skips_when_not_loaded(self):
        mock_patcher = MagicMock()
        mock_patcher.is_loaded = False

        with _patch_model_management():
            force_offload_model(mock_patcher, "TestModel")

            mock_patcher.unpatch_model.assert_not_called()

    def test_force_offload_passes_warm_flag(self):
        """NTH-004: warm=True must be forwarded to unpatch_model."""
        mock_patcher = MagicMock()
        mock_patcher.is_loaded = True

        with _patch_model_management():
            force_offload_model(mock_patcher, "TestModel", warm=True)

        mock_patcher.unpatch_model.assert_called_once_with(unpatch_weights=True, warm=True)


class TestGenerateAudioProgressReporting:
    """Phase 2 (2026-08-15 progress plan): generate_audio() must drive the
    standard ComfyUI ProgressBar through the vendored progress_callback."""

    @staticmethod
    def _run(progress_callback_capture: list, generate_side_effect=None):
        """Run generate_audio with a mocked model/processor; capture the
        progress_callback kwarg handed to model.generate."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_output = MagicMock()
        mock_output.speech_outputs = [torch.randn(24000)]
        if generate_side_effect is not None:
            mock_model.generate.side_effect = generate_side_effect
        else:
            mock_model.generate.return_value = mock_output

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.randint(0, 100, (1, 10))}
        mock_processor.tokenizer = MagicMock()

        with patch("ComfyUI_VibeVoice.modules.generation.ProgressBarWithConsole") as mock_pbar_cls, \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.throw_exception_if_processing_interrupted") as mock_interrupt, \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio", return_value=_mock_voice_sample()):
            mock_pbar = MagicMock()
            mock_pbar.total = 10
            mock_pbar_cls.return_value = mock_pbar

            # Capture the callback the wrapper passes into the vendored loop.
            def _capture(**kwargs):
                progress_callback_capture.append(kwargs.get("progress_callback"))
                return mock_output

            if generate_side_effect is None:
                mock_model.generate.side_effect = _capture

            result = generate_audio(
                model=mock_model,
                processor=mock_processor,
                text="[1] Hello world",
                voice_samples=[{"waveform": torch.randn(1, 1, 24000), "sample_rate": 24000}],
                speaker_ids=[1],
                inference_steps=10,
            )
            return result, mock_model, mock_pbar_cls, mock_pbar, mock_interrupt

    def test_generate_receives_callable_progress_callback(self):
        """T2.1: model.generate must receive a callable progress_callback kwarg."""
        captured = []
        self._run(captured)
        assert captured, "model.generate was not called"
        assert captured[0] is not None and callable(captured[0])

    def test_callback_maps_to_update_absolute_with_total(self):
        """T2.2: callback(current, total) -> pbar.update_absolute(current, total=total)."""
        captured = []
        _, _, _, mock_pbar, _ = self._run(captured)
        mock_pbar.update_absolute.reset_mock()

        captured[0](3, 10)
        mock_pbar.update_absolute.assert_called_once_with(3, total=10)

    def test_callback_checks_interrupt(self):
        """T2.3: the callback must call throw_exception_if_processing_interrupted;
        a raised interrupt propagates to the caller. The callback is invoked
        INSIDE the patch context so the interrupt mock is still active."""
        import comfy.model_management as mm

        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_output = MagicMock()
        mock_output.speech_outputs = [torch.randn(24000)]
        mock_model.generate.return_value = mock_output

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.randint(0, 100, (1, 10))}
        mock_processor.tokenizer = MagicMock()

        with patch("ComfyUI_VibeVoice.modules.generation.ProgressBarWithConsole") as mock_pbar_cls, \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.throw_exception_if_processing_interrupted") as mock_interrupt, \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio", return_value=_mock_voice_sample()):
            mock_pbar = MagicMock()
            mock_pbar.total = 10
            mock_pbar_cls.return_value = mock_pbar
            mock_interrupt.side_effect = mm.InterruptProcessingException()

            generate_audio(
                model=mock_model,
                processor=mock_processor,
                text="[1] Hello world",
                voice_samples=[{"waveform": torch.randn(1, 1, 24000), "sample_rate": 24000}],
                speaker_ids=[1],
                inference_steps=10,
            )

            cb = mock_model.generate.call_args.kwargs.get("progress_callback")
            assert cb is not None and callable(cb)
            with pytest.raises(mm.InterruptProcessingException):
                cb(1, 10)
            mock_interrupt.assert_called()

    def test_final_update_reaches_total_on_success(self):
        """T2.4: after successful generation the bar is driven to 100%."""
        captured = []
        _, _, mock_pbar_cls, mock_pbar, _ = self._run(captured)

        # Initial estimate = inference_steps.
        mock_pbar_cls.assert_called_once_with(10)
        # The very last update_absolute call must be the final (total) event.
        final_call = mock_pbar.update_absolute.call_args_list[-1]
        assert final_call.args == (10,), f"final update must be (total,), got {final_call}"

    def test_final_update_sent_even_when_generate_raises(self):
        """T2.5: the finally-block final update fires when model.generate raises."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.generate.side_effect = RuntimeError("boom")

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.randint(0, 100, (1, 10))}
        mock_processor.tokenizer = MagicMock()

        with patch("ComfyUI_VibeVoice.modules.generation.ProgressBarWithConsole") as mock_pbar_cls, \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.throw_exception_if_processing_interrupted"), \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio", return_value=_mock_voice_sample()):
            mock_pbar = MagicMock()
            mock_pbar.total = 10
            mock_pbar_cls.return_value = mock_pbar

            with pytest.raises(RuntimeError, match="boom"):
                generate_audio(
                    model=mock_model,
                    processor=mock_processor,
                    text="[1] Hello world",
                    voice_samples=[{"waveform": torch.randn(1, 1, 24000), "sample_rate": 24000}],
                    speaker_ids=[1],
                    inference_steps=10,
                )

            # The final absolute update must still have been sent.
            final_call = mock_pbar.update_absolute.call_args_list[-1]
            assert final_call.args == (10,)


class TestGenerateAudioStreamingGuard:
    """Regression (2026-08-16): generate_audio() must reject streaming
    (realtime) models/processors with a clear, actionable error instead of
    failing deep inside the processor with
    ``VibeVoiceStreamingProcessor.__call__() got an unexpected keyword
    argument 'text'``."""

    @staticmethod
    def _make_streaming_processor():
        class VibeVoiceStreamingProcessor:  # noqa: N801 - name is the contract
            pass

        return VibeVoiceStreamingProcessor()

    @staticmethod
    def _make_streaming_model():
        class VibeVoiceStreamingForConditionalGenerationInference:  # noqa: N801
            pass

        return VibeVoiceStreamingForConditionalGenerationInference()

    def test_rejects_streaming_processor(self):
        with pytest.raises(ValueError, match="canonical VibeVoice TTS realtime"):
            generate_audio(
                model=MagicMock(),
                processor=self._make_streaming_processor(),
                text="[1] Hello world",
                voice_samples=[_mock_voice_sample()],
                speaker_ids=[1],
            )

    def test_rejects_streaming_model(self):
        with pytest.raises(ValueError, match="canonical VibeVoice TTS realtime"):
            generate_audio(
                model=self._make_streaming_model(),
                processor=MagicMock(),
                text="[1] Hello world",
                voice_samples=[_mock_voice_sample()],
                speaker_ids=[1],
            )

    def test_guard_fires_before_script_parsing(self):
        """The guard must run before any other work (empty text still raises
        the streaming error, not the empty-script error)."""
        with pytest.raises(ValueError, match="canonical VibeVoice TTS realtime"):
            generate_audio(
                model=self._make_streaming_model(),
                processor=self._make_streaming_processor(),
                text="",
                voice_samples=[],
                speaker_ids=[],
            )

    def test_non_streaming_inputs_not_rejected_by_guard(self):
        """Plain MagicMock model/processor must pass the guard (it then fails
        later for unrelated mock reasons, proving the guard did not fire)."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_output = MagicMock()
        mock_output.speech_outputs = [torch.randn(24000)]
        mock_model.generate.return_value = mock_output

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.randint(0, 100, (1, 10))}
        mock_processor.tokenizer = MagicMock()

        with patch("ComfyUI_VibeVoice.modules.generation.ProgressBarWithConsole") as mock_pbar_cls, \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.throw_exception_if_processing_interrupted"), \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio", return_value=_mock_voice_sample()):
            mock_pbar = MagicMock()
            mock_pbar.total = 10
            mock_pbar_cls.return_value = mock_pbar

            waveform, sr = generate_audio(
                model=mock_model,
                processor=mock_processor,
                text="[1] Hello world",
                voice_samples=[{"waveform": torch.randn(1, 1, 24000), "sample_rate": 24000}],
                speaker_ids=[1],
                inference_steps=10,
            )
        assert sr == 24000
        assert waveform.ndim == 3


class TestGenerateAudioNoneSpeechOutputs:
    """Fix A: generate_audio must raise a clear RuntimeError (not AttributeError)
    when the model produces no speech outputs (None / empty).

    Regression for the int8-convrot crash: corrupt/over-quantized weights make
    the AR loop emit no speech_diffusion_id token, so speech_outputs[0] is None
    and the old code crashed with `AttributeError: 'NoneType' object has no
    attribute 'ndim'`.
    """

    def _run_with_speech_outputs(self, speech_outputs):
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_output = MagicMock()
        mock_output.speech_outputs = speech_outputs
        mock_model.generate.return_value = mock_output

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.randint(0, 100, (1, 10))}
        mock_processor.tokenizer = MagicMock()

        with patch("ComfyUI_VibeVoice.modules.generation.ProgressBarWithConsole"), \
             _patch_model_management(), \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio", return_value=_mock_voice_sample()):
            return generate_audio(
                model=mock_model,
                processor=mock_processor,
                text="[1] Hello world",
                voice_samples=[{"waveform": torch.randn(1, 1, 24000), "sample_rate": 24000}],
                speaker_ids=[1],
                inference_steps=10,
            )

    def test_none_first_output_raises_runtime_error(self):
        """speech_outputs=[None] must raise RuntimeError, not AttributeError."""
        with pytest.raises(RuntimeError, match="produced no audio"):
            self._run_with_speech_outputs([None])

    def test_empty_speech_outputs_raises_runtime_error(self):
        """speech_outputs=[] must raise RuntimeError, not IndexError."""
        with pytest.raises(RuntimeError, match="produced no audio"):
            self._run_with_speech_outputs([])

    def test_none_speech_outputs_raises_runtime_error(self):
        """speech_outputs=None must raise RuntimeError, not TypeError."""
        with pytest.raises(RuntimeError, match="produced no audio"):
            self._run_with_speech_outputs(None)

    def test_error_message_mentions_quantization_guidance(self):
        """The error message should guide the user toward a better checkpoint."""
        with pytest.raises(RuntimeError, match="over-quantized"):
            self._run_with_speech_outputs([None])

    def test_valid_output_still_returns_waveform(self):
        """A valid tensor output must still pass through unchanged (guard is
        only for None/empty, not a false positive)."""
        waveform, sr = self._run_with_speech_outputs([torch.randn(24000)])
        assert sr == 24000
        assert waveform.ndim == 3
