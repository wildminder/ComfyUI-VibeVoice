"""Tests for modules/patcher.py - VibeVoicePatcher lifecycle."""

import torch
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from ComfyUI_VibeVoice.modules.patcher import VibeVoicePatcher, VibeVoiceASRPatcher


def _create_patcher(handler, attention_mode="sdpa", dtype=None):
    """Create a VibeVoicePatcher with ModelPatcher.__init__ mocked."""
    with patch("comfy.model_patcher.ModelPatcher.__init__"):
        patcher = VibeVoicePatcher(
            handler,
            attention_mode=attention_mode,
            dtype=dtype,
            load_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            size=1000,
        )
    # Set attributes that ModelPatcher.__init__ would normally set
    patcher.load_device = torch.device("cpu")
    patcher.offload_device = torch.device("cpu")
    return patcher


class TestVibeVoicePatcherInit:
    """Test VibeVoicePatcher initialization."""

    def test_patcher_init(self):
        handler = MagicMock()
        handler.cache_key = "test_key"
        patcher = _create_patcher(handler, attention_mode="sdpa")
        assert patcher.attention_mode == "sdpa"
        assert patcher.cache_key == "test_key"

    def test_patcher_init_with_dtype(self):
        handler = MagicMock()
        handler.cache_key = "test_key"
        patcher = _create_patcher(handler, attention_mode="eager", dtype=torch.float16)
        assert patcher.target_dtype == torch.float16

    def test_patcher_default_dtype_none(self):
        handler = MagicMock()
        handler.cache_key = "test_key"
        patcher = _create_patcher(handler, attention_mode="sdpa")
        assert patcher.target_dtype is None


class TestVibeVoicePatcherIsLoaded:
    """Test VibeVoicePatcher.is_loaded property."""

    def test_is_loaded_false_initial(self):
        handler = MagicMock()
        handler.cache_key = "test_key"
        handler.model = None
        patcher = _create_patcher(handler)
        patcher.model = handler
        assert patcher.is_loaded is False

    def test_is_loaded_true(self):
        handler = MagicMock()
        handler.cache_key = "test_key"
        handler.model = MagicMock()
        handler.model.model = MagicMock()  # Not None
        patcher = _create_patcher(handler)
        patcher.model = handler
        assert patcher.is_loaded is True

    def test_is_loaded_false_when_model_none(self):
        handler = MagicMock()
        handler.cache_key = "test_key"
        # Use a real object so .model = None stays None (MagicMock auto-creates attrs)
        class FakeHandler:
            model = None
            model_pack_name = "TestModel"
        handler = FakeHandler()
        patcher = _create_patcher(handler)
        patcher.model = handler
        assert patcher.is_loaded is False


class TestVibeVoicePatcherPatchModel:
    """Test VibeVoicePatcher.patch_model."""

    def test_patch_model_loads_when_not_loaded(self):
        handler = MagicMock()
        handler.cache_key = "test_key"
        handler.model_pack_name = "TestModel"
        handler.model = None

        mock_inner_model = MagicMock()
        handler.load_model = MagicMock()

        def side_effect_load(device, attn):
            handler.model = mock_inner_model
        handler.load_model.side_effect = side_effect_load

        patcher = _create_patcher(handler)
        patcher.model = handler

        with patch("comfy.model_patcher.ModelPatcher.patch_model") as mock_super_patch:
            patcher.patch_model()

            handler.load_model.assert_called_once()
            mock_inner_model.to.assert_called()
            # Verify super().patch_model() is called with load_weights=True (default)
            # so ComfyUI can properly track model_loaded_weight_memory
            call_kwargs = mock_super_patch.call_args.kwargs
            assert call_kwargs.get("load_weights", True) is True

    def test_patch_model_skips_load_when_already_loaded(self):
        handler = MagicMock()
        handler.cache_key = "test_key"
        handler.model_pack_name = "TestModel"

        mock_inner_model = MagicMock()
        handler.model = mock_inner_model

        patcher = _create_patcher(handler)
        patcher.model = handler

        with patch("comfy.model_patcher.ModelPatcher.patch_model"):
            patcher.patch_model()

            handler.load_model.assert_not_called()
            mock_inner_model.to.assert_called()

    def test_patch_model_with_device_to(self):
        """When device_to is specified, model is moved to that device."""
        handler = MagicMock()
        handler.cache_key = "test_key"
        handler.model_pack_name = "TestModel"

        mock_inner_model = MagicMock()
        handler.model = mock_inner_model

        patcher = _create_patcher(handler)
        patcher.model = handler

        target_device = torch.device("cuda:0")
        with patch("comfy.model_patcher.ModelPatcher.patch_model") as mock_super_patch:
            patcher.patch_model(device_to=target_device)

            handler.load_model.assert_not_called()
            mock_inner_model.to.assert_called_with(target_device)
            mock_super_patch.assert_called_once()
            # Verify device_to is passed through to super
            call_kwargs = mock_super_patch.call_args.kwargs
            assert call_kwargs.get("device_to") == target_device

    def test_patch_model_passes_lowvram_memory(self):
        """lowvram_model_memory is passed through to super().patch_model()."""
        handler = MagicMock()
        handler.cache_key = "test_key"
        handler.model_pack_name = "TestModel"
        mock_inner_model = MagicMock()
        handler.model = mock_inner_model

        patcher = _create_patcher(handler)
        patcher.model = handler

        with patch("comfy.model_patcher.ModelPatcher.patch_model") as mock_super_patch:
            patcher.patch_model(lowvram_model_memory=1024)

            call_kwargs = mock_super_patch.call_args.kwargs
            assert call_kwargs.get("lowvram_model_memory") == 1024

    def test_patch_model_passes_force_patch_weights(self):
        """force_patch_weights is passed through to super().patch_model()."""
        handler = MagicMock()
        handler.cache_key = "test_key"
        handler.model_pack_name = "TestModel"
        mock_inner_model = MagicMock()
        handler.model = mock_inner_model

        patcher = _create_patcher(handler)
        patcher.model = handler

        with patch("comfy.model_patcher.ModelPatcher.patch_model") as mock_super_patch:
            patcher.patch_model(force_patch_weights=True)

            call_kwargs = mock_super_patch.call_args.kwargs
            assert call_kwargs.get("force_patch_weights") is True


class TestVibeVoicePatcherUnpatchModel:
    """Test VibeVoicePatcher.unpatch_model."""

    def test_unpatch_clears_model(self):
        from ComfyUI_VibeVoice.modules.loader import LOADED_MODELS_CACHE
        LOADED_MODELS_CACHE.clear()
        LOADED_MODELS_CACHE["test_key"] = ("model", "processor")

        # Use a real object to track state changes (MagicMock auto-creates attrs)
        class FakeHandler:
            def __init__(self):
                self.cache_key = "test_key"
                self.model_pack_name = "TestModel"
                self.model = MagicMock()
                self.model.model = MagicMock()
                self.model.processor = MagicMock()

        handler = FakeHandler()

        patcher = _create_patcher(handler)
        patcher.model = handler

        with patch("comfy.model_patcher.ModelPatcher.unpatch_model"):
            patcher.unpatch_model(unpatch_weights=True)

            # unpatch_model sets self.model.model = None and self.model.processor = None
            # self.model is the handler, so handler.model (inner model) is set to None
            assert handler.model is None
            assert "test_key" not in LOADED_MODELS_CACHE

    def test_unpatch_no_clear_when_no_unpatch_weights(self):
        from ComfyUI_VibeVoice.modules.loader import LOADED_MODELS_CACHE
        LOADED_MODELS_CACHE.clear()
        LOADED_MODELS_CACHE["test_key"] = ("model", "processor")

        handler = MagicMock()
        handler.cache_key = "test_key"
        handler.model = MagicMock()
        handler.model.model = MagicMock()
        handler.model.model_pack_name = "TestModel"

        patcher = _create_patcher(handler)
        patcher.model = handler

        with patch("comfy.model_patcher.ModelPatcher.unpatch_model"):
            patcher.unpatch_model(unpatch_weights=False)

            assert handler.model.model is not None
            assert "test_key" in LOADED_MODELS_CACHE


class TestVibeVoiceASRPatcher:
    """CRIT-001: ASR patcher must clear the ASR cache, not the TTS cache."""

    def _build_asr_patcher(self, handler, attention_mode="sdpa"):
        with patch("comfy.model_patcher.ModelPatcher.__init__"):
            patcher = VibeVoiceASRPatcher(
                handler,
                attention_mode=attention_mode,
                load_device=torch.device("cpu"),
                offload_device=torch.device("cpu"),
                size=1024,
            )
        patcher.load_device = torch.device("cpu")
        patcher.offload_device = torch.device("cpu")
        patcher.model = handler
        return patcher

    def test_asr_unpatch_clears_asr_cache_only(self):
        from ComfyUI_VibeVoice.modules.asr_loader import LOADED_ASR_MODELS_CACHE
        from ComfyUI_VibeVoice.modules.loader import LOADED_MODELS_CACHE

        LOADED_ASR_MODELS_CACHE.clear()
        LOADED_MODELS_CACHE.clear()
        # Same cache key present in both caches to prove the ASR patcher
        # only touches the ASR-specific cache.
        LOADED_ASR_MODELS_CACHE["tiny"] = ("m", "p")
        LOADED_MODELS_CACHE["tiny"] = ("m", "p")

        class FakeHandler:
            def __init__(self):
                self.cache_key = "tiny"
                self.model_pack_name = "TestASR"
                self.model = MagicMock()
                self.model.model = MagicMock()
                self.model.processor = MagicMock()

        handler = FakeHandler()
        patcher = self._build_asr_patcher(handler)

        with patch("comfy.model_patcher.ModelPatcher.unpatch_model"):
            patcher.unpatch_model(unpatch_weights=True)

        assert "tiny" not in LOADED_ASR_MODELS_CACHE
        # TTS cache must remain untouched.
        assert "tiny" in LOADED_MODELS_CACHE

        LOADED_ASR_MODELS_CACHE.clear()
        LOADED_MODELS_CACHE.clear()
