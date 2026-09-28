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
    patcher.pinned = set()
    patcher.is_injected = False
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

        def side_effect_load(device, attention_mode="sdpa"):
            handler.model = mock_inner_model
        handler.load_model.side_effect = side_effect_load

        patcher = _create_patcher(handler)
        patcher.model = handler

        with patch("comfy.model_patcher.ModelPatcher.patch_model") as mock_super_patch:
            patcher.patch_model()

            handler.load_model.assert_called_once()
            # Plan 2026-08-18 D6/RC-5: no bulk pre-move; the H2D transfer is
            # owned by super().patch_model() -> ModelPatcher.load().
            mock_inner_model.to.assert_not_called()
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
            # Plan 2026-08-18 D6/RC-5: no bulk pre-move.
            mock_inner_model.to.assert_not_called()

    def test_patch_model_with_device_to(self):
        """device_to is forwarded to super; no bulk pre-move happens here."""
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
            # Plan 2026-08-18 D6/RC-5: the move is delegated to super, not
            # done as a bulk .to() here.
            mock_inner_model.to.assert_not_called()
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

    def test_unpatch_default_keeps_model(self):
        """Plan 2026-08-18 D5/RC-6: the default offload is NON-destructive."""
        from ComfyUI_VibeVoice.modules.loader import LOADED_MODELS_CACHE
        LOADED_MODELS_CACHE.clear()
        LOADED_MODELS_CACHE["test_key"] = ("model", "processor")

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

            # Default offload keeps the model in RAM and the cache intact.
            assert handler.model is not None
            assert "test_key" in LOADED_MODELS_CACHE

    def test_unpatch_destroy_clears_model(self):
        """Plan 2026-08-18 D5: destroy=True keeps the old destructive path."""
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
            patcher.unpatch_model(unpatch_weights=True, destroy=True)

            # destroy=True sets self.model.model = None and self.model.processor = None
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
            # Plan 2026-08-18 D5: cache eviction happens on the explicit
            # destroy path (the default offload is non-destructive).
            patcher.unpatch_model(unpatch_weights=True, destroy=True)

        assert "tiny" not in LOADED_ASR_MODELS_CACHE
        # TTS cache must remain untouched.
        assert "tiny" in LOADED_MODELS_CACHE

        LOADED_ASR_MODELS_CACHE.clear()
        LOADED_MODELS_CACHE.clear()


class TestCloneCompatibility:
    """core's ModelPatcher.clone() reconstructs the subclass with POSITIONAL
    args — (model, load_device, offload_device, size, ...) — at
    comfy/model_patcher.py:446.

    With ``__init__(self, model, attention_mode="eager", dtype=None, *args,
    **kwargs)`` the first two POSITIONAL args bind to attention_mode and
    dtype, ``offload_device`` is never forwarded, and the call dies with
    ``TypeError: ModelPatcher.__init__() missing 1 required positional
    argument: 'offload_device'``. The two extras are keyword-only for that
    reason; this class is the lock.
    """

    @staticmethod
    def _real_base_record():
        """A stand-in ModelPatcher.__init__ that RECORDS what it was given.

        The bug is an argument-routing fault, so the test has to look at the
        call, not merely at the absence of an exception. Using a real
        ``super().__init__`` (rather than a MagicMock) is also what makes the
        failure mode real: a mocked base swallows the broken call entirely and
        the test would pass either way.
        """
        return patch(
            "comfy.model_patcher.ModelPatcher.__init__",
            autospec=True,
            side_effect=lambda model, *a, **kw: None,
        )

    def test_clone_forwards_devices_and_size_positionally(self):
        handler = MagicMock()
        handler.cache_key = "clone-test"
        load_device = torch.device("cuda")
        offload_device = torch.device("cpu")

        with self._real_base_record() as base_init:
            VibeVoicePatcher(
                handler,
                attention_mode="sage",
                dtype=torch.bfloat16,
                load_device=load_device,
                offload_device=offload_device,
                size=1000,
            )
        # Exactly what core's clone() does on a second, un-keyed construction.
        with self._real_base_record() as clone_init:
            clone = VibeVoicePatcher(handler, load_device, offload_device, 1000)

        # autospec records the bound `self` first, so args[0] is the patcher,
        # args[1] the model, and everything after is what core passed on.
        args, kwargs = clone_init.call_args
        assert args[1] is handler
        assert args[2:] == (load_device, offload_device, 1000), (
            "the positional device/size arguments must reach "
            "ModelPatcher.__init__ untouched. A positional extra in the "
            "subclass signature swallows load_device into attention_mode and "
            "then drops offload_device, which is exactly the TypeError core "
            "raises during clone eviction."
        )
        assert kwargs == {}

    def test_clone_without_keywords_gets_eager_defaults(self):
        """A core-constructed clone has no attention_mode/dtype, so it must
        fall back to the class defaults rather than to a device object."""
        handler = MagicMock()
        handler.cache_key = "clone-test"
        with self._real_base_record():
            clone = VibeVoicePatcher(
                handler, torch.device("cuda"), torch.device("cpu"), 1000
            )
        assert clone.attention_mode == "eager"
        assert clone.target_dtype is None

    def test_asr_patcher_inherits_the_keyword_only_signature(self):
        """VibeVoiceASRPatcher subclasses the same __init__, so the fix must
        reach it — otherwise ComfyUI still crashes on an ASR clone."""
        import inspect

        for cls in (VibeVoicePatcher, VibeVoiceASRPatcher):
            spec = inspect.getfullargspec(cls.__init__)
            assert spec.kwonlyargs == ["attention_mode", "dtype"], (
                f"{cls.__name__}.__init__ must take its extras keyword-only"
            )
            assert spec.args == ["self", "model"], (
                f"{cls.__name__}.__init__ must not declare a positional "
                f"parameter after `model` — core's clone() passes the devices "
                f"positionally and they would bind here"
            )
            assert spec.varargs == "args"

    def test_keyword_construction_still_works(self):
        """Every construction site in modules/ passes the extras by keyword
        (generation.py x2, asr_generation.py x2); guard that here."""
        handler = MagicMock()
        handler.cache_key = "clone-test"
        patcher = _create_patcher(handler, attention_mode="sdpa", dtype=torch.float16)
        assert patcher.attention_mode == "sdpa"
        assert patcher.target_dtype is torch.float16
