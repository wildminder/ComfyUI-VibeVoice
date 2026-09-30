"""Tests for modules/patcher.py - VibeVoicePatcher lifecycle."""

import torch
import pytest
import importlib
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import comfy.model_patcher

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


# ====================================================================
# T7 — offload / destroy safety across BOTH patcher classes
# ====================================================================
#
# The four production call sites that reach the patcher's unpatch_model are:
#   modules/generation.py:589      patcher.unpatch_model(unpatch_weights=True, warm=True)
#   modules/generation.py:594      patcher.unpatch_model(unpatch_weights=True, destroy=True)
#   modules/asr_generation.py:837  patcher.unpatch_model(unpatch_weights=True, destroy=True)
#   modules/model_registry.py:325  patcher.unpatch_model(unpatch_weights=True, destroy=True)
# (each read; model_registry.py is deliberately left unchanged). All four pass
# only `unpatch_weights` plus `warm`/`destroy`, and all four land on the
# patcher's OWN method — so the dynamic class only has to absorb the two flags
# core's dynamic unpatch_model does not declare.

# core's spelling, deliberately preserved: comfy/model_patcher.py:1822 and :1825
# both say "dymamic". Matched as a SUBSTRING, never as a whole message.
_DYNAMIC_PIN_TYPO = "dymamic weight loading"


def _model_cache(cache_attr: str) -> dict:
    """Return the handler cache dict named by ``loader`` / ``asr_loader``."""
    module, _, attr = cache_attr.rpartition(".")
    return getattr(importlib.import_module(module), attr)


class _FakeDynamicBase(comfy.model_patcher.ModelPatcher):
    """Aimdo-free stand-in for ``comfy.model_patcher.ModelPatcherDynamic``.

    A REAL ``ModelPatcherDynamic`` cannot be constructed under pytest, so these
    tests could not exercise the dynamic class at all: its ``__init__`` calls
    ``register_load_device`` (comfy/model_patcher.py:1760-1772, :1780-1787),
    which builds six ``comfy_aimdo.host_buffer.HostBuffer`` objects, and the
    aimdo native library is only initialised by ``main.py`` — never under
    pytest — so that construction dies with
    ``AttributeError: 'NoneType' object has no attribute 'hostbuf_allocate'``
    (verified live while writing this). It would also leave a ``__del__``
    (:1800-1801) dereferencing pin state nothing established, which is exactly
    the ``PytestUnraisableExceptionWarning`` class the plan caps.

    This stand-in reproduces the protocol surface the port depends on, copied
    from core: ``is_dynamic() -> True`` (:1803-1804), the
    vbar-term-plus-``model_loaded_weight_memory`` ``loaded_size()``
    (:1815-1817), the raising pin API (:1820-1825), ``unpin_all_weights()``
    (:1827-1828), the ``assert not load_weights`` ``patch_model``
    (:2132-2137), and the ``device_to``-ignoring ``unpatch_model`` (:2140-2146).
    """

    def __new__(cls, model=None, load_device=None, offload_device=None, size=0,
                weight_inplace_update=False, fast_disk=False):
        # Core declares this exact six-parameter __new__ (comfy/model_patcher.py:1753)
        # and reroutes a CPU load_device to a plain ModelPatcher inside it. The
        # stand-in deliberately does NOT reroute: the pack's own __new__ forwards
        # precisely these six arguments here, and a test that wanted the reroute
        # would be testing core, not the pack.
        return super().__new__(cls)

    def __init__(self, model, load_device, offload_device, size=0,
                 weight_inplace_update=False, fast_disk=False):
        super().__init__(model, load_device, offload_device, size,
                         weight_inplace_update, fast_disk)
        if not hasattr(self.model, "dynamic_vbars"):
            self.model.dynamic_vbars = {}
        if not hasattr(self.model, "dynamic_pins"):
            self.model.dynamic_pins = {}
        self.non_dynamic_delegate_model = None
        # Counts how many times the warm path released pins, so a test can tell
        # the vbar warm route apart from the legacy "move the tree" route.
        self.pin_releases = 0

    def is_dynamic(self):
        return True

    def _vbar_get(self, create=False):
        # The vbar term is always absent here: no aimdo, nothing is paged.
        return self.model.dynamic_vbars.get(self.load_device, None)

    def loaded_size(self):
        vbar = self._vbar_get()
        return (vbar.loaded_size() if vbar is not None else 0) + self.model.model_loaded_weight_memory

    def pin_weight_to_device(self, key):
        raise RuntimeError("pin_weight_to_device invalid for dymamic weight loading")

    def unpin_weight(self, key):
        raise RuntimeError("unpin_weight invalid for dymamic weight loading")

    def unpin_all_weights(self):
        self.pin_releases += 1

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True,
                    force_patch_weights=False):
        assert not load_weights
        return super().patch_model(load_weights=load_weights,
                                   force_patch_weights=force_patch_weights)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        # core hard-codes device_to=None here and frees via partially_unload_ram
        super().unpatch_model(device_to=None, unpatch_weights=False)
        if unpatch_weights:
            self.partially_unload_ram(1e32)
            self.partially_unload(None, 1e32)


@pytest.fixture
def dynamic_core_alias(monkeypatch):
    """Point the pack's resolver at the aimdo-free dynamic stand-in.

    ``make_dynamic_patcher_class`` reads BOTH names off ``comfy.model_patcher``
    (``resolve_core_patcher_class`` -> ``CoreModelPatcher``, plus the
    ``ModelPatcherDynamic`` the alias is tested against), and under pytest the
    alias is still the legacy class because ``main.py`` never runs. Both are
    rebound with ``monkeypatch`` so the real minting path is exercised rather
    than stubbed out.
    """
    monkeypatch.setattr(comfy.model_patcher, "ModelPatcherDynamic", _FakeDynamicBase)
    monkeypatch.setattr(comfy.model_patcher, "CoreModelPatcher", _FakeDynamicBase)
    return _FakeDynamicBase


class _Handler:
    """Minimal stand-in for an external TTS/ASR model handler."""

    def __init__(self, cache_key="t7_key", model_pack_name="T7Model"):
        self.cache_key = cache_key
        self.model_pack_name = model_pack_name
        self.size = 1024
        self.model = torch.nn.Linear(4, 4)
        self.processor = object()
        # The value core's ModelPatcher.patch_model writes after a real load;
        # the dynamic is_loaded is built on loaded_size(), which reads it.
        self.model_loaded_weight_memory = 64


def _build_legacy(handler, legacy_cls=VibeVoicePatcher):
    """Instantiate a LEGACY patcher with a REAL ``ModelPatcher.__init__``.

    The file's older ``_create_patcher`` helper mocks ``__init__`` out, which
    leaves the object without ``pinned`` / ``is_injected`` / ``model_options`` /
    ``callbacks``; core's ``__del__`` (:1748-1750) then walks straight into an
    ``AttributeError`` and the interpreter reports it as a
    ``PytestUnraisableExceptionWarning``. ``_Handler`` is a real object, not a
    MagicMock, so the genuine constructor works and no attribute has to be
    faked — the plan caps warning growth, so these tests must not add to it.
    """
    return legacy_cls(
        handler,
        attention_mode="sdpa",
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        size=handler.size,
    )


def _build_dynamic(handler, legacy_cls=VibeVoicePatcher):
    """Mint and instantiate the dynamic sibling of ``legacy_cls``."""
    from ComfyUI_VibeVoice.modules.patcher import make_dynamic_patcher_class

    dynamic_cls = make_dynamic_patcher_class(legacy_cls)
    return dynamic_cls(
        handler,
        attention_mode="sdpa",
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        size=handler.size,
    )


def load_side_effect(weight_memory: int = 128):
    """A stand-in for ``model_management.load_models_gpu``.

    Calls ``patcher.patch_model()`` — the no-arg form BOTH patcher classes
    accept (the dynamic one defaults ``load_weights=False``, which is what
    core's ``assert not load_weights`` at comfy/model_patcher.py:2132-2137
    requires) — and then records residency the way core's
    ``ModelPatcher.patch_model`` records it, in
    ``self.model.model_loaded_weight_memory``.

    That last step is not decoration: the dynamic class's ``is_loaded`` is
    built on ``loaded_size()``, so if residency is never recorded it correctly
    reports False and a "did the load work?" assertion would be meaningless.
    """

    def _load(models, **kwargs):
        patcher = models[0]
        result = patcher.patch_model()
        patcher.model.model_loaded_weight_memory = weight_memory
        return result

    return _load


def _core_offload_stubs():
    """Stub the four core methods the dynamic offload path calls into."""
    return (
        patch("comfy.model_patcher.ModelPatcher.unpatch_model"),
        patch("comfy.model_patcher.ModelPatcher.partially_unload_ram"),
        patch("comfy.model_patcher.ModelPatcher.partially_unload"),
    )


class TestForceOffloadDynamicSafety:
    """T7: force_offload_model must work on the dynamic class, not just legacy."""

    def test_force_offload_dynamic_does_not_raise(self, dynamic_core_alias):
        """The warm call site (generation.py:589) must not TypeError.

        ``ModelPatcherDynamic.unpatch_model`` declares no ``warm``/``destroy``
        (comfy/model_patcher.py:2140); the dynamic class absorbs both. No
        ``RuntimeError`` mentioning core's "dymamic weight loading" may escape:
        that string is only ever raised by pin_weight_to_device / unpin_weight,
        i.e. by a path the pack must never take.
        """
        from ComfyUI_VibeVoice.modules.generation import force_offload_model
        from ComfyUI_VibeVoice.modules.loader import LOADED_MODELS_CACHE

        LOADED_MODELS_CACHE.clear()
        handler = _Handler()
        patcher = _build_dynamic(handler)
        assert patcher.is_dynamic() is True

        core_unpatch, _ram, _unload = _core_offload_stubs()
        with core_unpatch as core_unpatch_call, _ram, _unload, \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.unload_all_models"), \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.soft_empty_cache"), \
             patch("ComfyUI_VibeVoice.modules.generation.gc"):
            force_offload_model(patcher, "T7Model", warm=True)

        # Warm under vbar releases pins; it must not move tensors, and it must
        # stay non-destructive.
        assert patcher.pin_releases == 1
        assert handler.model is not None

        for call in core_unpatch_call.call_args_list:
            assert _DYNAMIC_PIN_TYPO not in str(call), (
                f"the offload path reached core's pin API: {call}"
            )

    def test_dynamic_warm_path_never_touches_intermediate_device(self, dynamic_core_alias):
        """The dynamic warm path must not call intermediate_device() at all.

        Under vbar, placement belongs to the vbar; core's dynamic unpatch_model
        hard-codes ``device_to=None`` (comfy/model_patcher.py:2142), so a
        ``.to(intermediate_device())`` there would be meaningless at best.
        """
        from ComfyUI_VibeVoice.modules.generation import force_offload_model

        handler = _Handler()
        patcher = _build_dynamic(handler)

        _core, _ram, _unload = _core_offload_stubs()
        with _core, _ram, _unload, \
             patch("ComfyUI_VibeVoice.modules.patcher.model_management.intermediate_device") as intermediate, \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.unload_all_models"), \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.soft_empty_cache"), \
             patch("ComfyUI_VibeVoice.modules.generation.gc"):
            force_offload_model(patcher, "T7Model", warm=True)

        intermediate.assert_not_called()

    def test_dynamic_patch_model_rejects_load_weights(self, dynamic_core_alias):
        """``patch_model()`` with no args is the only supported shape.

        Core's dynamic ``patch_model`` asserts ``not load_weights``
        (comfy/model_patcher.py:2132-2137); the pack raises a named ValueError
        instead of a bare assert, pointing at load_models_gpu.
        """
        handler = _Handler()
        patcher = _build_dynamic(handler)

        with pytest.raises(ValueError, match="load_models_gpu"):
            patcher.patch_model(load_weights=True)

    def test_dynamic_warm_and_destroy_each_reach_core_once(self, dynamic_core_alias):
        """Both flag combinations land on core's unpatch_model exactly once."""
        handler = _Handler()
        patcher = _build_dynamic(handler)

        for flag in ("warm", "destroy"):
            _core, _ram, _unload = _core_offload_stubs()
            with _core as core_unpatch, _ram, _unload:
                patcher.unpatch_model(unpatch_weights=True, **{flag: True})
            assert core_unpatch.call_count == 1, flag


class TestForceOffloadLegacyUnchanged:
    """T7: the legacy warm path (modules/patcher.py:188-202) must not move."""

    def test_legacy_warm_moves_tree_to_intermediate_device(self):
        """Regression lock: warm offload still moves tensors, it does not free."""
        from ComfyUI_VibeVoice.modules.generation import force_offload_model
        from ComfyUI_VibeVoice.modules.loader import LOADED_MODELS_CACHE

        LOADED_MODELS_CACHE.clear()
        handler = _Handler(cache_key="t7_legacy_warm")
        # A MagicMock records the exact device the tree was moved to; a real
        # nn.Module cannot express "moved to a patched device" in an assert.
        # The reference is kept because unpatch_model REPLACES handler.model
        # with the return value of .to(), so handler.model is a different mock
        # by the time the assertion runs.
        tree = MagicMock()
        handler.model = tree
        patcher = _build_legacy(handler)
        patcher.model = handler

        offload_target = torch.device("cpu")
        with patch("ComfyUI_VibeVoice.modules.patcher.model_management.intermediate_device",
                   return_value=offload_target) as intermediate, \
             patch("comfy.model_patcher.ModelPatcher.unpatch_model") as core_unpatch, \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.unload_all_models"), \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.soft_empty_cache"), \
             patch("ComfyUI_VibeVoice.modules.generation.gc"):
            force_offload_model(patcher, "T7Model", warm=True)

        intermediate.assert_called_once_with()
        tree.to.assert_called_once_with(offload_target)
        # The legacy warm path keeps the GPU slot but retains the weights.
        assert core_unpatch.call_args.kwargs["unpatch_weights"] is False
        assert patcher._warm_offloaded is True
        assert not hasattr(patcher, "pin_releases"), (
            "the legacy class must not grow the dynamic class's pin bookkeeping"
        )


class TestDestroyEvictsRegistry:
    """T7: the destroy sequence (modules/patcher.py:215-231) runs on BOTH classes."""

    _CASES = [
        (VibeVoicePatcher, "ComfyUI_VibeVoice.modules.loader.LOADED_MODELS_CACHE"),
        (VibeVoiceASRPatcher, "ComfyUI_VibeVoice.modules.asr_loader.LOADED_ASR_MODELS_CACHE"),
    ]

    @pytest.mark.parametrize(
        "legacy_cls, cache_name", _CASES, ids=["tts", "asr"],
    )
    def test_destroy_evicts_cache_on_legacy(self, legacy_cls, cache_name):
        from ComfyUI_VibeVoice.modules.generation import force_offload_model

        cache = _model_cache(cache_name)
        cache.clear()
        handler = _Handler(cache_key=f"t7_{legacy_cls.__name__}")
        # The class under test decides the cache (`_model_cache`), so build the
        # class the case names rather than always the TTS one.
        patcher = _build_legacy(handler, legacy_cls)
        patcher.model = handler
        cache[handler.cache_key] = patcher

        with patch("comfy.model_patcher.ModelPatcher.unpatch_model"), \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.unload_all_models"), \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.soft_empty_cache"), \
             patch("ComfyUI_VibeVoice.modules.generation.gc"):
            force_offload_model(patcher, "T7Model", warm=False)

        assert handler.cache_key not in cache
        assert handler.model is None
        assert handler.processor is None

    @pytest.mark.parametrize(
        "legacy_cls, cache_name", _CASES, ids=["tts", "asr"],
    )
    def test_destroy_evicts_cache_on_dynamic(self, dynamic_core_alias, legacy_cls, cache_name):
        from ComfyUI_VibeVoice.modules.generation import force_offload_model

        cache = _model_cache(cache_name)
        cache.clear()
        handler = _Handler(cache_key=f"t7_dyn_{legacy_cls.__name__}")
        patcher = _build_dynamic(handler, legacy_cls=legacy_cls)
        cache[handler.cache_key] = patcher
        assert patcher.is_dynamic() is True

        _core, _ram, _unload = _core_offload_stubs()
        with _core, _ram, _unload, \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.unload_all_models"), \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.soft_empty_cache"), \
             patch("ComfyUI_VibeVoice.modules.generation.gc"):
            force_offload_model(patcher, "T7Model", warm=False)

        assert handler.cache_key not in cache
        assert handler.model is None
        assert handler.processor is None

    def test_tts_destroy_never_evicts_the_asr_cache(self, dynamic_core_alias):
        """AUD-012, on the dynamic class too: cache isolation must hold."""
        from ComfyUI_VibeVoice.modules.loader import LOADED_MODELS_CACHE
        from ComfyUI_VibeVoice.modules.asr_loader import LOADED_ASR_MODELS_CACHE
        from ComfyUI_VibeVoice.modules.generation import force_offload_model

        LOADED_MODELS_CACHE.clear()
        LOADED_ASR_MODELS_CACHE.clear()
        # A COLLIDING key, seeded in the ASR cache only.
        handler = _Handler(cache_key="t7_collision")
        patcher = _build_dynamic(handler, legacy_cls=VibeVoicePatcher)
        LOADED_ASR_MODELS_CACHE["t7_collision"] = patcher

        _core, _ram, _unload = _core_offload_stubs()
        with _core, _ram, _unload, \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.unload_all_models"), \
             patch("ComfyUI_VibeVoice.modules.generation.model_management.soft_empty_cache"), \
             patch("ComfyUI_VibeVoice.modules.generation.gc"):
            force_offload_model(patcher, "T7Model", warm=False)

        assert "t7_collision" in LOADED_ASR_MODELS_CACHE
        assert "t7_collision" not in LOADED_MODELS_CACHE


class TestPinApiIsNeverCalled:
    """T7: the pack must never reach core's pin API.

    ``pin_weight_to_device`` / ``unpin_weight`` raise unconditionally under the
    dynamic protocol (comfy/model_patcher.py:1820-1825) — under vbar, pinning
    is deferred to ops time. Today the pack has zero call sites
    (``grep -rn "pin_weight_to_device\\|unpin_weight" modules/ nodes/`` returns
    nothing), so this assertion is vacuously true; that is the point. It is a
    LOCK, and it must fail loudly the moment anyone adds such a call.
    """

    BANNED = ("pin_weight_to_device", "unpin_weight")

    def test_no_call_sites_in_pack_sources(self):
        pack_root = Path(__file__).resolve().parents[1]
        searched = sorted(
            [p for p in (pack_root / "modules").glob("*.py")]
            + [p for p in (pack_root / "nodes").glob("*.py")]
        )
        assert searched, "the glob found no pack sources — the lock is looking nowhere"

        offenders = []
        for path in searched:
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                for banned in self.BANNED:
                    # A CALL is `name(`. This skips prose in comments and
                    # docstrings, and the `unpin_all_weights` the dynamic warm
                    # path legitimately relies on.
                    if f"{banned}(" in stripped:
                        offenders.append(f"{path.relative_to(pack_root)}:{lineno}: {stripped}")
        assert not offenders, (
            "core's dynamic pin API raises unconditionally "
            "(comfy/model_patcher.py:1820-1825); the pack must not call it:\n"
            + "\n".join(offenders)
        )
