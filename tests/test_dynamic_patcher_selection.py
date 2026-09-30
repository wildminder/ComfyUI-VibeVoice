"""Coverage for the dynamic-VRAM patcher selector (plan 2026-09-29, T6).

``select_patcher_class`` is the WHOLE invariant-2 guarantee: only the dense
external single-file family may leave the legacy patcher. This module always
drives the REAL selector -- never a hand-written stand-in -- because a stub
would test the stub, not the guarantee.

Two things are true of the test environment and shape every test here:

* ``comfy.model_patcher.CoreModelPatcher`` is still the legacy ``ModelPatcher``
  under pytest, because ``main.py:300`` (the aimdo rebind) never runs. Tests
  therefore MONKEYPATCH the alias; they cannot rely on the host.
* ``torch.cuda.is_available()`` is True in this interpreter, so the CUDA test
  runs for real. The T6 selector tests construct NO dynamic patcher -- the
  selector is allocation-free by construction (``register_load_device`` would
  build six real aimdo host buffers, and ``__del__`` would then dereference pin
  state that never existed). The T9 tests further down DO build a real
  ``ModelPatcherDynamic`` subclass, via the ``dynamic_runtime`` fixture, which
  stubs exactly one thing: ``comfy_aimdo.host_buffer.HostBuffer``. Everything
  else -- the class, ``load``, the vbar bookkeeping -- is core's own code.
"""

import inspect
from unittest.mock import patch

import pytest
import torch
from torch import nn

import comfy.model_patcher
import comfy_aimdo.control
import comfy_aimdo.host_buffer

from ComfyUI_VibeVoice.modules import comfy_stream, external_loader
from ComfyUI_VibeVoice.modules.asr_generation import ExternalVibeVoiceASRModelHandler
from ComfyUI_VibeVoice.modules.comfy_stream import (
    convert_tree_for_streaming,
    streaming_conversion,
)
from ComfyUI_VibeVoice.modules.model_registry import (
    FAMILY_ASR,
    clear_active_keys,
    clear_bundle_registry,
    evict_if_changed,
    identity_for_external,
    register_model_bundle,
)
from ComfyUI_VibeVoice.modules.patcher import (
    VibeVoiceASRPatcher,
    VibeVoicePatcher,
    dynamic_vram_available,
    load_to_device,
    resolve_core_patcher_class,
    select_patcher_class,
)
from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_ASR_PATCHER_CACHE

DYNAMIC = comfy.model_patcher.ModelPatcherDynamic

# Exactly what modules/external_loader.py emits today, plus one hostile unknown.
EMITTED_FAMILIES = ("gguf_block", "convrot_int8", "fp8_resident")
# "gguf_resident" is NOT a label the loader emits -- the emitted set is
# dense / gguf_block / convrot_int8 / fp8_resident. It is included deliberately
# as a hostile unknown family: anything that is not exactly "dense" must return
# the legacy class, so a typo'd or future label degrades to legacy rather than
# silently opening the dynamic branch.
NON_DENSE_FAMILIES = EMITTED_FAMILIES + ("gguf_resident",)


@pytest.fixture
def aimdo_enabled(monkeypatch):
    """Pretend ``main.py:300`` ran: the alias names the dynamic patcher."""
    monkeypatch.setattr(comfy.model_patcher, "CoreModelPatcher", DYNAMIC)
    return DYNAMIC


class TestPatcherSelection:
    """The selector is pure: (weight_family, load_device) -> class."""

    def test_dense_cuda_aimdo_selects_dynamic(self, aimdo_enabled):
        """The one positive case: dense + CUDA + aimdo live -> dynamic class."""
        selected = select_patcher_class("dense", torch.device("cuda"))
        assert issubclass(selected, DYNAMIC)
        assert selected is not VibeVoicePatcher
        assert selected.__mro__[1] is aimdo_enabled

    @pytest.mark.parametrize(
        "family", [f for f in NON_DENSE_FAMILIES if f != "gguf_block"]
    )
    def test_quant_families_follow_core_availability(self, aimdo_enabled, family):
        """2026-09-30: quant families follow core's availability, nothing else.

        The old lock ("quant families NEVER open the dynamic branch — a stop
        condition") came from the pre-aimdo port and measured wrong: those
        families CLONE every tensor into host RAM and then fully H2D it,
        because a legacy patcher has nowhere to page from. With aimdo live,
        gguf / convrot_int8 / fp8_resident now take the same dynamic class as
        dense — the decision is core's, exactly like comfy/sd.py:2403.
        """
        selected = select_patcher_class(family, torch.device("cuda"))
        assert issubclass(selected, DYNAMIC)
        assert selected is not VibeVoicePatcher
        assert selected.__mro__[1] is aimdo_enabled

    def test_unknown_and_missing_family_follow_core_availability(self, aimdo_enabled):
        """weight_family is no longer consulted: even an unrecognised label
        (or none at all — the standard-directory dropdown loaders) gets the
        dynamic class when core resolved one."""
        for family in (None, "", "some_future_family"):
            selected = select_patcher_class(family, torch.device("cuda"))
            assert issubclass(selected, DYNAMIC), family

    def test_gguf_block_stays_legacy_while_its_install_copies(self, aimdo_enabled):
        """gguf_block is the one family still excluded, for a MEASURED reason.

        ``_install_gguf_weights`` materialises every raw block into a private
        host copy (GGUFTensor.from_reader_tensor clones the reader's view) and
        then unmaps the file, so a dynamic patcher would page from host RAM —
        no RAM win, different offload order. QA flagged this as a real
        regression when the whitelist was removed wholesale (2026-09-30).
        """
        assert (
            select_patcher_class("gguf_block", torch.device("cuda"))
            is VibeVoicePatcher
        )

    def test_dense_on_cpu_selects_legacy(self, aimdo_enabled):
        """A CPU load device reroutes ModelPatcherDynamic.__new__ to legacy."""
        assert select_patcher_class("dense", torch.device("cpu")) is VibeVoicePatcher

    def test_dense_without_aimdo_selects_legacy(self):
        """No rebind -> CoreModelPatcher is still ModelPatcher -> legacy."""
        assert resolve_core_patcher_class() is comfy.model_patcher.ModelPatcher
        selected = select_patcher_class("dense", torch.device("cuda"))
        assert selected is VibeVoicePatcher

    @pytest.mark.parametrize("attention_mode", ("eager", "sdpa", "sage"))
    def test_selection_is_orthogonal_to_attention_mode(self, aimdo_enabled, attention_mode):
        """The selector takes no attention argument, so it cannot depend on one."""
        selected = select_patcher_class("dense", torch.device("cuda"))
        # The class object is literally the same one every time -- the
        # attention mode is a per-instance __init__ argument, never captured on
        # the class.
        assert selected is select_patcher_class("dense", torch.device("cuda"))
        assert "attention_mode" not in vars(selected)
        assert issubclass(selected, DYNAMIC)

    def test_selection_ignores_bundle_dtype(self, aimdo_enabled):
        """dtype lives on the bundle and is passed to __init__, not the selector."""
        results = {
            select_patcher_class("dense", torch.device("cuda"))
            for _ in range(2)
        }
        # Same call, same answer -- the selector has no dtype parameter at all,
        # so bf16 and fp16 bundles cannot diverge.
        assert len(results) == 1
        assert "dtype" not in inspect.signature(select_patcher_class).parameters

    def test_asr_legacy_class_is_honoured(self, aimdo_enabled):
        """The TTS/ASR split rides on legacy_cls, not on a second selector."""
        assert (
            select_patcher_class("dense", torch.device("cuda"), legacy_cls=VibeVoiceASRPatcher)
            is not VibeVoiceASRPatcher
        )
        # 2026-09-30: with the family whitelist gone, the ASR class reaches the
        # dynamic branch on core's availability alone — the same rule as TTS,
        # still keyed on legacy_cls so the two families keep separate caches.
        asr_selected = select_patcher_class(
            "fp8_resident", torch.device("cuda"), legacy_cls=VibeVoiceASRPatcher
        )
        assert issubclass(asr_selected, DYNAMIC)
        assert "ASR" in asr_selected.__name__
        assert asr_selected is not select_patcher_class(
            "fp8_resident", torch.device("cuda")
        )

    @pytest.mark.parametrize("bad_device", [None, "", "not-a-device"])
    def test_unusable_device_falls_back_to_legacy(self, aimdo_enabled, bad_device):
        """The probe is failure-open: anything it cannot read is legacy."""
        assert dynamic_vram_available(DYNAMIC, bad_device) is False
        assert select_patcher_class("dense", bad_device) is VibeVoicePatcher

    def test_minted_class_is_cached_per_base(self, aimdo_enabled):
        """Repeated selections return the same class object (stable identity)."""
        first = select_patcher_class("dense", torch.device("cuda"))
        second = select_patcher_class("dense", torch.device("cuda"))
        assert first is second


class TestStreamingConversionSuppression:
    """The conversion must fire on BOTH routes.

    CORRECTED 2026-09-29 (QA round 1, verified against core): the earlier
    "one weight owner" premise was false. ``make_streaming``'s forward calls
    ``comfy.ops.cast_bias_weight`` (modules/comfy_stream.py:96), and
    ``cast_bias_weight`` is the vbar CONSUMER (``comfy/ops.py:374``) — the two
    mechanisms are producer and consumer. ``comfy_cast_weights`` is also the
    exact attribute core's dynamic ``load()`` gates on to take the vbar branch
    (``comfy/model_patcher.py:1967``); suppressing the conversion would push
    every module into the eager ``else`` branch (:2000-2009), which stashes a
    full host copy in ``self.backup``. The ``streaming_conversion`` context
    manager stays as an inert, tested seam; no loader enters it anymore.
    """

    @staticmethod
    def _tiny_model() -> nn.Module:
        return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))

    def test_conversion_is_on_by_default_and_on_both_routes(self):
        """The flag is True with no context, and the conversion always converts."""
        assert comfy_stream._STREAMING_CONVERSION_ENABLED is True
        plain = self._tiny_model()
        assert convert_tree_for_streaming(plain) != {}
        assert type(plain[0]) is not nn.Linear
        # Exact type: _ComfyStreamLinear SUBCLASSES nn.Linear, so isinstance
        # would be true either way and would prove nothing.
        assert type(plain[1]) is not nn.Linear

    def test_loader_never_suppresses_and_preserves_file_views_on_dynamic(self):
        """Source lock on the corrected dense assign sites.

        Exactly two dense assign sites, both passing ``preserve_file_views``
        (the dynamic route's clone gate), and NO suppression context anywhere.
        The generation-time load paths still carry no streaming_conversion
        reference (the conversion has long since fired there).
        """
        source = inspect.getsource(external_loader)
        assert "streaming_conversion(" not in source.replace(
            "def streaming_conversion", ""
        ).replace("streaming_conversion,", "")
        assert source.count("preserve_file_views=dynamic_route") == 2
        assert "_load_state_dict_into_model_from_memory" in source

        from ComfyUI_VibeVoice.modules import asr_generation, generation

        for module in (generation, asr_generation):
            assert "streaming_conversion" not in inspect.getsource(module)

    def test_conversion_context_manager_still_works_as_a_seam(self):
        """The seam stays green for the loader-level port that may use it later."""
        suppressed = self._tiny_model()
        with streaming_conversion(False):
            assert convert_tree_for_streaming(suppressed) == {}
        assert type(suppressed[0]) is nn.Linear
        # The flag is restored on exit, so a later load is unaffected.
        assert comfy_stream._STREAMING_CONVERSION_ENABLED is True

    def test_flag_is_restored_on_exception(self):
        """A failed load must not leak the suppression into the next bundle."""
        with pytest.raises(RuntimeError):
            with streaming_conversion(False):
                raise RuntimeError("boom")
        assert comfy_stream._STREAMING_CONVERSION_ENABLED is True


# ====================================================================
# T9 - coexistence and core-integration contracts.
#
# T6 above is about the SELECTOR (a pure function). What is left is the part
# that only shows up when a patcher actually meets ComfyUI core: a legacy
# patcher and a dynamic one alive in the same session, the cache keying that
# survives the route change, and the two core entry points
# (`clone(disable_dynamic=True)` and `deepclone_multigpu`) that hard-require
# `cached_patcher_init`.
#
# ENVIRONMENT LIMIT, measured not assumed: `comfy_aimdo` binds its native
# library at IMPORT time (`lib = control.lib`,
# comfy_aimdo/host_buffer.py:6) and only `main.py` ever loads it, which never
# runs under pytest. So:
#   * constructing a real `ModelPatcherDynamic` raises
#     `AttributeError: 'NoneType' object has no attribute 'hostbuf_allocate'`
#     in `register_load_device` (comfy/model_patcher.py:1784) -> stubbed below;
#   * the vbar-backed device load itself raises
#     `AttributeError: ... 'get_devctx'` in comfy_aimdo/control.py:210 and
#     CANNOT be stubbed, because it is the behaviour under test.
# Therefore the dynamic patcher's weights are put on the device through the
# core-built NON-DYNAMIC DELEGATE, which shares the same handler object and is
# a real, fully-executed load. The dynamic route's actual VRAM behaviour is
# NOT covered here and is not claimed to be.
# ====================================================================


class _StubHostBuffer:
    """Allocation-free stand-in for `comfy_aimdo.host_buffer.HostBuffer`.

    Only the ALLOCATOR is replaced. `register_load_device`
    (comfy/model_patcher.py:1780-1787) still runs and still populates
    `model.dynamic_pins` in full -- which is precisely what
    `ModelPatcherDynamic.__del__` (:1800-1801) -> `unpin_all_weights` (:1827)
    -> `partially_unload_ram` dereferences on teardown. A test double that
    skipped `__init__` would leave those pins unestablished and raise from
    `__del__`, which is the `PytestUnraisableExceptionWarning` this suite must
    not grow.
    """

    def __init__(self, prewarm=0, max_mmap=0, mark_cold=False):
        self.prewarm = prewarm
        self.max_mmap = max_mmap
        self.mark_cold = mark_cold


@pytest.fixture
def dynamic_runtime(monkeypatch):
    """A constructible dynamic patcher: aimdo's allocator stubbed, alias rebound."""
    monkeypatch.setattr(comfy_aimdo.host_buffer, "HostBuffer", _StubHostBuffer)
    monkeypatch.setattr(comfy.model_patcher, "CoreModelPatcher", DYNAMIC)
    return DYNAMIC


@pytest.fixture
def registry_sandbox():
    """Isolate every module-level cache these tests write to.

    There is no autouse cache-clearing fixture in conftest.py, so a test that
    leaves a patcher or a bundle in `VIBEVOICE_ASR_PATCHER_CACHE` /
    `_BUNDLE_REGISTRY` / `_ACTIVE_KEYS` would poison every later test.
    """
    VIBEVOICE_ASR_PATCHER_CACHE.clear()
    clear_bundle_registry()
    clear_active_keys()
    yield VIBEVOICE_ASR_PATCHER_CACHE
    VIBEVOICE_ASR_PATCHER_CACHE.clear()
    clear_bundle_registry()
    clear_active_keys()


def _external_handler(name: str, weight_family: str) -> ExternalVibeVoiceASRModelHandler:
    """A synthetic external handler -- two tiny Linear layers, no checkpoint."""
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    return ExternalVibeVoiceASRModelHandler(
        model, None, name, {"weight_family": weight_family}
    )


def _build(patcher_cls, handler, attention_mode="sdpa"):
    return patcher_cls(
        handler,
        torch.device("cuda"),
        torch.device("cpu"),
        size=handler.size,
        attention_mode=attention_mode,
        dtype=None,
    )


def _put_weights_on_device(patcher):
    """Load a (possibly dynamic) patcher's handler through a real load path.

    Prefers the dynamic patcher's own `load_models_gpu` route; falls back to
    the core-built non-dynamic delegate, which wraps the SAME handler object,
    when aimdo's native library is unbound (always, under pytest -- see the
    ENVIRONMENT LIMIT note above).
    """
    if patcher.is_dynamic() and comfy_aimdo.control.lib is None:
        load_to_device(patcher.clone(disable_dynamic=True))
    else:
        load_to_device(patcher)
    return patcher.loaded_size()


class TestDynamicAndLegacyCoexistence:
    """T9.1 - a dynamic and a legacy patcher live in one session, in turn."""

    def test_dynamic_and_legacy_coexist_in_one_session(self, dynamic_runtime, registry_sandbox):
        """Dense (dynamic) loads, is evicted, then GGUF (legacy) loads and wins.

        The point is not that both classes exist -- T6 covers that. It is that
        the *registry bookkeeping* is indifferent to which class served the
        load: eviction keys off the cache key, never the patcher class, and the
        dense entry is genuinely gone (not merely overwritten) at the end.
        """
        # -- pass 1: dense -> dynamic --------------------------------------
        dense_key = identity_for_external(
            "/models/VibeVoice-1.5B.safetensors", "VibeVoice-1.5B", "sdpa",
            prefix="asr_external",
        )
        dense_handler = _external_handler("dense_1p5b", "dense")
        dense_cls = select_patcher_class(
            "dense", torch.device("cuda"), legacy_cls=VibeVoiceASRPatcher
        )
        assert issubclass(dense_cls, DYNAMIC), "dense must select the dynamic class here"
        dense_patcher = _build(dense_cls, dense_handler)
        assert dense_patcher.is_dynamic() is True

        registry_sandbox[dense_key] = dense_patcher
        register_model_bundle(dense_key, {"weight_family": "dense", "dynamic_vram_route": True})
        evict_if_changed(FAMILY_ASR, dense_key, (registry_sandbox,))

        dense_loaded = _put_weights_on_device(dense_patcher)
        assert dense_loaded > 0, "the dense load must actually put weights on the device"
        assert dense_handler.model_loaded_weight_memory > 0
        assert list(registry_sandbox) == [dense_key]

        # -- the active model changes -> the dense patcher is evicted -------
        gguf_key = identity_for_external(
            "/models/VibeVoice-ASR-Q4_K_M.gguf", "VibeVoice-ASR", "sdpa",
            prefix="asr_external",
        )
        assert gguf_key != dense_key
        evicted = evict_if_changed(FAMILY_ASR, gguf_key, (registry_sandbox,))
        assert evicted == [dense_key]
        assert dense_key not in registry_sandbox, "the dense patcher must be gone, not shadowed"

        # -- pass 2: gguf -> legacy ----------------------------------------
        gguf_handler = _external_handler("asr_gguf", "gguf_block")
        # 2026-09-30: the legacy arm is produced the way CORE produces it — a
        # CPU load device, which ModelPatcherDynamic.__new__ reroutes to the
        # plain ModelPatcher — instead of by the (removed) family whitelist.
        gguf_cls = select_patcher_class(
            "gguf_block", torch.device("cpu"), legacy_cls=VibeVoiceASRPatcher
        )
        assert gguf_cls is VibeVoiceASRPatcher, "a CPU load device selects the legacy class"
        gguf_patcher = _build(gguf_cls, gguf_handler)
        assert gguf_patcher.is_dynamic() is False

        registry_sandbox[gguf_key] = gguf_patcher
        register_model_bundle(gguf_key, {"weight_family": "gguf_block", "dynamic_vram_route": False})
        evict_if_changed(FAMILY_ASR, gguf_key, (registry_sandbox,))

        gguf_loaded = _put_weights_on_device(gguf_patcher)
        assert gguf_loaded > 0
        assert gguf_patcher.is_dynamic() is False
        assert gguf_patcher.loaded_size() == gguf_handler.model_loaded_weight_memory, (
            "the legacy class's loaded_size() IS model_loaded_weight_memory "
            "(comfy/model_patcher.py:412-413)"
        )

        # -- exactly the live one survives ---------------------------------
        assert list(registry_sandbox) == [gguf_key]
        assert registry_sandbox[gguf_key] is gguf_patcher


class TestNonDynamicDelegate:
    """T9.2 - the batch-1 `cached_patcher_init` rebuild path really loads."""

    def test_non_dynamic_delegate_loads(self, dynamic_runtime, registry_sandbox):
        """`clone(disable_dynamic=True)` is a REAL, LOADABLE, non-dynamic patcher.

        This is the batch-1 requirement at comfy/model_patcher.py:438-441: core
        calls `cached_patcher_init[0](*cached_patcher_init[1],
        disable_dynamic=True)` to get a pristine patcher, and raises RuntimeError
        outright when `cached_patcher_init` is unset. A delegate that merely
        *constructed* would not prove the factory works -- so it is loaded.
        """
        handler = _external_handler("dense_1p5b", "dense")
        patcher = _build(select_patcher_class(
            "dense", torch.device("cuda"), legacy_cls=VibeVoiceASRPatcher), handler)
        assert patcher.is_dynamic() is True
        assert patcher.cached_patcher_init is not None, (
            "core raises RuntimeError without it (comfy/model_patcher.py:438-441)"
        )

        delegate = patcher.clone(disable_dynamic=True)

        assert delegate.is_dynamic() is False, (
            "the delegate must speak the legacy protocol -- load_to_device "
            "dispatches on is_dynamic() (modules/patcher.py:570-575)"
        )
        loaded = _put_weights_on_device(delegate)
        assert loaded > 0, "the delegate must actually load, not just construct"
        # Same handler object -> the load is visible from the dynamic patcher too.
        assert handler.model_loaded_weight_memory > 0
        assert patcher.loaded_size() == handler.model_loaded_weight_memory
        # The rebuild produced a genuine VibeVoice patcher, not a bare core one.
        assert isinstance(delegate, comfy.model_patcher.ModelPatcher)
        assert not isinstance(delegate, DYNAMIC)


class TestRegistryKeyStability:
    """T9.3 - the cache key must not encode the patcher route."""

    def test_registry_key_stability_across_patcher_class(
        self, dynamic_runtime, registry_sandbox, tmp_path
    ):
        """One checkpoint, one key -- whichever class actually served it.

        The invariant is CROSS-CLASS, so the test must compare two different
        classes. Building a real patcher of each class from the same
        checkpoint identity (dense on CUDA -> dynamic, dense on CPU ->
        legacy, both through the real `select_patcher_class`) and keying each
        the way `asr_generation.py:342` does -- from the bundle's recorded
        build fields -- is what would fail if a future change let the route
        leak into the key. Calling `identity_for_external` twice with
        identical arguments would only prove determinism, which is already
        covered by `test_key_degrades_deterministically_for_unreadable_paths`.
        """
        weight_path = tmp_path / "VibeVoice-1.5B.safetensors"
        weight_path.write_bytes(b"x" * 128)

        def key_from_bundle(bundle):
            """Key a load the way modules/asr_generation.py:342 does."""
            return identity_for_external(
                bundle["source_path"],
                bundle["config_name"],
                bundle["attention_mode"],
                use_llm_4bit=bundle["use_llm_4bit"],
                dtype_str=bundle["dtype_str"],
                prefix="asr_external",
            )

        def bundle_for(load_device):
            handler = _external_handler("dense_1p5b", "dense")
            cls = select_patcher_class(
                "dense", load_device, legacy_cls=VibeVoiceASRPatcher
            )
            patcher = _build(cls, handler, attention_mode="sdpa")
            bundle = {
                "source_path": str(weight_path),
                "config_name": "VibeVoice-1.5B",
                "attention_mode": "sdpa",
                "use_llm_4bit": False,
                "dtype_str": "bf16",
                "weight_family": "dense",
                "dynamic_vram_route": patcher.is_dynamic(),
            }
            return patcher, cls, bundle

        cuda_device = torch.device("cuda")
        dynamic_patcher, dynamic_cls, dynamic_bundle = bundle_for(cuda_device)
        legacy_patcher, legacy_cls, legacy_bundle = bundle_for(torch.device("cpu"))

        # The two sides must genuinely be different classes, or the comparison
        # below would be the tautology this test used to assert.
        assert dynamic_patcher.is_dynamic() is True
        assert legacy_patcher.is_dynamic() is False
        assert type(dynamic_patcher) is not type(legacy_patcher), (
            "the cross-class comparison needs two distinct patcher classes"
        )
        assert dynamic_bundle["dynamic_vram_route"] is True
        assert legacy_bundle["dynamic_vram_route"] is False

        dense_key = key_from_bundle(dynamic_bundle)
        legacy_key = key_from_bundle(legacy_bundle)
        assert dense_key == legacy_key, (
            "the key must be stable across the patcher class, otherwise "
            "switching routes silently rebuilds and double-loads the model"
        )

        # Both patchers are registrable under that ONE key, so flipping the
        # route is a no-op for eviction: nothing is dropped, and the live
        # entry survives. This is the operational point of the invariant --
        # a route flip must not silently rebuild and double-load the model.
        registry_sandbox[dense_key] = dynamic_patcher
        register_model_bundle(dense_key, dynamic_bundle)
        evicted = evict_if_changed(FAMILY_ASR, legacy_key, (registry_sandbox,))
        assert evicted == [], (
            "same checkpoint + same identity must not evict across a route "
            "flip; a differing key here is the regression this guards"
        )
        assert list(registry_sandbox) == [dense_key]
        assert registry_sandbox[dense_key] is dynamic_patcher

        # And the same identity through the REAL selector yields the same key,
        # whichever class it picks.
        assert select_patcher_class("dense", torch.device("cuda"),
                                    legacy_cls=VibeVoiceASRPatcher) is not VibeVoiceASRPatcher
        # 2026-09-30: the family no longer decides — the CPU device does
        # (core reroutes ModelPatcherDynamic.__new__ to the plain ModelPatcher).
        assert select_patcher_class("gguf_block", torch.device("cpu"),
                                    legacy_cls=VibeVoiceASRPatcher) is VibeVoiceASRPatcher

    def test_key_degrades_deterministically_for_unreadable_paths(self):
        """A missing file yields a stable placeholder key, not a raise."""
        missing = "/models/does-not-exist.safetensors"
        first = identity_for_external(missing, "VibeVoice-1.5B", "sdpa", prefix="asr_external")
        second = identity_for_external(missing, "VibeVoice-1.5B", "sdpa", prefix="asr_external")
        assert first == second
        assert "does-not-exist.safetensors" in first


class TestDeepcloneMultigpu:
    """T9.4 - the second core entry point that hard-requires cached_patcher_init."""

    def test_deepclone_multigpu_path_raises_no_loader_error(self, dynamic_runtime, registry_sandbox):
        """`deepclone_multigpu` must get past its RuntimeError guard.

        core (comfy/model_patcher.py:509-516) raises RuntimeError when
        `cached_patcher_init is None` with the message "...does not support
        multigpu (cached_patcher_init is not initialized)... or have the custom
        loader register a cached_patcher_init factory". Reaching the branch is
        asserted directly and by message, so a pass cannot come from silently
        missing the code path.
        """
        handler = _external_handler("dense_1p5b", "dense")
        patcher = _build(select_patcher_class(
            "dense", torch.device("cuda"), legacy_cls=VibeVoiceASRPatcher), handler)
        assert patcher.cached_patcher_init is not None

        # The guard's own failure mode, pinned so a regression is legible.
        patcher.cached_patcher_init = None
        with pytest.raises(RuntimeError, match="cached_patcher_init"):
            patcher.deepclone_multigpu()
        # Restore, then drive the real branch.
        fn, fn_args = _cached_init_of(patcher, handler)

        patcher.cached_patcher_init = (fn, fn_args)
        try:
            clone = patcher.deepclone_multigpu()
        except RuntimeError as e:
            pytest.fail(
                "deepclone_multigpu reached the cached_patcher_init guard with a "
                f"factory registered -- that is the loader error this test exists "
                f"to prevent: {e}"
            )
        except Exception as e:  # noqa: BLE001 - any other failure is still a failure
            pytest.fail(f"deepclone_multigpu raised {type(e).__name__}: {e}")
        # A pass must mean the branch RAN, not that it returned quietly: the
        # clone is built from the pristine model the factory produced
        # (comfy/model_patcher.py:522-527), so it is a real patcher.
        assert isinstance(clone, comfy.model_patcher.ModelPatcher)
        assert clone is not patcher


def _cached_init_of(patcher, handler):
    """The (fn, args) pair a dynamic patcher registers, rebuilt from scratch.

    Read straight off the constructor rather than captured before the None
    assignment above, so the test uses the real factory the real class built.
    """
    cls = select_patcher_class("dense", torch.device("cuda"), legacy_cls=VibeVoiceASRPatcher)
    fresh = _build(cls, _external_handler(handler.model_pack_name, "dense"))
    return fresh.cached_patcher_init


# ====================================================================
# b1 reconcile pass — foundations coverage.
#
# The classes above cover the SELECTOR and the core-integration contracts.
# What follows covers the per-instance foundations the minted class must get
# right against core's fixed signatures: the resolver's call-time lookup, the
# `__init__` shape core reconstructs positionally, the `is_loaded` contract
# core does not define, the `load_weights` guard, the `warm=`/`destroy=`
# adapter, and `load_to_device`'s dispatch contract.
# ====================================================================

import comfy.model_management as model_management  # noqa: E402


class _PoisonHandler:
    """Handler whose inner model raises if anything enumerates its parameters.

    `is_loaded` must not walk the tree: under vbar the weight functions own
    residency, and a `parameters()` probe is both a per-call tree walk and
    meaningless for a never-materialised module.
    """

    def __init__(self, size=1024):
        self.cache_key = "poison"
        self.model_pack_name = "Poison"
        self.size = size
        self.processor = None
        self.model_loaded_weight_memory = 0
        self.parameters_were_read = False
        self.model = self

    def parameters(self):
        self.parameters_were_read = True
        raise AssertionError("is_loaded must not enumerate parameters")

    def load_model(self, device, attention_mode="sdpa"):
        raise AssertionError("not used by these tests")


class TestResolverIsCallTimeOnly:
    """T1: the alias is read on every call, never captured or memoised."""

    def test_rebind_between_two_calls_is_visible_to_the_second(self, monkeypatch):
        """The whole reason the resolver exists (main.py:300 rebinds post-import).

        A module-level `from ... import CoreModelPatcher`, or any memo, would
        answer with the pre-rebind class for the second call.
        """
        legacy = comfy.model_patcher.ModelPatcher
        monkeypatch.setattr(comfy.model_patcher, "CoreModelPatcher", legacy)
        first = resolve_core_patcher_class()

        monkeypatch.setattr(comfy.model_patcher, "CoreModelPatcher", DYNAMIC)
        second = resolve_core_patcher_class()

        assert first is legacy
        assert second is DYNAMIC
        assert first is not second, (
            "resolve_core_patcher_class() cached its result; a stale class "
            "silently opts the node out of DynamicVRAM for the whole session"
        )

    def test_missing_alias_falls_back_to_the_legacy_class(self, monkeypatch):
        """A core that drops the alias must not break the resolver."""
        monkeypatch.delattr(comfy.model_patcher, "CoreModelPatcher", raising=False)
        assert resolve_core_patcher_class() is comfy.model_patcher.ModelPatcher


class TestMintedClassFoundations:
    """T2/T3: the per-instance contract against core's fixed signatures."""

    @staticmethod
    def _dynamic_cls():
        return select_patcher_class(
            "dense", torch.device("cuda"), legacy_cls=VibeVoiceASRPatcher
        )

    def test_init_signature_matches_legacy_patcher(self, dynamic_runtime):
        """Byte parity with the legacy `__init__`.

        core's ModelPatcher.clone() reconstructs the class POSITIONALLY,
        (model, load_device, offload_device, size, ...), at
        comfy/model_patcher.py:446. A parameter declared before *args would
        bind load_device to attention_mode and then die in
        ModelPatcher.__init__ with "missing 1 required positional argument:
        'offload_device'". The two signatures must therefore never drift.
        """
        cls = self._dynamic_cls()
        assert inspect.signature(cls.__init__) == inspect.signature(
            VibeVoicePatcher.__init__
        )
        spec = inspect.getfullargspec(cls.__init__)
        assert spec.args == ["self", "model"]
        assert spec.varargs == "args"
        assert spec.kwonlyargs == ["attention_mode", "dtype"]

    def test_core_positional_reconstruction_does_not_raise(self, dynamic_runtime):
        """The exact shape core's clone() builds, driven end to end.

        This is the clone-eviction crash: it surfaces from
        comfy/model_management.py, not from clone() itself, so it is a hard
        crash on a stock install rather than a test-only failure.
        """
        handler = _external_handler("dense_1p5b", "dense")
        cls = self._dynamic_cls()
        rebuilt = cls(handler, torch.device("cuda"), torch.device("cpu"), handler.size)
        assert rebuilt.offload_device == torch.device("cpu")
        assert rebuilt.attention_mode == "eager", "the keyword-only default must apply"
        assert rebuilt.target_dtype is None

    def test_accepts_our_own_keywords_at_construction(self, dynamic_runtime):
        """Regression lock on the fixed-signature `__new__`.

        ModelPatcherDynamic.__new__ (comfy/model_patcher.py:1754-1757) takes
        exactly six named parameters and no **kwargs, and Python dispatches
        through `__new__` FIRST — so attention_mode/dtype (which every call
        site passes) raise TypeError there unless they are filtered.
        """
        handler = _external_handler("dense_1p5b", "dense")
        patcher = self._dynamic_cls()(
            handler,
            torch.device("cuda"),
            torch.device("cpu"),
            size=handler.size,
            attention_mode="sage",
            dtype=torch.float16,
        )
        assert patcher.attention_mode == "sage"
        assert patcher.target_dtype is torch.float16

    def test_cached_patcher_init_is_a_two_tuple(self, dynamic_runtime):
        """core indexes cached_patcher_init[0] and [1] (:439, :521)."""
        patcher = _build(
            self._dynamic_cls(), _external_handler("dense_1p5b", "dense")
        )
        cpi = patcher.cached_patcher_init
        assert isinstance(cpi, tuple) and len(cpi) == 2
        fn, args = cpi
        assert callable(fn)
        assert isinstance(args, tuple) and len(args) == 1
        assert "disable_dynamic" in inspect.signature(fn).parameters, (
            "core calls this factory as fn(*args, disable_dynamic=True) "
            "(comfy/model_patcher.py:439); the keyword name is load-bearing"
        )

    def test_is_loaded_does_not_touch_parameters(self, dynamic_runtime):
        """Core defines no is_loaded on either patcher, so ours must stand alone."""
        handler = _PoisonHandler()
        patcher = self._dynamic_cls()(
            handler, torch.device("cuda"), torch.device("cpu"), size=handler.size
        )
        assert patcher.is_loaded is False
        assert handler.parameters_were_read is False, (
            "is_loaded must answer from references + loaded_size(), not from a "
            "walk of the module tree"
        )

    def test_is_loaded_false_when_handler_has_no_model(self, dynamic_runtime):
        handler = _PoisonHandler()
        handler.model = None
        patcher = self._dynamic_cls()(
            handler, torch.device("cuda"), torch.device("cpu"), size=handler.size
        )
        assert patcher.is_loaded is False

    def test_core_defines_no_is_loaded(self):
        """The premise: `super().is_loaded` would raise AttributeError."""
        assert not hasattr(comfy.model_patcher.ModelPatcher, "is_loaded")
        assert not hasattr(DYNAMIC, "is_loaded")


class TestLoadWeightsGuard:
    """core's dynamic override (:2132-2137) asserts `not load_weights`.

    Two failure modes to cover: an omitted argument (core DEFAULTS it to True)
    and an explicit True. The first is a bare AssertionError from deep inside
    core; the second must be actionable for whoever wired the graph.
    """

    @staticmethod
    def _patcher(dynamic_runtime):
        return _build(
            select_patcher_class("dense", torch.device("cuda"), legacy_cls=VibeVoiceASRPatcher),
            _external_handler("dense_1p5b", "dense"),
        )

    def test_default_never_passes_load_weights_true(self, dynamic_runtime):
        patcher = self._patcher(dynamic_runtime)
        with patch("comfy.model_patcher.ModelPatcherDynamic.patch_model") as base:
            patcher.patch_model()
        assert base.call_args.kwargs["load_weights"] is False, (
            "core's dynamic patch_model defaults load_weights=True and then "
            "asserts `not load_weights`, so omitting it is itself a crash"
        )

    def test_explicit_true_raises_actionable_value_error(self, dynamic_runtime):
        patcher = self._patcher(dynamic_runtime)
        with patch("comfy.model_patcher.ModelPatcherDynamic.patch_model"):
            with pytest.raises(ValueError) as exc:
                patcher.patch_model(load_weights=True)
        message = str(exc.value)
        assert "load_models_gpu" in message, (
            "the message must name the API to use instead, not surface core's "
            "bare AssertionError"
        )

    def test_forwards_own_load_device_not_the_callers(self, dynamic_runtime):
        """ModelPatcherDynamic.load asserts device_to == self.load_device (:1870)."""
        patcher = self._patcher(dynamic_runtime)
        with patch("comfy.model_patcher.ModelPatcherDynamic.patch_model") as base:
            patcher.patch_model(device_to=torch.device("cpu"))
        assert base.call_args.kwargs["device_to"] == patcher.load_device


class TestUnpatchAdapter:
    """core's dynamic unpatch_model is `(self, device_to, unpatch_weights)`.

    The pack's call sites pass warm=/destroy=, which core does not accept, so
    the adapter must consume them rather than forward them.
    """

    @staticmethod
    def _patcher(dynamic_runtime):
        return _build(
            select_patcher_class("dense", torch.device("cuda"), legacy_cls=VibeVoiceASRPatcher),
            _external_handler("dense_1p5b", "dense"),
        )

    def test_accepts_warm_and_destroy_kwargs(self, dynamic_runtime):
        patcher = self._patcher(dynamic_runtime)
        with patch("comfy.model_patcher.ModelPatcherDynamic.unpatch_model"):
            patcher.unpatch_model(warm=True)
            patcher.unpatch_model(destroy=True)

    def test_forwards_only_device_and_weights_to_core(self, dynamic_runtime):
        """Neither `warm` nor `destroy` may reach core.

        core's dynamic signature is exactly
        `(self, device_to=None, unpatch_weights=True)`; the adapter consumes
        the pack's two extra flags. The assertion is on the resolved pair
        rather than on call style, because the adapter forwards them
        positionally -- what matters is that exactly these two cross.
        """
        patcher = self._patcher(dynamic_runtime)
        with patch("comfy.model_patcher.ModelPatcherDynamic.unpatch_model") as base:
            patcher.unpatch_model(device_to=torch.device("cpu"), unpatch_weights=False)
        args, kwargs = base.call_args
        assert set(kwargs) <= {"device_to", "unpatch_weights"}
        assert len(args) + len(kwargs) == 2, (
            f"only device_to and unpatch_weights may reach core, got "
            f"args={args} kwargs={kwargs}"
        )
        forwarded = dict(zip(("device_to", "unpatch_weights"), args)) or kwargs
        assert forwarded.get("device_to") == torch.device("cpu")
        assert forwarded.get("unpatch_weights") is False

    def test_warm_releases_vbar_pins(self, dynamic_runtime):
        """There is no intermediate_device() notion under vbar.

        `unpin_all_weights()` (comfy/model_patcher.py:1827-1828 ->
        partially_unload_ram) is the warm equivalent; the legacy .to() path is
        meaningless once a vbar owns placement.
        """
        patcher = self._patcher(dynamic_runtime)
        released = []
        patcher.unpin_all_weights = lambda: released.append(1)
        with patch("comfy.model_patcher.ModelPatcherDynamic.unpatch_model"):
            patcher.unpatch_model(warm=True)
        assert released == [1]

    def test_destroy_evicts_the_registry_entry(self, dynamic_runtime):
        """The destructive path must unregister and drop the cache entry."""
        from ComfyUI_VibeVoice.modules.asr_loader import LOADED_ASR_MODELS_CACHE
        from ComfyUI_VibeVoice.modules import model_registry

        patcher = self._patcher(dynamic_runtime)
        cache_key = patcher.cache_key
        LOADED_ASR_MODELS_CACHE[cache_key] = ("model", "processor")
        try:
            with patch.object(model_registry, "unregister_from_comfy") as unreg, \
                 patch("comfy.model_patcher.ModelPatcherDynamic.unpatch_model"):
                patcher.unpatch_model(destroy=True)
            assert unreg.call_args.args == (patcher,), (
                "unregister BEFORE nulling the refs, so nothing downstream "
                "touches a destroyed model"
            )
            assert cache_key not in LOADED_ASR_MODELS_CACHE
            assert patcher.model.model is None
            assert patcher.model.processor is None
        finally:
            LOADED_ASR_MODELS_CACHE.pop(cache_key, None)


class TestLoadToDeviceContract:
    """T5: dispatch on the METHOD, one patcher per call, never force anything.

    `load_to_device` is the seam every construction site in generation.py /
    asr_generation.py routes through, so its contract is checked directly here
    rather than only through the integration paths above.
    """

    class _FakePatcher:
        def __init__(self, dynamic):
            self._dynamic = dynamic

        def is_dynamic(self):
            return self._dynamic

    def test_routes_dynamic_to_load_models_gpu(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            model_management, "load_models_gpu",
            lambda models, **kw: seen.append((models, kw)),
        )
        monkeypatch.setattr(
            model_management, "load_model_gpu", lambda m: seen.append(("legacy", m))
        )
        patcher = self._FakePatcher(True)
        load_to_device(patcher)
        assert seen == [([patcher], {"memory_required": 0})]

    def test_routes_legacy_to_load_model_gpu(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            model_management, "load_models_gpu",
            lambda models, **kw: seen.append(("dynamic", models)),
        )
        monkeypatch.setattr(
            model_management, "load_model_gpu", lambda m: seen.append(("legacy", m))
        )
        patcher = self._FakePatcher(False)
        load_to_device(patcher)
        assert seen == [("legacy", patcher)]

    def test_never_batches(self, monkeypatch):
        """load_models_gpu is all-or-nothing for dynamic (model_management.py:962-965).

        Batching a dynamic patcher with anything else silently clears
        free_for_dynamic, so every call must carry exactly one patcher.
        """
        calls = []
        monkeypatch.setattr(
            model_management, "load_models_gpu",
            lambda models, **kw: calls.append(models),
        )
        load_to_device(self._FakePatcher(True))
        load_to_device(self._FakePatcher(True))
        assert len(calls) == 2
        assert all(len(c) == 1 for c in calls)

    def test_never_passes_force_flags(self, monkeypatch):
        """force_patch_weights / force_full_load both assert (model_patcher.py:1864/:1868)."""
        seen = []
        monkeypatch.setattr(
            model_management, "load_models_gpu",
            lambda models, **kw: seen.append(kw),
        )
        load_to_device(self._FakePatcher(True), memory_required=256)
        assert seen[0].get("force_patch_weights") is None
        assert seen[0].get("force_full_load") is None
        assert seen[0]["memory_required"] == 256

    def test_dispatches_on_the_method_not_isinstance(self, monkeypatch):
        """ModelPatcherDynamic.__new__ reroutes a CPU load_device to a legacy
        ModelPatcher, so a dynamic patcher is not reliably an instance of the
        dynamic subclass; the predicate core uses is `is_dynamic()`."""
        seen = []
        monkeypatch.setattr(
            model_management, "load_model_gpu", lambda m: seen.append(("legacy", m))
        )
        monkeypatch.setattr(
            model_management, "load_models_gpu",
            lambda models, **kw: seen.append(("dynamic", models)),
        )
        load_to_device(self._FakePatcher(False))
        assert seen[-1][0] == "legacy", (
            "an object that does not positively declare itself dynamic must "
            "stay on the legacy call shape"
        )


class TestPreserveFileViews:
    """The clone gate in ``_load_state_dict_into_model_from_memory``.

    THE fix for the host-RAM spike: under aimdo, ``comfy.utils.load_torch_file``
    hands back zero-copy views into a ``ModelMMAP`` mapping, each storage
    tagged ``_comfy_tensor_file_slice`` — the same views the native Load
    Diffusion Model node keeps for the model's whole life and reads
    disk->VRAM from at every forward (``read_tensor_file_slice_into``).
    Cloning them before assign materialised the entire checkpoint as private
    host RAM (measured 1.26x / 1.74x of file size). On the dynamic route the
    views must survive; on the legacy route the clone stays (ghost-RAM +
    "Pin error." flood, user-confirmed fix).
    """

    @staticmethod
    def _model_and_dict():
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return model, sd

    @staticmethod
    def _tag(tensor):
        """Mimic comfy.utils.load_safetensors' file-slice tag on the storage.

        Returns the tag tuple so tests can compare attribute values, not
        storage identity.
        """
        tag = ("file", "lock", 0, tensor.numel() * 4)
        tensor.untyped_storage()._comfy_tensor_file_slice = tag
        return tag

    def test_dynamic_route_preserves_tagged_views(self):
        """preserve_file_views=True: the view (and its tag) reach the parameter."""
        model, sd = self._model_and_dict()
        tag = self._tag(sd["0.weight"])
        view_ptr = sd["0.weight"].data_ptr()
        external_loader._load_state_dict_into_model_from_memory(
            model, sd, preserve_file_views=True,
        )
        param_storage = model[0].weight.untyped_storage()
        assert getattr(param_storage, "_comfy_tensor_file_slice", None) == tag
        # Same memory, not a copy: the whole point of the dynamic route.
        assert param_storage.data_ptr() == view_ptr

    def test_legacy_route_clones_tagged_views(self):
        """preserve_file_views=False: byte-identical legacy behaviour."""
        model, sd = self._model_and_dict()
        self._tag(sd["0.weight"])
        view_ptr = sd["0.weight"].data_ptr()
        external_loader._load_state_dict_into_model_from_memory(
            model, sd, preserve_file_views=False,
        )
        param_storage = model[0].weight.untyped_storage()
        # Clone severs the mapping: new storage, no tag.
        assert getattr(param_storage, "_comfy_tensor_file_slice", None) is None
        assert param_storage.data_ptr() != view_ptr

    def test_dynamic_route_never_clones_untagged_tensors(self):
        """On the dynamic route even owned tensors are assigned as-is: the vbar
        read path handles plain CPU storages via cast_to_gathered's fallback."""
        model, sd = self._model_and_dict()
        before = sd["0.weight"].data_ptr()
        external_loader._load_state_dict_into_model_from_memory(
            model, sd, preserve_file_views=True,
        )
        assert model[0].weight.untyped_storage().data_ptr() == before

    def test_tag_survives_assign_storage_sharing(self):
        """assign=True wraps the SAME storage in nn.Parameter — the tag core
        reads at load time must survive that wrap."""
        t = torch.zeros(4, 4)
        tag = self._tag(t)
        p = torch.nn.Parameter(t, requires_grad=False)
        assert getattr(p.untyped_storage(), "_comfy_tensor_file_slice", None) == tag
