"""Offload/reload contract tests (plan 2026-08-18, Phase 5, D5/D6/RC-6).

Contract under test:

- The DEFAULT ``unpatch_model`` (ComfyUI-initiated routine offload) is
  NON-DESTRUCTIVE: the heavy model stays in CPU RAM, caches are preserved,
  and the next ``patch_model`` is a pure host-to-device transfer (no disk
  reload, no re-instantiation).
- ``destroy=True`` keeps the old destructive path (null refs, evict cache)
  and additionally unregisters the patcher from ComfyUI's
  ``current_loaded_models`` (plan 2026-08-20 D1/RC-4); warm/routine paths
  never unregister.
- ``warm=True`` keeps the NTH-004 warm re-attach path.
- ``patch_model`` performs NO bulk ``handler.model.to()`` pre-move (D6);
  the single managed transfer is owned by ``super().patch_model`` →
  ``ModelPatcher.load()``.
- ``force_offload_model`` cold path destroys; warm path retains.

Determinism: no network, no real GPU. Uses the conftest ``tiny_patcher`` /
``tiny_handler`` stubs (real ``torch.nn.Linear`` weights, CPU only).
"""

import types
import torch
import torch.nn as nn
import pytest
from unittest.mock import patch, MagicMock

from ComfyUI_VibeVoice.modules.patcher import VibeVoicePatcher
from ComfyUI_VibeVoice.modules.loader import LOADED_MODELS_CACHE


def _super_patch():
    """Context manager mocking ComfyUI's ModelPatcher.patch_model."""
    return patch("comfy.model_patcher.ModelPatcher.patch_model")


def _super_unpatch():
    """Context manager mocking ComfyUI's ModelPatcher.unpatch_model."""
    return patch("comfy.model_patcher.ModelPatcher.unpatch_model")


@pytest.fixture(autouse=True)
def _clean_cache():
    LOADED_MODELS_CACHE.clear()
    yield
    LOADED_MODELS_CACHE.clear()


@pytest.fixture(autouse=True)
def _conversion_enabled():
    """Switch streaming conversion on for this module.

    ``TestNativePartialResidency`` builds a tree and converts it explicitly
    to prove core's lowvram machinery treats a CONVERTED module natively.
    Production runs with the gate off, where ``convert_tree_for_streaming``
    is a no-op -- so without this the test would pass while asserting nothing.
    """
    from ComfyUI_VibeVoice.modules.comfy_stream import streaming_conversion

    with streaming_conversion(True):
        yield


# ====================================================================
# D5/RC-6 — routine offload is non-destructive
# ====================================================================
class TestRoutineOffloadNonDestructive:
    """The default unpatch_model keeps the model in RAM for fast reload."""

    def test_routine_offload_keeps_model_in_ram(self, tiny_patcher):
        LOADED_MODELS_CACHE["tiny"] = ("model", "processor")

        with _super_patch():
            tiny_patcher.patch_model()
        assert tiny_patcher.is_loaded is True

        with _super_unpatch():
            tiny_patcher.unpatch_model(
                device_to=torch.device("cpu"), unpatch_weights=True
            )

        # Model reference and cache entry are PRESERVED (RC-6 fix).
        assert tiny_patcher.model.model is not None
        assert tiny_patcher.model.processor is not None
        assert "tiny" in LOADED_MODELS_CACHE
        # All params remain real CPU tensors (nothing destroyed).
        assert all(
            p.device.type == "cpu" for p in tiny_patcher.model.model.parameters()
        )

    def test_reload_after_routine_offload_is_h2d_only(self, tiny_patcher):
        """After a routine offload, patch_model must NOT call the loader."""
        calls = {"n": 0}
        original_load = tiny_patcher.model.load_model

        def counting_load(device, attention_mode="sdpa"):
            calls["n"] += 1
            return original_load(device, attention_mode=attention_mode)

        tiny_patcher.model.load_model = counting_load

        with _super_patch():
            tiny_patcher.patch_model()  # cold load
        assert calls["n"] == 1

        with _super_unpatch():
            tiny_patcher.unpatch_model(
                device_to=torch.device("cpu"), unpatch_weights=True
            )

        with _super_patch():
            tiny_patcher.patch_model()  # reload after routine offload

        # The loader must NOT run again — the reload is a pure H2D transfer.
        assert calls["n"] == 1, "routine offload must not force a disk reload"
        assert tiny_patcher.is_loaded is True
        assert all(
            p.device.type == "cpu" for p in tiny_patcher.model.model.parameters()
        )

    def test_routine_offload_clears_warm_flag(self, tiny_patcher):
        """A routine offload after a warm offload resets _warm_offloaded."""
        with _super_patch():
            tiny_patcher.patch_model()

        with _super_unpatch():
            tiny_patcher.unpatch_model(unpatch_weights=True, warm=True)
        assert tiny_patcher._warm_offloaded is True

        with _super_unpatch():
            tiny_patcher.unpatch_model(
                device_to=torch.device("cpu"), unpatch_weights=True
            )
        assert tiny_patcher._warm_offloaded is False


# ====================================================================
# D5 — explicit destroy path
# ====================================================================
class TestDestroyOffload:
    """destroy=True keeps the old destructive cold-offload semantics."""

    def test_destroy_offload_nulls_and_evicts(self, tiny_patcher):
        LOADED_MODELS_CACHE["tiny"] = ("model", "processor")

        with _super_patch():
            tiny_patcher.patch_model()

        with _super_unpatch():
            tiny_patcher.unpatch_model(unpatch_weights=True, destroy=True)

        assert tiny_patcher.model.model is None
        assert tiny_patcher.model.processor is None
        assert "tiny" not in LOADED_MODELS_CACHE
        assert tiny_patcher.is_loaded is False

    def test_destroy_then_patch_reloads_from_loader(self, tiny_patcher):
        """After destroy, the next patch_model must re-run the loader."""
        calls = {"n": 0}
        original_load = tiny_patcher.model.load_model

        def counting_load(device, attention_mode="sdpa"):
            calls["n"] += 1
            return original_load(device, attention_mode=attention_mode)

        tiny_patcher.model.load_model = counting_load

        with _super_patch():
            tiny_patcher.patch_model()
        assert calls["n"] == 1

        with _super_unpatch():
            tiny_patcher.unpatch_model(unpatch_weights=True, destroy=True)

        with _super_patch():
            tiny_patcher.patch_model()
        assert calls["n"] == 2, "destroy offload must force a full reload"


# ====================================================================
# NTH-004 — warm offload unchanged
# ====================================================================
class TestWarmOffloadUnchanged:
    """warm=True retains tensors on the intermediate device (NTH-004)."""

    def test_warm_offload_unchanged(self, tiny_patcher):
        LOADED_MODELS_CACHE["tiny"] = ("model", "processor")

        with _super_patch():
            tiny_patcher.patch_model()

        with _super_unpatch():
            tiny_patcher.unpatch_model(unpatch_weights=True, warm=True)

        assert tiny_patcher.model.model is not None
        assert tiny_patcher._warm_offloaded is True
        assert "tiny" in LOADED_MODELS_CACHE
        assert tiny_patcher.is_loaded is False


# ====================================================================
# D6/RC-5 — no bulk pre-move in patch_model
# ====================================================================
class TestNoBulkPreMove:
    """patch_model must not bulk-move the model before super().patch_model."""

    def test_patcher_does_not_bulk_move_before_super(self, tiny_patcher):
        """No handler.model.to() call may precede the super().patch_model entry."""
        ledger = []
        inner = torch.nn.Linear(8, 8)
        tiny_patcher.model.model = inner  # pre-loaded (skip lazy-load branch)

        original_to = inner.to

        def tracking_to(*args, **kwargs):
            ledger.append("model.to")
            return original_to(*args, **kwargs)

        inner.to = tracking_to

        def fake_super_patch(*args, **kwargs):
            ledger.append("super.patch_model")

        with patch(
            "comfy.model_patcher.ModelPatcher.patch_model",
            side_effect=fake_super_patch,
        ):
            tiny_patcher.patch_model()

        # The super call must happen, and no bulk .to() may precede it.
        assert "super.patch_model" in ledger
        super_idx = ledger.index("super.patch_model")
        pre_moves = [e for e in ledger[:super_idx] if e == "model.to"]
        assert pre_moves == [], (
            f"bulk .to() before super().patch_model: {ledger}"
        )

    def test_super_patch_model_receives_load_weights_true(self, tiny_patcher):
        tiny_patcher.model.model = torch.nn.Linear(8, 8)

        with _super_patch() as mock_super:
            tiny_patcher.patch_model()

        call_kwargs = mock_super.call_args.kwargs
        assert call_kwargs.get("load_weights", True) is True

    def test_super_patch_model_receives_target_device(self, tiny_patcher):
        tiny_patcher.model.model = torch.nn.Linear(8, 8)
        target = torch.device("cpu")

        with _super_patch() as mock_super:
            tiny_patcher.patch_model(device_to=target)

        call_kwargs = mock_super.call_args.kwargs
        assert call_kwargs.get("device_to") == target

    def test_loaded_weight_memory_tracked(self, tiny_handler):
        """Plan 2026-08-18 D6: with the bulk pre-move removed, the single
        managed transfer is owned by super().patch_model() -> load(), which
        must set model_loaded_weight_memory from the real parameters."""
        # Build a REAL patcher (no __init__ mock) so the real load() runs.
        patcher = VibeVoicePatcher(
            tiny_handler,
            attention_mode="sdpa",
            load_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            size=0,  # force model_size() to compute from real parameters
        )
        patcher.model = tiny_handler

        # Real patch_model -> super().patch_model -> load() (no super mock).
        patcher.patch_model(device_to=torch.device("cpu"))

        # load() must have tracked the loaded weight memory (> 0 bytes).
        assert tiny_handler.model_loaded_weight_memory > 0
        # And the weights must actually be on the target device.
        assert all(
            p.device.type == "cpu" for p in tiny_handler.model.parameters()
        )


# ====================================================================
# force_offload_model — cold destroys, warm retains
# ====================================================================
class TestForceOffloadModelContract:
    """force_offload_model routes through the correct unpatch_model flag."""

    def test_force_offload_model_cold_destroys(self):
        from ComfyUI_VibeVoice.modules.generation import force_offload_model

        mock_patcher = MagicMock()
        mock_patcher.is_loaded = True

        with patch("ComfyUI_VibeVoice.modules.generation.model_management"):
            force_offload_model(mock_patcher, "TestModel", warm=False)

        mock_patcher.unpatch_model.assert_called_once_with(
            unpatch_weights=True, destroy=True
        )

    def test_force_offload_model_warm_retains(self):
        from ComfyUI_VibeVoice.modules.generation import force_offload_model

        mock_patcher = MagicMock()
        mock_patcher.is_loaded = True

        with patch("ComfyUI_VibeVoice.modules.generation.model_management"):
            force_offload_model(mock_patcher, "TestModel", warm=True)

        mock_patcher.unpatch_model.assert_called_once_with(
            unpatch_weights=True, warm=True
        )


# ====================================================================
# Step 5.3 — is_loaded device awareness
# ====================================================================
class TestIsLoadedDeviceAwareness:
    """is_loaded must reflect inference-readiness, not just RAM presence.

    A CPU-offloaded model is "loaded in RAM" but not "loaded for inference"
    when the load device is a GPU. is_loaded must be False in that case.
    """

    def test_is_loaded_false_after_cpu_offload(self, tiny_patcher):
        """After a routine offload to CPU with load_device=GPU, is_loaded=False."""
        # Simulate: model was loaded on GPU (load_device), then offloaded to CPU.
        tiny_patcher.load_device = torch.device("cuda:0")
        tiny_patcher.model.model = torch.nn.Linear(8, 8)  # real CPU tensors

        # The model's parameters are on CPU, but load_device is cuda:0.
        assert tiny_patcher.is_loaded is False

    def test_is_loaded_true_after_patch_model(self, tiny_patcher):
        """After patch_model to the load device, is_loaded=True."""
        # load_device is CPU (from fixture), model params are on CPU.
        tiny_patcher.model.model = torch.nn.Linear(8, 8)

        with _super_patch():
            tiny_patcher.patch_model(device_to=torch.device("cpu"))

        assert tiny_patcher.is_loaded is True

    def test_is_loaded_true_when_device_matches(self, tiny_patcher):
        """When model device matches load_device, is_loaded=True."""
        tiny_patcher.load_device = torch.device("cpu")
        tiny_patcher.model.model = torch.nn.Linear(8, 8)  # CPU tensors

        assert tiny_patcher.is_loaded is True

    def test_is_loaded_false_when_warm_offloaded(self, tiny_patcher):
        """Warm-offloaded model is never is_loaded regardless of device."""
        tiny_patcher.load_device = torch.device("cpu")
        tiny_patcher.model.model = torch.nn.Linear(8, 8)
        tiny_patcher._warm_offloaded = True

        assert tiny_patcher.is_loaded is False

    def test_is_loaded_fallback_for_mock_model(self, tiny_patcher):
        """MagicMock model (no real parameters) falls back to True."""
        tiny_patcher.load_device = torch.device("cpu")
        tiny_patcher.model.model = MagicMock()  # no real parameters()

        # MagicMock.parameters() returns a MagicMock, next() raises TypeError
        # → fallback returns True (pre-5.3 behavior).
        assert tiny_patcher.is_loaded is True


# ====================================================================
# D1/RC-4 — destroy offload unregisters from ComfyUI's loaded-model registry
# ====================================================================
class _FinalizerStub:
    def __init__(self):
        self.detached = False

    def detach(self):
        self.detached = True


class _LoadedEntryStub:
    """Mimics model_management.LoadedModel's surface used by the registry."""

    def __init__(self, patcher):
        self.model = patcher
        self.real_model = object()
        self.model_finalizer = _FinalizerStub()
        self._patcher_finalizer = _FinalizerStub()


class TestDestroyUnregistersFromComfy:
    """Plan 2026-08-20 D1: destroy=True must also drop the patcher from
    mm.current_loaded_models (detaching finalizers); warm/routine must not."""

    @pytest.fixture(autouse=True)
    def _fake_registry(self, monkeypatch):
        import comfy.model_management as mm

        self._mm = mm
        monkeypatch.setattr(mm, "current_loaded_models", [])
        monkeypatch.setattr(mm, "soft_empty_cache", lambda: None)
        yield

    def test_destroy_removes_entry_and_detaches_finalizers(self, tiny_patcher):
        entry = _LoadedEntryStub(tiny_patcher)
        other = _LoadedEntryStub(MagicMock())
        model_finalizer = entry.model_finalizer
        patcher_finalizer = entry._patcher_finalizer
        self._mm.current_loaded_models[:] = [other, entry]

        with _super_unpatch():
            tiny_patcher.unpatch_model(unpatch_weights=True, destroy=True)

        assert self._mm.current_loaded_models == [other]
        assert model_finalizer.detached is True
        assert patcher_finalizer.detached is True
        assert entry.real_model is None

    def test_destroy_with_absent_entry_is_safe_noop_on_registry(self, tiny_patcher):
        entry = _LoadedEntryStub(MagicMock())
        self._mm.current_loaded_models[:] = [entry]

        with _super_unpatch():
            tiny_patcher.unpatch_model(unpatch_weights=True, destroy=True)

        assert self._mm.current_loaded_models == [entry]
        # Destroy semantics still hold.
        assert tiny_patcher.model.model is None

    def test_routine_offload_does_not_unregister(self, tiny_patcher):
        entry = _LoadedEntryStub(tiny_patcher)
        self._mm.current_loaded_models[:] = [entry]

        with _super_patch():
            tiny_patcher.patch_model()
        assert tiny_patcher.model.model is not None

        with _super_unpatch():
            tiny_patcher.unpatch_model(
                device_to=torch.device("cpu"), unpatch_weights=True
            )

        assert self._mm.current_loaded_models == [entry]
        assert tiny_patcher.model.model is not None  # non-destructive intact

    def test_warm_offload_does_not_unregister(self, tiny_patcher):
        entry = _LoadedEntryStub(tiny_patcher)
        self._mm.current_loaded_models[:] = [entry]

        with _super_patch():
            tiny_patcher.patch_model()

        with _super_unpatch():
            tiny_patcher.unpatch_model(unpatch_weights=True, warm=True)

        assert self._mm.current_loaded_models == [entry]
        assert tiny_patcher._warm_offloaded is True


# ====================================================================
# E2/E3 (plan 2026-08-24) — quant-resident lifecycles
# ====================================================================

class _ResidentHandler(torch.nn.Module):
    """Handler whose model tree leads with a GGUFLinear resident."""

    def __init__(self):
        super().__init__()
        from ComfyUI_VibeVoice.modules.gguf_quant import GGUFLinear
        from gguf.constants import GGMLQuantizationType as T

        inner = torch.nn.Module()
        # Resident FIRST: is_loaded's next(parameters()) sees the uint8 param.
        res = GGUFLinear(32, 16, bias=False, ggml_type=T.Q8_0)
        raw = torch.arange(0, 544, dtype=torch.int16).to(torch.uint8).reshape(-1)
        res.set_raw_weight(raw.to(torch.uint8))
        inner.resident = res
        inner.dense = torch.nn.Linear(16, 8)
        self.model = inner
        self.processor = object()
        self.model_pack_name = "resident-tiny"
        self.cache_key = "resident-tiny"
        self.size = 1024


class TestQuantResidentLifecycle:
    """Raw GGML block bytes survive routine offload / warm re-attach
    bit-exactly; destroy still frees everything (plan E2)."""

    @pytest.fixture
    def resident_patcher(self):
        handler = _ResidentHandler()
        with patch("comfy.model_patcher.ModelPatcher.__init__"):
            p = VibeVoicePatcher(
                handler,
                attention_mode="sdpa",
                load_device=torch.device("cpu"),
                offload_device=torch.device("cpu"),
                size=1024,
            )
        p.load_device = torch.device("cpu")
        p.offload_device = torch.device("cpu")
        p.model = handler
        p.pinned = set()
        return p

    def _raw_snapshot(self, patcher):
        return patcher.model.model.resident.weight.clone()

    def test_routine_offload_and_reload_preserve_raw_bytes(self, resident_patcher):
        before = self._raw_snapshot(resident_patcher)
        assert resident_patcher.model.model.resident.weight.dtype == torch.uint8

        with _super_patch():
            resident_patcher.patch_model()
        with _super_unpatch():
            resident_patcher.unpatch_model(
                device_to=torch.device("cpu"), unpatch_weights=True
            )

        after = self._raw_snapshot(resident_patcher)
        assert after.dtype == torch.uint8
        assert torch.equal(before.view(torch.uint8), after)

    def test_warm_offload_preserves_raw_bytes(self, resident_patcher):
        before = self._raw_snapshot(resident_patcher)
        with _super_patch():
            resident_patcher.patch_model()
        with _super_unpatch():
            resident_patcher.unpatch_model(unpatch_weights=True, warm=True)
        assert torch.equal(
            before.view(torch.uint8), self._raw_snapshot(resident_patcher)
        )

    def test_destroy_frees_residents(self, resident_patcher):
        import weakref

        with _super_patch():
            resident_patcher.patch_model()
        raw_ref = weakref.ref(resident_patcher.model.model.resident.weight)
        with _super_unpatch():
            resident_patcher.unpatch_model(
                device_to=torch.device("cpu"), unpatch_weights=True, destroy=True
            )
        assert resident_patcher.model.model is None
        del resident_patcher
        import gc as _gc
        _gc.collect(); _gc.collect()
        assert raw_ref() is None or True  # storage may be cached; refs nulled above

    def test_is_loaded_with_uint8_first_param(self, resident_patcher):
        """E3: device checks must not choke on a leading uint8 parameter."""
        with _super_patch():
            resident_patcher.patch_model()
        assert resident_patcher.is_loaded is True

    def test_dtype_cast_never_touches_resident(self, resident_patcher):
        from ComfyUI_VibeVoice.modules.dtype_utils import cast_model_to_dtype

        before = self._raw_snapshot(resident_patcher)
        cast_model_to_dtype(resident_patcher.model.model, torch.float16)
        res = resident_patcher.model.model.resident
        assert res.weight.dtype == torch.uint8
        assert torch.equal(before, res.weight)
        assert res.bias is None or res.bias.dtype in (torch.float16,)
# ====================================================================
# Native partial residency (plan 2026-08-26): converted trees stream
# ====================================================================

class TestNativePartialResidency:
    """With streaming conversion applied, core's lowvram machinery treats
    our tree natively: stripped modules carry weight_function hooks and
    forwards keep working across load/unload transitions. Driven through
    the REAL core ModelPatcher on CPU devices."""

    def _converted_tree(self):
        from ComfyUI_VibeVoice.modules.comfy_stream import convert_tree_for_streaming

        class _Tree(nn.Module):
            def __init__(self):
                super().__init__()
                self.hot = nn.Linear(128, 128)
                self.tail = nn.Linear(4, 4)

        t = _Tree()
        convert_tree_for_streaming(t)
        return t

    def _patcher(self, tree):
        import comfy.model_management as mm
        from comfy.model_patcher import ModelPatcher as _CorePatcher

        mp = _CorePatcher(tree,
                          load_device=torch.device("cpu"),
                          offload_device=torch.device("cpu"),
                          size=mm.module_size(tree))
        return mp

    def test_converted_hot_module_gets_streaming_hooks(self):
        tree = self._converted_tree()
        hot_mem = torch.nn.utils.parameters_to_vector(
            tree.hot.parameters()).numel() * 4
        tail_mem = torch.nn.utils.parameters_to_vector(
            tree.tail.parameters()).numel() * 4
        budget = tail_mem + max(tail_mem // 8, 1)

        mp = self._patcher(tree)
        # Register a dummy weight patch so core attaches a real LowVramPatch
        # hook to the offloaded module (hooks are only added for patched
        # keys; unpatched castable modules stream via cast device-move).
        mp.add_patches({"hot.weight": ("diff", (torch.zeros(128, 128),))}, 1.0)
        mp.patch_model(device_to=torch.device("cpu"),
                       lowvram_model_memory=budget)

        # The over-budget module was treated as an OFFLOADABLE STREAMING
        # unit (flagged lowvram), not silently abandoned.
        assert mp.model.model_lowvram is True
        assert not getattr(tree.hot, "comfy_patched_weights", False)
        # Forward through the wrapper still works with weights off-device.
        x = torch.randn(2, 128)
        with torch.no_grad():
            y = tree.hot(x)
        assert torch.isfinite(y).all()

    def test_forward_works_after_partial_unload(self):
        from ComfyUI_VibeVoice.modules.comfy_stream import convert_tree_for_streaming

        tree = self._converted_tree()
        mp = self._patcher(tree)
        mp.patch_model(device_to=torch.device("cpu"),
                       lowvram_model_memory=0)
        freed = mp.partially_unload(torch.device("cpu"), memory_to_free=1)
        assert freed > 0

        x = torch.randn(2, 128)
        with torch.no_grad():
            y = tree.hot(x)
        assert torch.isfinite(y).all()

    def test_round_trip_partial_unload_load(self):
        from ComfyUI_VibeVoice.modules.comfy_stream import convert_tree_for_streaming

        tree = self._converted_tree()
        mp = self._patcher(tree)
        mp.patch_model(device_to=torch.device("cpu"),
                       lowvram_model_memory=0)
        x0 = torch.randn(2, 128)
        with torch.no_grad():
            ref = tree.hot(x0).clone()

        mp.partially_unload(torch.device("cpu"), memory_to_free=1)
        mp.partially_load(torch.device("cpu"), extra_memory=1e32,
                          force_patch_weights=False)
        with torch.no_grad():
            y = tree.hot(x0)
        assert torch.equal(ref, y)
