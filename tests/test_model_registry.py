"""Tests for modules/model_registry.py — active-model tracking and eviction.

Covers plan 2026-08-20 Phase A:
- unregister_from_comfy: removes only matching patcher entries from a faked
  model_management.current_loaded_models, detaches finalizers, survives None
  finalizers, no-op when absent.
- evict_patcher: strict ordered ledger (unregister -> destroy -> cache pop ->
  gc -> soft_empty_cache); cache pop happens even if unpatch_model raises.
- evict_if_changed: same-key strict no-op; different key evicts all other-key
  entries across all provided caches and updates the active key; empty caches
  safe; family isolation (tts eviction never touches asr caches).
- identity_for_external: differs for different paths / mtime changes; stable
  for identical inputs; basename-only (no path separators); missing-file
  fallback is deterministic.

Also pins the patcher size-estimation contract (``VibeVoiceASRPatcher
._estimate_size``) and the shared ``lowvram_model_memory`` forwarding
contract, which together decide what ComfyUI believes the model costs.
"""

import inspect
import os
import weakref

import pytest
import torch
from unittest.mock import patch, MagicMock

import comfy.model_management as model_management

from ComfyUI_VibeVoice.modules import model_registry
from ComfyUI_VibeVoice.modules.asr_generation import ExternalVibeVoiceASRModelHandler
from ComfyUI_VibeVoice.modules.model_registry import (
    clear_active_keys,
    evict_if_changed,
    evict_patcher,
    get_active,
    identity_for_external,
    set_active,
    unregister_from_comfy,
)
from ComfyUI_VibeVoice.modules.patcher import VibeVoicePatcher, VibeVoiceASRPatcher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _FinalizerStub:
    """Stands in for weakref.finalize objects with detach recording."""

    def __init__(self):
        self.detached = False

    def detach(self):
        self.detached = True


class _LoadedEntryStub:
    """Mimics comfy.model_management.LoadedModel's surface used by the registry."""

    def __init__(self, patcher, with_finalizers=True):
        self.model = patcher
        self.real_model = object()
        self.model_finalizer = _FinalizerStub() if with_finalizers else None
        self._patcher_finalizer = _FinalizerStub() if with_finalizers else None


@pytest.fixture(autouse=True)
def _fresh_registry(monkeypatch):
    """Fresh active-key store + fake current_loaded_models per test."""
    clear_active_keys()
    monkeypatch.setattr(model_management, "current_loaded_models", [])
    yield
    clear_active_keys()


# ====================================================================
# unregister_from_comfy
# ====================================================================

class TestUnregisterFromComfy:
    def test_removes_only_matching_entries(self):
        target = MagicMock()
        other = MagicMock()
        entry_target = _LoadedEntryStub(target)
        entry_other = _LoadedEntryStub(other)
        model_management.current_loaded_models[:] = [entry_other, entry_target, entry_other]

        removed = unregister_from_comfy(target)

        assert removed == [entry_target]
        assert model_management.current_loaded_models == [entry_other, entry_other]

    def test_detaches_finalizers_and_clears_real_model(self):
        target = MagicMock()
        entry = _LoadedEntryStub(target)
        model_finalizer = entry.model_finalizer
        patcher_finalizer = entry._patcher_finalizer
        model_management.current_loaded_models.append(entry)

        unregister_from_comfy(target)

        assert model_finalizer.detached is True
        assert patcher_finalizer.detached is True
        assert entry.model_finalizer is None
        assert entry._patcher_finalizer is None
        assert entry.real_model is None

    def test_survives_none_finalizers(self):
        target = MagicMock()
        entry = _LoadedEntryStub(target, with_finalizers=False)
        model_management.current_loaded_models.append(entry)

        removed = unregister_from_comfy(target)

        assert removed == [entry]
        assert model_management.current_loaded_models == []

    def test_noop_when_patcher_absent(self):
        entry = _LoadedEntryStub(MagicMock())
        model_management.current_loaded_models.append(entry)

        removed = unregister_from_comfy(MagicMock())

        assert removed == []
        assert len(model_management.current_loaded_models) == 1

    def test_dead_weakref_entry_is_left_alone(self):
        """An entry whose weakref died (loaded.model is None) is not matched."""
        dead_entry = _LoadedEntryStub(None)
        model_management.current_loaded_models.append(dead_entry)

        removed = unregister_from_comfy(MagicMock())

        assert removed == []
        assert model_management.current_loaded_models == [dead_entry]

    def test_works_against_real_weakref_property(self):
        """LoadedModel.model is a property over a weakref — verify via property."""

        class _RealishLoaded:
            def __init__(self, patcher):
                self._ref = weakref.ref(patcher)
                self.real_model = None
                self.model_finalizer = _FinalizerStub()
                self._patcher_finalizer = None

            @property
            def model(self):
                return self._ref()

        target = MagicMock()
        entry = _RealishLoaded(target)
        model_finalizer = entry.model_finalizer
        model_management.current_loaded_models.append(entry)

        removed = unregister_from_comfy(target)

        assert removed == [entry]
        assert model_finalizer.detached is True


# ====================================================================
# evict_patcher
# ====================================================================

class TestEvictPatcher:
    def test_order_ledger(self):
        calls = []
        patcher = MagicMock()
        patcher.unpatch_model.side_effect = lambda *a, **k: calls.append("destroy")
        cache_dict = {("k1"): patcher}

        with patch.object(model_registry, "unregister_from_comfy",
                          side_effect=lambda p: calls.append("unregister")), \
             patch("ComfyUI_VibeVoice.modules.model_registry.gc.collect",
                   side_effect=lambda: calls.append("gc")), \
             patch.object(model_management, "soft_empty_cache",
                          side_effect=lambda: calls.append("soft_empty_cache")):
            errors = evict_patcher(patcher, cache_dict, "k1")

        assert errors == []
        assert calls == ["unregister", "destroy", "gc", "soft_empty_cache"]
        assert "k1" not in cache_dict

    def test_cache_pop_even_when_unpatch_raises(self):
        patcher = MagicMock()
        patcher.unpatch_model.side_effect = RuntimeError("boom")
        cache_dict = {"k1": patcher}

        with patch.object(model_registry, "unregister_from_comfy"), \
             patch("ComfyUI_VibeVoice.modules.model_registry.gc.collect"), \
             patch.object(model_management, "soft_empty_cache"):
            errors = evict_patcher(patcher, cache_dict, "k1")

        assert "k1" not in cache_dict
        assert any("unpatch_model" in e for e in errors)

    def test_never_raises(self):
        patcher = MagicMock()
        # Every attribute access works, but make unregister raise too.
        with patch.object(model_registry, "unregister_from_comfy",
                          side_effect=RuntimeError("unreg boom")), \
             patch("ComfyUI_VibeVoice.modules.model_registry.gc.collect",
                   side_effect=RuntimeError("gc boom")), \
             patch.object(model_management, "soft_empty_cache",
                          side_effect=RuntimeError("cache boom")):
            errors = evict_patcher(patcher, {"k": patcher}, "k")

        assert len(errors) >= 3

    def test_none_cache_dict_is_safe(self):
        patcher = MagicMock()
        with patch.object(model_registry, "unregister_from_comfy"), \
             patch("ComfyUI_VibeVoice.modules.model_registry.gc.collect"), \
             patch.object(model_management, "soft_empty_cache"):
            errors = evict_patcher(patcher, None, "k")
        assert errors == []


# ====================================================================
# evict_if_changed / active-key store
# ====================================================================

class TestActiveKeys:
    def test_set_get(self):
        assert get_active("tts") is None
        set_active("tts", "key_a")
        assert get_active("tts") == "key_a"
        assert get_active("asr") is None


class TestEvictIfChanged:
    def test_same_key_strict_noop(self):
        set_active("tts", "k1")
        cache = {"k1": MagicMock()}
        with patch.object(model_registry, "evict_patcher") as mock_evict:
            evicted = evict_if_changed("tts", "k1", (cache,))
        assert evicted == []
        mock_evict.assert_not_called()
        assert get_active("tts") == "k1"
        assert "k1" in cache

    def test_different_key_evicts_all_others_across_caches(self):
        set_active("tts", "old")
        old_p = MagicMock()
        other_p = MagicMock()
        cache_a = {"old": old_p, "other": other_p}
        cache_b = {"other2": MagicMock()}

        evicted = evict_if_changed("tts", "new", (cache_a, cache_b))

        assert sorted(evicted) == ["old", "other", "other2"]
        assert cache_a == {} and cache_b == {}
        assert get_active("tts") == "new"

    def test_first_call_with_stale_entries_sweeps_them(self):
        """active=None differs from any key → stale leftovers are evicted."""
        stale = MagicMock()
        cache = {"stale": stale}
        evicted = evict_if_changed("tts", "new", (cache,))
        assert evicted == ["stale"]
        assert cache == {}
        assert get_active("tts") == "new"

    def test_empty_caches_safe(self):
        evicted = evict_if_changed("asr", "k", ({},))
        assert evicted == []
        assert get_active("asr") == "k"

    def test_family_isolation(self):
        """TTS eviction never touches ASR caches and vice versa (AUD-012)."""
        tts_cache = {"t_old": MagicMock()}
        asr_cache = {"a_old": MagicMock()}
        set_active("tts", "t_old")

        evict_if_changed("tts", "t_new", (tts_cache,))
        assert tts_cache == {}
        assert asr_cache == {"a_old"} or list(asr_cache.keys()) == ["a_old"]
        assert get_active("asr") is None

        evicted = evict_if_changed("asr", "a_new", (asr_cache,))
        assert evicted == ["a_old"]


# ====================================================================
# identity_for_external
# ====================================================================

class TestIdentityForExternal:
    def test_deterministic_for_identical_inputs(self, tmp_path):
        f = tmp_path / "weights.safetensors"
        f.write_bytes(b"x")
        k1 = identity_for_external(str(f), "Cfg", "sdpa", False, "auto")
        k2 = identity_for_external(str(f), "Cfg", "sdpa", False, "auto")
        assert k1 == k2

    def test_differs_for_different_paths(self, tmp_path):
        fa = tmp_path / "a.safetensors"
        fb = tmp_path / "b.safetensors"
        fa.write_bytes(b"x")
        fb.write_bytes(b"y")
        assert identity_for_external(str(fa), "Cfg", "sdpa") != \
            identity_for_external(str(fb), "Cfg", "sdpa")

    def test_differs_after_mtime_change(self, tmp_path):
        f = tmp_path / "w.safetensors"
        f.write_bytes(b"x")
        k_before = identity_for_external(str(f), "Cfg", "sdpa")
        os.utime(f, ns=(1000, 2000))
        os.utime(f, ns=(5000000000, 5000000000))
        k_after = identity_for_external(str(f), "Cfg", "sdpa")
        assert k_before != k_after

    def test_differs_for_settings_dimensions(self, tmp_path):
        f = tmp_path / "w.safetensors"
        f.write_bytes(b"x")
        base = identity_for_external(str(f), "Cfg", "sdpa", False, "auto")
        assert base != identity_for_external(str(f), "Cfg", "eager", False, "auto")
        assert base != identity_for_external(str(f), "Cfg", "sdpa", True, "auto")
        assert base != identity_for_external(str(f), "Cfg", "sdpa", False, "fp32")
        assert base != identity_for_external(str(f), "Cfg2", "sdpa", False, "auto")

    def test_basename_only_no_separators(self, tmp_path):
        f = tmp_path / "sub"
        f.mkdir()
        wf = f / "weights.safetensors"
        wf.write_bytes(b"x")
        key = identity_for_external(str(wf), "Cfg", "sdpa")
        assert ":" not in key
        assert "\\" not in key
        assert "/" not in key
        assert "weights.safetensors" in key

    def test_missing_file_fallback_is_deterministic(self):
        ghost = "Z:/no/such/file.safetensors"
        k1 = identity_for_external(ghost, "Cfg", "sdpa")
        k2 = identity_for_external(ghost, "Cfg", "sdpa")
        assert k1 == k2
        # And it still distinguishes paths by basename.
        assert k1 != identity_for_external("Z:/other/name.safetensors", "Cfg", "sdpa")

    def test_empty_path_is_safe(self):
        k1 = identity_for_external("", "Cfg", "sdpa")
        k2 = identity_for_external("", "Cfg", "sdpa")
        assert k1 == k2

    def test_prefix_namespacing(self, tmp_path):
        f = tmp_path / "w.safetensors"
        f.write_bytes(b"x")
        tts_key = identity_for_external(str(f), "Cfg", "sdpa", prefix="external")
        asr_key = identity_for_external(str(f), "Cfg", "sdpa", prefix="asr_external")
        assert tts_key.startswith("external_")
        assert asr_key.startswith("asr_external_")
        assert tts_key != asr_key


# ====================================================================
# Patcher size estimation + lowvram contract (verify-only)
# ====================================================================

class _SizedModel(torch.nn.Module):
    """Tiny real module whose parameter byte sum is exactly predictable."""

    def __init__(self, elements=8, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(elements, dtype=dtype))


class TestASRPatcherSizeEstimate:
    """``ExternalVibeVoiceASRModelHandler._estimate_size`` precedence.

    hint (``size_gb``) > parameter byte sum > 15 GB fallback. This value is
    ComfyUI's only view of what the model costs, so a wrong one either
    starves it or lets it evict other models.
    """

    def test_size_gb_hint_wins_over_parameter_sum(self):
        model = _SizedModel()  # 8 x fp32 = 32 bytes
        size = ExternalVibeVoiceASRModelHandler._estimate_size(model, {"size_gb": 2.0})
        assert size == int(2.0 * (1024 ** 3))
        assert size > 32, "the hint must beat the parameter sum"

    def test_parameter_sum_used_without_hint(self):
        model = _SizedModel(elements=8)  # 8 x fp32 = 32 bytes
        assert ExternalVibeVoiceASRModelHandler._estimate_size(model) == 32
        # An empty bundle is the real external-ASR shape: the loader builds
        # the bundle at modules/external_loader.py and sets no size_gb, which
        # is correct — the model is fully materialised by then, so the real
        # parameter sum is a better answer than a static config guess.
        assert ExternalVibeVoiceASRModelHandler._estimate_size(model, {"is_asr": True}) == 32
        assert ExternalVibeVoiceASRModelHandler._estimate_size(model, {"size_gb": None}) == 32

    def test_falls_back_to_15gb_when_no_parameters(self):
        assert ExternalVibeVoiceASRModelHandler._estimate_size(torch.nn.Module()) == \
            int(15.0 * (1024 ** 3))

    def test_unparsable_hint_falls_through_to_parameters(self):
        model = _SizedModel(elements=8)
        assert ExternalVibeVoiceASRModelHandler._estimate_size(
            model, {"size_gb": "not-a-number"}) == 32


def _make_patcher(cls):
    """Build a patcher of ``cls`` with core's ``ModelPatcher.__init__`` stubbed."""
    handler = MagicMock()
    handler.cache_key = "k"
    handler.model = MagicMock()  # pre-set → patch_model skips the lazy load
    with patch("comfy.model_patcher.ModelPatcher.__init__"):
        patcher = cls(
            handler,
            attention_mode="sdpa",
            dtype=None,
            load_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            size=1000,
        )
    # Attributes core's __init__ would normally set (it is stubbed out).
    patcher.model = handler
    patcher.load_device = torch.device("cpu")
    patcher.offload_device = torch.device("cpu")
    patcher.pinned = set()
    return patcher


class TestPatcherLowvramContract:
    """TTS and ASR patchers must expose the same lowvram contract.

    ``lowvram_model_memory`` is ComfyUI's per-model memory allowance. Both
    patchers take it and hand it to core untouched; neither stores it, and no
    caller in this pack ever sets it — so core's own default applies and the
    weights are fully resident on the load device.
    """

    @pytest.mark.parametrize("cls", [VibeVoicePatcher, VibeVoiceASRPatcher])
    def test_signature_matches_and_forwards_value(self, cls):
        assert inspect.signature(cls.patch_model) == \
            inspect.signature(VibeVoicePatcher.patch_model)

        patcher = _make_patcher(cls)
        with patch("comfy.model_patcher.ModelPatcher.patch_model") as mock_super:
            patcher.patch_model(lowvram_model_memory=123)

        assert mock_super.call_args.kwargs["lowvram_model_memory"] == 123
        # Forwarded, not intercepted: the patcher keeps no attribute for it.
        assert not hasattr(patcher, "lowvram_model_memory")

    def test_lowvram_model_memory_is_declared_and_forwarded_only(self):
        """No module in this pack ever *sets* the value — only forwards it."""
        modules_dir = os.path.join(ROOT, "modules")
        hits = {}
        assignments = []
        for name in sorted(os.listdir(modules_dir)):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(modules_dir, name), "r", encoding="utf-8") as fh:
                lines = fh.readlines()
            count = sum(1 for line in lines if "lowvram_model_memory" in line)
            if count:
                hits[name] = count
            for lineno, line in enumerate(lines, 1):
                stripped = line.strip()
                if "lowvram_model_memory" not in stripped:
                    continue
                if stripped.startswith(("def ", "lowvram_model_memory=")) or (
                    stripped.startswith("lowvram_model_memory=")
                ):
                    continue
                if "lowvram_model_memory=" in stripped and not stripped.startswith("def "):
                    # A keyword argument in a super() forward is fine; an
                    # attribute store (``self.lowvram_model_memory = ...``) is
                    # not, and that is the thing this test exists to forbid.
                    lhs = stripped.split("=", 1)[0]
                    if "self." in lhs or "patcher." in lhs:
                        assignments.append(f"{name}:{lineno}: {stripped}")
        # Every mention lives in the patcher module: the legacy signature, its
        # docstring, the legacy forward (which names the parameter twice — once
        # as key, once as value), and the DynamicVRAM sibling class's
        # signature + forward added by the 2026-09-29 T6 selector port.
        assert set(hits) == {"patcher.py"}, hits
        assert hits["patcher.py"] >= 3, hits
        assert assignments == [], assignments
