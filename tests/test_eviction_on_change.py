"""Tests for unload-before-load eviction on model change (plan 2026-08-20).

Phase C coverage:
- Order ledger (RC-5): the external loader node evicts any previously active
  model BEFORE building the new one.
- Model change frees the old model: handler refs nulled, cache swept, gc run;
  the old weights become collectible once the last holder drops them.
- Same-model no-op: identical identity -> zero eviction calls, patcher reused.
- Dropdown <-> external cross-eviction (G3, single-active per family).
- ASR family isolation (TTS eviction never touches ASR caches, AUD-012).
- cleanup_old_models routes through model_registry.evict_patcher (C5 superset).
- P1 amendment: loader-node REQUEST identity == consumer BUNDLE identity.
- P2: q4 / dtype switches produce new keys and evict (old key collision fix).
"""

import os
import weakref

import pytest
import torch
from unittest.mock import MagicMock, patch

import comfy.model_management as comfy_mm

from ComfyUI_VibeVoice.modules import model_registry
from ComfyUI_VibeVoice.modules.model_registry import (
    FAMILY_ASR,
    FAMILY_TTS,
    clear_active_keys,
    get_active,
    identity_for_external,
    set_active,
)
from ComfyUI_VibeVoice.modules.attention_utils import resolve_attention_mode
from ComfyUI_VibeVoice.modules.loader import cleanup_old_models, LOADED_MODELS_CACHE
from ComfyUI_VibeVoice.modules.asr_loader import LOADED_ASR_MODELS_CACHE
from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_ASR_PATCHER_CACHE, VIBEVOICE_PATCHER_CACHE
from ComfyUI_VibeVoice.modules.generation import load_vibevoice_from_external, load_vibevoice_model
from ComfyUI_VibeVoice.modules.asr_generation import load_asr_from_external
from ComfyUI_VibeVoice.nodes.external_loader_node import VibeVoiceExternalLoaderNode

NODE = "ComfyUI_VibeVoice.nodes.external_loader_node"
GEN = "ComfyUI_VibeVoice.modules.generation"
ASRGEN = "ComfyUI_VibeVoice.modules.asr_generation"
LOADER = "ComfyUI_VibeVoice.modules.loader"
MR = "ComfyUI_VibeVoice.modules.model_registry"


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch):
    """Isolate global registries and neuter accelerator cache flushes."""
    VIBEVOICE_PATCHER_CACHE.clear()
    VIBEVOICE_ASR_PATCHER_CACHE.clear()
    LOADED_MODELS_CACHE.clear()
    LOADED_ASR_MODELS_CACHE.clear()
    clear_active_keys()
    monkeypatch.setattr(comfy_mm, "soft_empty_cache", lambda: None)
    monkeypatch.setattr(comfy_mm, "current_loaded_models", [])
    yield
    VIBEVOICE_PATCHER_CACHE.clear()
    VIBEVOICE_ASR_PATCHER_CACHE.clear()
    LOADED_MODELS_CACHE.clear()
    LOADED_ASR_MODELS_CACHE.clear()
    clear_active_keys()


# ====================================================================
# Helpers
# ====================================================================

def _write_weights(tmp_path, tag):
    f = tmp_path / f"weights_{tag}.safetensors"
    f.write_bytes(b"fake-weights-" + tag.encode())
    return f


def _make_tts_bundle(tmp_path, tag="a", name="VibeVoice-1.5B", attn="sdpa",
                     q4=False, dtype="auto"):
    """A bundle shaped exactly like load_external_vibevoice_model output."""
    f = _write_weights(tmp_path, tag)
    st = os.stat(f)
    return {
        "model": torch.nn.Linear(8, 8),
        "processor": object(),
        "config": object(),
        "model_name": name,
        "source_path": str(f),
        "source_mtime_ns": st.st_mtime_ns,
        "source_size": st.st_size,
        "attention_mode": resolve_attention_mode(attn, q4),
        "use_llm_4bit": bool(q4),
        "dtype_str": dtype,
        "is_streaming": False,
        "is_asr": False,
    }, f


def _make_asr_bundle(tmp_path, tag="asr", name="VibeVoice-ASR", attn="sdpa",
                     dtype="auto"):
    f = _write_weights(tmp_path, tag)
    st = os.stat(f)
    return {
        "model": torch.nn.Linear(8, 8),
        "processor": object(),
        "config": object(),
        "model_name": name,
        "source_path": str(f),
        "source_mtime_ns": st.st_mtime_ns,
        "source_size": st.st_size,
        "attention_mode": resolve_attention_mode(attn, False),
        "use_llm_4bit": False,
        "dtype_str": dtype,
        "is_streaming": False,
        "is_asr": True,
    }


def _gen_load(bundle, dtype="fp32", attention_mode="sdpa"):
    """Run load_vibevoice_from_external with GPU loading stubbed."""
    with patch(f"{GEN}.model_management.load_model_gpu"), \
         patch(f"{MR}.gc.collect") as mock_gc:
        result = load_vibevoice_from_external(
            bundle, device="cpu", dtype=dtype, attention_mode=attention_mode
        )
    return result, mock_gc


def _node_execute(weight_path, config_name="VibeVoice-1.5B", attn="sdpa",
                  q4=False, dtype="auto", bundle=None):
    """Run the loader node with path resolution + heavy build stubbed."""
    with patch(f"{NODE}.folder_paths.get_full_path_or_raise",
               return_value=str(weight_path)), \
         patch(f"{NODE}.load_external_vibevoice_model", return_value=bundle or {}):
        return VibeVoiceExternalLoaderNode.execute(
            model_file=os.path.basename(weight_path),
            config_name=config_name,
            attention_mode=attn,
            quantize_llm_4bit=q4,
            dtype=dtype,
        )


# ====================================================================
# RC-5: eviction precedes the CPU build
# ====================================================================

class TestEvictionOrderLedger:
    def test_loader_node_evicts_before_build(self, tmp_path):
        events = []
        stale = MagicMock()
        VIBEVOICE_PATCHER_CACHE["stale_key"] = stale
        set_active(FAMILY_TTS, "stale_key")

        f = _write_weights(tmp_path, "new")
        bundle, _ = _make_tts_bundle(tmp_path, tag="new")

        def _record_build(**kwargs):
            events.append("build")
            return bundle

        with patch(f"{NODE}.folder_paths.get_full_path_or_raise",
                   return_value=str(f)), \
             patch(f"{NODE}.load_external_vibevoice_model",
                   side_effect=_record_build), \
             patch(f"{MR}.evict_patcher",
                   side_effect=lambda *a, **k: events.append("evict")):
            # Inline execute (not the _node_execute helper) so the loader
            # mock above is the one actually used.
            VibeVoiceExternalLoaderNode.execute(
                model_file=os.path.basename(f),
                config_name="VibeVoice-1.5B",
                attention_mode="sdpa",
                quantize_llm_4bit=False,
                dtype="auto",
            )

        assert events == ["evict", "build"], (
            "eviction must be recorded BEFORE the loader builds the new model"
        )

    def test_first_run_no_stale_no_eviction_call(self, tmp_path):
        f = _write_weights(tmp_path, "first")
        with patch(f"{NODE}.folder_paths.get_full_path_or_raise",
                   return_value=str(f)), \
             patch(f"{NODE}.load_external_vibevoice_model", return_value={}), \
             patch(f"{MR}.evict_patcher") as mock_evict:
            _node_execute(f)
        mock_evict.assert_not_called()


# ====================================================================
# RC-1: changing the model frees the old one
# ====================================================================

class TestModelChangeFreesOldModel:
    def test_change_destroys_handler_refs_and_sweeps_cache(self, tmp_path):
        bundle_a, _ = _make_tts_bundle(tmp_path, tag="a")
        bundle_b, _ = _make_tts_bundle(tmp_path, tag="b")

        (patcher_a, _, _), _ = _gen_load(bundle_a)
        handler_a = patcher_a.model
        old_model = bundle_a["model"]
        old_ref = weakref.ref(old_model)

        (patcher_b, _, _), mock_gc = _gen_load(bundle_b)

        # Old handler refs nulled by the destroy path.
        assert handler_a.model is None
        assert handler_a.processor is None
        # Cache contains ONLY the new key.
        assert list(VIBEVOICE_PATCHER_CACHE.keys()) == [patcher_b.cache_key]
        assert patcher_a.cache_key not in VIBEVOICE_PATCHER_CACHE
        # Eviction ran a garbage collection pass.
        assert mock_gc.called
        # Active bookkeeping follows the new model.
        assert get_active(FAMILY_TTS) == patcher_b.cache_key

        # Once the test drops its own holders (standing in for ComfyUI's
        # execution cache), the old weights are collectible.
        del old_model, bundle_a
        import gc as _gc
        _gc.collect()
        _gc.collect()
        assert old_ref() is None, "old model tensors must be collectible"

    def test_switching_back_reloads_once_more(self, tmp_path):
        bundle_a, _ = _make_tts_bundle(tmp_path, tag="a")
        bundle_b, _ = _make_tts_bundle(tmp_path, tag="b")

        (p1, _, _), _ = _gen_load(bundle_a)
        _gen_load(bundle_b)          # evicts A
        with patch(f"{MR}.evict_patcher", wraps=model_registry.evict_patcher) as spy:
            (p3, _, _), _ = _gen_load(dict(bundle_a))  # switch back -> rebuild
        assert spy.called, "switching back must evict B first"
        assert p3 is not p1
        assert list(VIBEVOICE_PATCHER_CACHE.keys()) == [p3.cache_key]


# ====================================================================
# G5: same-model re-run is a strict no-op
# ====================================================================

class TestSameModelNoOp:
    def test_same_bundle_reuses_patcher_without_eviction(self, tmp_path):
        bundle, _ = _make_tts_bundle(tmp_path, tag="a")
        (p1, _, _), _ = _gen_load(bundle)

        with patch(f"{MR}.evict_patcher") as mock_evict:
            (p2, _, _), _ = _gen_load(bundle)

        mock_evict.assert_not_called()
        assert p2 is p1
        assert list(VIBEVOICE_PATCHER_CACHE.keys()) == [p1.cache_key]

    def test_consumer_widget_attention_change_does_not_fork_patcher(self, tmp_path):
        """The TTS node's own attention widget must NOT fork a second patcher
        for the same weights (the bundle records the built mode)."""
        bundle, _ = _make_tts_bundle(tmp_path, tag="a")
        (p1, _, _), _ = _gen_load(bundle, attention_mode="eager")

        with patch(f"{MR}.evict_patcher") as mock_evict:
            (p2, _, _), _ = _gen_load(bundle, attention_mode="sdpa")

        mock_evict.assert_not_called()
        assert p2 is p1


# ====================================================================
# G3: dropdown <-> external cross-eviction
# ====================================================================

class TestDropdownExternalCrossEviction:
    def test_dropdown_evicted_when_external_loaded(self, tmp_path):
        dd_patcher = MagicMock()
        dd_key = "TestModel_attn_sdpa_q4_0"
        VIBEVOICE_PATCHER_CACHE[dd_key] = dd_patcher
        set_active(FAMILY_TTS, dd_key)

        bundle, _ = _make_tts_bundle(tmp_path, tag="ext")
        (ext_patcher, _, _), _ = _gen_load(bundle)

        assert dd_key not in VIBEVOICE_PATCHER_CACHE
        assert list(VIBEVOICE_PATCHER_CACHE.keys()) == [ext_patcher.cache_key]

    def test_external_evicted_when_dropdown_loaded(self, tmp_path):
        bundle, _ = _make_tts_bundle(tmp_path, tag="ext")
        (ext_patcher, _, _), _ = _gen_load(bundle)

        mock_patcher = MagicMock()
        mock_patcher.model.model = MagicMock()
        mock_patcher.model.processor = MagicMock()

        with patch(f"{GEN}.VibeVoiceModelHandler") as mock_handler_cls, \
             patch(f"{GEN}.VibeVoicePatcher", return_value=mock_patcher), \
             patch(f"{GEN}.model_management.load_model_gpu"):
            mock_handler = MagicMock()
            mock_handler.size = 1000
            mock_handler_cls.return_value = mock_handler
            _, _, _ = load_vibevoice_model(
                model_name="TestModel", device="cpu", dtype="fp32",
                attention_mode="sdpa", quantize_4bit=False,
            )

        assert ext_patcher.cache_key not in VIBEVOICE_PATCHER_CACHE
        assert "TestModel_attn_sdpa_q4_0" in VIBEVOICE_PATCHER_CACHE


# ====================================================================
# Family isolation
# ====================================================================

class TestFamilyIsolation:
    def test_asr_load_never_touches_tts_cache(self, tmp_path):
        tts_patcher = MagicMock()
        VIBEVOICE_PATCHER_CACHE["tts_old"] = tts_patcher
        set_active(FAMILY_TTS, "tts_old")

        bundle = _make_asr_bundle(tmp_path)

        with patch("comfy.model_patcher.ModelPatcher.patch_model"), \
             patch(f"{ASRGEN}.model_management.load_model_gpu",
                   side_effect=lambda p: p.patch_model()):
            patcher, _, _ = load_asr_from_external(
                bundle, device="cpu", dtype="fp32", attention_mode="sdpa"
            )

        assert VIBEVOICE_PATCHER_CACHE == {"tts_old": tts_patcher}
        assert list(VIBEVOICE_ASR_PATCHER_CACHE.keys()) == [patcher.cache_key]
        assert get_active(FAMILY_TTS) == "tts_old"


# ====================================================================
# C5: cleanup_old_models routes through evict_patcher
# ====================================================================

class TestCleanupOldModelsSuperset:
    def test_cleanup_routes_through_evict_patcher(self):
        pm = MagicMock()
        VIBEVOICE_PATCHER_CACHE["k1"] = pm
        VIBEVOICE_PATCHER_CACHE["keep"] = MagicMock()

        def _pop_like_real(patcher, cache_dict, key):
            cache_dict.pop(key, None)

        with patch(f"{MR}.evict_patcher", side_effect=_pop_like_real) as mock_evict, \
             patch(f"{LOADER}.model_management"):
            cleanup_old_models(keep_cache_key="keep")

        mock_evict.assert_called_once_with(pm, VIBEVOICE_PATCHER_CACHE, "k1")
        assert "k1" not in VIBEVOICE_PATCHER_CACHE
        assert "keep" in VIBEVOICE_PATCHER_CACHE


# ====================================================================
# P1: request identity == bundle identity
# ====================================================================

class TestRequestBundleIdentityConsistency:
    def test_node_request_key_equals_consumer_cache_key(self, tmp_path):
        """End-to-end: whatever key the loader node computes pre-build MUST be
        the key the consumer derives from the produced bundle."""
        f = _write_weights(tmp_path, "id")

        captured_keys = []
        real_identity = identity_for_external

        def _spy_identity(*args, **kwargs):
            key = real_identity(*args, **kwargs)
            captured_keys.append(key)
            return key

        bundle, _ = _make_tts_bundle(tmp_path, tag="id")

        # identity_for_external is imported into each caller's namespace,
        # so spy on BOTH call sites (loader node + consumer).
        with patch(f"{NODE}.folder_paths.get_full_path_or_raise",
                   return_value=str(f)), \
             patch(f"{NODE}.load_external_vibevoice_model", return_value=bundle), \
             patch(f"{NODE}.identity_for_external", side_effect=_spy_identity), \
             patch(f"{GEN}.identity_for_external", side_effect=_spy_identity), \
             patch(f"{GEN}.model_management.load_model_gpu"):
            _node_execute(f)                       # loader side
            (patcher, _, _), _ = _gen_load(bundle)  # consumer side

        assert len(captured_keys) >= 2
        assert captured_keys[-1] == patcher.cache_key
        assert get_active(FAMILY_TTS) == patcher.cache_key

    def test_second_identical_run_performs_no_eviction(self, tmp_path):
        """Run 1 loads; run 2 (identical settings) must be a pure cache hit."""
        f = _write_weights(tmp_path, "stable")
        bundle, _ = _make_tts_bundle(tmp_path, tag="stable")

        with patch(f"{NODE}.folder_paths.get_full_path_or_raise",
                   return_value=str(f)), \
             patch(f"{NODE}.load_external_vibevoice_model", return_value=bundle):
            _node_execute(f)
        (p1, _, _), _ = _gen_load(bundle)

        with patch(f"{MR}.evict_patcher") as mock_evict, \
             patch(f"{NODE}.folder_paths.get_full_path_or_raise",
                   return_value=str(f)), \
             patch(f"{NODE}.load_external_vibevoice_model", return_value=bundle):
            _node_execute(f)
            (p2, _, _), _ = _gen_load(bundle)

        mock_evict.assert_not_called()
        assert p2 is p1


# ====================================================================
# P1 (ASR): request identity == bundle identity across the namespace
# ====================================================================

class TestASRRequestBundleIdentityConsistency:
    """Regression: the loader node computes its pre-build eviction identity
    with the DEFAULT 'external' prefix while ``load_asr_from_external`` keys
    the built model under 'asr_external'. The mismatch made every identical
    re-run of an external ASR model spuriously evict + rebuild the live
    patcher (acceptance criterion 3 broken for the ASR branch)."""

    def _asr_gen_load(self, bundle):
        """Run load_asr_from_external with GPU loading stubbed."""
        with patch("comfy.model_patcher.ModelPatcher.patch_model"), \
             patch(f"{ASRGEN}.model_management.load_model_gpu",
                   side_effect=lambda p: p.patch_model()), \
             patch(f"{MR}.gc.collect"):
            return load_asr_from_external(
                bundle, device="cpu", dtype="fp32", attention_mode="sdpa"
            )

    def test_node_request_key_uses_consumer_prefix(self, tmp_path):
        f = _write_weights(tmp_path, "asr_id")
        bundle = _make_asr_bundle(tmp_path, tag="asr_id")

        with patch(f"{NODE}.folder_paths.get_full_path_or_raise",
                   return_value=str(f)), \
             patch(f"{NODE}.load_external_vibevoice_model", return_value=bundle):
            _node_execute(f, config_name="VibeVoice-ASR")

        assert get_active(FAMILY_ASR) is not None
        assert get_active(FAMILY_ASR).startswith("asr_external_"), (
            "node request identity must live in the consumer's 'asr_external' "
            f"namespace, got {get_active(FAMILY_ASR)!r}"
        )

    def test_second_identical_asr_run_performs_no_eviction(self, tmp_path):
        f = _write_weights(tmp_path, "asr_stable")
        bundle = _make_asr_bundle(tmp_path, tag="asr_stable")

        with patch(f"{NODE}.folder_paths.get_full_path_or_raise",
                   return_value=str(f)), \
             patch(f"{NODE}.load_external_vibevoice_model", return_value=bundle):
            _node_execute(f, config_name="VibeVoice-ASR")
        (p1, _, _) = self._asr_gen_load(bundle)

        with patch(f"{MR}.evict_patcher") as mock_evict, \
             patch(f"{NODE}.folder_paths.get_full_path_or_raise",
                   return_value=str(f)), \
             patch(f"{NODE}.load_external_vibevoice_model", return_value=bundle):
            _node_execute(f, config_name="VibeVoice-ASR")
            (p2, _, _) = self._asr_gen_load(bundle)

        mock_evict.assert_not_called()
        assert p2 is p1, "identical ASR re-run must reuse the cached patcher"
        assert list(VIBEVOICE_ASR_PATCHER_CACHE.keys()) == [p1.cache_key]


# ====================================================================
# P2: q4 / dtype switches collide no more
# ====================================================================

class TestQuantizationDtypeSwitchKeys:
    def test_q4_switch_same_file_new_key_and_eviction(self, tmp_path):
        bundle_fp, f = _make_tts_bundle(tmp_path, tag="same", q4=False)
        bundle_q4, _ = _make_tts_bundle(tmp_path, tag="same", q4=True)

        (p1, _, _), _ = _gen_load(bundle_fp)
        (p2, _, _), _ = _gen_load(bundle_q4)

        assert p1.cache_key != p2.cache_key
        assert p1.cache_key not in VIBEVOICE_PATCHER_CACHE
        assert list(VIBEVOICE_PATCHER_CACHE.keys()) == [p2.cache_key]

    def test_dtype_switch_same_file_new_key_and_eviction(self, tmp_path):
        bundle_auto, _ = _make_tts_bundle(tmp_path, tag="same", dtype="auto")
        bundle_fp32, _ = _make_tts_bundle(tmp_path, tag="same", dtype="fp32")

        (p1, _, _), _ = _gen_load(bundle_auto)
        (p2, _, _), _ = _gen_load(bundle_fp32)

        assert p1.cache_key != p2.cache_key
        assert list(VIBEVOICE_PATCHER_CACHE.keys()) == [p2.cache_key]
