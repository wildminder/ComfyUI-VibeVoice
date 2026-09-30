"""FIX 1 regression tests: eviction must release the node-output bundle.

The external loader node returns the model bundle as a node OUTPUT
(``nodes/external_loader_node.VibeVoiceModel.Output``), and ComfyUI's
execution cache holds that dict strongly (``execution.CacheEntry.outputs``)
until the loader node leaves the prompt or re-executes. So popping our own
patcher-cache entry frees nothing on its own: the cached bundle still pins the
live ``nn.Module`` and its tensors. These tests pin the keyed bundle registry
that makes eviction NEUTRALIZE the bundle.

Each eviction test deliberately keeps the bundle alive in a stand-in holder
for the whole test (standing in for ComfyUI's output cache). The holder is
NOT what frees the model — the release step is.
"""

import gc
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
    evict_if_changed,
    identity_for_external,
    register_model_bundle,
)
from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_ASR_PATCHER_CACHE, VIBEVOICE_PATCHER_CACHE
from ComfyUI_VibeVoice.modules.generation import load_vibevoice_from_external
from ComfyUI_VibeVoice.modules.asr_generation import load_asr_from_external

MR = "ComfyUI_VibeVoice.modules.model_registry"
GEN = "ComfyUI_VibeVoice.modules.generation"
ASRGEN = "ComfyUI_VibeVoice.modules.asr_generation"


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch):
    """Isolate global registries and neuter accelerator cache flushes."""
    def _reset():
        VIBEVOICE_PATCHER_CACHE.clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()
        clear_active_keys()
        model_registry.clear_bundle_registry()

    _reset()
    monkeypatch.setattr(comfy_mm, "soft_empty_cache", lambda: None)
    monkeypatch.setattr(comfy_mm, "current_loaded_models", [])
    yield
    _reset()


# ====================================================================
# Helpers
# ====================================================================

def _write_weights(tmp_path, tag, ext="gguf"):
    f = tmp_path / f"weights_{tag}.{ext}"
    f.write_bytes(b"fake-weights-" + tag.encode())
    return f


def _make_bundle(tmp_path, tag="a", name="VibeVoice-1.5B", attn="sdpa",
                 q4=False, dtype="auto", is_asr=False):
    """A bundle shaped exactly like the external loader node's output."""
    f = _write_weights(tmp_path, tag)
    return {
        "model": torch.nn.Linear(8, 8),
        "processor": object(),
        "model_name": name,
        "source_path": str(f),
        "attention_mode": attn,
        "use_llm_4bit": bool(q4),
        "dtype_str": dtype,
        "is_asr": bool(is_asr),
    }


def _cache_key_for(bundle, is_asr=False):
    """The consumer's REAL cache key, so the test tracks production."""
    return identity_for_external(
        bundle["source_path"],
        bundle["model_name"],
        bundle["attention_mode"],
        use_llm_4bit=bundle["use_llm_4bit"],
        dtype_str=bundle["dtype_str"],
        prefix="asr_external" if is_asr else "external",
    )


# ====================================================================
# TTS: eviction frees the weights the output cache is pinning
# ====================================================================

class TestTTSBundleReleasedOnEviction:
    def test_tts_bundle_model_freed_on_eviction(self, tmp_path):
        bundle = _make_bundle(tmp_path, tag="tts")
        # ComfyUI's output cache: the bundle stays strongly held for the whole
        # test. This holder is deliberately NOT what frees the model.
        comfy_output_cache = {"node_1": bundle}
        del bundle

        key = _cache_key_for(comfy_output_cache["node_1"])
        register_model_bundle(key, comfy_output_cache["node_1"])

        VIBEVOICE_PATCHER_CACHE[key] = MagicMock()
        model_registry.set_active(FAMILY_TTS, key)

        wr = weakref.ref(comfy_output_cache["node_1"]["model"])
        assert wr() is not None

        evicted = evict_if_changed(FAMILY_TTS, "some_other_key", (VIBEVOICE_PATCHER_CACHE,))
        gc.collect()

        assert key in evicted
        assert wr() is None, (
            "eviction must neutralize the output-cached bundle, otherwise the "
            "external->default/sharded transition leaks the whole model"
        )
        # The cached dict survives (debuggable), only the heavy fields are nulled.
        cached = comfy_output_cache["node_1"]
        assert cached["model"] is None
        assert cached["processor"] is None
        assert cached["model_name"] == "VibeVoice-1.5B"
        assert cached["source_path"]
        assert cached["attention_mode"] == "sdpa"
        assert cached["use_llm_4bit"] is False
        assert cached["dtype_str"] == "auto"
        assert cached["is_asr"] is False


# ====================================================================
# ASR: the registry-level twin of the TTS release test
# ====================================================================

class TestASRBundleReleasedOnEviction:
    def test_asr_bundle_model_freed_on_eviction(self, tmp_path):
        bundle = _make_bundle(tmp_path, tag="asr", name="VibeVoice-ASR", is_asr=True)
        comfy_output_cache = {"node_2": bundle}
        del bundle

        key = _cache_key_for(comfy_output_cache["node_2"], is_asr=True)
        assert key.startswith("asr_external_"), "ASR key namespace must be preserved"
        register_model_bundle(key, comfy_output_cache["node_2"])

        VIBEVOICE_ASR_PATCHER_CACHE[key] = MagicMock()
        model_registry.set_active(FAMILY_ASR, key)

        wr = weakref.ref(comfy_output_cache["node_2"]["model"])
        assert wr() is not None

        evicted = evict_if_changed(FAMILY_ASR, "some_other_key", (VIBEVOICE_ASR_PATCHER_CACHE,))
        gc.collect()

        assert key in evicted
        assert wr() is None, "ASR eviction must neutralize the output-cached bundle"
        cached = comfy_output_cache["node_2"]
        assert cached["model"] is None
        assert cached["processor"] is None
        assert cached["is_asr"] is True
        assert cached["model_name"] == "VibeVoice-ASR"


# ====================================================================
# ASR end-to-end: the REAL consumer does the registering
# ====================================================================

class TestASRConsumerWiring:
    """The ASR counterpart of ``test_nulled_bundle_fails_loudly_not_silently``.

    Registry-level tests can pass while the ASR consumer never calls
    ``register_model_bundle`` at all (the TTS equivalent is the
    ``register_model_bundle(cache_key, model_bundle)`` line in
    ``modules/generation.py``). These drive the real
    ``load_asr_from_external`` so a regression in the ASR wiring is caught
    here rather than on a user's external ASR swap.
    """

    def _run_asr_consumer(self, bundle):
        """Drive load_asr_from_external with GPU loading stubbed out.

        The load stub patches ``load_models_gpu``, not ``load_model_gpu``:
        core's ``load_model_gpu(model)`` is literally ``load_models_gpu([model])``
        (comfy/model_management.py:1041-1042), resolved as a module global at
        call time. One patch therefore intercepts both the legacy
        ``load_to_device`` branch and the dynamic one, which calls
        ``load_models_gpu([patcher], memory_required=...)``, without the test
        having to know which class the selector picked.
        """
        with patch("comfy.model_patcher.ModelPatcher.patch_model"), \
             patch(f"{ASRGEN}.model_management.load_models_gpu",
                   side_effect=lambda models: models[0].patch_model()), \
             patch(f"{MR}.gc.collect"):
            return load_asr_from_external(
                bundle, device="cpu", dtype="fp32", attention_mode="sdpa"
            )

    def test_asr_consumer_registers_bundle_under_its_patcher_key(self, tmp_path):
        bundle = _make_bundle(tmp_path, tag="e2e", name="VibeVoice-ASR",
                              is_asr=True)
        # ComfyUI's output cache holds the node output for the whole test.
        comfy_output_cache = {"node_asr": bundle}
        del bundle

        patcher, _, _ = self._run_asr_consumer(comfy_output_cache["node_asr"])
        key = patcher.cache_key
        assert key in VIBEVOICE_ASR_PATCHER_CACHE

        # The consumer must have registered THE SAME dict under THE SAME key
        # the patcher is cached under — that is the only thing eviction can
        # act on.
        assert model_registry.release_model_bundles(key) == 1

    def test_asr_consumer_bundle_freed_by_eviction(self, tmp_path):
        bundle = _make_bundle(tmp_path, tag="e2e2", name="VibeVoice-ASR",
                              is_asr=True)
        comfy_output_cache = {"node_asr2": bundle}
        del bundle

        patcher, _, _ = self._run_asr_consumer(comfy_output_cache["node_asr2"])
        key = patcher.cache_key
        model_registry.set_active(FAMILY_ASR, key)

        wr = weakref.ref(comfy_output_cache["node_asr2"]["model"])
        assert wr() is not None

        evicted = evict_if_changed(
            FAMILY_ASR, "a_different_key", (VIBEVOICE_ASR_PATCHER_CACHE,)
        )
        gc.collect()

        assert key in evicted
        assert wr() is None, (
            "the real ASR consumer must register a bundle that eviction can "
            "neutralize, or external ASR -> anything leaks the whole model"
        )
        assert comfy_output_cache["node_asr2"]["model"] is None

    def test_nulled_asr_bundle_fails_loudly_not_silently(self, tmp_path):
        bundle = _make_bundle(tmp_path, tag="loudasr", name="VibeVoice-ASR",
                              is_asr=True)
        key = _cache_key_for(bundle, is_asr=True)
        register_model_bundle(key, bundle)
        comfy_output_cache = {"node_asr3": bundle}   # output-cache holder
        del bundle

        model_registry.release_model_bundles(key)
        with patch(f"{ASRGEN}.evict_if_changed"):
            with pytest.raises(ValueError) as exc:
                load_asr_from_external(
                    comfy_output_cache["node_asr3"], device="cpu",
                    dtype="fp32", attention_mode="sdpa",
                )
        assert "missing required key 'model'" in str(exc.value)


# ====================================================================
# Scoping: a live sibling key is never collateral damage
# ====================================================================

class TestReleaseScoping:
    def test_release_is_scoped_to_the_evicted_key(self, tmp_path):
        """K1 is the superseded model, K2 the still-connected one: only K1's
        weights may be released."""
        bundle_k1 = _make_bundle(tmp_path, tag="k1")
        bundle_k2 = _make_bundle(tmp_path, tag="k2")
        key1 = _cache_key_for(bundle_k1)
        key2 = _cache_key_for(bundle_k2)
        assert key1 != key2

        register_model_bundle(key1, bundle_k1)
        register_model_bundle(key2, bundle_k2)

        VIBEVOICE_PATCHER_CACHE[key1] = MagicMock()
        VIBEVOICE_PATCHER_CACHE[key2] = MagicMock()
        model_registry.set_active(FAMILY_TTS, key1)

        wr1 = weakref.ref(bundle_k1["model"])
        wr2 = weakref.ref(bundle_k2["model"])
        del bundle_k1, bundle_k2

        # K2 becomes the new active model -> only K1 is swept.
        evicted = evict_if_changed(FAMILY_TTS, key2, (VIBEVOICE_PATCHER_CACHE,))
        gc.collect()

        assert evicted == [key1]
        assert wr1() is None, "the evicted key's weights must be freed"
        assert wr2() is not None, "the still-active key's weights must survive"
        assert key2 in VIBEVOICE_PATCHER_CACHE

    def test_release_returns_1_for_known_key_and_0_otherwise(self, tmp_path):
        bundle = _make_bundle(tmp_path, tag="ret")
        key = _cache_key_for(bundle)
        register_model_bundle(key, bundle)
        assert model_registry.release_model_bundles(key) == 1
        assert model_registry.release_model_bundles(key) == 0
        assert model_registry.release_model_bundles("never_registered") == 0

    def test_reregistering_same_bundle_does_not_neutralize_it(self, tmp_path):
        """The reused-patcher path re-registers the SAME dict."""
        bundle = _make_bundle(tmp_path, tag="reuse")
        key = _cache_key_for(bundle)
        model = bundle["model"]

        register_model_bundle(key, bundle)
        register_model_bundle(key, bundle)   # same object, no-op replace

        assert bundle["model"] is model, (
            "re-registering the live bundle must never null it out from under "
            "its own consumer"
        )
        assert model_registry.release_model_bundles(key) == 1
        assert bundle["model"] is None


# ====================================================================
# A still-connected consumer of a nulled bundle fails LOUDLY
# ====================================================================

class TestNulledBundleFailsLoudly:
    def test_nulled_bundle_rebuilds_from_source_path(self, tmp_path):
        """A released bundle is rebuilt, not rejected.

        The loader node's output is cached by ComfyUI and can outlive the
        registry entry that backed it, so a consumer can legitimately be
        handed a bundle whose heavy fields were already released.
        """
        bundle = _make_bundle(tmp_path, tag="rebuild")
        key = _cache_key_for(bundle)
        src = bundle["source_path"]
        register_model_bundle(key, bundle)
        comfy_output_cache = {"node_3": bundle}   # output-cache holder
        del bundle

        model_registry.release_model_bundles(key)
        assert comfy_output_cache["node_3"]["model"] is None

        rebuilt = {"model": torch.nn.Linear(8, 8), "processor": object(),
                   "model_name": "VibeVoice-1.5B"}
        with patch(f"{GEN}.evict_if_changed"):
            with patch(
                "ComfyUI_VibeVoice.modules.external_loader."
                "load_external_vibevoice_model", return_value=rebuilt
            ) as reload_:
                patcher, model, processor = load_vibevoice_from_external(
                    comfy_output_cache["node_3"], device="cpu",
                    dtype="fp32", attention_mode="sdpa",
                )
        assert model is rebuilt["model"] and processor is rebuilt["processor"]
        assert reload_.call_args[0][0] == src

    def test_nulled_bundle_without_source_still_raises(self, tmp_path):
        """With no source to rebuild from, failing loudly is correct."""
        bundle = _make_bundle(tmp_path, tag="nosrc")
        bundle["source_path"] = ""
        bundle["model"] = None
        comfy_output_cache = {"node_3": bundle}

        with patch(f"{GEN}.evict_if_changed"):
            with pytest.raises(ValueError) as exc:
                load_vibevoice_from_external(
                    comfy_output_cache["node_3"], device="cpu",
                    dtype="fp32", attention_mode="sdpa",
                )
        assert "missing required key 'model'" in str(exc.value)


# ====================================================================
# Key format stability
# ====================================================================

class TestIdentityKeyFormat:
    def test_identity_for_external_key_format_is_stable(self, tmp_path):
        f = _write_weights(tmp_path, "fmt")
        st = os.stat(f)

        key = identity_for_external(
            str(f), "VibeVoice-1.5B", "sdpa",
            use_llm_4bit=False, dtype_str="auto",
        )
        assert key == (
            f"external_VibeVoice-1.5B@{f.name}@{st.st_mtime_ns}@{st.st_size}"
            "_attn_sdpa_q4_0_dtype_auto"
        )

        asr_key = identity_for_external(
            str(f), "VibeVoice-ASR", "sage",
            use_llm_4bit=True, dtype_str="bf16", prefix="asr_external",
        )
        assert asr_key == (
            f"asr_external_VibeVoice-ASR@{f.name}@{st.st_mtime_ns}@{st.st_size}"
            "_attn_sage_q4_1_dtype_bf16"
        )

    def test_missing_path_degrades_to_placeholders(self):
        key = identity_for_external(
            r"C:\nope\missing.gguf", "VibeVoice-1.5B", "sdpa",
        )
        assert key == (
            "external_VibeVoice-1.5B@missing.gguf@0@0_attn_sdpa_q4_0_dtype_auto"
        )
