"""Does a REPEATED external load hold two full models at once? (round 2)

The single-load path measures 1.00x file (plan §11.8, t5 §8.1), so a live
2x observation for a 16.6 GB checkpoint has to come from somewhere other
than one fat load. This file tests the candidate that survived: the loader
NODE re-executed with an IDENTICAL identity.

The mechanism it pins: the node's unload-before-load gate
(``evict_if_changed``) is a deliberate no-op for a same-key re-run, but the
node then rebuilt the model unconditionally. The outgoing weights stayed
reachable from two places — ComfyUI's node output cache and the live
patcher's handler (``ExternalVibeVoiceASRModelHandler.__init__`` stores
``self.model`` independently of the bundle) — so the rebuild overlapped a
second full copy.

The fix short-circuits the build on a same-key cache hit
(``model_registry.get_live_bundle``), which keeps exactly one model AND
keeps the node's output consistent with the patcher the consumer reuses.

Nothing here loads a large checkpoint — every model is a stub ``nn.Linear``.
"""

import gc
import os
import weakref

import pytest
import torch
from unittest.mock import patch

from ComfyUI_VibeVoice.modules import model_registry
from ComfyUI_VibeVoice.modules.model_registry import (
    FAMILY_ASR,
    clear_active_keys,
    get_live_bundle,
    identity_for_external,
    set_active,
)
from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_ASR_PATCHER_CACHE
from ComfyUI_VibeVoice.nodes.external_loader_node import VibeVoiceExternalLoaderNode

NODE = "ComfyUI_VibeVoice.nodes.external_loader_node"


@pytest.fixture(autouse=True)
def _isolated_state():
    VIBEVOICE_ASR_PATCHER_CACHE.clear()
    clear_active_keys()
    model_registry.clear_bundle_registry()
    yield
    VIBEVOICE_ASR_PATCHER_CACHE.clear()
    clear_active_keys()
    model_registry.clear_bundle_registry()


def _asr_bundle(tmp_path, tag):
    """A bundle shaped like load_external_vibevoice_asr_model output."""
    f = tmp_path / f"weights_{tag}.safetensors"
    f.write_bytes(b"fake-weights-" + tag.encode())
    st = os.stat(f)
    return {
        "model": torch.nn.Linear(8, 8),
        "processor": object(),
        "config": object(),
        "model_name": "VibeVoice-ASR",
        "source_path": str(f),
        "source_mtime_ns": st.st_mtime_ns,
        "source_size": st.st_size,
        "attention_mode": "sdpa",
        "use_llm_4bit": False,
        "dtype_str": "auto",
        "is_streaming": False,
        "is_asr": True,
    }


def _execute(weight_path, bundle, dtype="auto"):
    """Run the loader node once, with the heavy build stubbed."""
    with patch(f"{NODE}.folder_paths.get_full_path_or_raise",
               return_value=str(weight_path)), \
         patch(f"{NODE}.load_external_vibevoice_model", return_value=bundle):
        out = VibeVoiceExternalLoaderNode.execute(
            model_file=os.path.basename(str(weight_path)),
            config_name="VibeVoice-ASR",
            attention_mode="sdpa",
            quantize_llm_4bit=False,
            dtype=dtype,
        )
    return out.args[0]   # io.NodeOutput -> the bundle


def _key_for(bundle, dtype="auto"):
    return identity_for_external(
        bundle["source_path"], "VibeVoice-ASR", "sdpa",
        use_llm_4bit=False, dtype_str=dtype, prefix="asr_external",
    )


def _live_patcher_for(bundle, key):
    """A REAL VibeVoiceASRPatcher whose handler holds the model.

    A MagicMock would be a false negative: the real
    ``unpatch_model(destroy=True)`` is what nulls ``handler.model``.
    """
    from ComfyUI_VibeVoice.modules.asr_generation import (
        ExternalVibeVoiceASRModelHandler,
    )
    from ComfyUI_VibeVoice.modules.patcher import VibeVoiceASRPatcher

    handler = ExternalVibeVoiceASRModelHandler(
        bundle["model"], object(), "VibeVoice-ASR", bundle
    )
    with patch("comfy.model_patcher.ModelPatcher.__init__", return_value=None):
        patcher = VibeVoiceASRPatcher(
            handler, attention_mode="sdpa",
            load_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            size=1024,
        )
    patcher.model = handler
    patcher.pinned = set()
    patcher.load_device = torch.device("cpu")
    patcher.offload_device = torch.device("cpu")
    patcher.is_injected = False
    patcher.cache_key = key
    patcher.model_loaded_weight_memory = 0
    patcher.was_pinned = False
    patcher.model_lowvram = False
    patcher.model_usage = 0.0
    return patcher


class TestSameKeyReloadHoldsOneModel:
    """The regression this round fixes."""

    def test_identical_reload_does_not_build_a_second_model(self, tmp_path):
        b1 = _asr_bundle(tmp_path, "one")
        b2 = _asr_bundle(tmp_path, "two")   # would be a SECOND model

        out1 = _execute(b1["source_path"], b1)
        key = _key_for(b1)
        model_registry.register_model_bundle(key, out1)
        set_active(FAMILY_ASR, key)
        first_ref = weakref.ref(out1["model"])

        # A live patcher, as a real session has: it holds the model
        # independently of the bundle.
        VIBEVOICE_ASR_PATCHER_CACHE[key] = _live_patcher_for(out1, key)

        builds = []

        def _spy(**kw):
            builds.append(kw)
            return b2

        with patch(f"{NODE}.folder_paths.get_full_path_or_raise",
                   return_value=b1["source_path"]), \
             patch(f"{NODE}.load_external_vibevoice_model", side_effect=_spy):
            out2 = VibeVoiceExternalLoaderNode.execute(
                model_file="weights_one.safetensors", config_name="VibeVoice-ASR",
                attention_mode="sdpa", quantize_llm_4bit=False, dtype="auto",
            ).args[0]
        gc.collect()

        assert not builds, (
            "an identical re-execution must NOT rebuild: the outgoing weights "
            "are still held by the output cache and the live patcher, so a "
            "rebuild is the ~2x host-RAM spike on a 16.6 GB checkpoint"
        )
        assert out2["model"] is out1["model"], (
            "the node must return the resident model, which is also the one "
            "the reused patcher wraps"
        )
        assert first_ref() is not None, "the live model must survive the re-run"
        assert out1["model"] is not None, "the live bundle must stay intact"

    def test_the_returned_bundle_is_the_one_the_patcher_holds(self, tmp_path):
        """Output consistency: node output and consumer patcher must agree.

        The consumer reuses its patcher on a same-key run, so a rebuilt
        bundle would be silently discarded in favour of the already-loaded
        weights. Returning the resident model keeps them the same object.
        """
        b1 = _asr_bundle(tmp_path, "one")
        out1 = _execute(b1["source_path"], b1)
        key = _key_for(b1)
        model_registry.register_model_bundle(key, out1)
        set_active(FAMILY_ASR, key)
        patcher = _live_patcher_for(out1, key)
        VIBEVOICE_ASR_PATCHER_CACHE[key] = patcher

        out2 = _execute(b1["source_path"], _asr_bundle(tmp_path, "two"))
        assert out2["model"] is patcher.model.model, (
            "the node must return exactly the model the reused patcher wraps"
        )


class TestShortCircuitIsNarrow:
    """The cache hit must not swallow a real load."""

    def test_a_different_file_still_builds(self, tmp_path):
        b1 = _asr_bundle(tmp_path, "one")
        b2 = _asr_bundle(tmp_path, "two")

        out1 = _execute(b1["source_path"], b1)
        key1 = _key_for(b1)
        model_registry.register_model_bundle(key1, out1)
        set_active(FAMILY_ASR, key1)
        VIBEVOICE_ASR_PATCHER_CACHE[key1] = _live_patcher_for(out1, key1)
        first_ref = weakref.ref(out1["model"])

        out2 = _execute(b2["source_path"], b2)
        gc.collect()

        assert out2["model"] is b2["model"], "a different file must really load"
        assert first_ref() is None, (
            "switching models must still fully release the previous one"
        )

    def test_a_changed_dtype_is_a_different_key_and_still_builds(self, tmp_path):
        """The key includes the requested dtype, so a dtype switch reloads."""
        b1 = _asr_bundle(tmp_path, "one")
        out1 = _execute(b1["source_path"], b1)
        model_registry.register_model_bundle(_key_for(b1, "auto"), out1)
        set_active(FAMILY_ASR, _key_for(b1, "auto"))

        b2 = _asr_bundle(tmp_path, "two")
        out2 = _execute(b1["source_path"], b2, dtype="fp32")
        assert out2["model"] is b2["model"], (
            "a dtype change produces a new key and must genuinely reload"
        )

    def test_a_neutralized_bundle_is_not_reused(self, tmp_path):
        """A retired bundle must not be served from cache."""
        b1 = _asr_bundle(tmp_path, "one")
        out1 = _execute(b1["source_path"], b1)
        key = _key_for(b1)
        model_registry.register_model_bundle(key, out1)
        set_active(FAMILY_ASR, key)
        # Something else evicted it: the weights are gone, so a cache hit
        # would hand back a dead model.
        model_registry.release_model_bundles(key)
        assert get_live_bundle(key) is None

        b2 = _asr_bundle(tmp_path, "two")
        out2 = _execute(b1["source_path"], b2)
        assert out2["model"] is b2["model"], (
            "a neutralized bundle must not be reused; the node must rebuild"
        )

    def test_first_load_always_builds(self, tmp_path):
        b1 = _asr_bundle(tmp_path, "one")
        out1 = _execute(b1["source_path"], b1)
        assert out1["model"] is b1["model"]
        assert get_live_bundle(_key_for(b1)) is None, (
            "the loader node does not register; the consumer does"
        )


class TestShortCircuitIsNotVacuous:
    """Non-vacuity: remove the guard and the double build must return."""

    def test_neutering_the_guard_rebuilds_and_double_holds(self, tmp_path):
        b1 = _asr_bundle(tmp_path, "one")
        b2 = _asr_bundle(tmp_path, "two")

        out1 = _execute(b1["source_path"], b1)
        key = _key_for(b1)
        model_registry.register_model_bundle(key, out1)
        set_active(FAMILY_ASR, key)
        patcher = _live_patcher_for(out1, key)
        VIBEVOICE_ASR_PATCHER_CACHE[key] = patcher
        first_ref = weakref.ref(out1["model"])

        # Neuter the guard: the node no longer recognises the cache hit.
        builds = []
        with patch(f"{NODE}.get_live_bundle", return_value=None), \
             patch(f"{NODE}.folder_paths.get_full_path_or_raise",
                   return_value=b1["source_path"]), \
             patch(f"{NODE}.load_external_vibevoice_model",
                   side_effect=lambda **kw: (builds.append(kw), b2)[1]):
            out2 = VibeVoiceExternalLoaderNode.execute(
                model_file="weights_one.safetensors", config_name="VibeVoice-ASR",
                attention_mode="sdpa", quantize_llm_4bit=False, dtype="auto",
            ).args[0]
        gc.collect()

        assert builds, "with the guard neutered the node must rebuild"
        assert out2["model"] is b2["model"]
        assert first_ref() is not None, (
            "with the guard neutered the outgoing model stays alive (held by "
            "the output cache and the live patcher) while the new one is built "
            "— that is the double hold the guard prevents. If this failed, the "
            "fixture is not exercising the leak and the positive tests prove "
            "nothing."
        )
        assert patcher.model.model is not None, (
            "the stale patcher still wraps the outgoing weights"
        )


class TestCacheHitCannotServeStaleWeights:
    """The short-circuit is only safe because the key is file-identity-aware.

    ``identity_for_external`` mixes in the weight file's ``mtime_ns`` and
    ``size``, so replacing a checkpoint on disk produces a DIFFERENT key and
    the resident model is not served. If that ever regresses, a user could
    load a stale model after swapping the file — a silent-wrong-output bug,
    strictly worse than the RAM spike this fixes.
    """

    def test_replacing_the_weight_file_yields_a_different_key(self, tmp_path):
        import time

        b1 = _asr_bundle(tmp_path, "one")
        key_before = _key_for(b1)

        # Same path, new bytes: a user re-exporting the checkpoint.
        time.sleep(0.01)
        with open(b1["source_path"], "wb") as fh:
            fh.write(b"fake-weights-one-REPLACED-and-longer")
        b1_after = _asr_bundle(tmp_path, "one")
        b1_after["source_path"] = b1["source_path"]
        b1_after["source_mtime_ns"] = os.stat(b1["source_path"]).st_mtime_ns
        b1_after["source_size"] = os.stat(b1["source_path"]).st_size

        assert _key_for(b1_after) != key_before, (
            "a changed weight file MUST produce a different key, or the "
            "cache-hit short-circuit would serve stale weights"
        )

    def test_a_replaced_file_is_reloaded_not_served_from_cache(self, tmp_path):
        import time

        b1 = _asr_bundle(tmp_path, "one")
        out1 = _execute(b1["source_path"], b1)
        key = _key_for(b1)
        model_registry.register_model_bundle(key, out1)
        set_active(FAMILY_ASR, key)

        time.sleep(0.01)
        with open(b1["source_path"], "wb") as fh:
            fh.write(b"fake-weights-one-REPLACED-and-longer")

        b2 = _asr_bundle(tmp_path, "two")
        b2["source_path"] = b1["source_path"]
        st = os.stat(b1["source_path"])
        b2["source_mtime_ns"] = st.st_mtime_ns
        b2["source_size"] = st.st_size

        out2 = _execute(b2["source_path"], b2)
        assert out2["model"] is b2["model"], (
            "a replaced checkpoint must be genuinely reloaded, never served "
            "from the cache"
        )
