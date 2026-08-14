"""Behavioral tests for the VibeVoicePatcher lifecycle (NTH-003).

Uses the tiny stub handler from conftest (``_TinyHandler``) to exercise the
real ``patch_model`` / ``unpatch_model`` device transitions and cache keying
without loading the multi-gigabyte VibeVoice weights.
"""

import torch
from unittest.mock import patch

from ComfyUI_VibeVoice.modules.patcher import VibeVoicePatcher


def _super_patch():
    """Context manager mocking ComfyUI's ModelPatcher.patch_model."""
    return patch("comfy.model_patcher.ModelPatcher.patch_model")


def _super_unpatch():
    """Context manager mocking ComfyUI's ModelPatcher.unpatch_model."""
    return patch("comfy.model_patcher.ModelPatcher.unpatch_model")


class TestPatchMovesToLoadDevice:
    """patch_model must move the inner model onto the load device."""

    def test_patch_moves_to_load_device(self, tiny_patcher):
        with _super_patch():
            tiny_patcher.patch_model()

        # The inner Linear should now live on the load device (cpu in the stub).
        inner = tiny_patcher.model.model
        assert next(inner.parameters()).device.type == "cpu"
        assert tiny_patcher.is_loaded is True

    def test_patch_moves_to_explicit_device(self, tiny_handler):
        """If a device_to is given, the model lands on that device."""
        with patch("comfy.model_patcher.ModelPatcher.__init__"):
            patcher = VibeVoicePatcher(
                tiny_handler,
                attention_mode="sdpa",
                load_device=torch.device("cpu"),
                offload_device=torch.device("cpu"),
                size=1024,
            )
        patcher.load_device = torch.device("cpu")
        patcher.offload_device = torch.device("cpu")
        patcher.model = tiny_handler

        with _super_patch():
            patcher.patch_model(device_to=torch.device("cpu"))

        assert next(tiny_handler.model.parameters()).device.type == "cpu"
        assert patcher.is_loaded is True


class TestUnpatchNullsAndClearsCache:
    """unpatch_model must null the model and clear its cache entry."""

    def test_unpatch_nulls_and_clears_cache(self, tiny_patcher):
        from ComfyUI_VibeVoice.modules.loader import LOADED_MODELS_CACHE

        LOADED_MODELS_CACHE.clear()
        LOADED_MODELS_CACHE["tiny"] = ("model", "processor")

        with _super_patch():
            tiny_patcher.patch_model()
        assert tiny_patcher.is_loaded is True

        with _super_unpatch():
            tiny_patcher.unpatch_model(unpatch_weights=True)

        assert tiny_patcher.is_loaded is False
        assert "tiny" not in LOADED_MODELS_CACHE
        LOADED_MODELS_CACHE.clear()

    def test_unpatch_without_weights_keeps_cache(self, tiny_patcher):
        from ComfyUI_VibeVoice.modules.loader import LOADED_MODELS_CACHE

        LOADED_MODELS_CACHE.clear()
        LOADED_MODELS_CACHE["tiny"] = ("model", "processor")

        with _super_patch():
            tiny_patcher.patch_model()

        with _super_unpatch():
            tiny_patcher.unpatch_model(unpatch_weights=False)

        # Without unpatch_weights the model reference and cache are preserved.
        assert LOADED_MODELS_CACHE.get("tiny") is not None
        LOADED_MODELS_CACHE.clear()


class TestCacheKey:
    """The patcher reads its cache key from the handler it wraps."""

    def test_cache_key_comes_from_handler(self, tiny_handler):
        tiny_handler.cache_key = "asr_VibeVoice-ASR_attn_sdpa"
        with patch("comfy.model_patcher.ModelPatcher.__init__"):
            patcher = VibeVoicePatcher(
                tiny_handler,
                attention_mode="sdpa",
                load_device=torch.device("cpu"),
                offload_device=torch.device("cpu"),
                size=1024,
            )
        assert patcher.cache_key == "asr_VibeVoice-ASR_attn_sdpa"


class TestWarmOffload:
    """NTH-004: warm offload retains tensors for fast re-attach (no reload)."""

    def test_warm_offload_keeps_tensors(self, tiny_patcher):
        from ComfyUI_VibeVoice.modules.loader import LOADED_MODELS_CACHE

        LOADED_MODELS_CACHE.clear()
        LOADED_MODELS_CACHE["tiny"] = ("model", "processor")

        with _super_patch():
            tiny_patcher.patch_model()
        assert tiny_patcher.is_loaded is True

        with _super_unpatch():
            tiny_patcher.unpatch_model(unpatch_weights=True, warm=True)

        # Tensors are retained on the offload device; cache entry is preserved.
        assert tiny_patcher.is_loaded is False
        assert tiny_patcher.model.model is not None
        assert next(tiny_patcher.model.model.parameters()).device.type == "cpu"
        assert "tiny" in LOADED_MODELS_CACHE
        LOADED_MODELS_CACHE.clear()

    def test_warm_reload_skips_reinstantiation(self, tiny_patcher):
        calls = {"n": 0}
        original_load = tiny_patcher.model.load_model

        def counting_load(device, attn="sdpa"):
            calls["n"] += 1
            return original_load(device, attn)

        tiny_patcher.model.load_model = counting_load

        with _super_patch():
            tiny_patcher.patch_model()  # first load -> load_model called
        assert calls["n"] == 1
        assert tiny_patcher.is_loaded is True

        with _super_unpatch():
            tiny_patcher.unpatch_model(unpatch_weights=True, warm=True)
        assert tiny_patcher.is_loaded is False

        with _super_patch():
            tiny_patcher.patch_model()  # warm re-attach -> load_model NOT called again
        assert calls["n"] == 1
        assert tiny_patcher.is_loaded is True
