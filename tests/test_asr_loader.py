"""Tests for modules/asr_loader.py - ASR model loading and caching."""

import torch
import pytest
from unittest.mock import patch, MagicMock

from ComfyUI_VibeVoice.modules.asr_loader import (
    VibeVoiceASRModelHandler,
    VibeVoiceASRLoader,
    _is_native_asr_checkpoint,
    LOADED_ASR_MODELS_CACHE,
    cleanup_asr_models,
)
from ComfyUI_VibeVoice.modules.base_loader import BaseVibeVoiceLoader


@pytest.fixture(autouse=True)
def mock_tts_folder():
    """Register a 'tts' folder with folder_paths for tests."""
    import os
    import folder_paths
    tts_path = os.path.join(folder_paths.models_dir, "tts")
    if "tts" not in folder_paths.folder_names_and_paths:
        supported_exts = folder_paths.supported_pt_extensions.union({".safetensors", ".json"})
        folder_paths.folder_names_and_paths["tts"] = ([tts_path], supported_exts)
    yield


class TestVibeVoiceASRModelHandler:
    """Test VibeVoiceASRModelHandler class."""

    def test_handler_init(self):
        handler = VibeVoiceASRModelHandler("VibeVoice-ASR")
        assert handler.model_name == "VibeVoice-ASR"
        assert handler.model is None
        assert handler.processor is None

    def test_handler_size_calculation(self):
        handler = VibeVoiceASRModelHandler("VibeVoice-ASR")
        # size_gb=15.0 → 15.0 * 1024^3 bytes
        assert handler.size == int(15.0 * (1024**3))

    def test_handler_is_torch_module(self):
        handler = VibeVoiceASRModelHandler("VibeVoice-ASR")
        assert isinstance(handler, torch.nn.Module)


class TestCleanupASRModels:
    """Test cleanup_asr_models function."""

    def test_cleanup_clears_all(self):
        LOADED_ASR_MODELS_CACHE.clear()
        LOADED_ASR_MODELS_CACHE["key1"] = "model1"
        LOADED_ASR_MODELS_CACHE["key2"] = "model2"

        with patch("ComfyUI_VibeVoice.modules.asr_loader.model_management"):
            cleanup_asr_models(keep_cache_key=None)

        assert len(LOADED_ASR_MODELS_CACHE) == 0

    def test_cleanup_keeps_specified_key(self):
        LOADED_ASR_MODELS_CACHE.clear()
        LOADED_ASR_MODELS_CACHE["key1"] = "model1"
        LOADED_ASR_MODELS_CACHE["key2"] = "model2"

        with patch("ComfyUI_VibeVoice.modules.asr_loader.model_management"):
            cleanup_asr_models(keep_cache_key="key1")

        assert "key1" in LOADED_ASR_MODELS_CACHE
        assert "key2" not in LOADED_ASR_MODELS_CACHE


class TestASRLoaderBaseInheritance:
    """IMP-004 / CRIT-001: ASR loader shares the base loader + path logic."""

    def test_asr_loader_is_base_loader(self):
        assert issubclass(VibeVoiceASRLoader, BaseVibeVoiceLoader)

    def test_resolve_paths_official_uses_base(self, tmp_path):
        with patch("ComfyUI_VibeVoice.modules.base_loader.folder_paths") as mock_fp, \
             patch.object(BaseVibeVoiceLoader, "_ensure_downloaded"):
            mock_fp.get_folder_paths.return_value = [str(tmp_path)]
            model_path, tokenizer_repo = VibeVoiceASRLoader._resolve_model_paths("VibeVoice-ASR")

        assert model_path == str(tmp_path / "VibeVoice" / "VibeVoice-ASR")
        assert tokenizer_repo == "Qwen/Qwen2.5-7B"

    def test_resolve_paths_local_dir(self, tmp_path):
        from ComfyUI_VibeVoice.modules.model_info import AVAILABLE_VIBEVOICE_MODELS

        with patch.dict(AVAILABLE_VIBEVOICE_MODELS, {"MyASR": {"type": "local_dir", "path": str(tmp_path)}}, clear=True):
            model_path, tokenizer_repo = VibeVoiceASRLoader._resolve_model_paths("MyASR")

        assert model_path == str(tmp_path)
        assert tokenizer_repo == "Qwen/Qwen2.5-7B"


class TestNativeAsrCheckpointDetection:
    """model_type-based branch: vibevoice_asr → native path, else vendored."""

    def _write_config(self, path, model_type):
        import json
        import os
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"model_type": model_type}, f)

    def test_native_asr_config_detected(self, tmp_path):
        self._write_config(str(tmp_path), "vibevoice_asr")
        assert _is_native_asr_checkpoint(str(tmp_path)) is True

    def test_vendored_asr_config_not_native(self, tmp_path):
        # Vendored checkpoints (streaming family, original VibeVoice-ASR)
        # share model_type "vibevoice" with TTS.
        self._write_config(str(tmp_path), "vibevoice")
        assert _is_native_asr_checkpoint(str(tmp_path)) is False

    def test_missing_config_not_native(self, tmp_path):
        assert _is_native_asr_checkpoint(str(tmp_path)) is False

    def test_load_model_branches_to_native(self, tmp_path):
        """A vibevoice_asr checkpoint routes through _load_native (which
        builds the model via transformers AutoProcessor/AutoModel), not the
        vendored from_pretrained path."""
        from ComfyUI_VibeVoice.modules.model_info import AVAILABLE_VIBEVOICE_MODELS

        self._write_config(str(tmp_path), "vibevoice_asr")
        LOADED_ASR_MODELS_CACHE.clear()
        fake_model, fake_processor = MagicMock(), MagicMock()
        with patch.dict(
            AVAILABLE_VIBEVOICE_MODELS,
            {"VibeVoice-ASR-HF": {"type": "local_dir", "path": str(tmp_path)}},
            clear=True,
        ), patch.object(
            VibeVoiceASRLoader, "_resolve_model_paths",
            return_value=(str(tmp_path), "Qwen/Qwen2.5-7B"),
        ), patch.object(
            VibeVoiceASRLoader, "_load_native",
            return_value=(fake_model, fake_processor),
        ) as mock_native, patch(
            "ComfyUI_VibeVoice.modules.asr_loader.VibeVoiceASRForConditionalGeneration.from_pretrained",
        ) as mock_vendored:
            model, processor = VibeVoiceASRLoader.load_model("VibeVoice-ASR-HF", "cpu")

        assert model is fake_model
        assert processor is fake_processor
        mock_native.assert_called_once()
        mock_vendored.assert_not_called()
        # The static loader is a PURE BUILDER: it must NOT register any cache
        # entry (cache ownership belongs to load_asr_model_patched under the
        # patcher key — the leak fix).
        assert len(LOADED_ASR_MODELS_CACHE) == 0
        LOADED_ASR_MODELS_CACHE.clear()


class TestASRLoadDeviceContract:
    """DF-003 for ASR: models load CPU-resident; the patcher owns the single
    H2D transfer after ComfyUI's VRAM arbitration. A force ``.to(device)`` or
    ``device_map`` inside the loader/handler bypassed that arbitration and
    OOM'd the user's 16 GB GPU at 15.14/15.99 GiB (2026-09-09)."""

    def test_handler_load_model_never_moves_device(self):
        """VibeVoiceASRModelHandler.load_model performs NO device move — the
        loader result is stored as-is."""
        handler = VibeVoiceASRModelHandler("VibeVoice-ASR-HF")
        fake_model = MagicMock()
        fake_model.to.return_value = fake_model
        with patch(
            "ComfyUI_VibeVoice.modules.asr_loader.VibeVoiceASRLoader.load_model",
            return_value=(fake_model, MagicMock()),
        ):
            handler.load_model(torch.device("cpu"), "auto", "sdpa")
        fake_model.to.assert_not_called()
        assert handler.model is fake_model

    def test_native_load_has_no_device_map_or_force_move(self):
        """_load_native passes no device_map and never calls model.to()."""
        LOADED_ASR_MODELS_CACHE.clear()
        fake_model = MagicMock()
        fake_model.to.return_value = fake_model
        fake_processor = MagicMock()
        with patch.dict(
            "ComfyUI_VibeVoice.modules.asr_loader.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-ASR-HF": {"type": "local_dir", "path": "some/dir"}},
        ), patch(
            "ComfyUI_VibeVoice.modules.asr_loader._is_native_asr_checkpoint",
            return_value=True,
        ), patch(
            "ComfyUI_VibeVoice.modules.asr_loader.VibeVoiceASRLoader._resolve_model_paths",
            return_value=("some/dir", "Qwen/Qwen2.5-7B"),
        ), patch(
            "transformers.AutoProcessor.from_pretrained", return_value=fake_processor,
        ), patch(
            "transformers.VibeVoiceAsrForConditionalGeneration.from_pretrained",
            return_value=fake_model,
        ) as mock_from_pretrained, patch(
            "ComfyUI_VibeVoice.modules.comfy_stream.convert_tree_for_streaming",
        ) as mock_convert:
            model, processor = VibeVoiceASRLoader.load_model(
                "VibeVoice-ASR-HF", torch.device("cpu"), "auto", "sdpa"
            )

        kwargs = mock_from_pretrained.call_args.kwargs
        assert "device_map" not in kwargs
        fake_model.to.assert_not_called()
        mock_convert.assert_called_once()
        assert model is fake_model
        assert processor is fake_processor

    def test_native_load_converts_tree_for_streaming(self):
        """The native tree gets convert_tree_for_streaming so core's
        partial-load machinery can stream it under a tight VRAM budget."""
        LOADED_ASR_MODELS_CACHE.clear()
        fake_model = MagicMock()
        with patch.dict(
            "ComfyUI_VibeVoice.modules.asr_loader.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-ASR-HF": {"type": "local_dir", "path": "some/dir"}},
        ), patch(
            "ComfyUI_VibeVoice.modules.asr_loader._is_native_asr_checkpoint",
            return_value=True,
        ), patch(
            "ComfyUI_VibeVoice.modules.asr_loader.VibeVoiceASRLoader._resolve_model_paths",
            return_value=("some/dir", "Qwen/Qwen2.5-7B"),
        ), patch(
            "transformers.AutoProcessor.from_pretrained", return_value=MagicMock(),
        ), patch(
            "transformers.VibeVoiceAsrForConditionalGeneration.from_pretrained",
            return_value=fake_model,
        ), patch(
            "ComfyUI_VibeVoice.modules.comfy_stream.convert_tree_for_streaming",
        ) as mock_convert:
            VibeVoiceASRLoader.load_model("VibeVoice-ASR-HF", torch.device("cpu"), "auto", "sdpa")
        mock_convert.assert_called_once_with(fake_model)


class TestASRSingleCacheOwnership:
    """Leak fix (2026-09-09): the static loader must be a PURE BUILDER.
    Registering the built model under a third key format created a cache
    entry that survived patcher destroy and pinned the superseded 17 GB
    tree in RAM/VRAM."""

    def _clear(self):
        LOADED_ASR_MODELS_CACHE.clear()

    def test_static_loader_writes_no_cache_entries_native(self, tmp_path):
        import json
        import os
        os.makedirs(tmp_path, exist_ok=True)
        with open(os.path.join(tmp_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"model_type": "vibevoice_asr"}, f)
        self._clear()
        fake_model, fake_processor = MagicMock(), MagicMock()
        with patch.dict(
            "ComfyUI_VibeVoice.modules.asr_loader.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-ASR-HF": {"type": "local_dir", "path": str(tmp_path)}},
        ), patch(
            "transformers.AutoProcessor.from_pretrained", return_value=fake_processor,
        ), patch(
            "transformers.VibeVoiceAsrForConditionalGeneration.from_pretrained",
            return_value=fake_model,
        ):
            VibeVoiceASRLoader.load_model("VibeVoice-ASR-HF", "cpu", "auto", "sdpa")
        assert len(LOADED_ASR_MODELS_CACHE) == 0
        self._clear()

    def test_static_loader_writes_no_cache_entries_vendored(self):
        self._clear()
        fake_model, fake_processor = MagicMock(), MagicMock()
        with patch.dict(
            "ComfyUI_VibeVoice.modules.asr_loader.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-ASR": {"type": "local_dir", "path": "some/dir"}},
        ), patch(
            "ComfyUI_VibeVoice.modules.asr_loader._is_native_asr_checkpoint",
            return_value=False,
        ), patch(
            "ComfyUI_VibeVoice.modules.asr_loader.VibeVoiceASRLoader._resolve_model_paths",
            return_value=("some/dir", "Qwen/Qwen2.5-7B"),
        ), patch(
            "ComfyUI_VibeVoice.modules.asr_loader.VibeVoiceASRProcessor.from_pretrained",
            return_value=fake_processor,
        ), patch(
            "ComfyUI_VibeVoice.modules.asr_loader.VibeVoiceASRForConditionalGeneration.from_pretrained",
            return_value=fake_model,
        ):
            VibeVoiceASRLoader.load_model("VibeVoice-ASR", "cpu", "auto", "sdpa")
        assert len(LOADED_ASR_MODELS_CACHE) == 0
        self._clear()

    def test_destroy_evicts_the_only_reference(self):
        """End-to-end with the real patched flow: the patched-flow cache
        entry must be the ONLY one, and destroy must leave BOTH caches
        empty (today a stale static-format entry survives destroy)."""
        from ComfyUI_VibeVoice.modules.asr_loader import VibeVoiceASRModelHandler
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_ASR_PATCHER_CACHE
        from ComfyUI_VibeVoice.modules.asr_generation import load_asr_model_patched

        self._clear()
        VIBEVOICE_ASR_PATCHER_CACHE.clear()

        def fake_load(self, device, attention_mode="sdpa"):
            self.model = torch.nn.Linear(8, 8)
            self.processor = object()

        try:
            with patch.object(VibeVoiceASRModelHandler, "load_model", fake_load), \
                 patch("comfy.model_patcher.ModelPatcher.patch_model"), \
                 patch("ComfyUI_VibeVoice.modules.asr_generation.model_management.load_models_gpu",
                       side_effect=lambda models: models[0].patch_model()), \
                 patch("ComfyUI_VibeVoice.modules.asr_generation.get_torch_device",
                       return_value=torch.device("cpu")), \
                 patch("ComfyUI_VibeVoice.modules.asr_generation.get_offload_device",
                       return_value=torch.device("cpu")):
                patcher, model, processor = load_asr_model_patched(
                    model_name="VibeVoice-ASR", device="cpu", dtype="fp32", attention_mode="sdpa"
                )
        finally:
            # One entry only, under the patcher key.
            assert list(LOADED_ASR_MODELS_CACHE.keys()) == ["asr_VibeVoice-ASR_attn_sdpa"]

            # Destroy via the real eviction sequence (evict_patcher:
            # unregister → unpatch(destroy) → patcher-cache pop).
            patcher.unpatch_model(unpatch_weights=True, destroy=True)
            VIBEVOICE_ASR_PATCHER_CACHE.pop("asr_VibeVoice-ASR_attn_sdpa", None)

            assert len(LOADED_ASR_MODELS_CACHE) == 0
            assert len(VIBEVOICE_ASR_PATCHER_CACHE) == 0
            self._clear()
            VIBEVOICE_ASR_PATCHER_CACHE.clear()


class TestASRSageAttentionParity:
    """Sage is EXCLUDED from the ASR path (modules/attention_utils
    .ASR_EXCLUDED_ATTENTION_MODES) and downgraded to sdpa before the model is
    built.

    Reason: the ASR processor left-pads every batch to the longest utterance, so
    a prefill step arrives with a real additive (B,1,S,S) mask, and sageattn has
    no attn_mask parameter — `sage_attention_forward` used the mask only to pick
    `is_causal` and then dropped it, letting every query attend to the pad
    columns. These tests pin the exclusion so it cannot be re-added by accident;
    the TTS family still uses sage and is unaffected."""

    @staticmethod
    def _native_patches(fake_model, fake_processor):
        """Build a real native-checkpoint dir (config.json model_type
        vibevoice_asr) and the patches to route the loader's native branch.
        Returns (tmp_path, registry_patch, processor_patch, fpg_patch,
        convert_patch) — route via _resolve_model_paths → tmp_path."""
        import json
        import os
        import tempfile

        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"model_type": "vibevoice_asr"}, f)

        return tmp, patch.dict(
            "ComfyUI_VibeVoice.modules.asr_loader.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-ASR-HF": {"type": "local_dir", "path": tmp}},
        ), patch(
            "transformers.AutoProcessor.from_pretrained", return_value=fake_processor,
        ), patch(
            "transformers.VibeVoiceAsrForConditionalGeneration.from_pretrained",
            return_value=fake_model,
        ), patch(
            "ComfyUI_VibeVoice.modules.comfy_stream.convert_tree_for_streaming",
        )

    def test_native_branch_never_applies_sage(self):
        """A sage request on the native ASR branch must not reach the kernel."""
        fake_model, fake_processor = MagicMock(), MagicMock()
        tmp, registry, proc, fpg, convert = self._native_patches(fake_model, fake_processor)
        with registry, proc, fpg, convert, \
             patch("ComfyUI_VibeVoice.modules.asr_loader.set_sage_attention") as mock_sage, \
             patch("ComfyUI_VibeVoice.modules.asr_loader.check_sage_attention_compatible",
                   return_value=True), \
             patch("ComfyUI_VibeVoice.modules.asr_loader.VibeVoiceASRLoader._resolve_model_paths",
                   return_value=(tmp, "Qwen/Qwen2.5-7B")):
            VibeVoiceASRLoader.load_model("VibeVoice-ASR-HF", "cpu", "auto", "sage")

        assert not mock_sage.called, (
            "sage must never be applied on the ASR path: the left-padded prefill "
            "mask cannot be forwarded to sageattn and would be silently dropped."
        )

    def test_vendored_branch_never_applies_sage(self):
        fake_model, fake_processor = MagicMock(), MagicMock()
        with patch.dict(
            "ComfyUI_VibeVoice.modules.asr_loader.AVAILABLE_VIBEVOICE_MODELS",
            {"VibeVoice-ASR": {"type": "local_dir", "path": "some/dir"}},
        ), patch(
            "ComfyUI_VibeVoice.modules.asr_loader._is_native_asr_checkpoint",
            return_value=False,
        ), patch(
            "ComfyUI_VibeVoice.modules.asr_loader.VibeVoiceASRLoader._resolve_model_paths",
            return_value=("some/dir", "Qwen/Qwen2.5-7B"),
        ), patch(
            "ComfyUI_VibeVoice.modules.asr_loader.VibeVoiceASRProcessor.from_pretrained",
            return_value=fake_processor,
        ), patch(
            "ComfyUI_VibeVoice.modules.asr_loader.VibeVoiceASRForConditionalGeneration.from_pretrained",
            return_value=fake_model,
        ), patch(
            "ComfyUI_VibeVoice.modules.comfy_stream.convert_tree_for_streaming",
        ), patch(
            "ComfyUI_VibeVoice.modules.asr_loader.set_sage_attention") as mock_sage, \
        patch("ComfyUI_VibeVoice.modules.asr_loader.check_sage_attention_compatible",
              return_value=True):
            VibeVoiceASRLoader.load_model("VibeVoice-ASR", "cpu", "auto", "sage")
        assert not mock_sage.called, (
            "the exclusion must reach the vendored ASR branch too, not just the "
            "native one."
        )

    def test_native_branch_sage_loads_with_sdpa_backend(self):
        """attn_implementation for the native load is the FOR-LOAD mapping
        (sage→sdpa); the sage post-load patch is unreachable on ASR."""
        fake_model, fake_processor = MagicMock(), MagicMock()
        tmp, registry, proc, fpg, convert = self._native_patches(fake_model, fake_processor)
        with registry, proc, fpg as mock_fpg, convert, \
             patch("ComfyUI_VibeVoice.modules.asr_loader.set_sage_attention"), \
             patch("ComfyUI_VibeVoice.modules.asr_loader.check_sage_attention_compatible",
                   return_value=True), \
             patch("ComfyUI_VibeVoice.modules.asr_loader.VibeVoiceASRLoader._resolve_model_paths",
                   return_value=(tmp, "Qwen/Qwen2.5-7B")):
            VibeVoiceASRLoader.load_model("VibeVoice-ASR-HF", "cpu", "auto", "sage")
        # The native per-subconfig dict routes sdpa to the LM (the "" key);
        # the tokenizer encoders always load eager.
        attn = mock_fpg.call_args.kwargs.get("attn_implementation")
        if isinstance(attn, dict):
            assert attn[""] == "sdpa"
            assert attn["acoustic_tokenizer_encoder_config"] == "eager"
        else:
            assert attn == "sdpa"

    def test_sage_on_incompatible_hardware_falls_back_instead_of_raising(self):
        """The ASR path must not raise on a machine that cannot run sage.

        The old contract raised RuntimeError("Incompatible hardware/setup")
        from inside the loader. The exclusion resolves sage to sdpa first, so
        the request is honoured as the nearest usable backend instead.
        """
        fake_model, fake_processor = MagicMock(), MagicMock()
        tmp, registry, proc, fpg, convert = self._native_patches(fake_model, fake_processor)
        with registry, proc, fpg, convert, \
             patch("ComfyUI_VibeVoice.modules.asr_loader.check_sage_attention_compatible",
                   return_value=False), \
             patch("ComfyUI_VibeVoice.modules.asr_loader.set_sage_attention"), \
             patch("ComfyUI_VibeVoice.modules.asr_loader.VibeVoiceASRLoader._resolve_model_paths",
                   return_value=(tmp, "Qwen/Qwen2.5-7B")):
            VibeVoiceASRLoader.load_model("VibeVoice-ASR-HF", "cpu", "auto", "sage")

    def test_sdpa_mode_never_calls_sage(self):
        fake_model, fake_processor = MagicMock(), MagicMock()
        tmp, registry, proc, fpg, convert = self._native_patches(fake_model, fake_processor)
        with registry, proc, fpg, convert, \
             patch("ComfyUI_VibeVoice.modules.asr_loader.set_sage_attention") as mock_sage, \
             patch("ComfyUI_VibeVoice.modules.asr_loader.VibeVoiceASRLoader._resolve_model_paths",
                   return_value=(tmp, "Qwen/Qwen2.5-7B")):
            VibeVoiceASRLoader.load_model("VibeVoice-ASR-HF", "cpu", "auto", "sdpa")
        mock_sage.assert_not_called()


class TestASRGenerationConfigNeutralization:
    """The ASR-HF checkpoint ships generation_config.json with BOTH
    max_length and max_new_tokens (=32768); transformers warns on every
    generate() when the node's max_new_tokens meets that preset. The loader
    neutralizes the preset (max_length/min_length → None) at load time —
    matching the TTS models, which carry no preset."""

    @staticmethod
    def _fake_model(generation_config=None):
        model = MagicMock()
        model.generation_config = generation_config
        return model

    def _native_patches(self, fake_model):
        fake_processor = MagicMock()
        return TestASRSageAttentionParity._native_patches(fake_model, fake_processor)

    def test_native_load_clears_preset_max_length(self):
        from types import SimpleNamespace
        gen_cfg = SimpleNamespace(max_length=32768, min_length=0, max_new_tokens=32768)
        fake_model = self._fake_model(gen_cfg)
        tmp, registry, proc, fpg, convert = self._native_patches(fake_model)
        with registry, proc, fpg, convert, \
             patch("ComfyUI_VibeVoice.modules.asr_loader.VibeVoiceASRLoader._resolve_model_paths",
                   return_value=(tmp, "Qwen/Qwen2.5-7B")):
            model, _ = VibeVoiceASRLoader.load_model("VibeVoice-ASR-HF", "cpu", "auto", "sdpa")
        assert gen_cfg.max_length is None
        assert gen_cfg.min_length is None
        assert gen_cfg.max_new_tokens == 32768
        assert model is fake_model

    def test_no_preset_leaves_config_untouched(self):
        from types import SimpleNamespace
        gen_cfg = SimpleNamespace(max_length=None, min_length=None, max_new_tokens=8192)
        fake_model = self._fake_model(gen_cfg)
        tmp, registry, proc, fpg, convert = self._native_patches(fake_model)
        with registry, proc, fpg, convert, \
             patch("ComfyUI_VibeVoice.modules.asr_loader.VibeVoiceASRLoader._resolve_model_paths",
                   return_value=(tmp, "Qwen/Qwen2.5-7B")):
            VibeVoiceASRLoader.load_model("VibeVoice-ASR-HF", "cpu", "auto", "sdpa")
        assert gen_cfg.max_length is None
        assert gen_cfg.min_length is None
        assert gen_cfg.max_new_tokens == 8192

    def test_missing_generation_config_is_tolerated(self):
        fake_model = MagicMock()
        del fake_model.generation_config  # no generation_config attr at all
        tmp, registry, proc, fpg, convert = self._native_patches(fake_model)
        with registry, proc, fpg, convert, \
             patch("ComfyUI_VibeVoice.modules.asr_loader.VibeVoiceASRLoader._resolve_model_paths",
                   return_value=(tmp, "Qwen/Qwen2.5-7B")):
            model, _ = VibeVoiceASRLoader.load_model("VibeVoice-ASR-HF", "cpu", "auto", "sdpa")
        assert model is fake_model
