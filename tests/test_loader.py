"""Tests for modules/loader.py - Model loading and caching."""

import os
import json
import torch
import pytest
from unittest.mock import patch, MagicMock

from ComfyUI_VibeVoice.modules.loader import (
    VibeVoiceModelHandler,
    VibeVoiceLoader,
    LOADED_MODELS_CACHE,
    cleanup_old_models,
)
from ComfyUI_VibeVoice.modules.base_loader import BaseVibeVoiceLoader
from ComfyUI_VibeVoice.modules.model_info import AVAILABLE_VIBEVOICE_MODELS, MODEL_CONFIGS


@pytest.fixture(autouse=True)
def mock_tts_folder():
    """Register a 'tts' folder with folder_paths for tests."""
    import folder_paths
    tts_path = os.path.join(folder_paths.models_dir, "tts")
    if "tts" not in folder_paths.folder_names_and_paths:
        supported_exts = folder_paths.supported_pt_extensions.union({".safetensors", ".json"})
        folder_paths.folder_names_and_paths["tts"] = ([tts_path], supported_exts)
    yield
    # Cleanup not needed — folder_paths is global


class TestVibeVoiceModelHandler:
    """Test VibeVoiceModelHandler class."""

    def test_handler_init(self):
        handler = VibeVoiceModelHandler("VibeVoice-1.5B", attention_mode="sdpa", use_llm_4bit=False)
        assert handler.model_pack_name == "VibeVoice-1.5B"
        assert handler.attention_mode == "sdpa"
        assert handler.use_llm_4bit is False
        assert handler.model is None
        assert handler.processor is None

    def test_handler_has_device_attribute(self):
        """Handler must have a device attribute for ComfyUI's ModelPatcher."""
        handler = VibeVoiceModelHandler("VibeVoice-1.5B")
        assert hasattr(handler, "device")
        # Initially None — ModelPatcher.__init__ will set it to offload_device

    def test_handler_cache_key(self):
        handler = VibeVoiceModelHandler("VibeVoice-1.5B", attention_mode="sdpa", use_llm_4bit=False)
        assert handler.cache_key == "VibeVoice-1.5B_attn_sdpa_q4_0"

    def test_handler_cache_key_4bit(self):
        handler = VibeVoiceModelHandler("VibeVoice-7B", attention_mode="eager", use_llm_4bit=True)
        assert handler.cache_key == "VibeVoice-7B_attn_eager_q4_1"

    def test_handler_size_calculation(self):
        handler = VibeVoiceModelHandler("VibeVoice-1.5B")
        assert handler.size == int(3.0 * (1024**3))

    def test_handler_size_large(self):
        handler = VibeVoiceModelHandler("VibeVoice-7B")
        assert handler.size == int(17.4 * (1024**3))

    def test_handler_is_torch_module(self):
        handler = VibeVoiceModelHandler("VibeVoice-1.5B")
        assert isinstance(handler, torch.nn.Module)


class TestHandlerSizeRefinement:
    """Plan 2026-08-18, Phase 6 (D7/RC-7): handler.size is refined from the
    real parameters after load, replacing the config-based size_gb estimate."""

    def test_handler_size_refined_after_load(self):
        """After load_model, handler.size equals the real parameter byte total."""
        handler = VibeVoiceModelHandler("VibeVoice-1.5B")
        # Config-based estimate before load.
        assert handler.size == int(3.0 * (1024**3))

        # A tiny model with a known parameter byte total.
        tiny = torch.nn.Linear(16, 16, bias=False)  # 16*16 = 256 floats
        expected_bytes = 256 * tiny.weight.element_size()

        with patch.object(VibeVoiceLoader, "load_model", return_value=(tiny, MagicMock())):
            handler.load_model(torch.device("cpu"), attention_mode="sdpa")

        assert handler.size == expected_bytes

    def test_handler_size_kept_when_params_empty(self):
        """If the model has no parameters, the config estimate is kept."""
        handler = VibeVoiceModelHandler("VibeVoice-1.5B")
        original_size = handler.size

        empty = torch.nn.Module()  # no parameters

        with patch.object(VibeVoiceLoader, "load_model", return_value=(empty, MagicMock())):
            handler.load_model(torch.device("cpu"), attention_mode="sdpa")

        assert handler.size == original_size

    def test_handler_size_kept_on_exception(self):
        """If parameter iteration raises, the config estimate is kept."""
        handler = VibeVoiceModelHandler("VibeVoice-1.5B")
        original_size = handler.size

        broken = MagicMock()
        broken.parameters.side_effect = RuntimeError("boom")

        with patch.object(VibeVoiceLoader, "load_model", return_value=(broken, MagicMock())):
            handler.load_model(torch.device("cpu"), attention_mode="sdpa")

        assert handler.size == original_size


class TestVibeVoiceLoaderResolvePaths:
    """Test VibeVoiceLoader._resolve_model_paths."""

    def test_resolve_local_dir(self, tmp_path):
        model_dir = tmp_path / "MyLocalModel"
        model_dir.mkdir()
        with patch("ComfyUI_VibeVoice.modules.loader.AVAILABLE_VIBEVOICE_MODELS", {
            "MyLocalModel": {"type": "local_dir", "path": str(model_dir)}
        }):
            model_path, config_path, preproc_path, tokenizer_dir = \
                VibeVoiceLoader._resolve_model_paths("MyLocalModel")
            assert model_path == str(model_dir)
            assert config_path == str(model_dir / "config.json")

    def test_resolve_standalone(self, tmp_path):
        model_file = tmp_path / "model.safetensors"
        with patch("ComfyUI_VibeVoice.modules.loader.AVAILABLE_VIBEVOICE_MODELS", {
            "model": {"type": "standalone", "path": str(model_file)}
        }):
            model_path, config_path, preproc_path, tokenizer_dir = \
                VibeVoiceLoader._resolve_model_paths("model")
            assert model_path is None
            assert config_path.endswith(".config.json")
            assert tokenizer_dir == str(tmp_path)

    def test_resolve_official(self, tmp_path):
        """Test official model path resolution (with download mocked)."""
        with patch("ComfyUI_VibeVoice.modules.loader.AVAILABLE_VIBEVOICE_MODELS", {
            "TestModel": {"type": "official", "repo_id": "test/repo"}
        }), patch("ComfyUI_VibeVoice.modules.loader.folder_paths") as mock_fp, \
             patch("huggingface_hub.snapshot_download") as mock_dl, \
             patch("os.path.exists", return_value=True):
            mock_fp.get_folder_paths.return_value = [str(tmp_path)]
            model_path, config_path, preproc_path, tokenizer_dir = \
                VibeVoiceLoader._resolve_model_paths("TestModel")
            assert "TestModel" in model_path
            assert config_path.endswith("config.json")
            # snapshot_download should NOT be called since we mocked os.path.exists to True
            mock_dl.assert_not_called()


class TestVibeVoiceLoaderInstantiateModel:
    """Test VibeVoiceLoader._instantiate_model."""

    def test_instantiate_model_non_streaming(self):
        """Test that non-streaming model is instantiated correctly."""
        config = MagicMock()
        config.decoder_config = MagicMock()
        config.torch_dtype = None
        with patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceForConditionalGeneration") as mock_cls:
            mock_model = MagicMock()
            mock_cls.return_value = mock_model

            model = VibeVoiceLoader._instantiate_model(
                config=config,
                is_streaming=False,
                attn_implementation="sdpa",
                final_load_dtype=torch.float16,
            )

            mock_cls.assert_called_once_with(config)
            assert model == mock_model
            # Verify attn_implementation was set on decoder_config
            assert config.decoder_config._attn_implementation == "sdpa"

    def test_instantiate_model_streaming(self):
        """Test that streaming model uses the correct class."""
        config = MagicMock()
        config.decoder_config = MagicMock()
        config.torch_dtype = None
        with patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceStreamingForConditionalGenerationInference") as mock_cls:
            mock_model = MagicMock()
            mock_cls.return_value = mock_model

            model = VibeVoiceLoader._instantiate_model(
                config=config,
                is_streaming=True,
                attn_implementation="sdpa",
                final_load_dtype=torch.float16,
            )

            mock_cls.assert_called_once_with(config)
            assert model == mock_model

    def test_instantiate_model_sets_dtype_on_config(self):
        """Test that dtype is set on config."""
        config = MagicMock()
        config.decoder_config = MagicMock()
        config.torch_dtype = None

        with patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceForConditionalGeneration"):
            VibeVoiceLoader._instantiate_model(
                config=config,
                is_streaming=False,
                attn_implementation="eager",
                final_load_dtype=torch.bfloat16,
            )

            assert config.torch_dtype == torch.bfloat16
            assert config.decoder_config.torch_dtype == torch.bfloat16

    def test_instantiate_model_real_config_no_deprecation(self):
        """With a REAL transformers PretrainedConfig (v5: torch_dtype is a
        deprecated property), _instantiate_model must record the dtype on the
        canonical ``dtype`` attribute and emit no torch_dtype deprecation.
        A capture handler is attached directly to the emitting transformers
        logger (its warning_once is lru_cache-wrapped, so the cache is
        cleared to keep the absence assertion non-vacuous)."""
        import logging as pylogging
        from transformers import PretrainedConfig

        config = PretrainedConfig()
        config.decoder_config = PretrainedConfig()

        logger = pylogging.getLogger("transformers.configuration_utils")
        records = []

        class _Capture(pylogging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Capture(level=pylogging.WARNING)
        logger.addHandler(handler)
        pylogging.Logger.warning_once.cache_clear()
        try:
            with patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceForConditionalGeneration"):
                VibeVoiceLoader._instantiate_model(
                    config=config,
                    is_streaming=False,
                    attn_implementation="eager",
                    final_load_dtype=torch.bfloat16,
                )
        finally:
            logger.removeHandler(handler)
            pylogging.Logger.warning_once.cache_clear()

        assert config.dtype == torch.bfloat16
        assert config.decoder_config.dtype == torch.bfloat16
        assert not any(
            "`torch_dtype` is deprecated" in m for m in records
        ), "torch_dtype deprecation warning was emitted"


class TestResolveCheckpointPath:
    """Test VibeVoiceLoader._resolve_checkpoint_path."""

    def test_standalone_returns_path_directly(self, tmp_path):
        """Standalone model returns the path directly with is_sharded=False."""
        ckpt_file = tmp_path / "model.safetensors"
        ckpt_file.write_text("dummy")
        model_info = {"path": str(ckpt_file)}
        result_path, is_sharded = VibeVoiceLoader._resolve_checkpoint_path(
            model_path=None, model_type="standalone", model_info=model_info
        )
        assert result_path == str(ckpt_file)
        assert is_sharded is False

    def test_standalone_raises_if_file_not_found(self):
        """Standalone model raises FileNotFoundError if file doesn't exist."""
        model_info = {"path": "/nonexistent/path.safetensors"}
        with pytest.raises(FileNotFoundError, match="Standalone checkpoint not found"):
            VibeVoiceLoader._resolve_checkpoint_path(
                model_path=None, model_type="standalone", model_info=model_info
            )

    def test_official_single_safetensors(self, tmp_path):
        """Official model with single model.safetensors file."""
        model_dir = tmp_path / "TestModel"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_text("dummy")
        result_path, is_sharded = VibeVoiceLoader._resolve_checkpoint_path(
            model_path=str(model_dir), model_type="official", model_info={}
        )
        assert result_path == str(model_dir / "model.safetensors")
        assert is_sharded is False

    def test_official_sharded_safetensors(self, tmp_path):
        """Official model with sharded safetensors (index.json present)."""
        model_dir = tmp_path / "TestModel"
        model_dir.mkdir()
        (model_dir / "model.safetensors.index.json").write_text('{"weight_map": {}}')
        result_path, is_sharded = VibeVoiceLoader._resolve_checkpoint_path(
            model_path=str(model_dir), model_type="official", model_info={}
        )
        assert result_path == str(model_dir / "model.safetensors.index.json")
        assert is_sharded is True

    def test_official_single_pytorch_bin(self, tmp_path):
        """Official model with single pytorch_model.bin file."""
        model_dir = tmp_path / "TestModel"
        model_dir.mkdir()
        (model_dir / "pytorch_model.bin").write_text("dummy")
        result_path, is_sharded = VibeVoiceLoader._resolve_checkpoint_path(
            model_path=str(model_dir), model_type="official", model_info={}
        )
        assert result_path == str(model_dir / "pytorch_model.bin")
        assert is_sharded is False

    def test_official_sharded_pytorch_bin(self, tmp_path):
        """Official model with sharded pytorch_model (index.json present)."""
        model_dir = tmp_path / "TestModel"
        model_dir.mkdir()
        (model_dir / "pytorch_model.bin.index.json").write_text('{"weight_map": {}}')
        result_path, is_sharded = VibeVoiceLoader._resolve_checkpoint_path(
            model_path=str(model_dir), model_type="official", model_info={}
        )
        assert result_path == str(model_dir / "pytorch_model.bin.index.json")
        assert is_sharded is True

    def test_official_prioritizes_safetensors_over_bin(self, tmp_path):
        """When both safetensors and bin exist, safetensors is preferred."""
        model_dir = tmp_path / "TestModel"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_text("dummy")
        (model_dir / "pytorch_model.bin").write_text("dummy")
        result_path, is_sharded = VibeVoiceLoader._resolve_checkpoint_path(
            model_path=str(model_dir), model_type="official", model_info={}
        )
        assert result_path == str(model_dir / "model.safetensors")
        assert is_sharded is False

    def test_official_no_checkpoint_raises(self, tmp_path):
        """Raises FileNotFoundError when no checkpoint file found."""
        model_dir = tmp_path / "TestModel"
        model_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No checkpoint file found"):
            VibeVoiceLoader._resolve_checkpoint_path(
                model_path=str(model_dir), model_type="official", model_info={}
            )

    def test_official_dir_not_found_raises(self):
        """Raises FileNotFoundError when model directory doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Model directory not found"):
            VibeVoiceLoader._resolve_checkpoint_path(
                model_path="/nonexistent/dir", model_type="official", model_info={}
            )


class TestLoadShardedStateDict:
    """Test VibeVoiceLoader._load_sharded_state_dict."""

    def test_load_sharded_merges_shards(self, tmp_path):
        """Test that sharded state dict is loaded and merged correctly."""
        # Create shard files (dummy content)
        (tmp_path / "model-00001-of-00002.safetensors").write_text("dummy1")
        (tmp_path / "model-00002-of-00002.safetensors").write_text("dummy2")

        # Create index file
        index_data = {
            "weight_map": {
                "layer1.weight": "model-00001-of-00002.safetensors",
                "layer1.bias": "model-00001-of-00002.safetensors",
                "layer2.weight": "model-00002-of-00002.safetensors",
                "layer2.bias": "model-00002-of-00002.safetensors",
            }
        }
        index_path = tmp_path / "model.safetensors.index.json"
        index_path.write_text(json.dumps(index_data))

        # Mock load_torch_file to return different dicts per shard
        shard1_data = {"layer1.weight": "tensor1", "layer1.bias": "tensor2"}
        shard2_data = {"layer2.weight": "tensor3", "layer2.bias": "tensor4"}

        def mock_load(path, device=None):
            if "00001" in path:
                return shard1_data
            elif "00002" in path:
                return shard2_data
            return {}

        with patch("ComfyUI_VibeVoice.modules.loader.comfy.utils.load_torch_file", side_effect=mock_load):
            result = VibeVoiceLoader._load_sharded_state_dict(
                index_path=str(index_path),
                model_dir=str(tmp_path),
                device=torch.device("cpu"),
            )

        assert "layer1.weight" in result
        assert "layer1.bias" in result
        assert "layer2.weight" in result
        assert "layer2.bias" in result
        assert result["layer1.weight"] == "tensor1"
        assert result["layer2.weight"] == "tensor3"

    def test_load_sharded_empty_weight_map_raises(self, tmp_path):
        """Test that empty weight_map raises ValueError."""
        index_path = tmp_path / "model.safetensors.index.json"
        index_path.write_text('{"weight_map": {}}')

        with pytest.raises(ValueError, match="empty weight_map"):
            VibeVoiceLoader._load_sharded_state_dict(
                index_path=str(index_path),
                model_dir=str(tmp_path),
                device=torch.device("cpu"),
            )

    def test_load_sharded_missing_shard_raises(self, tmp_path):
        """Test that missing shard file raises FileNotFoundError."""
        index_data = {
            "weight_map": {
                "layer1.weight": "model-00001-of-00002.safetensors",
                "layer2.weight": "model-00002-of-00002.safetensors",
            }
        }
        index_path = tmp_path / "model.safetensors.index.json"
        index_path.write_text(json.dumps(index_data))

        # Only create shard 1, not shard 2
        (tmp_path / "model-00001-of-00002.safetensors").write_text("dummy")

        with patch("ComfyUI_VibeVoice.modules.loader.comfy.utils.load_torch_file") as mock_load:
            mock_load.return_value = {}
            with pytest.raises(FileNotFoundError, match="Shard file not found"):
                VibeVoiceLoader._load_sharded_state_dict(
                    index_path=str(index_path),
                    model_dir=str(tmp_path),
                    device=torch.device("cpu"),
                )


class _TwoParam(torch.nn.Module):
    """Tiny real module for streaming-assign tests (no mocks).

    ``config`` carries the (untied) gate read by the post-assign fixups.
    """

    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(2, 2, bias=False)
        self.layer2 = torch.nn.Linear(2, 2, bias=False)
        self.config = MagicMock()
        self.config.decoder_config.tie_word_embeddings = False
        self.config.tie_word_embeddings = False


class TestVibeVoiceLoaderLoadStateDict:
    """Test VibeVoiceLoader._load_state_dict_into_model (streaming assign).

    Plan 2026-08-28: dense checkpoints (sharded or single-file) are applied
    per-tensor via ``_stream_apply_dense`` — no merged state dict, and every
    tensor is cloned into private memory, severing the checkpoint's file
    mapping (the mmap ghost behind the sharded 7B RAM bloat + Pin errors).
    These tests use REAL safetensors files so the mmap semantics are
    exercised end-to-end.
    """

    def test_load_state_dict_single_safetensors(self, tmp_path):
        """Single-file safetensors: streamed per-tensor into the model."""
        from safetensors.torch import save_file

        model_dir = tmp_path / "TestModel"
        model_dir.mkdir()
        w1 = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        w2 = torch.full((2, 2), 7.0)
        save_file(
            {"layer1.weight": w1, "layer2.weight": w2},
            str(model_dir / "model.safetensors"),
        )

        model = _TwoParam()
        result = VibeVoiceLoader._load_state_dict_into_model(
            model=model,
            model_path=str(model_dir),
            model_type="official",
            model_info={},
            device=torch.device("cpu"),
        )

        assert result is model
        assert torch.equal(model.layer1.weight.data, w1)
        assert torch.equal(model.layer2.weight.data, w2)
        assert model.layer1.weight.device.type == "cpu"

    def test_load_state_dict_standalone(self, tmp_path):
        """Standalone safetensors file: streamed per-tensor into the model."""
        from safetensors.torch import save_file

        ckpt_file = tmp_path / "checkpoint.safetensors"
        w1 = torch.ones(2, 2)
        w2 = torch.full((2, 2), 2.0)
        save_file({"layer1.weight": w1, "layer2.weight": w2}, str(ckpt_file))

        model = _TwoParam()
        result = VibeVoiceLoader._load_state_dict_into_model(
            model=model,
            model_path=None,
            model_type="standalone",
            model_info={"path": str(ckpt_file)},
            device=torch.device("cpu"),
        )

        assert result is model
        assert torch.equal(model.layer1.weight.data, w1)
        assert torch.equal(model.layer2.weight.data, w2)

    def test_load_state_dict_sharded(self, tmp_path):
        """Sharded safetensors: every shard streams into the model."""
        from safetensors.torch import save_file

        model_dir = tmp_path / "TestModel"
        model_dir.mkdir()

        w1 = torch.ones(2, 2)
        w2 = torch.full((2, 2), 2.0)
        save_file({"layer1.weight": w1}, str(model_dir / "model-00001-of-00002.safetensors"))
        save_file({"layer2.weight": w2}, str(model_dir / "model-00002-of-00002.safetensors"))
        index_data = {
            "weight_map": {
                "layer1.weight": "model-00001-of-00002.safetensors",
                "layer2.weight": "model-00002-of-00002.safetensors",
            }
        }
        (model_dir / "model.safetensors.index.json").write_text(json.dumps(index_data))

        model = _TwoParam()
        result = VibeVoiceLoader._load_state_dict_into_model(
            model=model,
            model_path=str(model_dir),
            model_type="official",
            model_info={},
            device=torch.device("cpu"),
        )

        assert result is model
        assert torch.equal(model.layer1.weight.data, w1)
        assert torch.equal(model.layer2.weight.data, w2)

    def test_load_state_dict_logs_missing_keys(self, tmp_path, caplog):
        """Keys the checkpoint omits are reported as missing."""
        import logging
        from safetensors.torch import save_file

        model_dir = tmp_path / "TestModel"
        model_dir.mkdir()
        save_file({"layer1.weight": torch.ones(2, 2)}, str(model_dir / "model.safetensors"))

        model = _TwoParam()
        with caplog.at_level(logging.WARNING):
            VibeVoiceLoader._load_state_dict_into_model(
                model=model,
                model_path=str(model_dir),
                model_type="official",
                model_info={},
                device=torch.device("cpu"),
            )

        assert any("Missing keys" in r.message for r in caplog.records)

    def test_assigned_weights_own_private_storage(self, tmp_path):
        """mmap severing: assigned params are private clones, not file views.

        safetensors tensors are zero-copy views into the mapped file; a view
        retained by an offloaded parameter would pin the whole file mapping
        in the process working set (ghost RAM + unstable cudaHostRegister
        pins). A private clone's storage covers exactly its own bytes, while
        a view's storage covers the file's entire data section.
        """
        from safetensors.torch import save_file

        model_dir = tmp_path / "TestModel"
        model_dir.mkdir()
        w1 = torch.ones(2, 2)
        w2 = torch.full((2, 2), 2.0)
        save_file(
            {"layer1.weight": w1, "layer2.weight": w2},
            str(model_dir / "model.safetensors"),
        )

        model = _TwoParam()
        VibeVoiceLoader._load_state_dict_into_model(
            model=model,
            model_path=str(model_dir),
            model_type="official",
            model_info={},
            device=torch.device("cpu"),
        )

        for param in (model.layer1.weight, model.layer2.weight):
            own_bytes = param.numel() * param.element_size()
            assert param.untyped_storage().nbytes() == own_bytes, (
                "assigned parameter still sits in the checkpoint's file "
                "mapping — the streaming assign must clone into private memory"
            )


class TestStreamApplyDense:
    """Direct unit tests for VibeVoiceLoader._stream_apply_dense."""

    def test_assigns_params_and_buffers_as_clones(self):
        model = _TwoParam()
        model.register_buffer("pos", torch.zeros(3))
        w = torch.ones(2, 2)
        pairs = [
            ("layer1.weight", w),
            ("layer2.weight", torch.full((2, 2), 2.0)),
            ("pos", torch.ones(3)),
        ]
        missing, unexpected = VibeVoiceLoader._stream_apply_dense(model, iter(pairs))

        assert unexpected == []
        assert missing == []
        assert torch.equal(model.layer1.weight.data, w)
        # Clone semantics: the model owns private storage, not the source.
        assert model.layer1.weight.data_ptr() != w.data_ptr()
        assert torch.equal(model.pos, torch.ones(3))

    def test_shape_mismatch_raises_friendly_error(self):
        model = _TwoParam()
        pairs = [("layer1.weight", torch.ones(3, 3))]
        with pytest.raises(ValueError, match="shapes do not match"):
            VibeVoiceLoader._stream_apply_dense(model, iter(pairs))

    def test_unexpected_and_missing_reported(self):
        model = _TwoParam()
        pairs = [("layer1.weight", torch.ones(2, 2)), ("bogus.key", torch.zeros(1))]
        missing, unexpected = VibeVoiceLoader._stream_apply_dense(model, iter(pairs))

        assert unexpected == ["bogus.key"]
        assert "layer2.weight" in missing


class TestCleanupOldModels:
    """Test cleanup_old_models function."""

    def test_cleanup_keeps_specified_key(self):
        from ComfyUI_VibeVoice.modules.loader import LOADED_MODELS_CACHE
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_PATCHER_CACHE

        LOADED_MODELS_CACHE.clear()
        VIBEVOICE_PATCHER_CACHE.clear()

        LOADED_MODELS_CACHE["key1"] = "model1"
        LOADED_MODELS_CACHE["key2"] = "model2"

        with patch("ComfyUI_VibeVoice.modules.loader.model_management"):
            cleanup_old_models(keep_cache_key="key1")

        assert "key1" in LOADED_MODELS_CACHE
        assert "key2" not in LOADED_MODELS_CACHE

    def test_cleanup_clears_all(self):
        from ComfyUI_VibeVoice.modules.loader import LOADED_MODELS_CACHE
        from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_PATCHER_CACHE

        LOADED_MODELS_CACHE.clear()
        VIBEVOICE_PATCHER_CACHE.clear()

        LOADED_MODELS_CACHE["key1"] = "model1"
        LOADED_MODELS_CACHE["key2"] = "model2"

        with patch("ComfyUI_VibeVoice.modules.loader.model_management"):
            cleanup_old_models(keep_cache_key=None)

        assert len(LOADED_MODELS_CACHE) == 0


class TestTTSLoaderBaseInheritance:
    """IMP-004: TTS loader shares the BaseVibeVoiceLoader."""

    def test_tts_loader_uses_base(self):
        assert isinstance(VibeVoiceLoader(), BaseVibeVoiceLoader)


# ====================================================================
# AUDIT PHASE B — B2: config loading & streaming detection
# ====================================================================
class TestLoadConfig:
    """B2: _load_config must pick the right config class + fallback."""

    def test_streaming_json_uses_streaming_config_class(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"model_type": "vibevoice_streaming"}))

        with patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceStreamingConfig") as mock_stream, \
             patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceConfig") as mock_base:
            VibeVoiceLoader._load_config(str(config_path), "SomeModel")
            mock_stream.from_pretrained.assert_called_once_with(str(config_path))
            mock_base.from_pretrained.assert_not_called()

    def test_non_streaming_json_uses_base_config_class(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"model_type": "vibevoice"}))

        with patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceStreamingConfig") as mock_stream, \
             patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceConfig") as mock_base:
            VibeVoiceLoader._load_config(str(config_path), "SomeModel")
            mock_base.from_pretrained.assert_called_once_with(str(config_path))
            mock_stream.from_pretrained.assert_not_called()

    def test_missing_config_falls_back_to_large_default(self):
        with patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceConfig") as mock_base:
            VibeVoiceLoader._load_config("/nonexistent/config.json", "VibeVoice-Large")
            mock_base.from_pretrained.assert_called_once()
            fallback_arg = mock_base.from_pretrained.call_args[0][0]
            assert "default_VibeVoice-Large_config.json" in fallback_arg

    def test_missing_config_falls_back_to_15b_default(self):
        with patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceConfig") as mock_base:
            VibeVoiceLoader._load_config("/nonexistent/config.json", "VibeVoice-1.5B")
            fallback_arg = mock_base.from_pretrained.call_args[0][0]
            assert "default_VibeVoice-1.5B_config.json" in fallback_arg


# ====================================================================
# AUDIT PHASE B — B3: tokenizer acquisition order
# ====================================================================
class TestLoadTokenizer:
    """B3 acquisition order, copy-free: existing file -> packaged direct
    load -> HF download -> RuntimeError."""

    def test_existing_tokenizer_no_packaged_no_download(self, tmp_path):
        (tmp_path / "tokenizer.json").write_text("{}")
        with patch("ComfyUI_VibeVoice.modules.loader.hf_hub_download") as mock_dl,              patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceTextTokenizerFast") as mock_tok:
            VibeVoiceLoader._load_tokenizer(str(tmp_path), "TestModel")
            mock_dl.assert_not_called()
            assert mock_tok.call_args[1]["tokenizer_file"] ==                 str(tmp_path / "tokenizer.json")

    def test_packaged_fallback_loaded_directly_no_side_effects(self, tmp_path):
        """Packaged tokenizer loads straight from the node folder; the
        user's model directory is never written to."""
        with patch("os.path.exists", side_effect=lambda p: (
            True if "configs" in p else os.path.isfile(p)
        )), patch("ComfyUI_VibeVoice.modules.loader.hf_hub_download") as mock_dl,              patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceTextTokenizerFast") as mock_tok:
            VibeVoiceLoader._load_tokenizer(str(tmp_path), "TestModel")
            mock_dl.assert_not_called()
            used_path = mock_tok.call_args[1]["tokenizer_file"]
            assert "configs" in used_path
            assert used_path.endswith("tokenizer.json")
            assert not (tmp_path / "tokenizer.json").exists()

    def test_download_fallback_second_repo_succeeds(self, tmp_path):
        calls = []

        def fake_download(repo_id=None, filename=None, local_dir=None):
            calls.append(repo_id)
            if repo_id == "Qwen/Qwen2.5-1.5B":
                raise RuntimeError("offline")
            with open(os.path.join(local_dir, "tokenizer.json"), "w") as f:
                f.write("{}")

        # Hide the packaged tokenizer so the download path runs.
        with patch("os.path.exists",
                   side_effect=lambda p: (False if "configs" in p
                                          else os.path.isfile(p))),              patch("ComfyUI_VibeVoice.modules.loader.hf_hub_download",
                   side_effect=fake_download),              patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceTextTokenizerFast"):
            VibeVoiceLoader._load_tokenizer(str(tmp_path), "TestModel")
            assert calls == ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-7B"]

    def test_all_sources_fail_raises_runtime_error(self, tmp_path):
        with patch("os.path.exists", return_value=False),              patch("ComfyUI_VibeVoice.modules.loader.hf_hub_download",
                   side_effect=RuntimeError("offline")),              patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceTextTokenizerFast"):
            with pytest.raises(RuntimeError, match="Could not get 'tokenizer.json'"):
                VibeVoiceLoader._load_tokenizer(str(tmp_path), "TestModel")


# ====================================================================
# AUDIT PHASE B — B6: load_model orchestration (fully mocked)
# ====================================================================
class TestLoadModelOrchestration:
    """B6: full load_model sequence, cache, and failure isolation."""

    def _model_registry(self, model_type="official"):
        return {"TestModel": {"type": model_type, "repo_id": "test/repo", "path": "x"}}

    def test_unknown_model_raises_value_error(self):
        with patch("ComfyUI_VibeVoice.modules.loader.AVAILABLE_VIBEVOICE_MODELS", {}):
            with pytest.raises(ValueError, match="Unknown VibeVoice model"):
                VibeVoiceLoader.load_model("Nope", torch.device("cpu"))

    def test_cache_hit_short_circuits(self):
        LOADED_MODELS_CACHE.clear()
        sentinel = ("cached_model", "cached_processor")
        LOADED_MODELS_CACHE["TestModel_attn_sdpa_q4_0"] = sentinel
        try:
            with patch("ComfyUI_VibeVoice.modules.loader.AVAILABLE_VIBEVOICE_MODELS",
                       self._model_registry()), \
                 patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._resolve_model_paths") as mock_rp:
                result = VibeVoiceLoader.load_model(
                    "TestModel", torch.device("cpu"), attention_mode="sdpa"
                )
                assert result == sentinel
                mock_rp.assert_not_called()  # no work done on cache hit
        finally:
            LOADED_MODELS_CACHE.clear()

    def test_full_sequence_and_cache_store(self):
        LOADED_MODELS_CACHE.clear()
        ledger = []

        fake_model = MagicMock()
        fake_model.to.return_value = fake_model
        fake_processor = MagicMock()
        fake_config = MagicMock(spec=[])  # not a VibeVoiceStreamingConfig instance

        # isinstance() needs a real class, not a MagicMock.
        class _FakeStreamingCfg:
            pass

        try:
            with patch("ComfyUI_VibeVoice.modules.loader.AVAILABLE_VIBEVOICE_MODELS",
                       self._model_registry()), \
                 patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceStreamingConfig", _FakeStreamingCfg), \
                 patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._resolve_model_paths",
                       side_effect=lambda n: (ledger.append("paths"), ("mp", "cp", "pp", "td"))[1]), \
                 patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._load_config",
                       side_effect=lambda cp, n: (ledger.append("config"), fake_config)[1]), \
                 patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._load_tokenizer",
                       side_effect=lambda td, n: (ledger.append("tokenizer"), MagicMock())[1]), \
                 patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._load_processor",
                       side_effect=lambda tok, pp, is_streaming=False: (
                           ledger.append("processor"), fake_processor)[1]), \
                  patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._instantiate_model",
                        side_effect=lambda **kw: (ledger.append("instantiate"), fake_model)[1]), \
                  patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._load_state_dict_into_model",
                        side_effect=lambda **kw: (ledger.append("state_dict"), fake_model)[1]), \
                  patch("ComfyUI_VibeVoice.modules.loader.cast_model_to_dtype_if_needed",
                        side_effect=lambda m, d: ledger.append("cast")):
                model, processor = VibeVoiceLoader.load_model(
                    "TestModel", torch.device("cpu"), attention_mode="sdpa"
                )

            assert model is fake_model
            assert processor is fake_processor
            assert ledger == ["paths", "config", "tokenizer", "processor",
                              "instantiate", "state_dict", "cast"]
            # Plan 2026-08-18 D4/RC-3: dtype applied via conditional cast
            # helper (not an unconditional .to()), eval'd, cached
            fake_model.eval.assert_called_once()
            assert LOADED_MODELS_CACHE["TestModel_attn_sdpa_q4_0"] == (fake_model, fake_processor)
        finally:
            LOADED_MODELS_CACHE.clear()

    def test_exception_mid_load_wraps_and_does_not_pollute_cache(self):
        LOADED_MODELS_CACHE.clear()

        class _FakeStreamingCfg:
            pass

        try:
            with patch("ComfyUI_VibeVoice.modules.loader.AVAILABLE_VIBEVOICE_MODELS",
                       self._model_registry()), \
                 patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceStreamingConfig", _FakeStreamingCfg), \
                 patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._resolve_model_paths",
                       return_value=("mp", "cp", "pp", "td")), \
                 patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._load_config",
                       return_value=MagicMock(spec=[])), \
                 patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._load_tokenizer",
                       return_value=MagicMock()), \
                 patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._load_processor",
                       return_value=MagicMock()), \
                 patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._instantiate_model",
                       side_effect=RuntimeError("boom")):
                with pytest.raises(RuntimeError, match="Failed to load model"):
                    VibeVoiceLoader.load_model(
                        "TestModel", torch.device("cpu"), attention_mode="sdpa"
                    )
            # Cache must NOT contain a partial entry.
            assert "TestModel_attn_sdpa_q4_0" not in LOADED_MODELS_CACHE
        finally:
            LOADED_MODELS_CACHE.clear()

    def test_4bit_builds_bnb_config_and_replaces_linears(self):
        LOADED_MODELS_CACHE.clear()
        fake_model = MagicMock()
        fake_model.to.return_value = fake_model

        class _FakeStreamingCfg:
            pass

        # AUD-014: patch the import target that actually resolves on this
        # transformers version (integrations on 5.x, utils on 4.x).
        try:
            import transformers.integrations.bitsandbytes as _bnb_mod
            _bnb_target = "transformers.integrations.bitsandbytes.replace_with_bnb_linear"
        except ImportError:
            _bnb_target = "transformers.utils.bitsandbytes.replace_with_bnb_linear"

        try:
            with patch("ComfyUI_VibeVoice.modules.loader.AVAILABLE_VIBEVOICE_MODELS",
                       self._model_registry()), \
                 patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceStreamingConfig", _FakeStreamingCfg), \
                 patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._resolve_model_paths",
                       return_value=("mp", "cp", "pp", "td")), \
                 patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._load_config",
                       return_value=MagicMock(spec=[])), \
                 patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._load_tokenizer",
                       return_value=MagicMock()), \
                 patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._load_processor",
                       return_value=MagicMock()), \
                 patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._instantiate_model",
                       return_value=fake_model), \
                 patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._load_state_dict_into_model",
                       return_value=fake_model), \
                 patch("ComfyUI_VibeVoice.modules.loader.BitsAndBytesConfig") as mock_bnb, \
                 patch(_bnb_target) as mock_replace:
                VibeVoiceLoader.load_model(
                    "TestModel", torch.device("cpu"),
                    attention_mode="sdpa", use_llm_4bit=True,
                )
                mock_bnb.assert_called_once()
                assert mock_bnb.call_args.kwargs["load_in_4bit"] is True
                mock_replace.assert_called_once()
                assert getattr(fake_model, "_llm_4bit") is True
        finally:
            LOADED_MODELS_CACHE.clear()
