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
        handler = VibeVoiceModelHandler("VibeVoice-Large", attention_mode="eager", use_llm_4bit=True)
        assert handler.cache_key == "VibeVoice-Large_attn_eager_q4_1"

    def test_handler_size_calculation(self):
        handler = VibeVoiceModelHandler("VibeVoice-1.5B")
        assert handler.size == int(3.0 * (1024**3))

    def test_handler_size_large(self):
        handler = VibeVoiceModelHandler("VibeVoice-Large")
        assert handler.size == int(17.4 * (1024**3))

    def test_handler_is_torch_module(self):
        handler = VibeVoiceModelHandler("VibeVoice-1.5B")
        assert isinstance(handler, torch.nn.Module)


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
             patch("ComfyUI_VibeVoice.modules.loader.snapshot_download") as mock_dl, \
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


class TestVibeVoiceLoaderLoadStateDict:
    """Test VibeVoiceLoader._load_state_dict_into_model."""

    def test_load_state_dict_single_safetensors(self, tmp_path):
        """Test loading state dict from a single safetensors file."""
        model_dir = tmp_path / "TestModel"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_text("dummy")

        model = MagicMock()
        model.load_state_dict.return_value = ([], [])

        with patch("ComfyUI_VibeVoice.modules.loader.comfy.utils.load_torch_file") as mock_load:
            mock_load.return_value = {"key": "value"}

            result = VibeVoiceLoader._load_state_dict_into_model(
                model=model,
                model_path=str(model_dir),
                model_type="official",
                model_info={},
                device=torch.device("cpu"),
            )

            mock_load.assert_called_once_with(
                str(model_dir / "model.safetensors"), device=torch.device("cpu")
            )
            model.load_state_dict.assert_called_once_with({"key": "value"}, strict=False)
            assert result == model

    def test_load_state_dict_standalone(self, tmp_path):
        """Test loading state dict for standalone model type."""
        ckpt_file = tmp_path / "checkpoint.safetensors"
        ckpt_file.write_text("dummy")

        model = MagicMock()
        model.load_state_dict.return_value = ([], [])

        model_info = {"path": str(ckpt_file)}

        with patch("ComfyUI_VibeVoice.modules.loader.comfy.utils.load_torch_file") as mock_load:
            mock_load.return_value = {"key": "value"}

            result = VibeVoiceLoader._load_state_dict_into_model(
                model=model,
                model_path=None,
                model_type="standalone",
                model_info=model_info,
                device=torch.device("cpu"),
            )

            mock_load.assert_called_once_with(str(ckpt_file), device=torch.device("cpu"))
            assert result == model

    def test_load_state_dict_sharded(self, tmp_path):
        """Test loading state dict from sharded safetensors checkpoint."""
        model_dir = tmp_path / "TestModel"
        model_dir.mkdir()

        # Create shard files (dummy content)
        (model_dir / "model-00001-of-00002.safetensors").write_text("dummy1")
        (model_dir / "model-00002-of-00002.safetensors").write_text("dummy2")

        index_data = {
            "weight_map": {
                "layer1.weight": "model-00001-of-00002.safetensors",
                "layer2.weight": "model-00002-of-00002.safetensors",
            }
        }
        (model_dir / "model.safetensors.index.json").write_text(json.dumps(index_data))

        model = MagicMock()
        model.load_state_dict.return_value = ([], [])

        shard1_data = {"layer1.weight": "tensor1"}
        shard2_data = {"layer2.weight": "tensor2"}

        def mock_load(path, device=None):
            if "00001" in path:
                return shard1_data
            elif "00002" in path:
                return shard2_data
            return {}

        with patch("ComfyUI_VibeVoice.modules.loader.comfy.utils.load_torch_file", side_effect=mock_load):
            result = VibeVoiceLoader._load_state_dict_into_model(
                model=model,
                model_path=str(model_dir),
                model_type="official",
                model_info={},
                device=torch.device("cpu"),
            )

            # Verify model.load_state_dict was called with merged dict
            called_state_dict = model.load_state_dict.call_args[0][0]
            assert "layer1.weight" in called_state_dict
            assert "layer2.weight" in called_state_dict
            assert called_state_dict["layer1.weight"] == "tensor1"
            assert called_state_dict["layer2.weight"] == "tensor2"
            assert result == model

    def test_load_state_dict_logs_missing_keys(self, tmp_path):
        """Test that missing keys are logged."""
        model_dir = tmp_path / "TestModel"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_text("dummy")

        model = MagicMock()
        model.load_state_dict.return_value = (["missing_key1", "missing_key2"], [])

        mock_state_dict = {"key": "value"}
        with patch("ComfyUI_VibeVoice.modules.loader.comfy.utils.load_torch_file", return_value=mock_state_dict):
            VibeVoiceLoader._load_state_dict_into_model(
                model=model,
                model_path=str(model_dir),
                model_type="official",
                model_info={},
                device=torch.device("cpu"),
            )

            model.load_state_dict.assert_called_once_with(mock_state_dict, strict=False)


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
