"""Tests for modules/base_loader.py - shared BaseVibeVoiceLoader (IMP-004).

Exercises the GPU-free helpers (path resolution, snapshot download, tokenizer
repo, sharded/single state-dict loading) in isolation.
"""

import json
import torch
import pytest
from unittest.mock import patch, MagicMock

from ComfyUI_VibeVoice.modules.base_loader import (
    BaseVibeVoiceLoader,
    iter_safetensors_tensors,
    place_tensor_on_device,
)


class TestResolveOfficialModelDir:
    def test_resolve_official_paths_uses_folder_paths(self, tmp_path):
        with patch("ComfyUI_VibeVoice.modules.base_loader.folder_paths") as mock_fp:
            mock_fp.get_folder_paths.return_value = [str(tmp_path)]
            path = BaseVibeVoiceLoader._resolve_official_model_dir("VibeVoice-ASR")
        assert path == str(tmp_path / "VibeVoice" / "VibeVoice-ASR")

    def test_resolve_official_paths_falls_back_to_models_dir(self, tmp_path):
        fake_fp = MagicMock()
        # No "tts" folder registered → falls back to <models_dir>/tts
        fake_fp.get_folder_paths.return_value = []
        fake_fp.models_dir = str(tmp_path)
        with patch("ComfyUI_VibeVoice.modules.base_loader.folder_paths", fake_fp):
            path = BaseVibeVoiceLoader._resolve_official_model_dir("VibeVoice-ASR")
        assert path == str(tmp_path / "tts" / "VibeVoice" / "VibeVoice-ASR")

    def test_resolve_official_prefers_existing_dir_in_secondary_root(self, tmp_path):
        """A model already downloaded under a secondary registered root
        (extra_model_paths.yaml) is picked up instead of re-downloading
        into the primary root."""
        first, second = tmp_path / "root1", tmp_path / "root2"
        model_dir = second / "VibeVoice" / "VibeVoice-ASR-Streaming-1.5B"
        model_dir.mkdir(parents=True)
        fake_fp = MagicMock()
        fake_fp.get_folder_paths.return_value = [str(first), str(second)]
        with patch("ComfyUI_VibeVoice.modules.base_loader.folder_paths", fake_fp):
            path = BaseVibeVoiceLoader._resolve_official_model_dir(
                "VibeVoice-ASR-Streaming-1.5B"
            )
        assert path == str(model_dir)

    def test_resolve_official_skips_dir_without_config(self, tmp_path):
        """An existing dir is preferred even when incomplete (no config.json):
        a partial download resumes in place instead of being orphaned."""
        first, second = tmp_path / "root1", tmp_path / "root2"
        partial = second / "VibeVoice" / "VibeVoice-ASR-Streaming-7B"
        partial.mkdir(parents=True)
        fake_fp = MagicMock()
        fake_fp.get_folder_paths.return_value = [str(first), str(second)]
        with patch("ComfyUI_VibeVoice.modules.base_loader.folder_paths", fake_fp):
            path = BaseVibeVoiceLoader._resolve_official_model_dir(
                "VibeVoice-ASR-Streaming-7B"
            )
        assert path == str(partial)


class TestEnsureDownloaded:
    def test_ensure_downloaded_triggers_snapshot(self, tmp_path):
        local_dir = tmp_path / "model"
        local_dir.mkdir()
        repo_id = "microsoft/VibeVoice-ASR"
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            BaseVibeVoiceLoader._ensure_downloaded(
                repo_id=repo_id, local_dir=str(local_dir), model_name="VibeVoice-ASR"
            )
        mock_dl.assert_called_once()
        _, kwargs = mock_dl.call_args
        assert kwargs.get("repo_id") == repo_id
        assert kwargs.get("local_dir") == str(local_dir)
        # AUD-013: the symlinks kwarg is only passed when the installed
        # huggingface_hub still accepts it (removed in >= 0.23 / 1.x).
        import inspect
        import huggingface_hub
        supports_symlinks = "local_dir_use_symlinks" in inspect.signature(
            huggingface_hub.snapshot_download
        ).parameters
        if supports_symlinks:
            assert kwargs.get("local_dir_use_symlinks") is False
        else:
            assert "local_dir_use_symlinks" not in kwargs

    def test_ensure_downloaded_skips_when_config_present(self, tmp_path):
        local_dir = tmp_path / "model"
        local_dir.mkdir()
        (local_dir / "config.json").write_text("{}")
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            BaseVibeVoiceLoader._ensure_downloaded(repo_id="x/y", local_dir=str(local_dir))
        mock_dl.assert_not_called()

    def test_ensure_downloaded_no_repo_returns_early(self, tmp_path):
        local_dir = tmp_path / "model"
        local_dir.mkdir()
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            BaseVibeVoiceLoader._ensure_downloaded(repo_id="", local_dir=str(local_dir))
        mock_dl.assert_not_called()


class TestTokenizerRepo:
    def test_tokenizer_repo_for_asr_uses_7b(self):
        assert BaseVibeVoiceLoader.tokenizer_repo_for("VibeVoice-ASR") == "Qwen/Qwen2.5-7B"

    def test_tokenizer_repo_for_small_uses_15b(self):
        assert BaseVibeVoiceLoader.tokenizer_repo_for("VibeVoice-1.5B") == "Qwen/Qwen2.5-1.5B"


class TestLoadStateDictSharded:
    def test_load_state_dict_sharded_single_file(self, tmp_path):
        (tmp_path / "model.safetensors").write_text("dummy")
        with patch(
            "ComfyUI_VibeVoice.modules.base_loader.comfy.utils.load_torch_file",
            return_value={"w": "v"},
        ):
            sd = BaseVibeVoiceLoader.load_state_dict_sharded(str(tmp_path), device=torch.device("cpu"))
        assert sd == {"w": "v"}

    def test_load_state_dict_sharded_merges_shards(self, tmp_path):
        (tmp_path / "model-00001-of-00002.safetensors").write_text("d1")
        (tmp_path / "model-00002-of-00002.safetensors").write_text("d2")
        index = {
            "weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        def mock_load(path, device=None):
            if "00001" in path:
                return {"a": 1}
            return {"b": 2}

        with patch(
            "ComfyUI_VibeVoice.modules.base_loader.comfy.utils.load_torch_file",
            side_effect=mock_load,
        ):
            sd = BaseVibeVoiceLoader.load_state_dict_sharded(str(tmp_path), device=torch.device("cpu"))
        assert sd == {"a": 1, "b": 2}

    def test_load_state_dict_sharded_empty_weight_map_raises(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
        with pytest.raises(ValueError, match="empty weight_map"):
            BaseVibeVoiceLoader.load_state_dict_sharded(str(tmp_path), device=torch.device("cpu"))

    def test_load_state_dict_sharded_missing_shard_raises(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors"}})
        )
        with patch(
            "ComfyUI_VibeVoice.modules.base_loader.comfy.utils.load_torch_file",
            return_value={},
        ):
            with pytest.raises(FileNotFoundError, match="Shard file not found"):
                BaseVibeVoiceLoader.load_state_dict_sharded(str(tmp_path), device=torch.device("cpu"))


class TestStreamingIterators:
    """Streaming tensor iterators (plan 2026-08-28: sharded mmap-ghost fix).

    Real safetensors files throughout — the iterators exist precisely to
    control how file mappings are consumed, so mocking them away would
    defeat the point.
    """

    @staticmethod
    def _write_sharded(tmp_path):
        from safetensors.torch import save_file

        w1 = torch.ones(2)
        w2 = torch.full((2,), 2.0)
        save_file({"a": w1}, str(tmp_path / "model-00001-of-00002.safetensors"))
        save_file({"b": w2}, str(tmp_path / "model-00002-of-00002.safetensors"))
        index = {
            "weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
        return w1, w2

    def test_iter_sharded_tensors_yields_all_keys(self, tmp_path):
        w1, w2 = self._write_sharded(tmp_path)
        pairs = dict(BaseVibeVoiceLoader.iter_sharded_tensors(str(tmp_path)))
        assert set(pairs) == {"a", "b"}
        assert torch.equal(pairs["a"], w1)
        assert torch.equal(pairs["b"], w2)

    def test_iter_sharded_tensors_matches_merged_dict(self, tmp_path):
        """Streaming and merged loads deliver identical tensors."""
        self._write_sharded(tmp_path)
        merged = BaseVibeVoiceLoader.load_state_dict_sharded(
            str(tmp_path), device=torch.device("cpu")
        )
        streamed = dict(BaseVibeVoiceLoader.iter_sharded_tensors(str(tmp_path)))
        assert set(merged) == set(streamed)
        for key in merged:
            assert torch.equal(merged[key], streamed[key]), key

    def test_iter_sharded_tensors_single_file_delegates(self, tmp_path):
        from safetensors.torch import save_file

        w = torch.arange(3, dtype=torch.float32)
        save_file({"only": w}, str(tmp_path / "model.safetensors"))
        pairs = dict(BaseVibeVoiceLoader.iter_sharded_tensors(str(tmp_path)))
        assert set(pairs) == {"only"}
        assert torch.equal(pairs["only"], w)

    def test_iter_sharded_tensors_empty_weight_map_raises(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
        with pytest.raises(ValueError, match="empty weight_map"):
            list(BaseVibeVoiceLoader.iter_sharded_tensors(str(tmp_path)))

    def test_iter_sharded_tensors_missing_shard_raises(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors"}})
        )
        with pytest.raises(FileNotFoundError, match="Shard file not found"):
            list(BaseVibeVoiceLoader.iter_sharded_tensors(str(tmp_path)))

    def test_iter_checkpoint_tensors_safetensors(self, tmp_path):
        from safetensors.torch import save_file

        w = torch.ones(2, 2)
        path = tmp_path / "single.safetensors"
        save_file({"w": w}, str(path))
        pairs = dict(BaseVibeVoiceLoader.iter_checkpoint_tensors(str(path)))
        assert set(pairs) == {"w"}
        assert torch.equal(pairs["w"], w)

    def test_iter_checkpoint_tensors_bin(self, tmp_path):
        w = torch.ones(2, 2)
        path = tmp_path / "single.bin"
        torch.save({"w": w}, str(path))
        pairs = dict(BaseVibeVoiceLoader.iter_checkpoint_tensors(str(path)))
        assert set(pairs) == {"w"}
        assert torch.equal(pairs["w"], w)


class TestSafetensorsIteratorArms:
    """``iter_safetensors_tensors`` has two read layers (2026-09-30 fp8 fix).

    The aimdo arm yields zero-copy ``load_safetensors`` file views (page
    cache, no private commit) so the consumer's mandatory clone is the only
    private copy — that clone is what halfed the 7B fp8 load spike. The
    aimdo-off fallback keeps the original ``safe_open`` stream. Both arms
    must deliver identical values; only the yielded storage's backing
    differs.
    """

    @staticmethod
    def _write_single(tmp_path):
        from safetensors.torch import save_file

        w = torch.randn(8, 4)
        path = tmp_path / "single.safetensors"
        save_file({"w": w}, str(path))
        return w

    @staticmethod
    def _has_file_slice(t):
        return hasattr(t.untyped_storage(), "_comfy_tensor_file_slice")

    @pytest.fixture(scope="class", autouse=True)
    def aimdo_runtime(self):
        """control.init() once for the class — the native runtime that
        production ComfyUI brings up at startup (same guard as the probes:
        no CUDA or a failed init skips the arm, the fallback arm still runs).
        """
        pytest.importorskip("comfy_aimdo")
        if not torch.cuda.is_available():
            pytest.skip("aimdo runtime needs CUDA on this install")
        from comfy_aimdo import control

        if not control.init():
            pytest.skip("aimdo control.init() failed")

    def test_fallback_arm_yields_owned_tensors(self, tmp_path, monkeypatch):
        import comfy.memory_management

        w = self._write_single(tmp_path)
        monkeypatch.setattr(
            comfy.memory_management, "aimdo_enabled", False, raising=False
        )
        pairs = dict(iter_safetensors_tensors(str(tmp_path / "single.safetensors")))
        assert set(pairs) == {"w"}
        assert torch.equal(pairs["w"], w)
        assert not self._has_file_slice(pairs["w"])

    def test_aimdo_arm_yields_file_views(self, tmp_path, monkeypatch):
        import comfy.memory_management

        w = self._write_single(tmp_path)
        monkeypatch.setattr(
            comfy.memory_management, "aimdo_enabled", True, raising=False
        )
        pairs = dict(iter_safetensors_tensors(str(tmp_path / "single.safetensors")))
        assert set(pairs) == {"w"}
        assert torch.equal(pairs["w"], w)
        assert self._has_file_slice(pairs["w"])

    def test_clone_severs_the_view(self, tmp_path, monkeypatch):
        """The consumer contract: clone before keeping. The clone must not
        carry the file mapping (pinning hazard), while the view does."""
        import comfy.memory_management

        self._write_single(tmp_path)
        monkeypatch.setattr(
            comfy.memory_management, "aimdo_enabled", True, raising=False
        )
        gen = iter_safetensors_tensors(str(tmp_path / "single.safetensors"))
        _, view = next(gen)
        gen.close()
        clone = view.clone()
        assert self._has_file_slice(view)
        assert not self._has_file_slice(clone)
        assert torch.equal(view, clone)

    def test_early_exit_does_not_raise(self, tmp_path, monkeypatch):
        """Breaking mid-stream closes cleanly — the finally's views.clear()
        must not fight GeneratorExit."""
        import comfy.memory_management

        self._write_single(tmp_path)
        monkeypatch.setattr(
            comfy.memory_management, "aimdo_enabled", True, raising=False
        )
        gen = iter_safetensors_tensors(str(tmp_path / "single.safetensors"))
        next(gen)
        gen.close()


class TestPlaceTensorOnDevice:
    """How a checkpoint tensor reaches the GPU decides whether the load
    touches the SSD and leaves host RAM full.

    ``Tensor.to(cuda)`` on a mapped view is a host-side read: 0.55 GB/s and
    +2.18 GB of machine RAM per 2 GB (report §F6). When aimdo mapped the file,
    core's ``read_tensor_file_slice_into`` DMAs the byte range into the
    destination instead: ~2.6 GB/s, page cache left clean. Every route goes
    through this helper so that choice is made in exactly one place.
    """

    CUDA = torch.device("cuda", 0)

    def _skip_without_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("no GPU in this environment")

    def test_non_cuda_device_is_a_no_op(self):
        t = torch.ones(4)
        assert place_tensor_on_device(t, None) is t
        assert place_tensor_on_device(t, torch.device("cpu")) is t

    def test_plain_tensor_falls_back_to_to(self):
        self._skip_without_gpu()
        t = torch.ones(8, dtype=torch.bfloat16)
        # No aimdo mapping on this storage -> the plain copy path.
        out = place_tensor_on_device(t, self.CUDA)
        assert out.device.type == "cuda"
        assert torch.equal(out.cpu(), t)

    def test_file_slice_prefers_the_core_dma(self):
        """With an aimdo mapping present, core's DMA is used, not .to()."""
        self._skip_without_gpu()
        import comfy.memory_management as mm

        t = torch.ones(8, dtype=torch.bfloat16)
        seen = {}

        class _Storage:
            _comfy_tensor_file_slice = object()

        real_storage = t.untyped_storage

        def _fake_read(tensor, destination, stream=None, destination2=None):
            seen["called"] = True
            seen["src_is_view"] = tensor is t
            destination.copy_(tensor)
            return True

        with patch.object(type(t), "untyped_storage",
                          lambda self: _Storage()),              patch.object(mm, "read_tensor_file_slice_into", _fake_read):
            out = place_tensor_on_device(t, self.CUDA)

        assert seen.get("called") is True
        assert seen["src_is_view"] is True
        assert out.device.type == "cuda"
        assert torch.equal(out.cpu(), t)

    def test_dma_declining_falls_back_instead_of_raising(self):
        """Core returns False for tensors it will not DMA; we still place it."""
        self._skip_without_gpu()
        import comfy.memory_management as mm

        t = torch.ones(8, dtype=torch.bfloat16)

        class _Storage:
            _comfy_tensor_file_slice = object()

        with patch.object(type(t), "untyped_storage",
                          lambda self: _Storage()),              patch.object(mm, "read_tensor_file_slice_into",
                          lambda *a, **k: False):
            out = place_tensor_on_device(t, self.CUDA)

        assert out.device.type == "cuda"
        assert torch.equal(out.cpu(), t)

    def test_dma_raising_falls_back(self):
        """A missing aimdo native library raises; a slow copy beats a crash."""
        self._skip_without_gpu()
        import comfy.memory_management as mm

        t = torch.ones(8, dtype=torch.bfloat16)

        class _Storage:
            _comfy_tensor_file_slice = object()

        def _boom(*a, **k):
            raise AttributeError("'NoneType' object has no attribute 'lib'")

        with patch.object(type(t), "untyped_storage",
                          lambda self: _Storage()),              patch.object(mm, "read_tensor_file_slice_into", _boom):
            out = place_tensor_on_device(t, self.CUDA)

        assert out.device.type == "cuda"
        assert torch.equal(out.cpu(), t)
