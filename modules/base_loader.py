"""Shared base loader for VibeVoice model families (TTS and ASR).

Centralizes the parts of model loading that are identical across families and
do not require a GPU:

- Official-model directory resolution under ``<tts>/VibeVoice/<name>``
- Lazy ``snapshot_download`` (skipped when the model is already present)
- Tokenizer-repo resolution
- Sharded / single checkpoint loading via ``comfy.utils.load_torch_file``

Family-specific loaders (``VibeVoiceLoader``, ``VibeVoiceASRLoader``) inherit
from this base and only add their own model/processor wiring. This removes the
duplicated download / discovery / sharded-load logic tracked in IMP-004.
"""

import os
import json
import logging
import torch

import comfy.utils
import folder_paths

from .model_info import get_tokenizer_repo
from .diagnostics import diagnostics_enabled


_DMA_ANNOUNCED = False


def _default_device() -> torch.device:
    """Return the CPU device (used when no explicit device is supplied)."""
    return torch.device("cpu")


def place_tensor_on_device(tensor, device):
    """Put a checkpoint tensor on ``device`` without staging it in host RAM.

    ``Tensor.to(cuda)`` on a memory-mapped view is a *host-side* read: the copy
    engine faults every page in through the CPU, which measured 0.55 GB/s and
    pushed the bytes through the page cache on the way (+2.18 GB of machine RAM
    per 2 GB read — report §F6, cold regions of a real 16.66 GB checkpoint).
    That is what makes a streamed load drive the SSD and leave RAM full.

    When ComfyUI mapped the file through aimdo, each view's storage carries the
    file reference and byte range, and core's own
    ``read_tensor_file_slice_into`` DMAs that range straight from the file into
    the destination buffer: the same bytes at ~2.6 GB/s with the page cache
    left clean (+0.13 GB per 2 GB). This is core's read primitive, not a second
    loader — it is the one core already uses to page weights into VRAM.

    Anything core cannot DMA (no aimdo mapping, non-contiguous, a tensor we
    built ourselves such as a dequantized weight) falls back to ``.to()``.
    """
    if device is None or getattr(device, "type", None) != "cuda":
        return tensor
    if getattr(tensor.untyped_storage(), "_comfy_tensor_file_slice", None) is None:
        return tensor.to(device)
    try:
        from comfy.memory_management import read_tensor_file_slice_into

        destination = torch.empty(
            tensor.shape, dtype=tensor.dtype, device=device
        )
        if read_tensor_file_slice_into(tensor, destination):
            global _DMA_ANNOUNCED
            if not _DMA_ANNOUNCED and diagnostics_enabled():
                _DMA_ANNOUNCED = True
                logging.info(
                    "[ComfyUI-VibeVoice] [vvload] weights are DMA'd file->VRAM (core aimdo); "
                    "host RAM should stay flat during a load"
                )
            return destination
    except Exception:
        # A missing/older aimdo native library raises here rather than
        # returning False; a plain copy is always correct, just slower.
        logging.debug("[ComfyUI-VibeVoice] file->device DMA unavailable, falling back to .to()",
                     exc_info=True)
    return tensor.to(device)


def iter_safetensors_tensors(ckpt_path: str):
    """Yield ``(key, tensor)`` one at a time from a safetensors file.

    Two read layers, selected by ComfyUI's core configuration:

    - aimdo ON: ``comfy.utils.load_torch_file`` routes to core's
      ``load_safetensors``, which maps the file once and hands back per-tensor
      ``torch.frombuffer`` views whose storages pin the mapping themselves
      (``_comfy_tensor_mmap_refs``). Holding all views costs only file-backed
      page cache — no private commit.
    - aimdo OFF: the standard ``safe_open`` stream, yielding zero-copy mmap
      tensors one tensor at a time.
    """
    import comfy.memory_management

    if getattr(comfy.memory_management, "aimdo_enabled", False):
        import comfy.utils

        views = comfy.utils.load_torch_file(ckpt_path)
        try:
            yield from views.items()
        finally:
            views.clear()
        return

    from safetensors import safe_open

    with safe_open(str(ckpt_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            yield key, f.get_tensor(key)


class BaseVibeVoiceLoader:
    """Base class holding GPU-free, reusable loader helpers."""

    # ------------------------------------------------------------------
    # Path discovery
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_official_model_dir(model_name: str) -> str:
        """Return ``<tts_folder>/VibeVoice/<model_name>`` for an official model.

        Registered tts roots are searched in order and the first candidate
        directory that already exists wins, so an official model placed
        manually under a secondary root (extra_model_paths.yaml) is picked up
        instead of re-downloaded into the primary root. When no candidate
        exists anywhere, the first tts folder's path is returned as the
        download target.
        """
        tts_paths = folder_paths.get_folder_paths("tts")
        if not tts_paths:
            tts_paths = [os.path.join(folder_paths.models_dir, "tts")]
        candidates = [os.path.join(p, "VibeVoice", model_name) for p in tts_paths]
        for candidate in candidates:
            if os.path.isdir(candidate):
                return candidate
        return candidates[0]

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_downloaded(repo_id: str, local_dir: str, model_name: str = "") -> None:
        """Download an official model via ``snapshot_download`` if not present.

        Skips the download when ``config.json`` already exists in ``local_dir``
        (the standard HuggingFace marker for a complete checkout).

        AUD-013: ``local_dir_use_symlinks`` was removed in huggingface_hub >= 0.23
        (and is absent in 1.x); passing it raises ``TypeError`` on the first
        official-model download. Only pass it when the installed version still
        accepts it.
        """
        if not repo_id:
            return
        if os.path.exists(os.path.join(local_dir, "config.json")):
            return
        logging.info(f"[ComfyUI-VibeVoice] Downloading official VibeVoice model: {model_name or local_dir}...")
        import inspect
        from huggingface_hub import snapshot_download

        kwargs = {"repo_id": repo_id, "local_dir": local_dir}
        if "local_dir_use_symlinks" in inspect.signature(snapshot_download).parameters:
            kwargs["local_dir_use_symlinks"] = False
        snapshot_download(**kwargs)

    # ------------------------------------------------------------------
    # Tokenizer repo
    # ------------------------------------------------------------------
    @staticmethod
    def tokenizer_repo_for(model_name: str) -> str:
        """Resolve the HuggingFace tokenizer repo for a model name."""
        return get_tokenizer_repo(model_name)

    # ------------------------------------------------------------------
    # Checkpoint resolution + loading
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_checkpoint_file(model_dir: str):
        """Resolve ``(checkpoint_path, is_sharded)`` for a model directory.

        Priority: single safetensors, sharded safetensors index, single
        pytorch_model.bin, sharded pytorch_model.bin index.

        Raises:
            FileNotFoundError: If no checkpoint file is found.
        """
        single_safetensors = os.path.join(model_dir, "model.safetensors")
        if os.path.isfile(single_safetensors):
            return single_safetensors, False

        sharded_safetensors_index = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.isfile(sharded_safetensors_index):
            return sharded_safetensors_index, True

        single_bin = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.isfile(single_bin):
            return single_bin, False

        sharded_bin_index = os.path.join(model_dir, "pytorch_model.bin.index.json")
        if os.path.isfile(sharded_bin_index):
            return sharded_bin_index, True

        raise FileNotFoundError(
            f"No checkpoint file found in model directory: {model_dir}. "
            f"Expected one of: model.safetensors, model.safetensors.index.json, "
            f"pytorch_model.bin, pytorch_model.bin.index.json"
        )

    @staticmethod
    def load_state_dict_sharded(local_dir: str, device=None) -> dict:
        """Load a checkpoint (sharded or single) into a single state dict.

        Args:
            local_dir: Directory containing the checkpoint file(s).
            device: Optional torch device to load tensors onto.

        Returns:
            A single merged state dict.

        Raises:
            FileNotFoundError: If the model directory or a referenced shard is missing.
            ValueError: If a sharded index has an empty ``weight_map``.
        """
        if device is None:
            device = _default_device()

        ckpt_path, is_sharded = BaseVibeVoiceLoader._resolve_checkpoint_file(local_dir)

        if not is_sharded:
            return comfy.utils.load_torch_file(ckpt_path, device=device)

        with open(ckpt_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        weight_map = index.get("weight_map", {})
        if not weight_map:
            raise ValueError(f"Sharded checkpoint index '{ckpt_path}' has empty weight_map")

        shard_filenames = sorted(set(weight_map.values()))
        logging.debug(f"[ComfyUI-VibeVoice] Loading {len(shard_filenames)} shards from {local_dir}")

        merged_state_dict = {}
        for shard_filename in shard_filenames:
            shard_path = os.path.join(local_dir, shard_filename)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(
                    f"Shard file not found: {shard_path} (referenced in {ckpt_path})"
                )
            logging.debug(f"[ComfyUI-VibeVoice] Loading shard: {shard_filename}")
            shard_state_dict = comfy.utils.load_torch_file(shard_path, device=device)
            merged_state_dict.update(shard_state_dict)
            del shard_state_dict

        logging.debug(
            f"[ComfyUI-VibeVoice] Merged {len(merged_state_dict)} parameters from {len(shard_filenames)} shards"
        )
        return merged_state_dict

    @staticmethod
    def iter_checkpoint_tensors(ckpt_path: str):
        """Yield ``(key, tensor)`` from a single-file checkpoint.

        safetensors files stream per-tensor (see
        :func:`iter_safetensors_tensors`); other formats (``.bin`` / ``.pt``)
        are pickle archives that fall back to ``comfy.utils.load_torch_file``.

        Args:
            ckpt_path: Path to a single checkpoint file.

        Yields:
            ``(key, tensor)`` pairs on CPU.
        """
        if str(ckpt_path).lower().endswith(".safetensors"):
            yield from iter_safetensors_tensors(ckpt_path)
            return
        state_dict = comfy.utils.load_torch_file(ckpt_path, device=_default_device())
        yield from state_dict.items()

    @staticmethod
    def iter_sharded_tensors(local_dir: str):
        """Yield ``(key, tensor)`` across every shard of a checkpoint directory.

        Streaming twin of :meth:`load_state_dict_sharded`: resolves the same
        checkpoint priority but never materializes a full merged dict in RAM.

        Args:
            local_dir: Directory containing the checkpoint file(s).

        Yields:
            ``(key, tensor)`` pairs on CPU.

        Raises:
            FileNotFoundError: If the directory or a referenced shard is missing.
            ValueError: If a sharded index has an empty ``weight_map``.
        """
        ckpt_path, is_sharded = BaseVibeVoiceLoader._resolve_checkpoint_file(local_dir)

        if not is_sharded:
            yield from BaseVibeVoiceLoader.iter_checkpoint_tensors(ckpt_path)
            return

        with open(ckpt_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        weight_map = index.get("weight_map", {})
        if not weight_map:
            raise ValueError(f"Sharded checkpoint index '{ckpt_path}' has empty weight_map")

        shard_filenames = sorted(set(weight_map.values()))
        logging.debug(f"[ComfyUI-VibeVoice] Loading {len(shard_filenames)} shards from {local_dir}")

        for shard_filename in shard_filenames:
            shard_path = os.path.join(local_dir, shard_filename)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(
                    f"Shard file not found: {shard_path} (referenced in {ckpt_path})"
                )
            logging.debug(f"[ComfyUI-VibeVoice] Loading shard: {shard_filename}")
            yield from BaseVibeVoiceLoader.iter_checkpoint_tensors(shard_path)