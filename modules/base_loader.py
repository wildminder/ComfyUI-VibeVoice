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

logger = logging.getLogger(__name__)


def _default_device() -> torch.device:
    """Return the CPU device (used when no explicit device is supplied)."""
    return torch.device("cpu")


class BaseVibeVoiceLoader:
    """Base class holding GPU-free, reusable loader helpers."""

    # ------------------------------------------------------------------
    # Path discovery
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_official_model_dir(model_name: str) -> str:
        """Return ``<tts_folder>/VibeVoice/<model_name>`` for an official model.

        Uses ComfyUI's ``folder_paths`` so the result follows the same layout
        the TTS and ASR loaders previously computed by hand.
        """
        tts_paths = folder_paths.get_folder_paths("tts")
        base_path = tts_paths[0] if tts_paths else os.path.join(folder_paths.models_dir, "tts")
        return os.path.join(base_path, "VibeVoice", model_name)

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
        logger.info(f"Downloading official VibeVoice model: {model_name or local_dir}...")
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

        # Sharded: read the weight_map and merge each shard.
        with open(ckpt_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        weight_map = index.get("weight_map", {})
        if not weight_map:
            raise ValueError(f"Sharded checkpoint index '{ckpt_path}' has empty weight_map")

        shard_filenames = sorted(set(weight_map.values()))
        logger.info(f"Loading {len(shard_filenames)} shards from {local_dir}")

        merged_state_dict = {}
        for shard_filename in shard_filenames:
            shard_path = os.path.join(local_dir, shard_filename)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(
                    f"Shard file not found: {shard_path} (referenced in {ckpt_path})"
                )
            logger.info(f"Loading shard: {shard_filename}")
            shard_state_dict = comfy.utils.load_torch_file(shard_path, device=device)
            merged_state_dict.update(shard_state_dict)
            del shard_state_dict

        logger.info(
            f"Merged {len(merged_state_dict)} parameters from {len(shard_filenames)} shards"
        )
        return merged_state_dict
