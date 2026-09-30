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


def iter_safetensors_tensors(ckpt_path: str):
    """Yield ``(key, tensor)`` one at a time from a safetensors file.

    Two read layers, selected by the same switch core's own loaders use:

    - aimdo ON (the DynamicVRAM install): ``comfy.utils.load_torch_file``
      routes to core's ``load_safetensors`` (``comfy/utils.py:165-166``),
      which maps the file once and hands back per-tensor ``torch.frombuffer``
      VIEWS whose storages pin the mapping themselves
      (``_comfy_tensor_mmap_refs``, ``comfy/utils.py:148-153``). Holding all
      views costs only file-backed page cache — no private commit — so the
      FULL view dict may exist for the duration of the stream without
      breaking the contract below; the only private bytes are the ones the
      CONSUMER clones. This deletes the read layer that used to sit beside
      that clone.
    - aimdo OFF: the original ``safe_open`` stream, one tensor materialised
      at a time.

    Per-tensor streaming contract (unchanged): a consumer that keeps a
    yielded tensor past the iteration MUST ``clone()`` it into memory it
    owns before handing it to a parameter that may later be
    ``cudaHostRegister``-pinned. Under the aimdo arm the yielded tensor IS a
    file view (kept views would pin the whole mapping / fail pinning); under
    the fallback arm it is an owned read. Every current consumer
    (``_stream_apply_dense``, ``_stream_apply_safetensors``) clones
    unconditionally, so the clone's size does not depend on the arm.

    MEASURED, 2026-09-29 (tests/probe_safetensors_aliases_mapping.py, this
    host): a ``safe_open`` read moves ``private`` by ~1x file on this host
    (charged private, NOT ws/uss, unlike a plain mmap read) — and that cost
    sits beside the consumer's mandatory clone, which is how the 7B fp8
    load reached 17.58GB private for a 9.47GB file (live ``[vvrss]``,
    2026-09-30). The aimdo arm removes the read half.
    """
    import comfy.memory_management

    if getattr(comfy.memory_management, "aimdo_enabled", False):
        import comfy.utils

        views = comfy.utils.load_torch_file(ckpt_path)
        try:
            yield from views.items()
        finally:
            # Prompt release of every view the consumer did not keep; a kept
            # view still self-pins the mapping via its storage refs, so a
            # mid-iteration close cannot dangle.
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
            logger.debug(f"Loading shard: {shard_filename}")
            shard_state_dict = comfy.utils.load_torch_file(shard_path, device=device)
            merged_state_dict.update(shard_state_dict)
            del shard_state_dict

        logger.debug(
            f"Merged {len(merged_state_dict)} parameters from {len(shard_filenames)} shards"
        )
        return merged_state_dict

    @staticmethod
    def iter_checkpoint_tensors(ckpt_path: str):
        """Yield ``(key, tensor)`` from a single-file checkpoint.

        safetensors files stream per-tensor (see
        :func:`iter_safetensors_tensors`); other formats (``.bin`` / ``.pt``)
        are pickle archives that cannot be streamed and fall back to
        ``comfy.utils.load_torch_file`` + item iteration.

        Tensors from a safetensors file are aimdo file views when aimdo is
        enabled, otherwise owned deserialised copies (see
        :func:`iter_safetensors_tensors` for the measurement); a ``.bin`` /
        ``.pt`` tensor may be a ``torch.load(mmap=True)`` view when
        ``comfy.utils.MMAP_TORCH_FILES`` is set. Either way a consumer that
        keeps a tensor must copy it into memory it owns first.

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
        checkpoint priority (single safetensors -> sharded index -> single
        bin -> sharded bin index) but never materializes a merged dict —
        peak host RAM is bounded by the model plus one tensor in flight, and
        each shard's file mapping can be released as soon as its tensors have
        been consumed.

        Tensors from safetensors shards are aimdo file views when aimdo is
        enabled, otherwise owned deserialised copies; a ``.bin`` shard read
        under ``MMAP_TORCH_FILES`` may yield genuine views. A consumer that
        keeps a tensor must copy it into memory it owns first (a retained
        view pins its whole shard mapping in the process working set).

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
        logger.info(f"Loading {len(shard_filenames)} shards from {local_dir}")

        for shard_filename in shard_filenames:
            shard_path = os.path.join(local_dir, shard_filename)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(
                    f"Shard file not found: {shard_path} (referenced in {ckpt_path})"
                )
            logger.debug(f"Loading shard: {shard_filename}")
            yield from BaseVibeVoiceLoader.iter_checkpoint_tensors(shard_path)
