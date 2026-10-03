"""Model loading and caching for VibeVoice models.

Handles:
- Model path resolution (official, local_dir, standalone)
- Config loading (with fallback to packaged defaults)
- Tokenizer loading (with download fallback)
- Processor setup
- Model instantiation with dtype/attention/quantization settings
- Model caching by cache key
"""

import os
import json
import gc
import logging
import contextlib
import torch

import comfy.utils
import folder_paths
import comfy.model_management as model_management

from transformers import BitsAndBytesConfig

from ..src.vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from ..src.vibevoice.modular.configuration_vibevoice_streaming import VibeVoiceStreamingConfig
from ..src.vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from ..src.vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from ..src.vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from ..src.vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
from ..src.vibevoice.processor.vibevoice_tokenizer_processor import VibeVoiceTokenizerProcessor
from ..src.vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast

from .model_info import AVAILABLE_VIBEVOICE_MODELS, MODEL_CONFIGS
from .base_loader import BaseVibeVoiceLoader, place_tensor_on_device
from .attention_utils import (
    SAGE_ATTENTION_AVAILABLE,
    ATTENTION_MODES,
    resolve_attention_mode,
    get_attn_implementation_for_load,
    check_sage_attention_compatible,
)
from .dtype_utils import resolve_dtype, get_dtype_str, cast_model_to_dtype_if_needed, set_config_dtype

if SAGE_ATTENTION_AVAILABLE:
    from ..src.vibevoice.modular.sage_attention_patch import set_sage_attention

from huggingface_hub import hf_hub_download


# Sentinel buffer values that must be restored to their intended initial state
# after meta-init materialization (regression fix, v2.3.1). The VibeVoice model
# registers ``speech_scaling_factor`` / ``speech_bias_factor`` as
# ``torch.tensor(float('nan'))`` and computes them at inference time
# (modeling_vibevoice.py:324-338). The diffusion-inversion gate at
# modeling_vibevoice.py:623 is ``if not torch.isnan(sf) and not torch.isnan(bf)``
# — zeroing these buffers makes the gate TRUE and applies ``speech / 0 - 0``,
# producing silent/garbage output. ``fix_std`` (persistent=False) is likewise
# restored to its config value (default 0.5 for the acoustic tokenizer).
#
# Keys are matched by the buffer's FINAL dotted component (e.g.
# ``speech_scaling_factor`` matches ``model.speech_scaling_factor``), so the
# lookup is robust to submodule nesting.
_SENTINEL_BUFFER_VALUES = {
    "speech_scaling_factor": float("nan"),
    "speech_bias_factor": float("nan"),
    "fix_std": 0.5,
}

# Subtrees a released checkpoint may omit without indicating a broken load.
#
# VibeVoice-Realtime-0.5B ships an acoustic-tokenizer DECODER only (605 keys,
# 276 of them the decoder). Its encoder is not part of the released TTS weights
# and the generate path never calls it — it exists so the tokenizer can encode
# audio for voice-cloning / training pipelines. Vanilla
# `from_pretrained` reports the same 276 keys as MISSING for that checkpoint,
# which is the reference behaviour; we match it instead of printing an
# alarming warning on every realtime load, where it could mask a genuinely
# missing key.
#
# A prefix is only treated as optional when the checkpoint supplied NO key
# under it. If any encoder key is present, the rest are reported normally.
OPTIONAL_ABSENT_PREFIXES = ("acoustic_tokenizer.encoder.",)

# Quant-storage dtypes that must NEVER reach a dense float loader.
#
# Single source of truth for both dense routes: the batch check
# (``external_loader._assert_dense_loadable``) and the per-tensor streaming
# assign (``VibeVoiceLoader._stream_apply_dense``). ``external_loader``
# imports this name from here (it already imports ``VibeVoiceLoader``), so the
# dependency runs one way only — ``loader`` must never import from
# ``external_loader``, which would be a cycle.
#
# Assigning int8/uint8/fp8 storages into float parameters either crashes
# cryptically or (fp8) silently misloads with scales ignored. Both are worse
# than a clear error at load time.
QUANT_STORAGE_DTYPES = frozenset({
    torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2,
})


def mark_optional_absent(
    missing_keys: list[str],
    assigned: set[str],
    known_missing: set[str] | None = None,
) -> set[str]:
    """Add deliberately-omitted keys to ``known_missing``.

    A prefix is only treated as optional when the checkpoint supplied NO key
    under it. If even one key arrived, the remaining gaps under that prefix are
    a real finding and stay reported.
    """
    known = set(known_missing or ())
    seen = {
        prefix
        for prefix in OPTIONAL_ABSENT_PREFIXES
        if any(key.startswith(prefix) for key in assigned)
    }
    for key in missing_keys:
        for prefix in OPTIONAL_ABSENT_PREFIXES:
            if key.startswith(prefix) and prefix not in seen:
                known.add(key)
                break
    return known


def _recompute_rope_buffers(model) -> int:
    """Recompute RoPE ``inv_freq`` buffers destroyed by meta-init (v2.3.2 fix).

    Rotary embeddings compute ``inv_freq`` / ``original_inv_freq`` from the
    config inside ``__init__`` — they are **non-persistent buffers that never
    appear in the checkpoint**. Under ``torch.device("meta")`` instantiation
    they are created on the meta device, and the generic zero-materialization
    in :meth:`VibeVoiceLoader._apply_state_dict` turns them into all-zeros.

    With ``inv_freq == 0`` RoPE yields ``cos(0)=1`` / ``sin(0)=0``, i.e. **no
    positional encoding**: the language model cannot order tokens, so it emits
    gibberish syllables and hits the speech-end token almost immediately
    (observed as a ~2 s garbled clip). A deterministic eager-vs-meta diff on the
    real VibeVoice-1.5B checkpoint showed these two buffers were the ONLY
    tensors that differed (all 1204 checkpoint params matched exactly).

    This helper re-runs each rotary module's own config-based computation on
    CPU (mirroring the eager ``__init__``) and re-registers the buffers.

    Args:
        model: The loaded model (post assign-load + buffer materialization).

    Returns:
        Number of rotary modules whose ``inv_freq`` was recomputed.
    """
    recomputed = 0
    for module in model.modules():
        buffers = dict(module.named_buffers(recurse=False))
        if "inv_freq" not in buffers:
            continue
        config = getattr(module, "config", None)
        if config is None:
            continue
        try:
            rope_type = getattr(module, "rope_type", "default")
            if rope_type == "default" and hasattr(module, "compute_default_rope_parameters"):
                inv_freq, _ = module.compute_default_rope_parameters(config, device="cpu")
            else:
                from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
                rope_init_fn = ROPE_INIT_FUNCTIONS.get(rope_type)
                if rope_init_fn is None:
                    logging.warning(
                        f"[VibeVoice TTS] No rope init function for type '{rope_type}' on "
                        f"{module.__class__.__name__}; leaving inv_freq as-is."
                    )
                    continue
                inv_freq, _ = rope_init_fn(config, device="cpu")
            inv_freq = inv_freq.to(device="cpu")
            module.register_buffer("inv_freq", inv_freq, persistent=False)
            if "original_inv_freq" in buffers:
                module.register_buffer(
                    "original_inv_freq", inv_freq.clone(), persistent=False
                )
            recomputed += 1
        except Exception as e:
            logging.warning(
                f"[VibeVoice TTS] Could not recompute RoPE inv_freq for "
                f"{module.__class__.__name__}: {e}"
            )
    if recomputed:
        logging.debug(f"[VibeVoice TTS] Recomputed RoPE inv_freq for {recomputed} rotary module(s).")
    return recomputed


def _assert_shapes_compatible(model, state_dict, max_reported: int = 5) -> None:
    """Raise a friendly error when checkpoint shapes contradict the model."""
    model_tensors = dict(model.named_parameters(remove_duplicate=False))
    model_tensors.update(dict(model.named_buffers()))

    mismatches = []
    for key, tensor in state_dict.items():
        model_tensor = model_tensors.get(key)
        if model_tensor is None or not hasattr(tensor, "shape"):
            continue
        if tuple(tensor.shape) != tuple(model_tensor.shape):
            mismatches.append(
                (key, tuple(tensor.shape), tuple(model_tensor.shape))
            )

    if not mismatches:
        return
    raise _shape_mismatch_error(model, mismatches, max_reported)


def _shape_mismatch_error(model, mismatches, max_reported: int = 5):
    """Build the friendly config/weights-mismatch ValueError."""
    lines = [
        f"  {key}: checkpoint shape {ckpt} vs model shape {mdl}"
        for key, ckpt, mdl in mismatches[:max_reported]
    ]
    if len(mismatches) > max_reported:
        lines.append(f"  ... and {len(mismatches) - max_reported} more")
    return ValueError(
        f"Checkpoint weight shapes do not match the "
        f"'{type(model).__name__}' model for {len(mismatches)} shared "
        f"key(s):\n" + "\n".join(lines) +
        "\nThis almost always means the selected config does not match the "
        "weight file. Verify config_name (or the sidecar config.json) "
        "matches the checkpoint's architecture, or use 'Auto-detect'."
    )

# Cache for loaded (model, processor) tuples, keyed by cache_key
LOADED_MODELS_CACHE = {}


def cleanup_old_models(keep_cache_key: str = None) -> None:
    """Remove all cached models except the one matching keep_cache_key."""
    from .utils import VIBEVOICE_PATCHER_CACHE
    from .model_registry import evict_patcher

    keys_to_remove = []
    for key in list(LOADED_MODELS_CACHE.keys()):
        if key != keep_cache_key:
            keys_to_remove.append(key)
            del LOADED_MODELS_CACHE[key]

    for key in list(VIBEVOICE_PATCHER_CACHE.keys()):
        if key != keep_cache_key:
            patcher = VIBEVOICE_PATCHER_CACHE.get(key)
            try:
                evict_patcher(patcher, VIBEVOICE_PATCHER_CACHE, key)
            except Exception as e:
                logging.warning(f"[VibeVoice TTS] Error cleaning up patcher {key}: {e}")

    if keys_to_remove:
        logging.debug(f"[VibeVoice TTS] Cleaned up cached models: {keys_to_remove}")
        gc.collect()
        model_management.soft_empty_cache()


class VibeVoiceModelHandler(torch.nn.Module):
    """A lightweight handler for a VibeVoice model.

    Acts as a container that ComfyUI's ModelPatcher can manage, while the
    actual heavy model is loaded on demand.
    """

    def __init__(
        self,
        model_pack_name: str,
        attention_mode: str = "eager",
        use_llm_4bit: bool = False,
        dtype_str: str = "auto",
    ):
        super().__init__()
        self.model_pack_name = model_pack_name
        self.attention_mode = attention_mode
        self.use_llm_4bit = use_llm_4bit
        self.dtype_str = dtype_str
        self.cache_key = f"{self.model_pack_name}_attn_{attention_mode}_q4_{int(use_llm_4bit)}"
        self.model = None
        self.processor = None
        self.device = None

        info = AVAILABLE_VIBEVOICE_MODELS.get(model_pack_name, {})
        size_gb = MODEL_CONFIGS.get(model_pack_name, {}).get("size_gb", 4.0)
        self.size = int(size_gb * (1024**3))

    def load_model(self, device, attention_mode: str = "eager"):
        """Load the model and processor into memory."""
        self.model, self.processor = VibeVoiceLoader.load_model(
            self.model_pack_name,
            device,
            attention_mode,
            use_llm_4bit=self.use_llm_4bit,
            dtype_str=self.dtype_str,
        )
        self._refine_size()

    def _refine_size(self) -> None:
        """Refine ``self.size`` from the real parameters after a load."""
        try:
            total = sum(
                p.numel() * p.element_size() for p in self.model.parameters()
            )
            if total > 0:
                self.size = total
        except Exception:
            pass


class VibeVoiceLoader(BaseVibeVoiceLoader):
    """Static loader class for VibeVoice models."""

    @staticmethod
    def _resolve_model_paths(model_name: str) -> tuple:
        """Resolve model paths based on the model type."""
        model_info = AVAILABLE_VIBEVOICE_MODELS[model_name]
        model_type = model_info["type"]

        model_path = None
        config_path = None
        preprocessor_config_path = None
        tokenizer_dir = None

        if model_type == "official":
            model_path = BaseVibeVoiceLoader._resolve_official_model_dir(model_name)
            BaseVibeVoiceLoader._ensure_downloaded(
                repo_id=model_info.get("repo_id", ""),
                local_dir=model_path,
                model_name=model_name,
            )
            config_path = os.path.join(model_path, "config.json")
            preprocessor_config_path = os.path.join(model_path, "preprocessor_config.json")
            tokenizer_dir = model_path

        elif model_type == "local_dir":
            model_path = model_info["path"]
            config_path = os.path.join(model_path, "config.json")
            preprocessor_config_path = os.path.join(model_path, "preprocessor_config.json")
            tokenizer_dir = model_path

        elif model_type == "standalone":
            model_path = None
            config_path = os.path.splitext(model_info["path"])[0] + ".config.json"
            preprocessor_config_path = os.path.splitext(model_info["path"])[0] + ".preprocessor.json"
            tokenizer_dir = os.path.dirname(model_info["path"])

        return model_path, config_path, preprocessor_config_path, tokenizer_dir

    @staticmethod
    def _load_config(config_path: str, model_name: str):
        """Load VibeVoice config from path, with fallback to packaged default."""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            model_type = config_data.get("model_type", "")

            if model_type == "vibevoice_streaming":
                logging.debug(f"[VibeVoice TTS] Detected streaming config for '{model_name}'")
                return VibeVoiceStreamingConfig.from_pretrained(config_path)
            else:
                return VibeVoiceConfig.from_pretrained(config_path)

        fallback_name = (
            "default_VibeVoice-Large_config.json"
            if "large" in model_name.lower()
            else "default_VibeVoice-1.5B_config.json"
        )
        fallback_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "vibevoice", "configs", fallback_name
        )
        logging.warning(f"[VibeVoice TTS] Config not found for '{model_name}'. Using fallback: {fallback_name}")
        return VibeVoiceConfig.from_pretrained(fallback_path)

    @staticmethod
    def _load_tokenizer(tokenizer_dir: str, model_name: str) -> "VibeVoiceTextTokenizerFast":
        """Load the VibeVoice text tokenizer."""
        tokenizer_file_path = os.path.join(tokenizer_dir, "tokenizer.json")

        if not os.path.exists(tokenizer_file_path):
            logging.debug(f"[VibeVoice TTS] 'tokenizer.json' not found in model directory: {tokenizer_dir}")

            packaged_configs_dir = os.path.join(
                os.path.dirname(__file__), "..", "src", "vibevoice", "configs"
            )
            packaged_tokenizer_path = os.path.join(packaged_configs_dir, "tokenizer.json")

            if os.path.exists(packaged_tokenizer_path):
                logging.debug("[VibeVoice TTS] Using pre-packaged tokenizer directly from the node folder...")
                return VibeVoiceTextTokenizerFast(tokenizer_file=packaged_tokenizer_path)

            repos_to_try = ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-7B"]
            download_successful = False
            last_error = None

            for repo_id in repos_to_try:
                logging.debug(f"[VibeVoice TTS] Attempting to download 'tokenizer.json' from Hugging Face repo '{repo_id}'...")
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename="tokenizer.json",
                        local_dir=tokenizer_dir,
                    )
                    download_successful = True
                    logging.debug("[VibeVoice TTS] Download successful.")
                    break
                except Exception as e:
                    logging.warning(f"[VibeVoice TTS] Failed to download from '{repo_id}': {e}")
                    last_error = e

            if not download_successful:
                error_message = (
                    f"FATAL: Could not get 'tokenizer.json'. All download attempts failed.\n"
                    f"Last error: {last_error}\n\n"
                    f"ACTION REQUIRED:\n"
                    f"1. Manually download 'tokenizer.json' from "
                    f"https://huggingface.co/{repos_to_try[0]}/blob/main/tokenizer.json\n"
                    f"2. Place the downloaded file in the following directory:\n   '{tokenizer_dir}'"
                )
                raise RuntimeError(error_message)

        return VibeVoiceTextTokenizerFast(tokenizer_file=tokenizer_file_path)

    @staticmethod
    def _load_processor(
        tokenizer,
        preprocessor_config_path: str,
        is_streaming: bool = False,
    ):
        """Load the VibeVoice processor with tokenizer and audio processor."""
        processor_config_data = {}
        if os.path.exists(preprocessor_config_path):
            with open(preprocessor_config_path, 'r', encoding='utf-8') as f:
                processor_config_data = json.load(f)

        audio_processor = VibeVoiceTokenizerProcessor()

        if is_streaming:
            processor = VibeVoiceStreamingProcessor(
                tokenizer=tokenizer,
                audio_processor=audio_processor,
                speech_tok_compress_ratio=processor_config_data.get("speech_tok_compress_ratio", 3200),
                db_normalize=processor_config_data.get("db_normalize", True),
            )
        else:
            processor = VibeVoiceProcessor(
                tokenizer=tokenizer,
                audio_processor=audio_processor,
                speech_tok_compress_ratio=processor_config_data.get("speech_tok_compress_ratio", 3200),
                db_normalize=processor_config_data.get("db_normalize", True),
            )
        return processor

    @staticmethod
    def _instantiate_model(
        config,
        is_streaming: bool,
        attn_implementation: str,
        final_load_dtype: torch.dtype,
        use_meta: bool = True,
    ):
        """Instantiate the model class directly, bypassing from_pretrained()."""
        if hasattr(config, 'decoder_config'):
            config.decoder_config._attn_implementation = attn_implementation

        set_config_dtype(config, final_load_dtype)
        if hasattr(config, 'decoder_config'):
            set_config_dtype(config.decoder_config, final_load_dtype)

        ctx = torch.device("meta") if use_meta else contextlib.nullcontext()
        with ctx:
            if is_streaming:
                model = VibeVoiceStreamingForConditionalGenerationInference(config)
            else:
                model = VibeVoiceForConditionalGeneration(config)

        return model

    @staticmethod
    def _resolve_checkpoint_path(model_path, model_type, model_info):
        """Resolve the checkpoint file path(s) from a model directory or standalone path."""
        if model_type == "standalone":
            ckpt_path = model_info["path"]
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Standalone checkpoint not found: {ckpt_path}")
            return ckpt_path, False

        if model_path is None or not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        single_safetensors = os.path.join(model_path, "model.safetensors")
        if os.path.isfile(single_safetensors):
            logging.debug(f"[VibeVoice TTS] Found single safetensors checkpoint: {single_safetensors}")
            return single_safetensors, False

        sharded_safetensors_index = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.isfile(sharded_safetensors_index):
            logging.debug(f"[VibeVoice TTS] Found sharded safetensors checkpoint: {sharded_safetensors_index}")
            return sharded_safetensors_index, True

        single_bin = os.path.join(model_path, "pytorch_model.bin")
        if os.path.isfile(single_bin):
            logging.debug(f"[VibeVoice TTS] Found single PyTorch checkpoint: {single_bin}")
            return single_bin, False

        sharded_bin_index = os.path.join(model_path, "pytorch_model.bin.index.json")
        if os.path.isfile(sharded_bin_index):
            logging.debug(f"[VibeVoice TTS] Found sharded PyTorch checkpoint: {sharded_bin_index}")
            return sharded_bin_index, True

        raise FileNotFoundError(
            f"No checkpoint file found in model directory: {model_path}. "
            f"Expected one of: model.safetensors, model.safetensors.index.json, "
            f"pytorch_model.bin, pytorch_model.bin.index.json"
        )

    @staticmethod
    def _load_sharded_state_dict(index_path, model_dir, device):
        """Load and merge a sharded checkpoint by delegating to the shared base."""
        return BaseVibeVoiceLoader.load_state_dict_sharded(model_dir, device)

    @staticmethod
    def _apply_state_dict(model, state_dict, known_missing=None):
        """Load a state dict into the model with assign semantics."""
        known_missing = known_missing or set()
        _assert_shapes_compatible(model, state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict, strict=False, assign=True
        )
        return VibeVoiceLoader._post_assign_fixups(
            model, missing_keys, unexpected_keys, known_missing
        )

    @staticmethod
    def _post_assign_fixups(model, missing_keys, unexpected_keys,
                            known_missing=None):
        """Post-assign model fixups shared by the batch and streaming loads."""
        known_missing = known_missing or set()

        config = getattr(model, "config", None)
        tied = False
        if config is not None and hasattr(model, "tie_weights"):
            decoder_config = getattr(config, "decoder_config", None)
            tied = bool(getattr(decoder_config, "tie_word_embeddings", False)) or \
                bool(getattr(config, "tie_word_embeddings", False))
            if tied:
                model.tie_weights()

        for name, param in list(model.named_parameters()):
            if param.is_meta:
                param_new = torch.zeros(param.shape, dtype=param.dtype, device="cpu")
                comfy.utils.set_attr_param(model, name, param_new)

        for name, buf in list(model.named_buffers()):
            if buf.is_meta:
                leaf = name.split(".")[-1]
                if leaf in _SENTINEL_BUFFER_VALUES:
                    val = _SENTINEL_BUFFER_VALUES[leaf]
                    buf_new = torch.tensor(
                        val, dtype=buf.dtype, device="cpu"
                    ).reshape(buf.shape)
                else:
                    buf_new = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu")
                comfy.utils.set_attr(model, name, buf_new)

        _recompute_rope_buffers(model)

        if tied and "lm_head.weight" in missing_keys:
            logging.debug(
                "[VibeVoice TTS] lm_head.weight absent from checkpoint (tied to input "
                "embeddings via tie_word_embeddings) — expected."
            )
        reported_missing = [
            k for k in missing_keys
            if k not in known_missing and not (tied and k == "lm_head.weight")
        ]
        if reported_missing:
            logging.warning(f"[VibeVoice TTS] Missing keys when loading state dict: {len(reported_missing)} keys")
            if len(reported_missing) < 20:
                logging.warning(f"[VibeVoice TTS] Missing keys: {reported_missing}")
            else:
                logging.warning(f"[VibeVoice TTS] First 10 missing keys: {reported_missing[:10]}")

        if unexpected_keys:
            logging.warning(f"[VibeVoice TTS] Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
            if len(unexpected_keys) < 20:
                logging.warning(f"[VibeVoice TTS] Unexpected keys: {unexpected_keys}")
            else:
                logging.warning(f"[VibeVoice TTS] First 10 unexpected keys: {unexpected_keys[:10]}")

        return missing_keys, unexpected_keys

    @staticmethod
    def _stream_apply_dense(
        model, tensor_pairs, known_missing=None, preserve_file_views: bool = True,
        target_device=None,
    ):
        """Assign a dense checkpoint per-tensor directly from zero-copy file views.

        ``target_device`` moves each tensor onto the accelerator as it is
        assigned, so the checkpoint is never materialised as a whole CPU model
        first. This is the one dense assign used by every load route.
        """
        known_missing = known_missing or set()
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        assigned = set()
        unexpected = []
        to_device = (
            target_device is not None and getattr(target_device, "type", None) == "cuda"
        )

        def _placed(tensor):
            if to_device:
                return place_tensor_on_device(tensor, target_device)
            return tensor if preserve_file_views else tensor.clone()

        for key, tensor in tensor_pairs:
            if tensor.dtype in QUANT_STORAGE_DTYPES:
                raise ValueError(
                    "Checkpoint contains quantized-weight tensors but carries "
                    "no executable quantization metadata (first seen at "
                    f"'{key}' [{tensor.dtype}]). Loading them as floats would "
                    "corrupt the model. Re-export it with *.comfy_quant "
                    "metadata (comfy-model-tools) or use the dense/BF16 "
                    "checkpoint."
                )
            target = params.get(key)
            if target is not None:
                if tuple(tensor.shape) != tuple(target.shape):
                    raise _shape_mismatch_error(
                        model, [(key, tuple(tensor.shape), tuple(target.shape))]
                    )
                comfy.utils.set_attr_param(model, key, _placed(tensor))
                assigned.add(key)
                continue
            target_buf = buffers.get(key)
            if target_buf is not None:
                if tuple(tensor.shape) != tuple(target_buf.shape):
                    raise _shape_mismatch_error(
                        model, [(key, tuple(tensor.shape), tuple(target_buf.shape))]
                    )
                comfy.utils.set_attr_buffer(model, key, _placed(tensor))
                assigned.add(key)
                continue
            unexpected.append(key)

        expected = set(model.state_dict().keys())
        missing_keys = [k for k in expected if k not in assigned]
        known_missing = mark_optional_absent(missing_keys, assigned, known_missing)

        return VibeVoiceLoader._post_assign_fixups(
            model, missing_keys, unexpected, known_missing
        )

    @staticmethod
    def _load_state_dict_into_model(
        model,
        model_path: str,
        model_type: str,
        model_info: dict,
        device,
    ):
        """Stream a dense checkpoint into the model using zero-copy file views."""
        ckpt_path, is_sharded = VibeVoiceLoader._resolve_checkpoint_path(
            model_path, model_type, model_info
        )

        if is_sharded:
            model_dir = model_path if model_type != "standalone" else os.path.dirname(ckpt_path)
            tensor_pairs = BaseVibeVoiceLoader.iter_sharded_tensors(model_dir)
        else:
            logging.debug(f"[VibeVoice TTS] Loading state dict from: {ckpt_path}")
            tensor_pairs = BaseVibeVoiceLoader.iter_checkpoint_tensors(ckpt_path)

        VibeVoiceLoader._stream_apply_dense(
            model,
            tensor_pairs,
            preserve_file_views=True,
            target_device=device,
        )

        return model

    @staticmethod
    def load_model(
        model_name: str,
        device,
        attention_mode: str = "eager",
        use_llm_4bit: bool = False,
        dtype_str: str = "auto",
    ):
        """Load a VibeVoice model, downloading if necessary. Caches the loaded model."""
        if model_name not in AVAILABLE_VIBEVOICE_MODELS:
            raise ValueError(
                f"Unknown VibeVoice model: {model_name}. "
                f"Available models: {list(AVAILABLE_VIBEVOICE_MODELS.keys())}"
            )

        attention_mode = resolve_attention_mode(attention_mode, use_llm_4bit)

        cache_key = f"{model_name}_attn_{attention_mode}_q4_{int(use_llm_4bit)}"
        if cache_key in LOADED_MODELS_CACHE:
            logging.debug(f"[VibeVoice TTS] Using cached model with {attention_mode} attention and q4={use_llm_4bit}")
            return LOADED_MODELS_CACHE[cache_key]

        model_info = AVAILABLE_VIBEVOICE_MODELS[model_name]
        model_type = model_info["type"]

        model_path, config_path, preprocessor_config_path, tokenizer_dir = \
            VibeVoiceLoader._resolve_model_paths(model_name)

        config = VibeVoiceLoader._load_config(config_path, model_name)
        is_streaming = isinstance(config, VibeVoiceStreamingConfig)
        if is_streaming:
            logging.debug(f"[VibeVoice TTS] Model '{model_name}' detected as streaming model")

        vibevoice_tokenizer = VibeVoiceLoader._load_tokenizer(tokenizer_dir, model_name)

        processor = VibeVoiceLoader._load_processor(
            vibevoice_tokenizer, preprocessor_config_path, is_streaming=is_streaming
        )

        load_device = model_management.get_torch_device() if not isinstance(device, torch.device) else device
        model_dtype = resolve_dtype(dtype_str, load_device)

        quant_config = None
        final_load_dtype = model_dtype
        if use_llm_4bit:
            bnb_compute_dtype = model_dtype
            if attention_mode == 'sage':
                bnb_compute_dtype, final_load_dtype = torch.float32, torch.float32
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=bnb_compute_dtype,
            )

        attn_implementation_for_load = get_attn_implementation_for_load(attention_mode)

        try:
            logging.debug(
                f"[VibeVoice TTS] Loading model '{model_name}' with dtype: {final_load_dtype} "
                f"and attention: '{attn_implementation_for_load}'"
            )

            model = VibeVoiceLoader._instantiate_model(
                config=config,
                is_streaming=is_streaming,
                attn_implementation=attn_implementation_for_load,
                final_load_dtype=final_load_dtype,
            )

            model = VibeVoiceLoader._load_state_dict_into_model(
                model=model,
                model_path=model_path,
                model_type=model_type,
                model_info=model_info,
                device=load_device,
            )

            cast_model_to_dtype_if_needed(model, final_load_dtype)

            if quant_config is not None:
                try:
                    from transformers.integrations.bitsandbytes import replace_with_bnb_linear
                except ImportError:
                    from transformers.utils.bitsandbytes import replace_with_bnb_linear
                replace_with_bnb_linear(
                    model,
                    quantization_config=quant_config,
                    modules_to_not_convert=None,
                )

            if attention_mode == "sage":
                if check_sage_attention_compatible():
                    set_sage_attention(model)
                else:
                    raise RuntimeError("Incompatible hardware/setup for SageAttention.")

            model.eval()
            setattr(model, "_llm_4bit", bool(quant_config))
            LOADED_MODELS_CACHE[cache_key] = (model, processor)
            logging.debug(f"[VibeVoice TTS] Successfully configured model '{model_name}' with {attention_mode} attention")
            return model, processor

        except Exception as e:
            logging.error(f"[VibeVoice TTS] Failed to load model '{model_name}' with {attention_mode} attention: {e}")
            raise RuntimeError(f"Failed to load model even with eager attention: {e}")