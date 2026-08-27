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
from .base_loader import BaseVibeVoiceLoader
from .attention_utils import (
    SAGE_ATTENTION_AVAILABLE,
    ATTENTION_MODES,
    resolve_attention_mode,
    get_attn_implementation_for_load,
    check_sage_attention_compatible,
)
from .dtype_utils import resolve_dtype, get_dtype_str, cast_model_to_dtype_if_needed

if SAGE_ATTENTION_AVAILABLE:
    from ..src.vibevoice.modular.sage_attention_patch import set_sage_attention

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

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
                # Non-default rope types resolve through the shared init table.
                from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
                rope_init_fn = ROPE_INIT_FUNCTIONS.get(rope_type)
                if rope_init_fn is None:
                    logger.warning(
                        f"No rope init function for type '{rope_type}' on "
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
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                f"Could not recompute RoPE inv_freq for "
                f"{module.__class__.__name__}: {e}"
            )
    if recomputed:
        logger.info(f"Recomputed RoPE inv_freq for {recomputed} rotary module(s).")
    return recomputed

# Cache for loaded (model, processor) tuples, keyed by cache_key
LOADED_MODELS_CACHE = {}


def cleanup_old_models(keep_cache_key: str = None) -> None:
    """Remove all cached models except the one matching keep_cache_key.

    Plan 2026-08-20 (C5): the patcher loop routes through
    ``model_registry.evict_patcher`` — a behavior superset of the previous
    inline destroy (it also unregisters the patcher from ComfyUI's
    ``model_management.current_loaded_models``). The ``LOADED_MODELS_CACHE``
    loop and ``keep_cache_key`` semantics are unchanged.

    Args:
        keep_cache_key: Cache key to preserve. If None, all are cleared.
    """
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
                # Explicit destructive free via the registry primitive:
                # unregister from ComfyUI -> destroy -> pop cache -> gc ->
                # soft_empty_cache (each step individually guarded).
                evict_patcher(patcher, VIBEVOICE_PATCHER_CACHE, key)
            except Exception as e:
                logger.warning(f"Error cleaning up patcher {key}: {e}")

    if keys_to_remove:
        logger.info(f"Cleaned up cached models: {keys_to_remove}")
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
        # DF-004/AUD-008: the user-selected dtype is threaded into the loader
        # so the final dtype is applied ON CPU during load (before the single
        # H2D transfer), instead of being re-resolved to "auto" here.
        self.dtype_str = dtype_str
        self.cache_key = f"{self.model_pack_name}_attn_{attention_mode}_q4_{int(use_llm_4bit)}"
        self.model = None
        self.processor = None
        # Device attribute — ComfyUI's ModelPatcher reads/writes this to track
        # which device the model is currently on. Initialized to CPU; will be
        # updated by ModelPatcher.load() when the model is moved to GPU.
        self.device = None

        info = AVAILABLE_VIBEVOICE_MODELS.get(model_pack_name, {})
        size_gb = MODEL_CONFIGS.get(model_pack_name, {}).get("size_gb", 4.0)
        self.size = int(size_gb * (1024**3))

    def load_model(self, device, attention_mode: str = "eager"):
        """Load the model and processor into memory.

        Device contract (DF-003 fix): the loader builds the model entirely on
        CPU and this handler performs NO device move. The single host-to-
        device transfer is owned by ``VibeVoicePatcher.patch_model``.

        Args:
            device: Target device (used for dtype-auto resolution inside the
                loader; NOT used for placement here).
            attention_mode: Attention implementation to use.
        """
        self.model, self.processor = VibeVoiceLoader.load_model(
            self.model_pack_name,
            device,
            attention_mode,
            use_llm_4bit=self.use_llm_4bit,
            dtype_str=self.dtype_str,
        )
        # Plan 2026-08-18, Phase 6 (D7/RC-7): refine the size estimate from
        # the real parameters now that the model is loaded. The __init__
        # estimate is config-based (size_gb) and can be inaccurate for
        # quantized / merged / GGUF checkpoints.
        self._refine_size()

    def _refine_size(self) -> None:
        """Refine ``self.size`` from the real parameters after a load.

        Only overwrites when a positive byte total can be computed; otherwise
        the config-based estimate from ``__init__`` is kept.
        """
        try:
            total = sum(
                p.numel() * p.element_size() for p in self.model.parameters()
            )
            if total > 0:
                self.size = total
        except Exception:
            # Keep the config-based estimate if the size cannot be computed.
            pass


class VibeVoiceLoader(BaseVibeVoiceLoader):
    """Static loader class for VibeVoice models."""

    @staticmethod
    def _resolve_model_paths(model_name: str) -> tuple:
        """Resolve model paths based on the model type.

        Args:
            model_name: Name of the model to resolve paths for.

        Returns:
            Tuple of (model_path, config_path, preprocessor_config_path, tokenizer_dir).
            model_path is None for standalone (state_dict) models.
        """
        model_info = AVAILABLE_VIBEVOICE_MODELS[model_name]
        model_type = model_info["type"]

        model_path = None
        config_path = None
        preprocessor_config_path = None
        tokenizer_dir = None

        if model_type == "official":
            # Delegate directory resolution + lazy download to the shared base.
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
            model_path = None  # None when loading from state_dict
            config_path = os.path.splitext(model_info["path"])[0] + ".config.json"
            preprocessor_config_path = os.path.splitext(model_info["path"])[0] + ".preprocessor.json"
            tokenizer_dir = os.path.dirname(model_info["path"])

        return model_path, config_path, preprocessor_config_path, tokenizer_dir

    @staticmethod
    def _load_config(config_path: str, model_name: str):
        """Load VibeVoice config from path, with fallback to packaged default.

        Automatically detects whether the config is for a streaming model
        (VibeVoiceStreamingConfig) or a standard model (VibeVoiceConfig)
        based on the model_type field in the config JSON.

        Args:
            config_path: Path to config.json.
            model_name: Model name (for fallback selection).

        Returns:
            VibeVoiceStreamingConfig or VibeVoiceConfig instance.
        """
        if os.path.exists(config_path):
            # Read the config to detect model type
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            model_type = config_data.get("model_type", "")

            if model_type == "vibevoice_streaming":
                logger.info(f"Detected streaming config for '{model_name}'")
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
        logger.warning(f"Config not found for '{model_name}'. Using fallback: {fallback_name}")
        return VibeVoiceConfig.from_pretrained(fallback_path)

    @staticmethod
    def _load_tokenizer(tokenizer_dir: str, model_name: str) -> "VibeVoiceTextTokenizerFast":
        """Load the VibeVoice text tokenizer.

        Acquisition order: tokenizer.json in the model directory → packaged
        tokenizer loaded DIRECTLY from the node folder (never copied into the
        user's model directory) → download from HuggingFace as last resort.

        Args:
            tokenizer_dir: Directory to find/create tokenizer.json.
            model_name: Model name (for logging).

        Returns:
            VibeVoiceTextTokenizerFast instance.

        Raises:
            RuntimeError: If tokenizer.json cannot be obtained.
        """
        tokenizer_file_path = os.path.join(tokenizer_dir, "tokenizer.json")

        if not os.path.exists(tokenizer_file_path):
            logger.info(f"'tokenizer.json' not found in model directory: {tokenizer_dir}")

            # Packaged fallback: load straight from the node folder. The
            # user's model directory stays untouched.
            packaged_configs_dir = os.path.join(
                os.path.dirname(__file__), "..", "src", "vibevoice", "configs"
            )
            packaged_tokenizer_path = os.path.join(packaged_configs_dir, "tokenizer.json")

            if os.path.exists(packaged_tokenizer_path):
                logger.info("Using pre-packaged tokenizer directly from the node folder...")
                return VibeVoiceTextTokenizerFast(tokenizer_file=packaged_tokenizer_path)

            # Download from HuggingFace if still missing
            repos_to_try = ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-7B"]
            download_successful = False
            last_error = None

            for repo_id in repos_to_try:
                logger.info(f"Attempting to download 'tokenizer.json' from Hugging Face repo '{repo_id}'...")
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename="tokenizer.json",
                        local_dir=tokenizer_dir,
                    )
                    download_successful = True
                    logger.info("Download successful.")
                    break
                except Exception as e:
                    logger.warning(f"Failed to download from '{repo_id}': {e}")
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
        """Load the VibeVoice processor with tokenizer and audio processor.

        Args:
            tokenizer: VibeVoiceTextTokenizerFast instance.
            preprocessor_config_path: Path to preprocessor_config.json.
            is_streaming: If True, use VibeVoiceStreamingProcessor.

        Returns:
            VibeVoiceStreamingProcessor or VibeVoiceProcessor instance.
        """
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
        """Instantiate the model class directly, bypassing from_pretrained().

        Fast-load contract (plan 2026-08-18, D1 — RC-1 elimination): by
        default the construction runs inside a ``torch.device("meta")``
        context, so every parameter/buffer allocation is virtual — zero RAM
        and zero random-init CPU work. The vendored models are meta-safe:
        the DPM scheduler computes its tables with numpy, and the
        ``.to(dtype)`` calls in ``__init__`` are guarded by ``is_meta``
        checks (verified by tests/test_meta_init_feasibility.py). Weights
        are bound afterwards by ``_apply_state_dict`` (assign semantics).

        ``use_meta=False`` is the escape hatch that restores the previous
        eager (random-init) construction.

        Args:
            config: VibeVoiceConfig or VibeVoiceStreamingConfig instance.
            is_streaming: If True, use streaming model class.
            attn_implementation: Attention implementation string.
            final_load_dtype: torch.dtype for the model.
            use_meta: Construct under a meta device context (default True).

        Returns:
            Model instance (weights not yet loaded).
        """
        # Set attention implementation on the decoder config
        if hasattr(config, 'decoder_config'):
            config.decoder_config._attn_implementation = attn_implementation

        # Set dtype on config
        config.torch_dtype = final_load_dtype
        if hasattr(config, 'decoder_config'):
            config.decoder_config.torch_dtype = final_load_dtype

        # Instantiate directly — meta context by default (zero alloc, zero RNG)
        ctx = torch.device("meta") if use_meta else contextlib.nullcontext()
        with ctx:
            if is_streaming:
                model = VibeVoiceStreamingForConditionalGenerationInference(config)
            else:
                model = VibeVoiceForConditionalGeneration(config)

        return model

    @staticmethod
    def _resolve_checkpoint_path(model_path, model_type, model_info):
        """Resolve the checkpoint file path(s) from a model directory or standalone path.

        For directory-based models (official, local_dir), checks in priority order:
        1. model.safetensors (single safetensors file)
        2. model.safetensors.index.json (sharded safetensors)
        3. pytorch_model.bin (single PyTorch checkpoint)
        4. pytorch_model.bin.index.json (sharded PyTorch checkpoint)

        For standalone models: returns the path directly.

        Args:
            model_path: Path to model directory (for official/local_dir) or None.
            model_type: "official", "local_dir", or "standalone".
            model_info: Model info dict (for standalone path).

        Returns:
            Tuple of (checkpoint_path, is_sharded).
            - checkpoint_path: Path to single checkpoint file or index file.
            - is_sharded: True if the checkpoint is sharded.

        Raises:
            FileNotFoundError: If no checkpoint file is found in the directory.
        """
        # Standalone models: path is already a file
        if model_type == "standalone":
            ckpt_path = model_info["path"]
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Standalone checkpoint not found: {ckpt_path}")
            return ckpt_path, False

        # Directory-based models: resolve checkpoint file from directory
        if model_path is None or not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        # Check for single safetensors file
        single_safetensors = os.path.join(model_path, "model.safetensors")
        if os.path.isfile(single_safetensors):
            logger.info(f"Found single safetensors checkpoint: {single_safetensors}")
            return single_safetensors, False

        # Check for sharded safetensors
        sharded_safetensors_index = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.isfile(sharded_safetensors_index):
            logger.info(f"Found sharded safetensors checkpoint: {sharded_safetensors_index}")
            return sharded_safetensors_index, True

        # Check for single PyTorch checkpoint
        single_bin = os.path.join(model_path, "pytorch_model.bin")
        if os.path.isfile(single_bin):
            logger.info(f"Found single PyTorch checkpoint: {single_bin}")
            return single_bin, False

        # Check for sharded PyTorch checkpoint
        sharded_bin_index = os.path.join(model_path, "pytorch_model.bin.index.json")
        if os.path.isfile(sharded_bin_index):
            logger.info(f"Found sharded PyTorch checkpoint: {sharded_bin_index}")
            return sharded_bin_index, True

        # No checkpoint found
        raise FileNotFoundError(
            f"No checkpoint file found in model directory: {model_path}. "
            f"Expected one of: model.safetensors, model.safetensors.index.json, "
            f"pytorch_model.bin, pytorch_model.bin.index.json"
        )

    @staticmethod
    def _load_sharded_state_dict(index_path, model_dir, device):
        """Load and merge a sharded checkpoint by delegating to the shared base.

        Retained with the original signature (``index_path`` is identified from
        ``model_dir`` by the base) so existing callers/tests are unaffected.

        Args:
            index_path: Path to the index file (model.safetensors.index.json
                        or pytorch_model.bin.index.json).
            model_dir: Directory containing the shard files.
            device: Target device for loading.

        Returns:
            Merged state dict containing all parameters from all shards.
        """
        return BaseVibeVoiceLoader.load_state_dict_sharded(model_dir, device)

    @staticmethod
    def _apply_state_dict(model, state_dict, known_missing=None):
        """Load a state dict into the model with assign semantics.

        Fast-load contract (plan 2026-08-18, D2/D3 — un-defers DF-005):

        1. ``load_state_dict(..., strict=False, assign=True)`` — checkpoint
           tensors REPLACE the (meta or random) parameter objects directly;
           no copy pass, no extra host RAM.
        2. Re-tie weights. ``assign=True`` breaks tied pairs (the checkpoint
           omits ``lm_head.weight``), so ``tie_weights()`` is re-invoked —
           the same mitigation transformers' ``from_pretrained`` applies.
        3. Materialize any parameter still on meta (checkpoint omitted the
           key) with zeros, mirroring ComfyUI's ``_zero_init_parameter``.

        Args:
            model: Instantiated model (meta- or eager-initialized).
            state_dict: Checkpoint state dict (CPU tensors).
            known_missing: Optional set of keys that are INTENTIONALLY absent
                from ``state_dict`` (e.g. quant-resident linear weights
                installed separately); excluded from missing-key warnings but
                still returned in ``missing_keys``.

        Returns:
            Tuple of (missing_keys, unexpected_keys).
        """
        known_missing = known_missing or set()
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict, strict=False, assign=True
        )

        # Re-tie weights broken by assign=True (e.g. lm_head.weight <->
        # embed_tokens.weight). Standard/ASR models gate on
        # decoder_config.tie_word_embeddings; streaming gates on the
        # top-level config flag — check both, and let each model's own
        # tie_weights() apply its internal guard.
        config = getattr(model, "config", None)
        if config is not None and hasattr(model, "tie_weights"):
            decoder_config = getattr(config, "decoder_config", None)
            tied = bool(getattr(decoder_config, "tie_word_embeddings", False)) or \
                bool(getattr(config, "tie_word_embeddings", False))
            if tied:
                model.tie_weights()

        # Materialize meta parameters the checkpoint did not cover (D3).
        for name, param in list(model.named_parameters()):
            if param.is_meta:
                param_new = torch.zeros(param.shape, dtype=param.dtype, device="cpu")
                comfy.utils.set_attr_param(model, name, param_new)

        # Materialize meta buffers (e.g. position_ids, attention_mask) that
        # were created on meta during instantiation but are not in the
        # checkpoint state dict. Without this, ComfyUI's unpatch_model →
        # model.to(device_to) raises NotImplementedError on meta buffers.
        #
        # Regression fix (v2.3.1): some buffers are SENTINELS that must NOT be
        # zero. speech_scaling_factor / speech_bias_factor are registered as
        # float('nan') and computed at inference time; the diffusion-inversion
        # gate (modeling_vibevoice.py:623) is ``if not torch.isnan(sf) and not
        # torch.isnan(bf)`` — zeroing them makes the gate TRUE and applies
        # ``speech / 0 - 0`` → silent output. fix_std (persistent=False) is
        # restored to its config value. See _SENTINEL_BUFFER_VALUES.
        for name, buf in list(model.named_buffers()):
            if buf.is_meta:
                # Match by the buffer's final dotted component (robust to
                # submodule nesting, e.g. "model.speech_scaling_factor").
                leaf = name.split(".")[-1]
                if leaf in _SENTINEL_BUFFER_VALUES:
                    val = _SENTINEL_BUFFER_VALUES[leaf]
                    buf_new = torch.tensor(
                        val, dtype=buf.dtype, device="cpu"
                    ).reshape(buf.shape)
                else:
                    buf_new = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu")
                comfy.utils.set_attr(model, name, buf_new)

        # Recompute RoPE inv_freq buffers (v2.3.2 fix). These are computed
        # from the config in the rotary module's __init__ and are NOT in the
        # checkpoint, so the zero-materialization above destroyed them. With
        # inv_freq == 0 the model loses positional encoding and emits
        # gibberish. Re-run each rotary module's config-based computation.
        _recompute_rope_buffers(model)

        # Native lowvram streaming (plan 2026-08-26): convert leaf modules so
        # core's partial load/offload machinery can stream them instead of
        # silently stranding them on CPU. Class-only swap; weights untouched.
        try:
            from .comfy_stream import convert_tree_for_streaming

            convert_tree_for_streaming(model)
        except Exception as e:
            logger.warning(
                f"Streaming conversion failed (continuing without it): {e}"
            )

        reported_missing = [k for k in missing_keys if k not in known_missing]
        if reported_missing:
            logger.warning(f"Missing keys when loading state dict: {len(reported_missing)} keys")
            if len(reported_missing) < 20:
                logger.warning(f"Missing keys: {reported_missing}")
            else:
                logger.warning(f"First 10 missing keys: {reported_missing[:10]}")

        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
            if len(unexpected_keys) < 20:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
            else:
                logger.warning(f"First 10 unexpected keys: {unexpected_keys[:10]}")

        return missing_keys, unexpected_keys

    @staticmethod
    def _load_state_dict_into_model(
        model,
        model_path: str,
        model_type: str,
        model_info: dict,
        device,
    ):
        """Load state dict into model using ComfyUI's loading utilities.

        Resolves the checkpoint file path(s) from the model directory or
        standalone path, then loads the state dict (handling sharded checkpoints)
        and loads it into the model.

        Device contract (DF-001/DF-002 fix): the state dict is ALWAYS loaded
        onto CPU, regardless of the ``device`` argument. Loading it directly
        onto CUDA caused a disk->VRAM->RAM->VRAM round-trip (full-model VRAM
        spike outside ComfyUI's arbitration, then a GPU->CPU copy into the
        CPU-resident parameters). The single host-to-device transfer is owned
        by ``VibeVoicePatcher.patch_model`` after ComfyUI has arbitrated VRAM.

        Args:
            model: Model instance (already instantiated).
            model_path: Path to model directory (for official/local_dir) or None.
            model_type: "official", "local_dir", or "standalone".
            model_info: Model info dict (for standalone path).
            device: Reserved for signature compatibility; NOT used for
                placement — the state dict is always loaded onto CPU.

        Returns:
            The model with loaded state dict (on CPU).
        """
        # Resolve checkpoint path (handles single-file and sharded checkpoints)
        ckpt_path, is_sharded = VibeVoiceLoader._resolve_checkpoint_path(
            model_path, model_type, model_info
        )

        # DF-001/DF-002: always load weights onto CPU. The patcher performs
        # the single H2D transfer after ComfyUI's VRAM arbitration.
        cpu_device = torch.device("cpu")

        if is_sharded:
            # Load and merge sharded checkpoint (onto CPU)
            model_dir = model_path if model_type != "standalone" else os.path.dirname(ckpt_path)
            state_dict = VibeVoiceLoader._load_sharded_state_dict(ckpt_path, model_dir, cpu_device)
        else:
            # Load single checkpoint file (onto CPU)
            logger.info(f"Loading state dict from: {ckpt_path}")
            state_dict = comfy.utils.load_torch_file(ckpt_path, device=cpu_device)

        # Assign-based load (D2/D3): checkpoint tensors replace the meta/random
        # parameter objects directly (no copy pass), tied weights are re-tied,
        # and any meta stragglers are zero-materialized. strict=False handles
        # missing/unexpected keys (e.g., tied weights, quantized layers).
        VibeVoiceLoader._apply_state_dict(model, state_dict)

        # Free the state dict immediately — the model now owns the tensors
        # (assign semantics), so the dict is dead weight (plan D7/RC-4).
        del state_dict

        return model

    @staticmethod
    def load_model(
        model_name: str,
        device,
        attention_mode: str = "eager",
        use_llm_4bit: bool = False,
        dtype_str: str = "auto",
    ):
        """Load a VibeVoice model, downloading if necessary. Caches the loaded model.

        Bypasses transformers' from_pretrained() to avoid meta device initialization
        issues in transformers 5.x. Instead, instantiates the model class directly
        and loads the state dict using ComfyUI's comfy.utils.load_torch_file().

        Args:
            model_name: Name of the model to load.
            device: Target device for the model.
            attention_mode: Attention implementation ("eager", "sdpa", "flash_attention_2", "sage").
            use_llm_4bit: Whether to quantize the LLM to 4-bit NF4.
            dtype_str: Dtype string ("auto", "bf16", "fp16", "fp32").

        Returns:
            Tuple of (model, processor).

        Raises:
            ValueError: If model_name is not found.
            RuntimeError: If model loading fails.
        """
        if model_name not in AVAILABLE_VIBEVOICE_MODELS:
            raise ValueError(
                f"Unknown VibeVoice model: {model_name}. "
                f"Available models: {list(AVAILABLE_VIBEVOICE_MODELS.keys())}"
            )

        # Resolve attention mode with fallback logic
        attention_mode = resolve_attention_mode(attention_mode, use_llm_4bit)

        cache_key = f"{model_name}_attn_{attention_mode}_q4_{int(use_llm_4bit)}"
        if cache_key in LOADED_MODELS_CACHE:
            logger.info(f"Using cached model with {attention_mode} attention and q4={use_llm_4bit}")
            return LOADED_MODELS_CACHE[cache_key]

        model_info = AVAILABLE_VIBEVOICE_MODELS[model_name]
        model_type = model_info["type"]

        # Resolve paths
        model_path, config_path, preprocessor_config_path, tokenizer_dir = \
            VibeVoiceLoader._resolve_model_paths(model_name)

        # Load config
        config = VibeVoiceLoader._load_config(config_path, model_name)

        # Detect if this is a streaming model
        is_streaming = isinstance(config, VibeVoiceStreamingConfig)
        if is_streaming:
            logger.info(f"Model '{model_name}' detected as streaming model")

        # Load tokenizer
        vibevoice_tokenizer = VibeVoiceLoader._load_tokenizer(tokenizer_dir, model_name)

        # Load processor (use streaming processor for streaming models)
        processor = VibeVoiceLoader._load_processor(
            vibevoice_tokenizer, preprocessor_config_path, is_streaming=is_streaming
        )

        # Determine dtype
        load_device = model_management.get_torch_device() if not isinstance(device, torch.device) else device
        model_dtype = resolve_dtype(dtype_str, load_device)

        # Quantization config
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

        # Attention implementation for loading (sage is applied post-load)
        attn_implementation_for_load = get_attn_implementation_for_load(attention_mode)

        try:
            logger.info(
                f"Loading model '{model_name}' with dtype: {final_load_dtype} "
                f"and attention: '{attn_implementation_for_load}'"
            )

            # Step 1: Instantiate model class directly (bypasses meta device init)
            model = VibeVoiceLoader._instantiate_model(
                config=config,
                is_streaming=is_streaming,
                attn_implementation=attn_implementation_for_load,
                final_load_dtype=final_load_dtype,
            )

            # Step 2: Load state dict using ComfyUI's loading utilities
            model = VibeVoiceLoader._load_state_dict_into_model(
                model=model,
                model_path=model_path,
                model_type=model_type,
                model_info=model_info,
                device=load_device,
            )

            # Step 3: Apply the final dtype ON CPU (DF-002/DF-003 fix) — but
            # only where needed (plan 2026-08-18, D4/RC-3). The model leaves
            # the loader on CPU; the single host-to-device transfer is owned
            # by VibeVoicePatcher.patch_model after ComfyUI's VRAM
            # arbitration. The conditional helper skips the pass entirely
            # when the checkpoint dtype already matches the target.
            cast_model_to_dtype_if_needed(model, final_load_dtype)

            # Step 4: Apply 4-bit quantization if requested (post-load)
            if quant_config is not None:
                # AUD-014: the module moved from transformers.utils.bitsandbytes
                # (<= 4.x) to transformers.integrations.bitsandbytes (5.x).
                try:
                    from transformers.integrations.bitsandbytes import replace_with_bnb_linear
                except ImportError:
                    from transformers.utils.bitsandbytes import replace_with_bnb_linear
                replace_with_bnb_linear(
                    model,
                    quantization_config=quant_config,
                    modules_to_not_convert=None,
                )

            # Apply SageAttention post-load
            if attention_mode == "sage":
                if check_sage_attention_compatible():
                    set_sage_attention(model)
                else:
                    raise RuntimeError("Incompatible hardware/setup for SageAttention.")

            model.eval()
            setattr(model, "_llm_4bit", bool(quant_config))
            LOADED_MODELS_CACHE[cache_key] = (model, processor)
            logger.info(f"Successfully configured model '{model_name}' with {attention_mode} attention")
            return model, processor

        except Exception as e:
            logger.error(f"Failed to load model '{model_name}' with {attention_mode} attention: {e}")
            raise RuntimeError(f"Failed to load model even with eager attention: {e}")
