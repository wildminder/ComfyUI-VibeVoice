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
import shutil
import logging
import torch

import comfy.utils
import folder_paths
import comfy.model_management as model_management

import transformers

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
from .dtype_utils import resolve_dtype, get_dtype_str

if SAGE_ATTENTION_AVAILABLE:
    from ..src.vibevoice.modular.sage_attention_patch import set_sage_attention

from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)

# Cache for loaded (model, processor) tuples, keyed by cache_key
LOADED_MODELS_CACHE = {}


def cleanup_old_models(keep_cache_key: str = None) -> None:
    """Remove all cached models except the one matching keep_cache_key.

    Args:
        keep_cache_key: Cache key to preserve. If None, all are cleared.
    """
    from .utils import VIBEVOICE_PATCHER_CACHE

    keys_to_remove = []
    for key in list(LOADED_MODELS_CACHE.keys()):
        if key != keep_cache_key:
            keys_to_remove.append(key)
            del LOADED_MODELS_CACHE[key]

    for key in list(VIBEVOICE_PATCHER_CACHE.keys()):
        if key != keep_cache_key:
            try:
                patcher = VIBEVOICE_PATCHER_CACHE[key]
                if hasattr(patcher, 'model') and patcher.model:
                    patcher.model.model = None
                    patcher.model.processor = None
                del VIBEVOICE_PATCHER_CACHE[key]
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

    def __init__(self, model_pack_name: str, attention_mode: str = "eager", use_llm_4bit: bool = False):
        super().__init__()
        self.model_pack_name = model_pack_name
        self.attention_mode = attention_mode
        self.use_llm_4bit = use_llm_4bit
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

        Args:
            device: Target device for the model.
            attention_mode: Attention implementation to use.
        """
        self.model, self.processor = VibeVoiceLoader.load_model(
            self.model_pack_name, device, attention_mode, use_llm_4bit=self.use_llm_4bit
        )
        if self.model.device != device:
            self.model.to(device)


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

        Attempts to find tokenizer.json in the model directory, copies a
        packaged fallback if available, or downloads from HuggingFace.

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

            # Try packaged fallback
            packaged_configs_dir = os.path.join(
                os.path.dirname(__file__), "..", "src", "vibevoice", "configs"
            )
            packaged_tokenizer_path = os.path.join(packaged_configs_dir, "tokenizer.json")

            if os.path.exists(packaged_tokenizer_path):
                try:
                    logger.info("Found pre-packaged tokenizer. Copying it to model directory...")
                    shutil.copyfile(packaged_tokenizer_path, tokenizer_file_path)
                except Exception as e:
                    logger.warning(f"Failed to copy pre-packaged tokenizer: {e}. Will attempt to download.")

            # Download from HuggingFace if still missing
            if not os.path.exists(tokenizer_file_path):
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
    ):
        """Instantiate model class directly, bypassing from_pretrained() meta device init.

        Transformers 5.x from_pretrained() unconditionally uses torch.device("meta")
        as an init context, which causes DPMSolverMultistepScheduler and other
        non-parameter tensor operations in __init__ to fail. By instantiating
        the model class directly, we avoid the meta device context entirely.

        Args:
            config: VibeVoiceConfig or VibeVoiceStreamingConfig instance.
            is_streaming: If True, use streaming model class.
            attn_implementation: Attention implementation string.
            final_load_dtype: torch.dtype for the model.

        Returns:
            Model instance (not yet loaded with weights).
        """
        # Set attention implementation on the decoder config
        if hasattr(config, 'decoder_config'):
            config.decoder_config._attn_implementation = attn_implementation

        # Set dtype on config
        config.torch_dtype = final_load_dtype
        if hasattr(config, 'decoder_config'):
            config.decoder_config.torch_dtype = final_load_dtype

        # Instantiate directly — no meta device context
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

        Args:
            model: Model instance (already instantiated).
            model_path: Path to model directory (for official/local_dir) or None.
            model_type: "official", "local_dir", or "standalone".
            model_info: Model info dict (for standalone path).
            device: Target device for loading.

        Returns:
            The model with loaded state dict.
        """
        # Resolve checkpoint path (handles single-file and sharded checkpoints)
        ckpt_path, is_sharded = VibeVoiceLoader._resolve_checkpoint_path(
            model_path, model_type, model_info
        )

        if is_sharded:
            # Load and merge sharded checkpoint
            model_dir = model_path if model_type != "standalone" else os.path.dirname(ckpt_path)
            state_dict = VibeVoiceLoader._load_sharded_state_dict(ckpt_path, model_dir, device)
        else:
            # Load single checkpoint file
            logger.info(f"Loading state dict from: {ckpt_path}")
            state_dict = comfy.utils.load_torch_file(ckpt_path, device=device)

        # Load state dict with strict=False to handle any missing/unexpected keys
        # (e.g., tied weights, quantized layers)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            logger.warning(f"Missing keys when loading state dict: {len(missing_keys)} keys")
            if len(missing_keys) < 20:
                logger.warning(f"Missing keys: {missing_keys}")
            else:
                logger.warning(f"First 10 missing keys: {missing_keys[:10]}")

        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
            if len(unexpected_keys) < 20:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
            else:
                logger.warning(f"First 10 unexpected keys: {unexpected_keys[:10]}")

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

            # Step 3: Move to target device and apply dtype
            model = model.to(device=load_device, dtype=final_load_dtype)

            # Step 4: Apply 4-bit quantization if requested (post-load)
            if quant_config is not None:
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
