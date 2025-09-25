import os
import torch
import gc
import json
import logging
from huggingface_hub import hf_hub_download, snapshot_download

import comfy.utils
import folder_paths
import comfy.model_management as model_management

import transformers
from packaging import version

_transformers_version = version.parse(transformers.__version__)
_DTYPE_ARG_SUPPORTED = _transformers_version >= version.parse("4.56.0")

from transformers import BitsAndBytesConfig
from ..vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from ..vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from ..vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from ..vibevoice.processor.vibevoice_tokenizer_processor import VibeVoiceTokenizerProcessor
from ..vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast

from .model_info import AVAILABLE_VIBEVOICE_MODELS, MODEL_CONFIGS
from .. import SAGE_ATTENTION_AVAILABLE
if SAGE_ATTENTION_AVAILABLE:
    from ..vibevoice.modular.sage_attention_patch import set_sage_attention

logger = logging.getLogger(__name__)

LOADED_MODELS = {}
VIBEVOICE_PATCHER_CACHE = {}

ATTENTION_MODES = ["eager", "sdpa", "flash_attention_2"]
if SAGE_ATTENTION_AVAILABLE:
    ATTENTION_MODES.append("sage")

def cleanup_old_models(keep_cache_key=None):
    global LOADED_MODELS, VIBEVOICE_PATCHER_CACHE
    keys_to_remove = []
    for key in list(LOADED_MODELS.keys()):
        if key != keep_cache_key:
            keys_to_remove.append(key)
            del LOADED_MODELS[key]
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
    def __init__(self, model_pack_name, attention_mode="eager", use_llm_4bit=False):
        super().__init__()
        self.model_pack_name = model_pack_name
        self.attention_mode = attention_mode
        self.use_llm_4bit = use_llm_4bit
        self.cache_key = f"{self.model_pack_name}_attn_{attention_mode}_q4_{int(use_llm_4bit)}"
        self.model = None
        self.processor = None
        info = AVAILABLE_VIBEVOICE_MODELS.get(model_pack_name, {})
        size_gb = MODEL_CONFIGS.get(model_pack_name, {}).get("size_gb", 4.0)
        self.size = int(size_gb * (1024**3))
    def load_model(self, device, attention_mode="eager"):
        self.model, self.processor = VibeVoiceLoader.load_model(self.model_pack_name, device, attention_mode, use_llm_4bit=self.use_llm_4bit)
        if self.model.device != device:
             self.model.to(device)

class VibeVoiceLoader:
    @staticmethod
    def _check_gpu_for_sage_attention():
        if not SAGE_ATTENTION_AVAILABLE: return False
        if not torch.cuda.is_available(): return False
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            logger.warning(f"Your GPU (compute capability {major}.x) does not support SageAttention, which requires CC 8.0+. Sage option will be disabled.")
            return False
        return True

    @staticmethod
    def load_model(model_name: str, device, attention_mode: str = "eager", use_llm_4bit: bool = False):
        if model_name not in AVAILABLE_VIBEVOICE_MODELS:
            raise ValueError(f"Unknown VibeVoice model: {model_name}. Available models: {list(AVAILABLE_VIBEVOICE_MODELS.keys())}")
        
        if use_llm_4bit and attention_mode in ["eager", "flash_attention_2"]:
            logger.warning(f"Attention mode '{attention_mode}' is not recommended with 4-bit quantization. Falling back to 'sdpa' for stability and performance.")
            attention_mode = "sdpa"
        if attention_mode not in ATTENTION_MODES:
            logger.warning(f"Unknown attention mode '{attention_mode}', falling back to eager")
            attention_mode = "eager"

        cache_key = f"{model_name}_attn_{attention_mode}_q4_{int(use_llm_4bit)}"
        if cache_key in LOADED_MODELS:
            logger.info(f"Using cached model with {attention_mode} attention and q4={use_llm_4bit}")
            return LOADED_MODELS[cache_key]

        model_info = AVAILABLE_VIBEVOICE_MODELS[model_name]
        model_type = model_info["type"]
        vibevoice_base_path = os.path.join(folder_paths.get_folder_paths("tts")[0], "VibeVoice")

        model_path_or_none = None
        config_path = None
        preprocessor_config_path = None
        tokenizer_dir = None

        if model_type == "official":
            model_path_or_none = os.path.join(vibevoice_base_path, model_name)
            if not os.path.exists(os.path.join(model_path_or_none, "model.safetensors.index.json")):
                logger.info(f"Downloading official VibeVoice model: {model_name}...")
                snapshot_download(repo_id=model_info["repo_id"], local_dir=model_path_or_none, local_dir_use_symlinks=False)
            config_path = os.path.join(model_path_or_none, "config.json")
            preprocessor_config_path = os.path.join(model_path_or_none, "preprocessor_config.json")
            tokenizer_dir = model_path_or_none
        elif model_type == "local_dir":
            model_path_or_none = model_info["path"]
            config_path = os.path.join(model_path_or_none, "config.json")
            preprocessor_config_path = os.path.join(model_path_or_none, "preprocessor_config.json")
            tokenizer_dir = model_path_or_none
        elif model_type == "standalone":
            model_path_or_none = None # IMPORTANT: This must be None when loading from state_dict
            config_path = os.path.splitext(model_info["path"])[0] + ".config.json"
            preprocessor_config_path = os.path.splitext(model_info["path"])[0] + ".preprocessor.json"
            tokenizer_dir = os.path.dirname(model_info["path"])

        if os.path.exists(config_path):
            config = VibeVoiceConfig.from_pretrained(config_path)
        else:
            fallback_name = "default_VibeVoice-Large_config.json" if "large" in model_name.lower() else "default_VibeVoice-1.5B_config.json"
            fallback_path = os.path.join(os.path.dirname(__file__), "..", "vibevoice", "configs", fallback_name)
            logger.warning(f"Config not found for '{model_name}'. Using fallback: {fallback_name}")
            config = VibeVoiceConfig.from_pretrained(fallback_path)

        # Processor & Tokenizer setup
        tokenizer_file_path = os.path.join(tokenizer_dir, "tokenizer.json")

        if not os.path.exists(tokenizer_file_path):
            logger.info(f"'tokenizer.json' not found in model directory: {tokenizer_dir}")

            packaged_configs_dir = os.path.join(os.path.dirname(__file__), "..", "vibevoice", "configs")
            packaged_tokenizer_path = os.path.join(packaged_configs_dir, "tokenizer.json")

            if os.path.exists(packaged_tokenizer_path):
                try:
                    import shutil
                    logger.info("Found pre-packaged tokenizer. Copying it to model directory...")
                    shutil.copyfile(packaged_tokenizer_path, tokenizer_file_path)
                except Exception as e:
                    logger.warning(f"Failed to copy pre-packaged tokenizer: {e}. Will attempt to download.")
            
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
                            local_dir=tokenizer_dir
                        )
                        download_successful = True
                        logger.info("Download successful.")
                        break  # Exit the loop on success
                    except Exception as e:
                        logger.warning(f"Failed to download from '{repo_id}': {e}")
                        last_error = e

                # Final Failure
                if not download_successful:
                    error_message = (
                        f"FATAL: Could not get 'tokenizer.json'. All download attempts failed.\n"
                        f"Last error: {last_error}\n\n"
                        f"ACTION REQUIRED:\n"
                        f"1. Manually download 'tokenizer.json' from https://huggingface.co/{repos_to_try[0]}/blob/main/tokenizer.json\n"
                        f"2. Place the downloaded file in the following directory:\n   '{tokenizer_dir}'"
                    )
                    raise RuntimeError(error_message)
        
        vibevoice_tokenizer = VibeVoiceTextTokenizerFast(tokenizer_file=tokenizer_file_path)
        
        processor_config_data = {}
        if os.path.exists(preprocessor_config_path):
            with open(preprocessor_config_path, 'r', encoding='utf-8') as f: processor_config_data = json.load(f)
        
        audio_processor = VibeVoiceTokenizerProcessor()
        processor = VibeVoiceProcessor(tokenizer=vibevoice_tokenizer, audio_processor=audio_processor, speech_tok_compress_ratio=processor_config_data.get("speech_tok_compress_ratio", 3200), db_normalize=processor_config_data.get("db_normalize", True))

        # Model Loading Prep
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported(): model_dtype = torch.bfloat16
        else: model_dtype = torch.float16
        quant_config = None
        final_load_dtype = model_dtype

        if use_llm_4bit:
            bnb_compute_dtype = model_dtype
            if attention_mode == 'sage': bnb_compute_dtype, final_load_dtype = torch.float32, torch.float32
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bnb_compute_dtype)

        attn_implementation_for_load = "sdpa" if attention_mode == "sage" else attention_mode
        
        try:
            logger.info(f"Loading model '{model_name}' with dtype: {final_load_dtype} and attention: '{attn_implementation_for_load}'")
            
            # UNIFIED MODEL LOADING LOGIC
            from_pretrained_kwargs = {
                "config": config,
                "attn_implementation": attn_implementation_for_load,
                "device_map": "auto" if quant_config else device,
                "quantization_config": quant_config,
            }
            if _DTYPE_ARG_SUPPORTED:
                from_pretrained_kwargs['dtype'] = final_load_dtype
            else:
                from_pretrained_kwargs['torch_dtype'] = final_load_dtype

            if model_type == "standalone":
                logger.info(f"Loading standalone model state_dict directly to device: {device}")
                # loading the state dict directly to the target device
                state_dict = comfy.utils.load_torch_file(model_info["path"], device=device)
                from_pretrained_kwargs["state_dict"] = state_dict

            model = VibeVoiceForConditionalGenerationInference.from_pretrained(model_path_or_none, **from_pretrained_kwargs)

            if attention_mode == "sage":
                if VibeVoiceLoader._check_gpu_for_sage_attention():
                    set_sage_attention(model)
                else:
                    raise RuntimeError("Incompatible hardware/setup for SageAttention.")
            
            model.eval()
            setattr(model, "_llm_4bit", bool(quant_config))
            LOADED_MODELS[cache_key] = (model, processor)
            logger.info(f"Successfully configured model '{model_name}' with {attention_mode} attention")
            return model, processor
            
        except Exception as e:
            # It's not ideal to automatically reload the model. Let the user decide what to do in case of an error.
            logger.error(f"Failed to load model '{model_name}' with {attention_mode} attention: {e}")
            # if attention_mode in ["sage", "flash_attention_2"]: return VibeVoiceLoader.load_model(model_name, device, "sdpa", use_llm_4bit)
            # elif attention_mode == "sdpa": return VibeVoiceLoader.load_model(model_name, device, "eager", use_llm_4bit)
            # else:
            raise RuntimeError(f"Failed to load model even with eager attention: {e}")