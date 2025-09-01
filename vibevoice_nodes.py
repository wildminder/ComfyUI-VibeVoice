import os
import re
import torch
import numpy as np
import random
from huggingface_hub import hf_hub_download, snapshot_download
import logging

import gc

import folder_paths
import comfy.model_management as model_management
import comfy.model_patcher
from comfy.utils import ProgressBar
from comfy.model_management import throw_exception_if_processing_interrupted

from transformers import set_seed, AutoTokenizer, BitsAndBytesConfig
from .vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from .vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from .vibevoice.processor.vibevoice_tokenizer_processor import VibeVoiceTokenizerProcessor
from .vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast

try:
    import librosa
except ImportError:
    print("VibeVoice Node: `librosa` is not installed. Resampling of reference audio will not be available.")
    librosa = None

logger = logging.getLogger(__name__)

LOADED_MODELS = {}
VIBEVOICE_PATCHER_CACHE = {}

MODEL_CONFIGS = {
    "VibeVoice-1.5B": {
        "repo_id": "microsoft/VibeVoice-1.5B",
        "size_gb": 3.0,
        "tokenizer_repo": "Qwen/Qwen2.5-1.5B"
    },
    "VibeVoice-Large": {
        "repo_id": "microsoft/VibeVoice-Large",
        "size_gb": 17.4,
        "tokenizer_repo": "Qwen/Qwen2.5-7B" 
    }
}

ATTENTION_MODES = ["eager", "sdpa", "flash_attention_2"]

def cleanup_old_models(keep_cache_key=None):
    """Clean up old models, optionally keeping one specific model loaded"""
    global LOADED_MODELS, VIBEVOICE_PATCHER_CACHE
    
    keys_to_remove = []
    
    # Clear LOADED_MODELS
    for key in list(LOADED_MODELS.keys()):
        if key != keep_cache_key:
            keys_to_remove.append(key)
            del LOADED_MODELS[key]
    
    # Clear VIBEVOICE_PATCHER_CACHE - but more carefully
    for key in list(VIBEVOICE_PATCHER_CACHE.keys()):
        if key != keep_cache_key:
            # Set the model/processor to None but don't delete the patcher itself
            # This lets ComfyUI's model management handle the patcher cleanup
            try:
                patcher = VIBEVOICE_PATCHER_CACHE[key]
                if hasattr(patcher, 'model') and patcher.model:
                    patcher.model.model = None
                    patcher.model.processor = None
                # Remove from our cache but let ComfyUI handle the rest
                del VIBEVOICE_PATCHER_CACHE[key]
            except Exception as e:
                logger.warning(f"Error cleaning up patcher {key}: {e}")
    
    if keys_to_remove:
        logger.info(f"Cleaned up cached models: {keys_to_remove}")
        gc.collect()
        model_management.soft_empty_cache()

class VibeVoiceModelHandler(torch.nn.Module):
    """A torch.nn.Module wrapper to hold the VibeVoice model and processor."""
    def __init__(self, model_pack_name, attention_mode="eager", use_llm_4bit=False):
        super().__init__()
        self.model_pack_name = model_pack_name
        self.attention_mode = attention_mode
        self.use_llm_4bit = use_llm_4bit
        self.cache_key = f"{model_pack_name}_attn_{attention_mode}"
        self.model = None
        self.processor = None
        self.size = int(MODEL_CONFIGS[model_pack_name].get("size_gb", 4.0) * (1024**3))

    def load_model(self, device, attention_mode="eager"):
        self.model, self.processor = VibeVoiceLoader.load_model(self.model_pack_name, device, attention_mode, use_llm_4bit=self.use_llm_4bit)
        self.model.to(device)

class VibeVoicePatcher(comfy.model_patcher.ModelPatcher):
    """Custom ModelPatcher for managing VibeVoice models in ComfyUI."""
    def __init__(self, model, attention_mode="eager", *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.attention_mode = attention_mode
        self.cache_key = model.cache_key
    
    @property
    def is_loaded(self):
        """Check if the model is currently loaded in memory."""
        return hasattr(self, 'model') and self.model is not None and hasattr(self.model, 'model') and self.model.model is not None

    def patch_model(self, device_to=None, *args, **kwargs):
        target_device = self.load_device
        if self.model.model is None:
            logger.info(f"Loading VibeVoice models for '{self.model.model_pack_name}' to {target_device}...")
            mode_names = {
                "eager": "Eager (Most Compatible)",
                "sdpa": "SDPA (Balanced Speed/Compatibility)", 
                "flash_attention_2": "Flash Attention 2 (Fastest)"
            }
            logger.info(f"Attention Mode: {mode_names.get(self.attention_mode, self.attention_mode)}")
            self.model.load_model(target_device, self.attention_mode)
        self.model.model.to(target_device)
        return super().patch_model(device_to=target_device, *args, **kwargs)

    def unpatch_model(self, device_to=None, unpatch_weights=True, *args, **kwargs):
        if unpatch_weights:
            logger.info(f"Offloading VibeVoice models for '{self.model.model_pack_name}' ({self.attention_mode}) to {device_to}...")
            self.model.model = None
            self.model.processor = None
            
            # Clear using the correct cache key
            if self.cache_key in LOADED_MODELS:
                del LOADED_MODELS[self.cache_key]
                logger.info(f"Cleared LOADED_MODELS cache for: {self.cache_key}")
            
            # DON'T delete from VIBEVOICE_PATCHER_CACHE here - let ComfyUI handle it
            # This prevents the IndexError in ComfyUI's model management
            
            # Force garbage collection
            gc.collect()
            model_management.soft_empty_cache()
            
        return super().unpatch_model(device_to, unpatch_weights, *args, **kwargs)

class VibeVoiceLoader:
    @staticmethod
    def get_model_path(model_name: str):
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown VibeVoice model: {model_name}")
        
        vibevoice_path = os.path.join(folder_paths.get_folder_paths("tts")[0], "VibeVoice")
        model_path = os.path.join(vibevoice_path, model_name)
        
        index_file = os.path.join(model_path, "model.safetensors.index.json")
        if not os.path.exists(index_file):
            print(f"Downloading VibeVoice model: {model_name}...")
            repo_id = MODEL_CONFIGS[model_name]["repo_id"]
            snapshot_download(repo_id=repo_id, local_dir=model_path)
        return model_path

    @staticmethod
    def _check_attention_compatibility(attention_mode: str, torch_dtype, device_name: str = ""):
        """Check if the requested attention mode is compatible with current setup."""
        
        # Check for SDPA availability (PyTorch 2.0+)
        if attention_mode == "sdpa":
            if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                logger.warning("SDPA not available (requires PyTorch 2.0+), falling back to eager")
                return "eager"
        
        # Check for Flash Attention availability  
        elif attention_mode == "flash_attention_2":
            if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                logger.warning("Flash Attention not available, falling back to eager")
                return "eager"
            elif torch_dtype == torch.float32:
                logger.warning("Flash Attention not recommended with float32, falling back to SDPA")
                return "sdpa" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else "eager"
        
        # Just informational messages, no forced fallbacks
        if device_name and torch.cuda.is_available():
            if "RTX 50" in device_name or "Blackwell" in device_name:
                if attention_mode == "flash_attention_2":
                    logger.info(f"Using Flash Attention on {device_name}")
                elif attention_mode == "sdpa":
                    logger.info(f"Using SDPA on {device_name}")
        
        return attention_mode

    @staticmethod
    def load_model(model_name: str, device, attention_mode: str = "eager", use_llm_4bit: bool = False):
        # Validate attention mode
        if attention_mode not in ATTENTION_MODES:
            logger.warning(f"Unknown attention mode '{attention_mode}', falling back to eager")
            attention_mode = "eager"
            if use_llm_4bit and attention_mode == "flash_attention_2":
                attention_mode = "sdpa"
        
        # Create cache key that includes attention mode
        cache_key = f"{model_name}_attn_{attention_mode}"
        
        if cache_key in LOADED_MODELS:
            logger.info(f"Using cached model with {attention_mode} attention")
            return LOADED_MODELS[cache_key]

        model_path = VibeVoiceLoader.get_model_path(model_name)
        
        logger.info(f"Loading VibeVoice model components from: {model_path}")

        
        tokenizer_repo = MODEL_CONFIGS[model_name].get("tokenizer_repo")
        try:
            tokenizer_file_path = hf_hub_download(repo_id=tokenizer_repo, filename="tokenizer.json")
        except Exception as e:
            raise RuntimeError(f"Could not download tokenizer.json for {tokenizer_repo}. Error: {e}")

        vibevoice_tokenizer = VibeVoiceTextTokenizerFast(tokenizer_file=tokenizer_file_path)
        audio_processor = VibeVoiceTokenizerProcessor()
        processor = VibeVoiceProcessor(tokenizer=vibevoice_tokenizer, audio_processor=audio_processor)
        torch_dtype = model_management.text_encoder_dtype(device)
        device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else ""
        
        # Check compatibility and potentially fall back to safer mode
        final_attention_mode = VibeVoiceLoader._check_attention_compatibility(
            attention_mode, torch_dtype, device_name
        )

        # Build optional 4-bit config (LLM only)
        quant_config = None
        if use_llm_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        
        logger.info(f"Requested attention mode: {attention_mode}")
        if final_attention_mode != attention_mode:
            logger.info(f"Using attention mode: {final_attention_mode} (automatic fallback)")
            # Update cache key to reflect actual mode used
            cache_key = f"{model_name}_attn_{final_attention_mode}"
            if cache_key in LOADED_MODELS:
                return LOADED_MODELS[cache_key]
        else:
            logger.info(f"Using attention mode: {final_attention_mode}")
        
        logger.info(f"Final attention implementation: {final_attention_mode}")

        # Modify config for non-flash attention modes
        if final_attention_mode in ["eager", "sdpa"]:
            import json
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Remove flash attention settings
                    removed_keys = []
                    for key in ['_attn_implementation', 'attn_implementation', 'use_flash_attention_2']:
                        if key in config:
                            config.pop(key)
                            removed_keys.append(key)
                    
                    if removed_keys:
                        with open(config_path, 'w') as f:
                            json.dump(config, f, indent=2)
                        logger.info(f"Removed FlashAttention settings from config.json: {removed_keys}")
                except Exception as e:
                    logger.warning(f"Could not modify config.json: {e}")

        try:
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if quant_config else torch_dtype,
                attn_implementation=final_attention_mode,
                device_map="auto" if quant_config else device,
                quantization_config=quant_config,   # <- forwarded if supported
            )
            model.eval()
            setattr(model, "_llm_4bit", bool(quant_config))
            
            # Store with the actual attention mode used (not the requested one)
            LOADED_MODELS[cache_key] = (model, processor)
            logger.info(f"Successfully loaded model with {final_attention_mode} attention")
            return model, processor
            
        except Exception as e:
            logger.error(f"Failed to load model with {final_attention_mode} attention: {e}")

            # Progressive fallback: flash -> sdpa -> eager
            if final_attention_mode == "flash_attention_2":
                logger.info("Attempting fallback to SDPA...")
                return VibeVoiceLoader.load_model(model_name, device, "sdpa")
            elif final_attention_mode == "sdpa":
                logger.info("Attempting fallback to eager...")
                return VibeVoiceLoader.load_model(model_name, device, "eager")
            else:
                # If eager fails, something is seriously wrong
                raise RuntimeError(f"Failed to load model even with eager attention: {e}")


def set_vibevoice_seed(seed: int):
    """Sets the seed for torch, numpy, and random, handling large seeds for numpy."""
    if seed == 0:
        seed = random.randint(1, 0xffffffffffffffff)
    
    MAX_NUMPY_SEED = 2**32 - 1
    numpy_seed = seed % MAX_NUMPY_SEED
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(numpy_seed)
    random.seed(seed)

def parse_script_1_based(script: str) -> tuple[list[tuple[int, str]], list[int]]:
    """
    Parses a 1-based speaker script into a list of (speaker_id, text) tuples
    and a list of unique speaker IDs in the order of their first appearance.
    Internally, it converts speaker IDs to 0-based for the model.
    """
    parsed_lines = []
    speaker_ids_in_script = [] # This will store the 1-based IDs from the script
    for line in script.strip().split("\n"):
        if not (line := line.strip()): continue
        match = re.match(r'^Speaker\s+(\d+)\s*:\s*(.*)$', line, re.IGNORECASE)
        if match:
            speaker_id = int(match.group(1))
            if speaker_id < 1:
                logger.warning(f"Speaker ID must be 1 or greater. Skipping line: '{line}'")
                continue
            text = ' ' + match.group(2).strip()
            # Internally, the model expects 0-based indexing for speakers
            internal_speaker_id = speaker_id - 1
            parsed_lines.append((internal_speaker_id, text))
            if speaker_id not in speaker_ids_in_script:
                speaker_ids_in_script.append(speaker_id)
        else:
            logger.warning(f"Could not parse line, skipping: '{line}'")
    return parsed_lines, sorted(list(set(speaker_ids_in_script)))

def preprocess_comfy_audio(audio_dict: dict, target_sr: int = 24000) -> np.ndarray:
    """
    Converts a ComfyUI AUDIO dict to a mono NumPy array, resampling if necessary.
    """
    if not audio_dict: return None
    waveform_tensor = audio_dict.get('waveform')
    if waveform_tensor is None or waveform_tensor.numel() == 0: return None
    
    waveform = waveform_tensor[0].cpu().numpy()
    original_sr = audio_dict['sample_rate']
    
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=0)

    # Check for invalid values
    if np.any(np.isnan(waveform)) or np.any(np.isinf(waveform)):
        logger.error("Audio contains NaN or Inf values, replacing with zeros")
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure audio is not completely silent or has extreme values
    if np.all(waveform == 0):
        logger.warning("Audio waveform is completely silent")
    
    # Normalize extreme values
    max_val = np.abs(waveform).max()
    if max_val > 10.0:
        logger.warning(f"Audio values are very large (max: {max_val}), normalizing")
        waveform = waveform / max_val

    if original_sr != target_sr:
        if librosa is None:
            raise ImportError("`librosa` package is required for audio resampling. Please install it with `pip install librosa`.")
        logger.warning(f"Resampling reference audio from {original_sr}Hz to {target_sr}Hz.")
        waveform = librosa.resample(y=waveform, orig_sr=original_sr, target_sr=target_sr)
    
    # Final check after resampling
    if np.any(np.isnan(waveform)) or np.any(np.isinf(waveform)):
        logger.error("Audio contains NaN or Inf after resampling, replacing with zeros")
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        
    return waveform.astype(np.float32)

def check_for_interrupt():
    try:
        throw_exception_if_processing_interrupted()
        return False
    except:
        return True

class VibeVoiceTTSNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(MODEL_CONFIGS.keys()), {
                    "tooltip": "Select the VibeVoice model to use. Models will be downloaded automatically if not present."
                }),
                "text": ("STRING", {
                    "multiline": True, 
                    "default": "Speaker 1: Hello from ComfyUI!\nSpeaker 2: VibeVoice sounds amazing.",
                    "tooltip": "The script for the conversation. Use 'Speaker 1:', 'Speaker 2:', etc. to assign lines to different voices. Each speaker line should be on a new line."
                }),
                "quantize_llm_4bit": ("BOOLEAN", {
                    "default": False, "label_on": "Q4 (LLM only)", "label_off": "Full precision",
                    "tooltip": "Quantize the Qwen2.5 LLM to 4-bit NF4 via bitsandbytes. Diffusion head stays BF16/FP32."
                }),
                "attention_mode": (["eager", "sdpa", "flash_attention_2"], {
                    "default": "sdpa",
                    "tooltip": "Attention implementation: Eager (safest), SDPA (balanced), Flash Attention 2 (fastest but may cause issues on some GPUs like RTX 5090)"
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 1.3, "min": 1.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Classifier-Free Guidance scale. Higher values increase adherence to the voice prompt but may reduce naturalness. Recommended: 1.3"
                }),
                "inference_steps": ("INT", {
                    "default": 10, "min": 1, "max": 50,
                    "tooltip": "Number of diffusion steps for audio generation. More steps can improve quality but take longer. Recommended: 10"
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True,
                    "tooltip": "Seed for reproducibility. Set to 0 for a random seed on each run."
                }),
                "do_sample": ("BOOLEAN", {
                    "default": True, "label_on": "Enabled (Sampling)", "label_off": "Disabled (Greedy)",
                    "tooltip": "Enable to use sampling methods (like temperature and top_p) for more varied output. Disable for deterministic (greedy) decoding."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Controls randomness. Higher values make the output more random and creative, while lower values make it more focused and deterministic. Active only if 'do_sample' is enabled."
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Nucleus sampling (Top-P). The model samples from the smallest set of tokens whose cumulative probability exceeds this value. Active only if 'do_sample' is enabled."
                }),
                "top_k": ("INT", {
                    "default": 0, "min": 0, "max": 500, "step": 1, 
                    "tooltip": "Top-K sampling. Restricts sampling to the K most likely next tokens. Set to 0 to disable. Active only if 'do_sample' is enabled."
                }),
                "force_offload": ("BOOLEAN", {
                    "default": False, "label_on": "Force Offload", "label_off": "Keep in VRAM",
                    "tooltip": "Force model to be offloaded from VRAM after generation. Useful to free up memory between generations but may slow down subsequent runs."
                }),
            },
            "optional": {
                "speaker_1_voice": ("AUDIO", {"tooltip": "Reference audio for 'Speaker 1' in the script."}),
                "speaker_2_voice": ("AUDIO", {"tooltip": "Reference audio for 'Speaker 2' in the script."}),
                "speaker_3_voice": ("AUDIO", {"tooltip": "Reference audio for 'Speaker 3' in the script."}),
                "speaker_4_voice": ("AUDIO", {"tooltip": "Reference audio for 'Speaker 4' in the script."}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate_audio"
    CATEGORY = "audio/tts"

    def generate_audio(self, model_name, text, attention_mode, cfg_scale, inference_steps, seed, do_sample, temperature, top_p, top_k, quantize_llm_4bit, force_offload, **kwargs):
        if not text.strip():
            logger.warning("VibeVoiceTTS: Empty text provided, returning silent audio.")
            return ({"waveform": torch.zeros((1, 1, 24000), dtype=torch.float32), "sample_rate": 24000},)

        # Create cache key that includes attention mode
        cache_key = f"{model_name}_attn_{attention_mode}_q4_{int(quantize_llm_4bit)}"
        
        # Clean up old models when switching to a different model
        if cache_key not in VIBEVOICE_PATCHER_CACHE:
            # Only keep models that are currently being requested
            cleanup_old_models(keep_cache_key=cache_key)
            
            model_handler = VibeVoiceModelHandler(model_name, attention_mode, use_llm_4bit=quantize_llm_4bit)
            patcher = VibeVoicePatcher(
                model_handler,
                attention_mode=attention_mode,
                load_device=model_management.get_torch_device(), 
                offload_device=model_management.unet_offload_device(),
                size=model_handler.size
            )
            VIBEVOICE_PATCHER_CACHE[cache_key] = patcher
        
        patcher = VIBEVOICE_PATCHER_CACHE[cache_key]
        model_management.load_model_gpu(patcher)
        model = patcher.model.model
        processor = patcher.model.processor
        
        if model is None or processor is None:
            raise RuntimeError("VibeVoice model and processor could not be loaded. Check logs for errors.")
        
        parsed_lines_0_based, speaker_ids_1_based = parse_script_1_based(text)
        if not parsed_lines_0_based:
            raise ValueError("Script is empty or invalid. Use 'Speaker 1:', 'Speaker 2:', etc. format.")
            
        full_script = "\n".join([f"Speaker {spk}: {txt}" for spk, txt in parsed_lines_0_based])
        
        speaker_inputs = {i: kwargs.get(f"speaker_{i}_voice") for i in range(1, 5)}
        voice_samples_np = [preprocess_comfy_audio(speaker_inputs[sid]) for sid in speaker_ids_1_based]
        
        if any(v is None for v in voice_samples_np):
            missing_ids = [sid for sid, v in zip(speaker_ids_1_based, voice_samples_np) if v is None]
            raise ValueError(f"Script requires voices for Speakers {missing_ids}, but they were not provided.")
        
        set_vibevoice_seed(seed)
        
        try:
            inputs = processor(
                text=[full_script], voice_samples=[voice_samples_np], padding=True,
                return_tensors="pt", return_attention_mask=True
            )
            
            # Validate inputs before moving to GPU
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if torch.any(torch.isnan(value)) or torch.any(torch.isinf(value)):
                        logger.error(f"Input tensor '{key}' contains NaN or Inf values")
                        raise ValueError(f"Invalid values in input tensor: {key}")
            
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            model.set_ddpm_inference_steps(num_steps=inference_steps)

            generation_config = {'do_sample': do_sample}
            if do_sample:
                generation_config['temperature'] = temperature
                generation_config['top_p'] = top_p
                if top_k > 0:
                    generation_config['top_k'] = top_k
            
            # Hardware-specific optimizations - only for eager mode
            if attention_mode == "eager":
                # Apply RTX 5090 / Blackwell compatibility fixes only for eager
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.cuda.empty_cache()
                
                # Apply additional tensor fixes for eager mode
                model = model.float()
                processed_inputs = {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        # Keep integer/boolean tensors as-is (token IDs, attention masks, etc.)
                        if v.dtype in [torch.int, torch.long, torch.int32, torch.int64, torch.bool, torch.uint8]:
                            processed_inputs[k] = v
                        # Keep tensors with "mask" in their name as boolean
                        elif "mask" in k.lower():
                            processed_inputs[k] = v.bool() if v.dtype != torch.bool else v
                        else:
                            # Convert float/bfloat16 tensors to float32
                            processed_inputs[k] = v.float()
                    else:
                        processed_inputs[k] = v
                inputs = processed_inputs
            
            with torch.no_grad():
                # Create progress bar for inference steps
                pbar = ProgressBar(inference_steps)
                
                def progress_callback(step, total_steps):
                    pbar.update(1)
                    # Check for interruption from ComfyUI
                    if model_management.interrupt_current_processing:
                        raise comfy.model_management.InterruptProcessingException()

                # Custom generation loop with interruption support
                try:
                    outputs = model.generate(
                        **inputs, max_new_tokens=None, cfg_scale=cfg_scale,
                        tokenizer=processor.tokenizer, generation_config=generation_config,
                        verbose=False, stop_check_fn=check_for_interrupt
                    )
                    # Note: The model.generate method doesn't support progress callbacks in the current VibeVoice implementation
                    # But we check for interruption at the start and end of generation
                    pbar.update(inference_steps - pbar.current)
                    
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    if "assertion" in error_msg or "cuda" in error_msg:
                        logger.error(f"CUDA assertion failed with {attention_mode} attention: {e}")
                        logger.error("This might be due to invalid input data, GPU memory issues, or incompatible attention mode.")
                        logger.error("Try restarting ComfyUI, using different audio files, or switching to 'eager' attention mode.")
                    raise e
                except comfy.model_management.InterruptProcessingException:
                    logger.info("VibeVoice generation interrupted by user")
                    raise
                finally:
                    pbar.update_absolute(inference_steps)

        except comfy.model_management.InterruptProcessingException:
            logger.info("VibeVoice TTS generation was cancelled")
            # Return silent audio on cancellation
            return ({"waveform": torch.zeros((1, 1, 24000), dtype=torch.float32), "sample_rate": 24000},)
        
        except Exception as e:
            logger.error(f"Error during VibeVoice generation with {attention_mode} attention: {e}")
            if "interrupt" in str(e).lower() or "cancel" in str(e).lower():
                logger.info("Generation was interrupted")
                return ({"waveform": torch.zeros((1, 1, 24000), dtype=torch.float32), "sample_rate": 24000},)
            raise

        output_waveform = outputs.speech_outputs[0]
        if output_waveform.ndim == 1: output_waveform = output_waveform.unsqueeze(0)
        if output_waveform.ndim == 2: output_waveform = output_waveform.unsqueeze(0)
        
        # Force offload model if requested
        if force_offload:
            logger.info(f"Force offloading VibeVoice model '{model_name}' from VRAM...")
            # Force offload by unpatching the model and freeing memory
            if patcher.is_loaded:
                patcher.unpatch_model(unpatch_weights=True)
            # Force unload all models to free memory
            model_management.unload_all_models()
            gc.collect()
            model_management.soft_empty_cache()
            logger.info("Model force offload completed")
            
        return ({"waveform": output_waveform.detach().cpu(), "sample_rate": 24000},)

NODE_CLASS_MAPPINGS = {"VibeVoiceTTS": VibeVoiceTTSNode}
NODE_DISPLAY_NAME_MAPPINGS = {"VibeVoiceTTS": "VibeVoice TTS"}
