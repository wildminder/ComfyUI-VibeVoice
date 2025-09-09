import os, re, torch, numpy as np, random, logging, gc
from huggingface_hub import hf_hub_download, snapshot_download
import folder_paths, comfy.model_management as model_management, comfy.model_patcher
from comfy.utils import ProgressBar
from comfy.model_management import throw_exception_if_processing_interrupted
import transformers
from packaging import version

_transformers_version = version.parse(transformers.__version__)
_DTYPE_ARG_SUPPORTED = _transformers_version >= version.parse("4.56.0")

from transformers import set_seed, AutoTokenizer, BitsAndBytesConfig
from .vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from .vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from .vibevoice.processor.vibevoice_tokenizer_processor import VibeVoiceTokenizerProcessor
from .vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast

from . import SAGE_ATTENTION_AVAILABLE
if SAGE_ATTENTION_AVAILABLE:
    from .vibevoice.modular.sage_attention_patch import set_sage_attention

try:
    import librosa
except ImportError:
    print("VibeVoice Node: `librosa` not installed. Resampling unavailable.")
    librosa = None

logger = logging.getLogger(__name__)

LOADED_MODELS, VIBEVOICE_PATCHER_CACHE = {}, {}

MODEL_CONFIGS = {
    "VibeVoice-1.5B": {"repo_id": "microsoft/VibeVoice-1.5B", "size_gb": 3.0, "tokenizer_repo": "Qwen/Qwen2.5-1.5B"},
    "VibeVoice-Large": {"repo_id": "aoi-ot/VibeVoice-Large", "size_gb": 17.4, "tokenizer_repo": "Qwen/Qwen2.5-7B"}
}

ATTENTION_MODES = ["eager", "sdpa", "flash_attention_2"]
if SAGE_ATTENTION_AVAILABLE:
    ATTENTION_MODES.append("sage")

def cleanup_old_models(keep_cache_key=None):
    global LOADED_MODELS, VIBEVOICE_PATCHER_CACHE
    
    for key in list(LOADED_MODELS.keys()):
        if key != keep_cache_key:
            del LOADED_MODELS[key]
    
    for key in list(VIBEVOICE_PATCHER_CACHE.keys()):
        if key != keep_cache_key:
            try:
                patcher = VIBEVOICE_PATCHER_CACHE[key]
                if hasattr(patcher, 'model') and patcher.model:
                    patcher.model.model, patcher.model.processor = None, None
                del VIBEVOICE_PATCHER_CACHE[key]
            except Exception as e:
                logger.warning(f"Error cleaning up patcher {key}: {e}")
    
    gc.collect()
    model_management.soft_empty_cache()

class VibeVoiceModelHandler(torch.nn.Module):
    def __init__(self, model_pack_name, attention_mode="eager", use_llm_4bit=False):
        super().__init__()
        self.model_pack_name = model_pack_name
        self.attention_mode = attention_mode
        self.use_llm_4bit = use_llm_4bit
        self.cache_key = f"{model_pack_name}_attn_{attention_mode}_q4_{int(use_llm_4bit)}"
        self.model, self.processor = None, None
        self.size = int(MODEL_CONFIGS[model_pack_name].get("size_gb", 4.0) * (1024**3))

    def load_model(self, device, attention_mode="eager"):
        self.model, self.processor = VibeVoiceLoader.load_model(self.model_pack_name, device, attention_mode, use_llm_4bit=self.use_llm_4bit)
        self.model.to(device)

class VibeVoicePatcher(comfy.model_patcher.ModelPatcher):
    def __init__(self, model, attention_mode="eager", *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.attention_mode, self.cache_key = attention_mode, model.cache_key
    
    @property
    def is_loaded(self):
        return hasattr(self, 'model') and self.model is not None and hasattr(self.model, 'model') and self.model.model is not None

    def patch_model(self, device_to=None, *args, **kwargs):
        target_device = self.load_device
        if self.model.model is None:
            logger.info(f"Loading VibeVoice models for '{self.model.model_pack_name}' to {target_device}...")
            mode_names = {"eager": "Eager (Most Compatible)", "sdpa": "SDPA (Balanced Speed/Compatibility)", 
                         "flash_attention_2": "Flash Attention 2 (Fastest)", "sage": "SageAttention (Quantized High-Performance)"}
            logger.info(f"Attention Mode: {mode_names.get(self.attention_mode, self.attention_mode)}")
            self.model.load_model(target_device, self.attention_mode)
        self.model.model.to(target_device)
        return super().patch_model(device_to=target_device, *args, **kwargs)

    def unpatch_model(self, device_to=None, unpatch_weights=True, *args, **kwargs):
        if unpatch_weights:
            logger.info(f"Offloading VibeVoice models for '{self.model.model_pack_name}' ({self.attention_mode}) to {device_to}...")
            self.model.model, self.model.processor = None, None
            
            if self.cache_key in LOADED_MODELS:
                del LOADED_MODELS[self.cache_key]
            
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
    def _check_gpu_for_sage_attention():
        if not SAGE_ATTENTION_AVAILABLE or not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            logger.warning(f"Your GPU (compute capability {major}.x) does not support SageAttention, which requires CC 8.0+. Sage option will be disabled.")
            return False
        return True
        
    @staticmethod
    def load_model(model_name: str, device, attention_mode: str = "eager", use_llm_4bit: bool = False):
        if use_llm_4bit and attention_mode in ["eager", "flash_attention_2"]:
            logger.warning(f"Attention mode '{attention_mode}' not recommended with 4-bit quantization. Falling back to 'sdpa'.")
            attention_mode = "sdpa"

        if attention_mode not in ATTENTION_MODES:
            logger.warning(f"Unknown attention mode '{attention_mode}', falling back to eager")
            attention_mode = "eager"

        cache_key = f"{model_name}_attn_{attention_mode}_q4_{int(use_llm_4bit)}"
        
        if cache_key in LOADED_MODELS:
            logger.info(f"Using cached model with {attention_mode} attention and q4={use_llm_4bit}")
            return LOADED_MODELS[cache_key]

        model_path = VibeVoiceLoader.get_model_path(model_name)
        logger.info(f"Loading VibeVoice model components from: {model_path}")

        tokenizer_repo = MODEL_CONFIGS[model_name].get("tokenizer_repo")
        tokenizer_file_path = os.path.join(model_path, "tokenizer.json")
        if not os.path.exists(tokenizer_file_path):
            logger.info(f"tokenizer.json not found in {model_path}. Downloading from '{tokenizer_repo}'...")
            try:
                hf_hub_download(repo_id=tokenizer_repo, filename="tokenizer.json", local_dir=model_path)
            except Exception as e:
                logger.error(f"Failed to download tokenizer.json: {e}")
                raise
        
        vibevoice_tokenizer = VibeVoiceTextTokenizerFast(tokenizer_file=tokenizer_file_path)
        audio_processor = VibeVoiceTokenizerProcessor()
        processor = VibeVoiceProcessor(tokenizer=vibevoice_tokenizer, audio_processor=audio_processor)
        
        model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        quant_config, final_load_dtype = None, model_dtype

        if use_llm_4bit:
            bnb_compute_dtype = model_dtype
            if attention_mode == 'sage':
                logger.info("Using SageAttention with 4-bit quant. Forcing fp32 compute dtype for stability.")
                bnb_compute_dtype, final_load_dtype = torch.float32, torch.float32
            else:
                 logger.info(f"Using {attention_mode} with 4-bit quant. Using {model_dtype} compute dtype.")
            
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bnb_compute_dtype
            )

        attn_implementation_for_load = "sdpa" if attention_mode == "sage" else attention_mode
        
        try:
            logger.info(f"Loading model with dtype: {final_load_dtype} and attention: '{attn_implementation_for_load}'")
            from_pretrained_kwargs = {
                "attn_implementation": attn_implementation_for_load,
                "device_map": "auto" if quant_config else device,
                "quantization_config": quant_config,
            }

            if _DTYPE_ARG_SUPPORTED:
                from_pretrained_kwargs['dtype'] = final_load_dtype
            else:
                from_pretrained_kwargs['torch_dtype'] = final_load_dtype
                
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(model_path, **from_pretrained_kwargs)
            
            if attention_mode == "sage" and VibeVoiceLoader._check_gpu_for_sage_attention():
                logger.info("Applying SageAttention patch to the model...")
                set_sage_attention(model)
            elif attention_mode == "sage":
                logger.error("Cannot apply SageAttention due to incompatible GPU. Falling back.")
                raise RuntimeError("Incompatible hardware/setup for SageAttention.")

            model.eval()
            setattr(model, "_llm_4bit", bool(quant_config))
            
            LOADED_MODELS[cache_key] = (model, processor)
            logger.info(f"Successfully configured model with {attention_mode} attention")
            return model, processor
            
        except Exception as e:
            logger.error(f"Failed to load model with {attention_mode} attention: {e}")
            if attention_mode in ["sage", "flash_attention_2"]:
                logger.info("Attempting fallback to SDPA...")
                return VibeVoiceLoader.load_model(model_name, device, "sdpa", use_llm_4bit)
            elif attention_mode == "sdpa":
                logger.info("Attempting fallback to eager...")
                return VibeVoiceLoader.load_model(model_name, device, "eager", use_llm_4bit)
            else:
                raise RuntimeError(f"Failed to load model even with eager attention: {e}")

def set_vibevoice_seed(seed: int):
    if seed == 0:
        seed = random.randint(1, 0xffffffffffffffff)
    
    MAX_NUMPY_SEED = 2**32 - 1
    numpy_seed = seed % MAX_NUMPY_SEED
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(numpy_seed)
    random.seed(seed)

def parse_script_1_based(script: str):
    parsed_lines, speaker_ids_in_script = [], []
    current_speaker, accumulated_text = 1, ""
    
    lines = script.strip().split("\n")
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        if not line:
            if accumulated_text:
                internal_speaker_id = current_speaker - 1
                parsed_lines.append((internal_speaker_id, ' ' + accumulated_text))
                if current_speaker not in speaker_ids_in_script:
                    speaker_ids_in_script.append(current_speaker)
                accumulated_text = ""
            continue
        
        match = re.match(r'^Speaker\s+(\d+)\s*:\s*(.*)$', line, re.IGNORECASE)
        if match:
            if accumulated_text:
                internal_speaker_id = current_speaker - 1
                parsed_lines.append((internal_speaker_id, ' ' + accumulated_text))
                if current_speaker not in speaker_ids_in_script:
                    speaker_ids_in_script.append(current_speaker)
                accumulated_text = ""
            
            speaker_id = int(match.group(1))
            if speaker_id < 1:
                logger.warning(f"Speaker ID must be 1 or greater. Skipping line: '{line}'")
                continue
            current_speaker = speaker_id
            text_content = match.group(2).strip()
            
            if text_content:
                accumulated_text = text_content
        else:
            accumulated_text = accumulated_text + " " + line if accumulated_text else line
    
    if accumulated_text:
        internal_speaker_id = current_speaker - 1
        parsed_lines.append((internal_speaker_id, ' ' + accumulated_text))
        if current_speaker not in speaker_ids_in_script:
            speaker_ids_in_script.append(current_speaker)
    
    return parsed_lines, sorted(list(set(speaker_ids_in_script)))

def preprocess_comfy_audio(audio_dict: dict, target_sr: int = 24000):
    if not audio_dict: return None
    waveform_tensor = audio_dict.get('waveform')
    if waveform_tensor is None or waveform_tensor.numel() == 0: return None
    
    waveform = waveform_tensor[0].cpu().numpy()
    original_sr = audio_dict['sample_rate']
    
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=0)

    if np.any(np.isnan(waveform)) or np.any(np.isinf(waveform)):
        logger.error("Audio contains NaN or Inf values, replacing with zeros")
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.all(waveform == 0):
        logger.warning("Audio waveform is completely silent")
    
    max_val = np.abs(waveform).max()
    if max_val > 10.0:
        logger.warning(f"Audio values are very large (max: {max_val}), normalizing")
        waveform = waveform / max_val

    if original_sr != target_sr:
        if librosa is None:
            raise ImportError("`librosa` package required for audio resampling. Install with `pip install librosa`.")
        logger.warning(f"Resampling reference audio from {original_sr}Hz to {target_sr}Hz.")
        waveform = librosa.resample(y=waveform, orig_sr=original_sr, target_sr=target_sr)
    
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
                "model_name": (list(MODEL_CONFIGS.keys()), {"tooltip": "Select the VibeVoice model to use. Models will be downloaded automatically if not present."}),
                "text": ("STRING", {"multiline": True, "default": "Speaker 1: Hello from ComfyUI!\nSpeaker 2: VibeVoice sounds amazing.", "tooltip": "The script for the conversation. Use 'Speaker 1:', 'Speaker 2:', etc. to assign lines to different voices. Each speaker line should be on a new line."}),
                "quantize_llm_4bit": ("BOOLEAN", {"default": False, "label_on": "Q4 (LLM only)", "label_off": "Full precision", "tooltip": "Quantize the Qwen2.5 LLM to 4-bit NF4 via bitsandbytes. Diffusion head stays BF16/FP32."}),
                "attention_mode": (ATTENTION_MODES, {"default": "sdpa", "tooltip": "Attention implementation: Eager (safest), SDPA (balanced), Flash Attention 2 (fastest), Sage (quantized)"}),
                "cfg_scale": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "Classifier-Free Guidance scale. Higher values increase adherence to the voice prompt but may reduce naturalness. Recommended: 1.3"}),
                "inference_steps": ("INT", {"default": 10, "min": 1, "max": 50, "tooltip": "Number of diffusion steps for audio generation. More steps can improve quality but take longer. Recommended: 10"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True, "tooltip": "Seed for reproducibility. Set to 0 for a random seed on each run."}),
                "do_sample": ("BOOLEAN", {"default": True, "label_on": "Enabled (Sampling)", "label_off": "Disabled (Greedy)", "tooltip": "Enable to use sampling methods (like temperature and top_p) for more varied output. Disable for deterministic (greedy) decoding."}),
                "temperature": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Controls randomness. Higher values make the output more random and creative, while lower values make it more focused and deterministic. Active only if 'do_sample' is enabled."}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Nucleus sampling (Top-P). The model samples from the smallest set of tokens whose cumulative probability exceeds this value. Active only if 'do_sample' is enabled."}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 500, "step": 1, "tooltip": "Top-K sampling. Restricts sampling to the K most likely next tokens. Set to 0 to disable. Active only if 'do_sample' is enabled."}),
                "force_offload": ("BOOLEAN", {"default": False, "label_on": "Force Offload", "label_off": "Keep in VRAM", "tooltip": "Force model to be offloaded from VRAM after generation. Useful to free up memory between generations but may slow down subsequent runs."}),
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
        actual_attention_mode = "sdpa" if quantize_llm_4bit and attention_mode in ["eager", "flash_attention_2"] else attention_mode
        
        cache_key = f"{model_name}_attn_{actual_attention_mode}_q4_{int(quantize_llm_4bit)}"
        
        if cache_key not in VIBEVOICE_PATCHER_CACHE:
            cleanup_old_models(keep_cache_key=cache_key)
            
            model_handler = VibeVoiceModelHandler(model_name, attention_mode, use_llm_4bit=quantize_llm_4bit)
            patcher = VibeVoicePatcher(
                model_handler, attention_mode=attention_mode,
                load_device=model_management.get_torch_device(), 
                offload_device=model_management.unet_offload_device(),
                size=model_handler.size
            )
            VIBEVOICE_PATCHER_CACHE[cache_key] = patcher
        
        patcher = VIBEVOICE_PATCHER_CACHE[cache_key]
        model_management.load_model_gpu(patcher)
        model, processor = patcher.model.model, patcher.model.processor
        
        if model is None or processor is None:
            raise RuntimeError("VibeVoice model and processor could not be loaded. Check logs for errors.")
        
        parsed_lines_0_based, speaker_ids_1_based = parse_script_1_based(text)
        if not parsed_lines_0_based:
            raise ValueError("Script is empty or invalid. Use 'Speaker 1:', 'Speaker 2:', etc. format.")
            
        full_script = "\n".join([f"Speaker {spk+1}: {txt}" for spk, txt in parsed_lines_0_based])
        
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
            
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and (torch.any(torch.isnan(value)) or torch.any(torch.isinf(value))):
                    logger.error(f"Input tensor '{key}' contains NaN or Inf values")
                    raise ValueError(f"Invalid values in input tensor: {key}")
            
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            model.set_ddpm_inference_steps(num_steps=inference_steps)

            generation_config = {'do_sample': do_sample}
            if do_sample:
                generation_config.update({'temperature': temperature, 'top_p': top_p})
                if top_k > 0:
                    generation_config['top_k'] = top_k

            with torch.no_grad():
                pbar = ProgressBar(inference_steps)
                
                def progress_callback(step, total_steps):
                    pbar.update(1)
                    if model_management.interrupt_current_processing:
                        raise comfy.model_management.InterruptProcessingException()

                try:
                    outputs = model.generate(
                        **inputs, max_new_tokens=None, cfg_scale=cfg_scale,
                        tokenizer=processor.tokenizer, generation_config=generation_config,
                        verbose=False, stop_check_fn=check_for_interrupt
                    )
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
        
        if force_offload and patcher.is_loaded:
            logger.info(f"Force offloading VibeVoice model '{model_name}' from VRAM...")
            patcher.unpatch_model(unpatch_weights=True)
            model_management.unload_all_models()
            gc.collect()
            model_management.soft_empty_cache()
            logger.info("Model force offload completed")
            
        return ({"waveform": output_waveform.detach().cpu(), "sample_rate": 24000},)

NODE_CLASS_MAPPINGS = {"VibeVoiceTTS": VibeVoiceTTSNode}
NODE_DISPLAY_NAME_MAPPINGS = {"VibeVoiceTTS": "VibeVoice TTS"}