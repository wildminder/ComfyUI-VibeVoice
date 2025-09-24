import torch
import gc
import logging

import comfy.model_management as model_management
from comfy.utils import ProgressBar

# Import from the dedicated model_info module
from .modules.model_info import AVAILABLE_VIBEVOICE_MODELS
from .modules.loader import VibeVoiceModelHandler, ATTENTION_MODES, VIBEVOICE_PATCHER_CACHE, cleanup_old_models
from .modules.patcher import VibeVoicePatcher
from .modules.utils import parse_script_1_based, preprocess_comfy_audio, set_vibevoice_seed, check_for_interrupt

logger = logging.getLogger(__name__)

class VibeVoiceTTSNode:
    @classmethod
    def INPUT_TYPES(cls):
        model_names = list(AVAILABLE_VIBEVOICE_MODELS.keys())
        if not model_names:
            model_names.append("No models found in models/tts/VibeVoice")

        return {
            "required": {
                "model_name": (model_names, {
                    "tooltip": "Select the VibeVoice model to use. Official models will be downloaded automatically."
                }),
                "text": ("STRING", {
                    "multiline": True, 
                    "default": "[1] Hello, this is a cloned voice.\n[2] And this is a generated voice, how cool is that?",
                    "tooltip": "The script for generation. Use '[1]' or 'Speaker 1:' for speakers. If a speaker in the script lacks a reference voice, it will be generated via zero-shot TTS."
                }),
                "quantize_llm_4bit": ("BOOLEAN", {
                    "default": False, "label_on": "Q4 (LLM only)", "label_off": "Full precision",
                    "tooltip": "Quantize the Qwen2.5 LLM to 4-bit NF4 via bitsandbytes. Diffusion head stays BF16/FP32."
                }),
                "attention_mode": (ATTENTION_MODES, {
                    "default": "sdpa",
                    "tooltip": "Attention implementation: Eager (safest), SDPA (balanced), Flash Attention 2 (fastest), Sage (quantized)"
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 1.3, "min": 0.1, "max": 50.0, "step": 0.05,
                    "tooltip": "Classifier-Free Guidance scale. Higher values increase adherence to the voice prompt but may reduce naturalness. Recommended: 1.3"
                }),
                "inference_steps": ("INT", {
                    "default": 10, "min": 1, "max": 500,
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
                "speaker_1_voice": ("AUDIO", {"tooltip": "Reference audio for 'Speaker 1' or '[1]' in the script."}),
                "speaker_2_voice": ("AUDIO", {"tooltip": "Reference audio for 'Speaker 2' or '[2]' in the script."}),
                "speaker_3_voice": ("AUDIO", {"tooltip": "Reference audio for 'Speaker 3' or '[3]' in the script."}),
                "speaker_4_voice": ("AUDIO", {"tooltip": "Reference audio for 'Speaker 4' or '[4]' in the script."}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate_audio"
    CATEGORY = "audio/tts"

    def generate_audio(self, model_name, text, attention_mode, cfg_scale, inference_steps, seed, do_sample, temperature, top_p, top_k, quantize_llm_4bit, force_offload, **kwargs):
        actual_attention_mode = attention_mode
        if quantize_llm_4bit and attention_mode in ["eager", "flash_attention_2"]:
            actual_attention_mode = "sdpa"
        
        cache_key = f"{model_name}_attn_{actual_attention_mode}_q4_{int(quantize_llm_4bit)}"
        
        if cache_key not in VIBEVOICE_PATCHER_CACHE:
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
            raise ValueError("Script is empty or invalid. Please provide text to generate.")
            
        # full_script = "\n".join([f"Speaker {spk+1}: {txt}" for spk, txt in parsed_lines_0_based]) # <-- REMOVED: This was the cause of the bug.
        
        speaker_inputs = {i: kwargs.get(f"speaker_{i}_voice") for i in range(1, 5)}
        voice_samples_np = [preprocess_comfy_audio(speaker_inputs.get(sid)) for sid in speaker_ids_1_based]

        set_vibevoice_seed(seed)
        
        try:
            inputs = processor(
                parsed_scripts=[parsed_lines_0_based],
                voice_samples=[voice_samples_np], 
                speaker_ids_for_prompt=[speaker_ids_1_based],
                padding=True,
                return_tensors="pt", 
                return_attention_mask=True
            )
            
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

            with torch.no_grad():
                pbar = ProgressBar(inference_steps)
                
                def progress_callback(step, total_steps):
                    pbar.update(1)
                    if model_management.interrupt_current_processing:
                        raise model_management.InterruptProcessingException()

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
                except model_management.InterruptProcessingException:
                    logger.info("VibeVoice generation interrupted by user")
                    raise
                finally:
                    pbar.update_absolute(inference_steps)

        except model_management.InterruptProcessingException:
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
        
        if force_offload:
            logger.info(f"Force offloading VibeVoice model '{model_name}' from VRAM...")
            if patcher.is_loaded:
                patcher.unpatch_model(unpatch_weights=True)
            model_management.unload_all_models()
            gc.collect()
            model_management.soft_empty_cache()
            logger.info("Model force offload completed")
            
        return ({"waveform": output_waveform.detach().cpu(), "sample_rate": 24000},)

NODE_CLASS_MAPPINGS = {"VibeVoiceTTS": VibeVoiceTTSNode}
NODE_DISPLAY_NAME_MAPPINGS = {"VibeVoiceTTS": "VibeVoice TTS"}