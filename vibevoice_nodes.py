import os
import re
import torch
import numpy as np
import random
from huggingface_hub import snapshot_download
import logging
import librosa

import folder_paths
import comfy.model_management as model_management
import comfy.model_patcher
from comfy.utils import ProgressBar


from transformers import set_seed
from .vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from .vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

logger = logging.getLogger("comfyui_vibevoice")

LOADED_MODELS = {}
VIBEVOICE_PATCHER_CACHE = {}

MODEL_CONFIGS = {
    "VibeVoice-1.5B": {
        "repo_id": "microsoft/VibeVoice-1.5B",
        "size_gb": 3.0,
    },
    "VibeVoice-Large-pt": {
        "repo_id": "WestZhang/VibeVoice-Large-pt",
        "size_gb": 14.0,
    }
}

class VibeVoiceModelHandler(torch.nn.Module):
    """A torch.nn.Module wrapper to hold the VibeVoice model and processor."""
    def __init__(self, model_pack_name):
        super().__init__()
        self.model_pack_name = model_pack_name
        self.model = None
        self.processor = None
        self.size = int(MODEL_CONFIGS[model_pack_name].get("size_gb", 4.0) * (1024**3))

    def load_model(self, device):
        self.model, self.processor = VibeVoiceLoader.load_model(self.model_pack_name)
        self.model.to(device)

class VibeVoicePatcher(comfy.model_patcher.ModelPatcher):
    """Custom ModelPatcher for managing VibeVoice models in ComfyUI."""
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    def patch_model(self, device_to=None, *args, **kwargs):
        target_device = self.load_device
        if self.model.model is None:
            logger.info(f"Loading VibeVoice models for '{self.model.model_pack_name}' to {target_device}...")
            self.model.load_model(target_device)
        self.model.model.to(target_device)
        return super().patch_model(device_to=target_device, *args, **kwargs)

    def unpatch_model(self, device_to=None, unpatch_weights=True, *args, **kwargs):
        if unpatch_weights:
            logger.info(f"Offloading VibeVoice models for '{self.model.model_pack_name}' to {device_to}...")
            self.model.model = None
            self.model.processor = None
            if self.model.model_pack_name in LOADED_MODELS:
                del LOADED_MODELS[self.model.model_pack_name]
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
    def load_model(model_name: str):
        if model_name in LOADED_MODELS:
            return LOADED_MODELS[model_name]

        model_path = VibeVoiceLoader.get_model_path(model_name)
        
        print(f"Loading VibeVoice model components from: {model_path}")
        processor = VibeVoiceProcessor.from_pretrained(model_path)
        
        torch_dtype = model_management.text_encoder_dtype(model_management.get_torch_device())

        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') and torch_dtype != torch.float32 else "eager",
        )
        model.eval()
        
        LOADED_MODELS[model_name] = (model, processor)
        return model, processor


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

    if original_sr != target_sr:
        logger.warning(f"Resampling reference audio from {original_sr}Hz to {target_sr}Hz.")
        waveform = librosa.resample(y=waveform, orig_sr=original_sr, target_sr=target_sr)
        
    return waveform.astype(np.float32)


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

    def generate_audio(self, model_name, text, cfg_scale, inference_steps, seed, do_sample, temperature, top_p, top_k, **kwargs):
        if not text.strip():
            logger.warning("VibeVoiceTTS: Empty text provided, returning silent audio.")
            return ({"waveform": torch.zeros((1, 1, 24000), dtype=torch.float32), "sample_rate": 24000},)

        cache_key = model_name
        if cache_key not in VIBEVOICE_PATCHER_CACHE:
            model_handler = VibeVoiceModelHandler(model_name)
            patcher = VibeVoicePatcher(
                model_handler, 
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
        
        inputs = processor(
            text=[full_script], voice_samples=[voice_samples_np], padding=True,
            return_tensors="pt", return_attention_mask=True
        )
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        model.set_ddpm_inference_steps(num_steps=inference_steps)

        generation_config = {'do_sample': do_sample}
        if do_sample:
            generation_config['temperature'] = temperature
            generation_config['top_p'] = top_p
            if top_k > 0:
                generation_config['top_k'] = top_k
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=None, cfg_scale=cfg_scale,
                tokenizer=processor.tokenizer, generation_config=generation_config,
                verbose=False
            )

        output_waveform = outputs.speech_outputs[0]
        if output_waveform.ndim == 1: output_waveform = output_waveform.unsqueeze(0)
        if output_waveform.ndim == 2: output_waveform = output_waveform.unsqueeze(0)
            
        return ({"waveform": output_waveform.detach().cpu(), "sample_rate": 24000},)

NODE_CLASS_MAPPINGS = {"VibeVoiceTTS": VibeVoiceTTSNode}
NODE_DISPLAY_NAME_MAPPINGS = {"VibeVoiceTTS": "VibeVoice TTS"}