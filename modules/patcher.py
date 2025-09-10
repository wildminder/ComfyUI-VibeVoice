import torch
import gc
import logging
import comfy.model_patcher
import comfy.model_management as model_management

from .loader import LOADED_MODELS, logger

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
                "flash_attention_2": "Flash Attention 2 (Fastest)",
                "sage": "SageAttention (Quantized High-Performance)",
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
            
            if self.cache_key in LOADED_MODELS:
                del LOADED_MODELS[self.cache_key]
                logger.info(f"Cleared LOADED_MODELS cache for: {self.cache_key}")
            
            gc.collect()
            model_management.soft_empty_cache()
            
        return super().unpatch_model(device_to, unpatch_weights, *args, **kwargs)