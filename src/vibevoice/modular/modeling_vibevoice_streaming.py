from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers.models.auto import AutoModel, AutoModelForCausalLM

from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutput, BaseModelOutputWithPast, ModelOutput
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers import modeling_utils
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from .modular_vibevoice_diffusion_head import VibeVoiceDiffusionHead
from ..schedule.dpm_solver import DPMSolverMultistepScheduler

from .configuration_vibevoice_streaming import VibeVoiceStreamingConfig
from .configuration_vibevoice import get_config_dtype


if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]


class BinaryClassifier(nn.Module):
    def __init__(self, hidden_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SpeechConnector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = LlamaRMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, features, **kwargs):    
        x = self.fc1(features)
        x = self.norm(x)
        x = self.fc2(x)
        return x


# @auto_docstring
class VibeVoiceStreamingPreTrainedModel(PreTrainedModel):
    config_class = VibeVoiceStreamingConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        # transformers 5.x builds the model on meta, loads the checkpoint, then
        # walks the whole graph through _initialize_missing_keys ->
        # initialize_weights() -> _init_weights(). PreTrainedModel._initialize_weights
        # only honours the per-parameter ``_is_hf_initialized`` flag for remote
        # code, so without this guard every checkpoint-loaded tensor is
        # overwritten with the initializer distribution. Measured against
        # VibeVoice-Realtime-0.5B that silently destroyed 8 tensors
        # (acoustic_connector.{fc1,fc2} and tts_eos_classifier.{fc1,fc2},
        # weight+bias) while reporting missing=276 unexpected=0. A module whose
        # own parameters are all flagged is already loaded and must be left
        # alone; a module with any unflagged parameter is genuinely new (e.g. the
        # checkpoint-absent acoustic-tokenizer encoder) and is still initialised.
        params = list(module.parameters(recurse=False))
        if params and all(getattr(p, "_is_hf_initialized", False) for p in params):
            return

        if isinstance(module, VibeVoiceDiffusionHead):
            module.initialize_weights()
            return

        # Use the language model's initializer_range if available
        if hasattr(self.config, 'language_model_config') and hasattr(self.config.language_model_config, 'initializer_range'):
            std = self.config.language_model_config.initializer_range
        elif hasattr(self.config, 'decoder_config') and hasattr(self.config.decoder_config, 'initializer_range'):
            std = self.config.decoder_config.initializer_range
        else:
            std = 0.02  # Default value
            
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()


# @auto_docstring
class VibeVoiceStreamingModel(VibeVoiceStreamingPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        dtype = get_config_dtype(config)
        if dtype is None:
            dtype = torch.float32
        
        # Initialize Qwen2 model for language modeling. 
        # The lower Transformer layers are only used for encoding text, while the upper Transformer layers are used for encoding text and generating speech. 
        # To keep the code clean, we constructs two language models.
        # The final norm layer of the first language_model is set to identity and will not be used in inference.
        lm_config = copy.deepcopy(config.decoder_config)
        lm_backbone_num_hidden_layers = getattr(lm_config, 'num_hidden_layers', 24) - config.tts_backbone_num_hidden_layers
        lm_config.num_hidden_layers = lm_backbone_num_hidden_layers
        self.language_model = AutoModel.from_config(lm_config)
        self.language_model.norm = nn.Identity()
        
        # We only need the Transformer layers here. Note that embed_tokens in tts_language_model is unused
        tts_lm_config = copy.deepcopy(lm_config)
        tts_lm_config.num_hidden_layers = config.tts_backbone_num_hidden_layers
        self.tts_language_model = AutoModel.from_config(tts_lm_config)

        # Marks the text that needs to be spoken by the TTS model.
        self.tts_input_types = nn.Embedding(num_embeddings=2, embedding_dim=config.decoder_config.hidden_size)
        
        # Initialize speech components if needed
        # Guard .to(dtype) calls for meta tensor safety (transformers 5.x from_pretrained
        # uses torch.device("meta") context, where .to(dtype) on meta tensors fails)
        self.acoustic_tokenizer = AutoModel.from_config(config.acoustic_tokenizer_config)
        self.acoustic_connector = SpeechConnector(config.acoustic_vae_dim, lm_config.hidden_size)
        if not any(p.is_meta for p in self.acoustic_tokenizer.parameters()):
            self.acoustic_tokenizer = self.acoustic_tokenizer.to(dtype)
            self.acoustic_connector = self.acoustic_connector.to(dtype)
        
        # Register scaling factors as buffers - use 1D tensors for FSDP compatibility
        self.register_buffer('speech_scaling_factor', torch.tensor(float('nan')))  
        self.register_buffer('speech_bias_factor', torch.tensor(float('nan')))

        # Initialize prediction head for speech generation
        self.prediction_head = AutoModel.from_config(config.diffusion_head_config)
        if not any(p.is_meta for p in self.prediction_head.parameters()):
            self.prediction_head = self.prediction_head.to(dtype)

        # Initialize noise scheduler
        self.noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=config.diffusion_head_config.ddpm_num_steps,
            beta_schedule=config.diffusion_head_config.ddpm_beta_schedule,
            prediction_type=config.diffusion_head_config.prediction_type
        )
    
    def get_input_embeddings(self):
        if hasattr(self.language_model, 'embed_tokens'):
            # If the language model has an embed_tokens attribute, return it
            return self.language_model.embed_tokens
        
        for name, attr in self.language_model.fullmap.items(): # parallel by nnscaler, the name is changed
            if attr.orig_name == 'embed_tokens.weight':
                return getattr(self.language_model, name)
        assert False, 'should not arrive here'

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value
    
    def set_speech_tokenizers(self, acoustic_tokenizer=None):
        """Set the speech tokenizers used for encoding and decoding speech."""
        self.acoustic_tokenizer = acoustic_tokenizer
        
        # Reset the encoder to evaluation mode
        if self.acoustic_tokenizer is not None:
            self.acoustic_tokenizer.eval()
    
    def forward(self, *args, **kwargs):
        """
        Intentionally not implemented.

        This streaming model is split into two explicit submodules:
          - `language_model`      for plain text processing (lower layers).
          - `tts_language_model`  for TTS-related upper layers.

        We deliberately avoid a unified `forward` to prevent accidental calls
        that mix responsibilities.

        To use the model:
          - Call `self.language_model(...)` for text embeddings / hidden states.
          - Call `self.tts_language_model(...)` for the TTS portion.
          - Use the dedicated inference class for combined generation logic.
        """
        raise RuntimeError(
            "VibeVoiceStreamingModel.forward is intentionally disabled. "
            "Use `model.language_model(...)` or `model.tts_language_model(...)` instead."
        )


AutoModel.register(VibeVoiceStreamingConfig, VibeVoiceStreamingModel, exist_ok=True)

__all__ = [
    "VibeVoiceStreamingPreTrainedModel",
    "VibeVoiceStreamingModel",
]