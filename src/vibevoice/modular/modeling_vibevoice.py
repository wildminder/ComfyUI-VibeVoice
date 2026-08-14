from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable
from tqdm import tqdm
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
from transformers.utils import logging


from .modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache, VibeVoiceAcousticTokenizerModel, VibeVoiceSemanticTokenizerModel
from .modular_vibevoice_diffusion_head import VibeVoiceDiffusionHead
from ..schedule.dpm_solver import DPMSolverMultistepScheduler

from .configuration_vibevoice import VibeVoiceConfig


logger = logging.get_logger(__name__)

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

@dataclass
class VibeVoiceCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    diffusion_loss: Optional[torch.FloatTensor] = None
    speech_token_num: Optional[int] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class VibeVoiceGenerationOutput(ModelOutput):
    """
    Output type for VibeVoice generation.
    
    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. 
        speech_outputs (`List[torch.FloatTensor]`, *optional*):
            List of generated speech waveforms or latents for each speech segment.
    """
    sequences: torch.LongTensor = None
    speech_outputs: Optional[List[torch.FloatTensor]] = None


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
class VibeVoicePreTrainedModel(PreTrainedModel):
    config_class = VibeVoiceConfig
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
class VibeVoiceModel(VibeVoicePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        if hasattr(config, 'torch_dtype') and config.torch_dtype is not None:
            if isinstance(config.torch_dtype, str):
                dtype = getattr(torch, config.torch_dtype)
            else:
                dtype = config.torch_dtype
        else:
            dtype = torch.float32
        
        # Initialize Qwen2 model for language modeling
        lm_config = config.decoder_config
        self.language_model = AutoModel.from_config(lm_config)
        
        # Initialize speech components if needed
        # Guard .to(dtype) calls for meta tensor safety (transformers 5.x from_pretrained
        # uses torch.device("meta") context, where .to(dtype) on meta tensors fails)
        self.acoustic_tokenizer = AutoModel.from_config(config.acoustic_tokenizer_config)
        self.semantic_tokenizer = AutoModel.from_config(config.semantic_tokenizer_config)
        if not any(p.is_meta for p in self.acoustic_tokenizer.parameters()):
            self.acoustic_tokenizer = self.acoustic_tokenizer.to(dtype)
            self.semantic_tokenizer = self.semantic_tokenizer.to(dtype)

        self.acoustic_connector = SpeechConnector(config.acoustic_vae_dim, lm_config.hidden_size)
        self.semantic_connector = SpeechConnector(config.semantic_vae_dim, lm_config.hidden_size)
        if not any(p.is_meta for p in self.acoustic_connector.parameters()):
            self.acoustic_connector = self.acoustic_connector.to(dtype)
            self.semantic_connector = self.semantic_connector.to(dtype)
        
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
    
    def set_speech_tokenizers(self, acoustic_tokenizer=None, semantic_tokenizer=None):
        """Set the speech tokenizers used for encoding and decoding speech."""
        self.acoustic_tokenizer = acoustic_tokenizer
        self.semantic_tokenizer = semantic_tokenizer
        
        # Reset the encoder to evaluation mode
        if self.acoustic_tokenizer is not None:
            self.acoustic_tokenizer.eval()
            
        if self.semantic_tokenizer is not None:
            self.semantic_tokenizer.eval()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward through language model
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        
        if not return_dict:
            return outputs
            
        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class VibeVoiceForConditionalGeneration(VibeVoicePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)
        self.model = VibeVoiceModel(config)
        self.vocab_size = config.decoder_config.vocab_size
        self.lm_head = nn.Linear(config.decoder_config.hidden_size, self.vocab_size, bias=False)

        self.post_init()
        
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_decoder(self, decoder):
        self.model.language_model = decoder

    def get_decoder(self):
        return self.model.language_model

    def tie_weights(self, missing_keys=None, recompute_mapping=True, **kwargs):
        """
        Tie the weights between the input embeddings and the output embeddings.

        Accepts missing_keys and recompute_mapping for transformers 5.x
        compatibility (PreTrainedModel.init_weights passes these kwargs).
        """
        if getattr(self.config.decoder_config, 'tie_word_embeddings', False):
            # The standard PreTrainedModel method will handle the tying.
            # It typically does a simple parameter object assignment, which is
            # CORRECT to do BEFORE FSDP wraps the model.
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()
            if hasattr(input_embeddings, 'weight'):
                output_embeddings.weight = input_embeddings.weight
            else:
                # maybe returned input_embeddings a tensor directly
                output_embeddings.weight = input_embeddings

            if getattr(output_embeddings, "bias", None) is not None:
                output_embeddings.bias.data = nn.functional.pad(
                    output_embeddings.bias.data,
                    (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]),
                    "constant",
                    0,
                )
            print("Tied input and output embeddings using standard assignment.")
        else:
            print("tie_word_embeddings is False, not tying weights.")

    # Also, ensure set_output_embeddings is safe, though your implementation looks okay.
    # The key is to avoid calling it after accelerator.prepare().
    def set_output_embeddings(self, new_embeddings):
        # Your current implementation using data.copy_ is good practice,
        # but the best way is to not call this after prepare().
        self.lm_head = new_embeddings

    def forward_speech_features(
            self, 
            speech_tensors=None, 
            speech_masks=None, 
            speech_type="audio", 
            return_unmask=False
        ):
        if speech_tensors is None:
            # Use config to get vae_dim instead of non-existent self.args
            vae_dim = self.config.acoustic_tokenizer_config.vae_dim
            audio_features = torch.zeros(1, 1, vae_dim).to(self.get_input_embeddings().weight)
            connect_features = self.model.acoustic_connector(audio_features)
            return audio_features, connect_features
        else:
            with torch.no_grad():
                if speech_type == "audio":
                    with torch.no_grad():
                        # encode() returns a VibeVoiceTokenizerEncoderOutput
                        # dataclass (with .mean and .std), NOT a tensor. Use
                        # .sample() to draw latents; it returns (x, std), so [0]
                        # extracts the sampled tensor.
                        encoder_output = self.model.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))
                    audio_tokens = encoder_output.sample(self.model.acoustic_tokenizer.std_dist_type)[0]

                elif speech_type == "vae":
                    # Use config to get vae_dim instead of non-existent self.args
                    vae_dim = self.config.acoustic_tokenizer_config.vae_dim
                    speech_mode = speech_tensors.reshape(speech_tensors.size(0), -1, vae_dim)

                    # gaussian sample from the speech_mode
                    batch_size = speech_mode.size(0)
                    value = self.model.acoustic_tokenizer.fix_std / 0.8
                    std = torch.randn(batch_size, dtype=speech_mode.dtype, device=speech_mode.device) * value
                    std = std.view(-1, *[1] * (speech_mode.dim() - 1))
                    audio_tokens = speech_mode + std * torch.randn(speech_mode.shape).to(speech_mode)
                else:
                    raise NotImplementedError(f"Speech type {speech_type} not implemented")
                
                if torch.isnan(self.model.speech_scaling_factor) or torch.isnan(self.model.speech_bias_factor):
                    scaling_factor = 1. / audio_tokens[speech_masks].flatten().std()
                    bias_factor = -audio_tokens[speech_masks].flatten().mean()
                    
                    # Only use distributed operations if the process group is initialized
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(scaling_factor, op=dist.ReduceOp.SUM)
                        dist.all_reduce(bias_factor, op=dist.ReduceOp.SUM)
                        world_size = dist.get_world_size()
                        self.model.speech_scaling_factor.copy_(scaling_factor / world_size)  
                        self.model.speech_bias_factor.copy_(bias_factor / world_size)
                        print(f"Speech scaling factor (distributed): {self.model.speech_scaling_factor}, bias factor: {self.model.speech_bias_factor}", flush=True)
                    else:
                        # Single process case
                        self.model.speech_scaling_factor.copy_(scaling_factor)  
                        self.model.speech_bias_factor.copy_(bias_factor)
                        print(f"Speech scaling factor (single process): {self.model.speech_scaling_factor}, bias factor: {self.model.speech_bias_factor}", flush=True)
                    
                audio_features = (audio_tokens + self.model.speech_bias_factor) * self.model.speech_scaling_factor
            
            connect_features = self.model.acoustic_connector(audio_features)
            if return_unmask:
                return audio_features, connect_features
            return audio_features[speech_masks], connect_features[speech_masks]
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # New arguments for speech processing and loss calculation
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speeches_loss_input: Optional[torch.FloatTensor] = None,
        speech_semantic_tensors: Optional[torch.FloatTensor] = None, 
        acoustic_input_mask: Optional[torch.BoolTensor] = None,
        acoustic_loss_mask: Optional[torch.BoolTensor] = None,
        ddpm_batch_mul: int = 1,
        **kwargs: Optional[Dict[str, Union[torch.Tensor, str]]],
        ) -> Union[Tuple, VibeVoiceCausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        x = self.get_input_embeddings()(input_ids)

        # Initialize speech feature containers so they are always defined, even
        # when no speech tensors are provided (avoids UnboundLocalError at return).
        speech_features = None
        speech_connect_features = None

        # Only compute semantic connect features when semantic speech tensors
        # are provided. During inference/generation the processor does not
        # produce speech_semantic_tensors, so this must be guarded to avoid
        # passing None into the SpeechConnector (which would crash with a
        # TypeError on the underlying nn.Linear). The value is only consumed
        # inside the training branch (speeches_loss_input is not None).
        if speech_semantic_tensors is not None:
            semantic_speech_all_connect_features = self.model.semantic_connector(speech_semantic_tensors)
        else:
            semantic_speech_all_connect_features = None
        if speeches_loss_input is not None:
            # only part audio need diffuse
            speech_all_features, speech_all_connect_features = self.forward_speech_features(
                    speech_tensors=speech_tensors.type_as(x) if speech_tensors is not None else None,
                    speech_masks=speech_masks,
                    speech_type=kwargs.get("speech_type", "audio"),
                    return_unmask=True
                )
            if speech_tensors is not None:
                if semantic_speech_all_connect_features is not None:
                    x[acoustic_input_mask] = (
                        speech_all_connect_features[speech_masks]
                        + semantic_speech_all_connect_features[speech_masks]
                    )
                else:
                    x[acoustic_input_mask] = speech_all_connect_features[speech_masks]

                # Select only the target segments' latents for diffusion loss.
                # Both masks are [num_segments, max_latent_len]; using 2D mask on [B,T,D] selects [N_true, D].
                target_latent_mask = speeches_loss_input & speech_masks
                speech_features = speech_all_features[target_latent_mask]
                speech_connect_features = speech_all_connect_features[target_latent_mask]
        else:
            speech_features, speech_connect_features = self.forward_speech_features(
                    speech_tensors=speech_tensors.type_as(x) if speech_tensors is not None else None,
                    speech_masks=speech_masks,
                    speech_type=kwargs.get("speech_type", "audio"),
                )
            if speech_tensors is not None:
                x[acoustic_input_mask] = speech_connect_features

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=x,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        # logits = logits.float()

        loss = None
        if labels is not None:
            # The custom CE loss with masking is calculated in the training script.
            # We leave the standard loss calculation here as None.
            pass

        # --- Diffusion Loss Calculation ---
        diffusion_loss = None
        # This block is executed only if we are in a training context that
        # involves speech (acoustic_loss_mask is provided). During inference /
        # generation acoustic_loss_mask is None, so we skip the loss computation
        # and fall through to the dummy-loss branch below.
        if acoustic_loss_mask is not None and speech_tensors is not None and acoustic_loss_mask.sum().item() > 0:
            condition_features = hidden_states[acoustic_loss_mask]
            
            speech_len, latent_size = speech_features.shape
            
            noise = torch.randn(
                (speech_len * ddpm_batch_mul, latent_size),
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )
            
            timesteps = torch.multinomial(
                torch.ones(self.config.diffusion_head_config.ddpm_num_steps),
                speech_len * ddpm_batch_mul,
                replacement=True,
            ).to(hidden_states.device)

            speech_features_repeated = speech_features.repeat_interleave(ddpm_batch_mul, dim=0)
            condition_features_repeated = condition_features.repeat_interleave(ddpm_batch_mul, dim=0)

            noisy_speech_features = self.model.noise_scheduler.add_noise(
                speech_features_repeated, noise, timesteps
            )
            
            model_output = self.model.prediction_head(
                noisy_speech_features, 
                timesteps.type_as(x), 
                condition_features_repeated
            )

            prediction_type = self.config.diffusion_head_config.prediction_type
            if prediction_type == "epsilon":
                target_for_loss = noise
            elif prediction_type == "v_prediction":
                target_for_loss = self.model.noise_scheduler.get_velocity(
                    speech_features_repeated, noise, timesteps
                )
            else:
                raise NotImplementedError(f"Prediction type {prediction_type} not implemented")

            diffusion_loss = F.mse_loss(model_output.float(), target_for_loss.float(), reduction='sum')
            if latent_size > 0 and ddpm_batch_mul > 0:
                diffusion_loss = diffusion_loss / latent_size / ddpm_batch_mul
            else:
                diffusion_loss = torch.tensor(0.0, device=diffusion_loss.device)
        
        else:
            # Dummy loss for DDP to work when there are no speech samples in a batch,
            # but we are in a speech context.
            diffusion_loss = sum(p.sum() for p in self.model.prediction_head.parameters()) * 0.0
            diffusion_loss += sum(p.sum() for p in self.model.acoustic_connector.parameters()) * 0.0
            diffusion_loss += sum(p.sum() for p in self.model.semantic_connector.parameters()) * 0.0
        # --- End Diffusion Loss Calculation ---

        # speech_len is only assigned inside the training branch above; derive it
        # safely from speech_features (always defined by this point) so the
        # inference/generation path (no acoustic_loss_mask) does not hit an
        # UnboundLocalError.
        speech_len = speech_features.shape[0] if speech_features is not None else 0

        if not return_dict:
            output = (logits, speech_len) + outputs.to_tuple()[1:]
            return (loss, diffusion_loss) + output

        return VibeVoiceCausalLMOutputWithPast(
            loss=loss,
            diffusion_loss=diffusion_loss,
            speech_token_num=speech_len if speech_tensors is not None else 0,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )

    # ------------------------------------------------------------------
    # Inference / generation support (non-streaming TTS)
    #
    # The non-streaming VibeVoiceForConditionalGeneration is the model class
    # used for the official multi-speaker TTS checkpoints (config model_type
    # "vibevoice"). It already contains the full diffusion infrastructure
    # (noise_scheduler, prediction_head, acoustic_tokenizer, acoustic_connector,
    # speech_scaling_factor, speech_bias_factor) but previously exposed no
    # generate() entry point. The methods below add a self-contained,
    # non-streaming generation path that mirrors the diffusion sampling loop
    # used by the streaming model's sample_speech_tokens().
    # ------------------------------------------------------------------

    def set_ddpm_inference_steps(self, num_steps=None):
        """Set the number of DDPM inference (diffusion) steps for generation.

        Mirrors the streaming model API so that modules/generation.py can call
        this uniformly. Falls back to the config default when num_steps is None.

        Args:
            num_steps: Number of diffusion steps. If None, uses
                config.diffusion_head_config.ddpm_num_inference_steps.
        """
        default = self.config.diffusion_head_config.ddpm_num_inference_steps
        self.ddpm_inference_steps = num_steps or default

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        acoustic_input_mask: Optional[torch.BoolTensor] = None,
        semantic_speech_tensors: Optional[torch.FloatTensor] = None,
        cfg_scale: float = 1.3,
        inference_steps: Optional[int] = None,
        return_speech: bool = True,
        **kwargs,
    ) -> "VibeVoiceGenerationOutput":
        """Non-streaming text-to-speech generation.

        Runs the language model forward pass to obtain condition features for
        the speech diffusion head, samples speech latents via the DDPM
        scheduler, and decodes the latents to a waveform with the acoustic
        tokenizer.

        Args:
            input_ids: Token ids of shape (batch, seq).
            attention_mask: Attention mask of shape (batch, seq).
            speech_tensors: Reference speech latents/audio for voice cloning.
            speech_masks: Masks for speech_tensors.
            acoustic_input_mask: Boolean mask marking acoustic (speech) positions.
            semantic_speech_tensors: Semantic speech features (optional).
            cfg_scale: Classifier-free guidance scale for diffusion sampling.
            inference_steps: Number of diffusion steps (defaults to
                self.ddpm_inference_steps or config default).
            return_speech: If True, decode latents to waveform and return them.

        Returns:
            VibeVoiceGenerationOutput with `sequences` (input ids) and
            `speech_outputs` (list of waveform tensors, or None).
        """
        device = next(self.parameters()).device
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # 1. Language-model forward pass to obtain condition features.
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            speech_tensors=speech_tensors,
            speech_masks=speech_masks,
            acoustic_input_mask=acoustic_input_mask,
            semantic_speech_tensors=semantic_speech_tensors,
            return_dict=True,
            **kwargs,
        )
        # The diffusion condition is the LM *hidden states* (dim = model hidden_size),
        # NOT the vocab logits. `forward` returns `hidden_states` (= last_hidden_state).
        hidden_states = outputs.hidden_states if outputs.hidden_states is not None else None
        if hidden_states is None:
            fwd = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                speech_tensors=speech_tensors,
                speech_masks=speech_masks,
                acoustic_input_mask=acoustic_input_mask,
                semantic_speech_tensors=semantic_speech_tensors,
                return_dict=True,
                **kwargs,
            )
            hidden_states = fwd.hidden_states

        # Condition features: hidden states at acoustic positions.
        if acoustic_input_mask is not None and acoustic_input_mask.any():
            condition = hidden_states[acoustic_input_mask]
        else:
            condition = hidden_states

        if condition.ndim == 3:
            condition = condition.reshape(-1, condition.shape[-1])

        # 2. Configure diffusion scheduler.
        num_steps = inference_steps or getattr(
            self, "ddpm_inference_steps",
            self.config.diffusion_head_config.ddpm_num_inference_steps
        )
        self.model.noise_scheduler.set_timesteps(num_steps)

        # 3. Sample speech latents via the diffusion head (CFG).
        vae_dim = self.config.acoustic_tokenizer_config.vae_dim
        speech = torch.randn(condition.shape[0], vae_dim, device=device, dtype=condition.dtype)
        neg_condition = torch.zeros_like(condition)

        for t in self.model.noise_scheduler.timesteps:
            # Classifier-free guidance: run conditioned + unconditioned together.
            cond_in = torch.cat([condition, neg_condition], dim=0).to(self.model.prediction_head.device)
            combined = torch.cat([speech, speech], dim=0)
            eps = self.model.prediction_head(
                combined,
                t.repeat(combined.shape[0]).to(combined.device),
                condition=cond_in,
            )
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            # Collapse the two branches into a single CFG prediction for the N samples.
            eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample

        sampled = speech

        # 4. Decode latents to waveform.
        speech_outputs = None
        if return_speech:
            # The diffusion head is trained on (and therefore outputs) the
            # *scaled* acoustic features `audio_features = (tokens + bias) *
            # scaling`. The VAE decoder expects the *raw* latent tokens, so we
            # must apply the INVERSE transform before decoding. (This mirrors
            # the streaming inference path: `latent / scaling - bias`.) The
            # original `(sampled + bias) * scaling` doubly-scaled the output and
            # produced a silent/garbage waveform.
            if not torch.isnan(self.model.speech_scaling_factor) and not torch.isnan(self.model.speech_bias_factor):
                sf = self.model.speech_scaling_factor.to(sampled.device)
                bf = self.model.speech_bias_factor.to(sampled.device)
                sampled = sampled / sf - bf
            # Decode expects (batch, vae_dim, frames) or (batch, frames, vae_dim).
            latents = sampled.unsqueeze(0)  # (1, N, vae_dim)
            audio = self.model.acoustic_tokenizer.decode(latents)
            if isinstance(audio, (list, tuple)):
                audio = audio[0]
            speech_outputs = [audio]

        return VibeVoiceGenerationOutput(
            sequences=input_ids,
            speech_outputs=speech_outputs,
        )

AutoModel.register(VibeVoiceConfig, VibeVoiceModel, exist_ok=True)
AutoModelForCausalLM.register(VibeVoiceConfig, VibeVoiceForConditionalGeneration, exist_ok=True)

__all__ = [
    "VibeVoiceModel",
    "VibeVoicePreTrainedModel",
    "VibeVoiceForConditionalGeneration",
    "VibeVoiceCausalLMOutputWithPast",
    "VibeVoiceGenerationOutput",
]