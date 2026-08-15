from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
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
                    else:
                        # Single process case
                        self.model.speech_scaling_factor.copy_(scaling_factor)
                        self.model.speech_bias_factor.copy_(bias_factor)
                    
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

    # ------------------------------------------------------------------
    # Option C helpers — autoregressive target-speech generation (BUG-006 fix)
    #
    # The non-streaming model is a SINGLE-LM design: `language_model` + `lm_head`
    # generate both text and target speech tokens (the Qwen backbone is split into
    # text layers + upper speech layers; see configuration_vibevoice_streaming.py:79).
    # The previous generate() mirrored ONLY the diffusion half and conditioned on the
    # *reference-prompt* hidden states, emitting reference-length latents -> gibberish.
    # The helpers + generate() below autoregressively generate the *target* speech
    # tokens via `lm_head`, condition the diffusion head on the *generated* positions,
    # and feed each produced latent back through `acoustic_connector` as the next-step
    # embedding (the single-LM analogue of the streaming tts_lm acoustic feedback).
    # ------------------------------------------------------------------

    def _build_prefix_embeds(self, input_ids, acoustic_input_mask, speech_tensors, speech_masks):
        """Embed `input_ids` and inject reference speech features at `acoustic_input_mask`.

        Mirrors the inference branch of the training `forward` (modeling_vibevoice.py:
        414-421): reference VAE tokens are replaced by `acoustic_connector` features so
        the LM "hears" the reference voice. As a side effect this also computes
        `speech_scaling_factor` / `speech_bias_factor` (used later to invert the
        diffusion output before VAE decode).
        """
        x = self.get_input_embeddings()(input_ids)
        if (
            speech_tensors is not None
            and acoustic_input_mask is not None
            and acoustic_input_mask.any()
        ):
            with torch.no_grad():
                _, connect_features = self.forward_speech_features(
                    speech_tensors=speech_tensors.type_as(x) if speech_tensors is not None else None,
                    speech_masks=speech_masks,
                    speech_type="audio",
                )
            x[acoustic_input_mask] = connect_features
        return x

    def _sample_one_latent(self, condition, neg_condition, cfg_scale, num_steps):
        """Sample ONE acoustic latent conditioned on `condition` (CFG).

        Reuses the 2N/2N classifier-free-guidance layout fixed in BUG-004, then applies
        the inverse scaling (`latent / scaling - bias`) so the VAE decoder receives raw
        latents (BUG-005 fix). Returns a `(B, vae_dim)` tensor.
        """
        # BUG-006 scheduler-index fix: reset the multistep scheduler's internal step
        # counter at the START of every latent. `generate()` calls set_timesteps() once
        # before the AR loop, but this method runs once PER AR step. The vendored
        # DPMSolverMultistepScheduler only resets `_step_index` inside set_timesteps();
        # without a per-latent reset, from the 2nd latent the final-step
        # `lower_order_final` guard never triggers and the 2nd-order update reads
        # `sigmas[step_index + 1]` -> IndexError. The streaming reference
        # (sample_speech_tokens, L891) does exactly this.
        self.model.noise_scheduler.set_timesteps(num_steps)
        head_dev = self.model.prediction_head.device
        condition = condition.to(head_dev)
        neg_condition = neg_condition.to(head_dev)
        cond_in = torch.cat([condition, neg_condition], dim=0)
        vae_dim = self.config.acoustic_tokenizer_config.vae_dim
        speech = torch.randn(condition.shape[0], vae_dim, device=head_dev, dtype=condition.dtype)
        for t in self.model.noise_scheduler.timesteps:
            combined = torch.cat([speech, speech], dim=0)
            eps = self.model.prediction_head(
                combined,
                t.repeat(combined.shape[0]).to(combined.device),
                condition=cond_in,
            )
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample
        # Invert the diffusion output scaling before VAE decode (BUG-005).
        sf = self.model.speech_scaling_factor
        bf = self.model.speech_bias_factor
        if not torch.isnan(sf) and not torch.isnan(bf):
            sf = sf.to(speech.device)
            bf = bf.to(speech.device)
            speech = speech / sf - bf
        return speech  # (B, vae_dim)

    def _decode_latent(self, latent):
        """Decode a single `(B, vae_dim)` latent to a waveform chunk (inverse scaling applied)."""
        latents = latent.reshape(latent.shape[0], 1, -1)  # (B, 1, vae_dim)
        audio = self.model.acoustic_tokenizer.decode(latents)
        if isinstance(audio, (list, tuple)):
            audio = audio[0]
        if audio is None:
            return None
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        return audio  # (B, T)

    @staticmethod
    def _sample_next_token(logits, do_sample=True, temperature=1.0, top_p=1.0, top_k=0):
        """Sample the next speech token id from `logits` (B, vocab) -> (B,)."""
        logits = logits.float()
        if not do_sample:
            return logits.argmax(dim=-1)
        if temperature > 0:
            logits = logits / max(temperature, 1e-6)
        if top_k and top_k > 0:
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, -float("inf"))
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumulative = torch.cumsum(torch.softmax(sorted_logits, -1), -1)
            remove = cumulative > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            logits = logits.masked_fill(remove.gather(-1, sorted_idx), -float("inf"))
        probs = torch.softmax(logits, -1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def sample_speech_tokens(self, condition, neg_condition, cfg_scale=3.0):
        """Sample acoustic latents conditioned on positive/negative LM hidden states.

        Faithful port of the original ``VibeVoiceForConditionalGenerationInference.
        sample_speech_tokens``: builds a 2N batch (positive || negative), runs the
        DPM-Solver diffusion head with real classifier-free guidance (the negative
        branch is the unconditional/negative forward, NOT zeros), and returns the
        scaled positive latents ``(N, acoustic_vae_dim)``.
        """
        self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)
        condition = torch.cat([condition, neg_condition], dim=0).to(self.model.prediction_head.device)
        speech = torch.randn(condition.shape[0], self.config.acoustic_vae_dim).to(condition)
        for t in self.model.noise_scheduler.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = self.model.prediction_head(
                combined,
                t.repeat(combined.shape[0]).to(combined),
                condition=condition,
            )
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample
        return speech[: len(speech) // 2]

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
        max_new_tokens: Optional[int] = None,
        do_sample: bool = True,
        temperature: float = 0.95,
        top_p: float = 0.95,
        top_k: int = 0,
        tokenizer: Optional[Any] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs,
    ) -> "VibeVoiceGenerationOutput":
        """Non-streaming text-to-speech generation (faithful original-protocol port).

        This is a faithful port of Microsoft's ``VibeVoiceForConditionalGeneration
        Inference.generate`` for the single-LM 1.5B checkpoint. The previous custom
        loop was replaced because it (a) decoded a latent on *every* AR step instead
        of only on ``speech_diffusion_id`` steps, (b) faked CFG with
        ``neg_condition = zeros`` instead of a real parallel negative forward, (c)
        fed ``acoustic_connector(latent)`` as the next embedding unconditionally
        (no semantic feedback, no token constraint), and (d) only terminated on EOS
        / a hard cap -- producing minutes of broken syllables (BUG-010).

        The protocol implemented here:

        * **Token-constrained control stream.** ``lm_head`` may only emit
          ``{speech_start_id, speech_end_id, speech_diffusion_id, eos, bos}``. This
          bounds generation to a short control sequence and is what prevents the
          runaway loop.
        * **Diffusion gated to diffusion steps.** A latent is sampled *only* when the
          emitted token is ``speech_diffusion_id``, conditioned on
          ``last_hidden_state[diffusion_indices, -1, :]``.
        * **Real CFG.** A parallel *negative* forward (conditioned on the
          ``speech_start_id`` / unconditional branch) produces ``neg_condition``;
          ``sample_speech_tokens`` combines them as
          ``uncond + cfg_scale * (cond - uncond)``.
        * **Semantic feedback.** Each decoded latent is re-encoded to semantic
          features; the next-step embedding is
          ``acoustic_connector(latent) + semantic_connector(semantic_features)``.
        * **Streaming caches.** ``acoustic_cache`` / ``semantic_cache`` are cleared
          (``set_to_zero``) on ``speech_end_id``.

        Args:
            input_ids / attention_mask / speech_tensors / speech_masks /
                acoustic_input_mask: from the processor. ``acoustic_input_mask`` marks
                reference (voice-clone) positions used to inject reference features.
            cfg_scale: Classifier-free guidance scale for diffusion sampling.
            inference_steps: DDPM steps (defaults to the model setting; also applied
                via ``set_ddpm_inference_steps``).
            return_speech: If True, decode latents to a waveform and return it.
            max_new_tokens: Hard cap on generated *control* tokens. ``None`` derives
                from ``max_position_embeddings - prompt_len`` (original default), then
                the loop is further bounded by ``max_length_times * prompt_len``.
            do_sample / temperature / top_p / top_k: ``temperature``/``top_p``/``top_k``
                are accepted for interface compatibility but unused -- the original
                protocol samples with a constrained softmax (multinomial) or argmax.
            tokenizer: processor tokenizer; supplies the speech control-token ids.
            progress_callback: Optional framework-agnostic hook invoked as
                ``progress_callback(current, total)`` — once with ``(0, max_steps)``
                before the AR loop and once after each completed AR step with
                ``(step, max_steps)``. ``current`` is monotonic non-decreasing and
                bounded by ``total``. ``None`` disables reporting. The callback may
                raise to interrupt generation (the exception propagates out of
                ``generate()``); callers use this for progress UI and cancellation.

        Returns:
            VibeVoiceGenerationOutput with ``sequences`` (input ids) and
            ``speech_outputs`` (list with one waveform tensor per sample, or None).
        """
        if input_ids is not None:
            device = input_ids.device
        else:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if speech_tensors is not None:
            speech_tensors = speech_tensors.to(device)
        if speech_masks is not None:
            speech_masks = speech_masks.to(device)
        if acoustic_input_mask is not None:
            acoustic_input_mask = acoustic_input_mask.to(device)

        # ------------------------------------------------------------------
        # Resolve speech control-token ids from the tokenizer (BUG-010: the 1.5B
        # single-LM class has no dedicated tts_eos_classifier, so it stops on a
        # discrete EOS token emitted by lm_head).
        # ------------------------------------------------------------------
        speech_start_id = speech_end_id = speech_diffusion_id = eos_id = None
        bos_id = None
        tok = tokenizer if tokenizer is not None else getattr(self, "tokenizer", None)
        if tok is not None:
            speech_start_id = getattr(tok, "speech_start_id", None)
            speech_end_id = getattr(tok, "speech_end_id", None)
            speech_diffusion_id = getattr(tok, "speech_diffusion_id", None)
            eos_id = getattr(tok, "eos_id", None)
            if eos_id is None:
                eos_id = getattr(tok, "eos_token_id", None)
            bos_id = getattr(tok, "bos_token_id", None)
        if eos_id is None:
            eos_id = getattr(getattr(self, "config", None), "eos_token_id", None)
        if speech_start_id is None or speech_end_id is None or speech_diffusion_id is None:
            raise ValueError(
                "generate() requires a tokenizer that exposes "
                "speech_start_id / speech_end_id / speech_diffusion_id "
                "(e.g. VibeVoiceTextTokenizer)."
            )

        # ------------------------------------------------------------------
        # Prefix embeddings (reference injected at acoustic_input_mask positions).
        # ------------------------------------------------------------------
        inputs_embeds = self._build_prefix_embeds(
            input_ids, acoustic_input_mask, speech_tensors, speech_masks
        )  # (B, S, H)
        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

        # ------------------------------------------------------------------
        # Length budget (control tokens). Mirrors the original:
        #   max_new_tokens = max_position_embeddings - prompt_len   (if auto)
        #   max_steps      = min(max_new_tokens, max_length_times * prompt_len)
        # The token constraint + EOS termination keep this from running away.
        # ------------------------------------------------------------------
        if max_new_tokens is None:
            max_position = getattr(self.config.decoder_config, "max_position_embeddings", None)
            if max_position:
                max_new_tokens = int(max_position) - seq_len
            else:
                max_new_tokens = min(int(seq_len * 12), 2000)
        max_new_tokens = max(1, min(int(max_new_tokens), 8192))
        max_length = seq_len + max_new_tokens
        max_length_times = int(kwargs.get("max_length_times", 2))
        max_steps = max(1, min(max_new_tokens, int(max_length_times * seq_len)))

        # Progress reporting: announce the loop budget before the first step.
        if progress_callback is not None:
            progress_callback(0, max_steps)

        # Diffusion steps.
        num_steps = inference_steps or getattr(
            self, "ddpm_inference_steps", self.config.diffusion_head_config.ddpm_num_inference_steps
        )
        self.set_ddpm_inference_steps(num_steps)
        self.model.noise_scheduler.set_timesteps(num_steps)

        # Streaming caches (cleared on speech_end_id).
        acoustic_cache = VibeVoiceTokenizerStreamingCache()
        semantic_cache = VibeVoiceTokenizerStreamingCache()

        # Token-constraint mask: only the 4 (+bos) control tokens are allowed.
        valid_tokens = [int(speech_start_id), int(speech_end_id), int(speech_diffusion_id), int(eos_id)]
        if bos_id is not None:
            valid_tokens.append(int(bos_id))
        valid_tokens_t = torch.tensor(valid_tokens, dtype=torch.long, device=device)
        constraint = torch.full((1, self.vocab_size), float("-inf"), device=device)
        constraint[0, valid_tokens_t] = 0.0

        embed = self.get_input_embeddings()
        head_dev = self.model.prediction_head.device
        lm_dev = getattr(self.lm_head.weight, "device", None) or device
        ac_dev = self.model.acoustic_tokenizer.device
        sf = self.model.speech_scaling_factor
        bf = self.model.speech_bias_factor

        # ------------------------------------------------------------------
        # Prefill (positive) forward -- build the KV cache over the prompt.
        # ------------------------------------------------------------------
        cache_position = torch.arange(0, seq_len, device=device)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        hidden = outputs.last_hidden_state  # (B, S, H)

        # ------------------------------------------------------------------
        # Negative (unconditional) pass bookkeeping. The negative sequence is
        # seeded with a single `speech_start_id` token and grows by one token per
        # AR step, mirroring the original `refresh_negative=False` semantics: the
        # negative forward is fed the SAME per-step embedding as the positive but
        # through its own (unconditionally-seeded) KV cache -> real CFG signal.
        # ------------------------------------------------------------------
        neg_input_ids = torch.full((batch_size, 1), int(speech_start_id), dtype=torch.long, device=device)
        neg_attention_mask = torch.ones((batch_size, 1), dtype=torch.long, device=device)
        neg_position_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        neg_past = None
        neg_cur_pos = 0
        # `last_embeds` is the embedding fed to the positive forward of the CURRENT
        # step; the negative forward reuses it (original's inputs_embeds override).
        # At step 0 the original falls back to the negative's own [speech_start_id]
        # embedding, so we start last_embeds as the prefix and use the neg_past is
        # None branch for the first negative forward.
        last_embeds = inputs_embeds

        audio_chunks = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        cur_pos = seq_len

        step = 0
        while cur_pos < max_length and step < max_steps and not finished.all():
            # 1) Sample the next control token (token-constrained).
            logits = self.lm_head(hidden[:, -1, :].to(lm_dev))  # (B, vocab)
            scores = logits.float() + constraint  # -inf everywhere except valid
            if do_sample:
                probs = torch.softmax(scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
            else:
                next_tokens = scores.argmax(dim=-1)  # (B,)
            next_tokens = next_tokens.clone()
            next_tokens[finished] = int(eos_id)

            # 2) Negative (unconditional) forward for this step.
            if neg_past is None:
                neg_embeds = embed(neg_input_ids)  # (B,1,H) == [speech_start_id]
            else:
                neg_embeds = last_embeds
            # BUG-011 fix: `neg_embeds` is a SINGLE token (B,1,H), so the RoPE
            # `position_ids` must be the CURRENT position only. Passing the
            # full-length `neg_position_ids` (B, step+1) made q/k broadcast
            # against full-length cos/sin and silently expand to seq-len
            # step+1 while v (never rotated) stayed length 1 -> the KV cache
            # accumulated step+1 keys but only 1 value per step, and SDPA's
            # `attn @ value` crashed with "Expected size for first two
            # dimensions of batch2 tensor to be: [B*H, keys] but got:
            # [B*H, values]" (e.g. [12, 3] but got: [12, 2] at step 1).
            # The attention mask stays full-length (it addresses the whole
            # cache); only position_ids must match the single input token.
            neg_out = self.model.language_model(
                inputs_embeds=neg_embeds.to(lm_dev),
                attention_mask=neg_attention_mask,
                position_ids=neg_position_ids[:, -1:],
                cache_position=torch.tensor([neg_cur_pos], device=device, dtype=torch.long),
                past_key_values=neg_past,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            neg_past = neg_out.past_key_values
            neg_hidden = neg_out.last_hidden_state[:, -1, :]  # (B, H)
            neg_attention_mask = torch.cat(
                [neg_attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=device)], dim=-1
            )
            neg_position_ids = torch.cat(
                [neg_position_ids, neg_position_ids[:, -1:] + 1], dim=-1
            )
            neg_cur_pos += 1
            neg_input_ids = torch.cat([neg_input_ids, next_tokens[:, None]], dim=-1)

            # 3) Default next embedding = plain token embedding.
            next_inputs_embeds = embed(next_tokens).unsqueeze(1)  # (B,1,H)

            # 4) EOS marks a sample finished (no more audio for it).
            eos_now = (next_tokens == int(eos_id))
            finished = finished | eos_now

            # 5) On speech_end: clear the streaming tokenizer caches.
            end_idx = (next_tokens == int(speech_end_id)).nonzero(as_tuple=False).squeeze(1)
            if end_idx.numel() > 0:
                acoustic_cache.set_to_zero(end_idx)
                semantic_cache.set_to_zero(end_idx)

            # 6) Diffusion is gated to speech_diffusion_id steps only.
            diffusion_idx = (~finished & (next_tokens == int(speech_diffusion_id))).nonzero(as_tuple=False).squeeze(1)
            if diffusion_idx.numel() > 0:
                pos_cond = hidden[:, -1, :][diffusion_idx].to(head_dev)  # (Nd, H)
                neg_cond = neg_hidden[diffusion_idx].to(head_dev)        # (Nd, H)
                latent = self.sample_speech_tokens(pos_cond, neg_cond, cfg_scale=cfg_scale)  # (Nd, vae_dim), scaled
                # Invert the diffusion scaling before VAE decode (BUG-005).
                if not torch.isnan(sf) and not torch.isnan(bf):
                    sf_d = sf.to(latent.device)
                    bf_d = bf.to(latent.device)
                    scaled_latent = latent / sf_d - bf_d
                else:
                    scaled_latent = latent
                scaled_latent = scaled_latent.unsqueeze(1)  # (Nd, 1, vae_dim)
                audio_chunk = self.model.acoustic_tokenizer.decode(
                    scaled_latent.to(ac_dev),
                    cache=acoustic_cache,
                    sample_indices=diffusion_idx.to(ac_dev),
                    use_cache=True,
                    debug=False,
                )
                # Normalize audio_chunk to (Nd, T) for storage + semantic encode.
                if audio_chunk is not None:
                    if isinstance(audio_chunk, (list, tuple)):
                        audio_chunk = audio_chunk[0]
                    if isinstance(audio_chunk, torch.Tensor):
                        if audio_chunk.dim() == 3:
                            audio_chunk = audio_chunk.squeeze(1)
                        if audio_chunk.dim() == 1:
                            audio_chunk = audio_chunk.unsqueeze(0)
                if audio_chunk is not None and isinstance(audio_chunk, torch.Tensor):
                    if return_speech:
                        for i, idx in enumerate(diffusion_idx.tolist()):
                            if not finished[idx]:
                                audio_chunks[idx].append(audio_chunk[i])
                    # Semantic feedback: encode the chunk, combine with acoustic connector.
                    # acoustic_tokenizer.decode yields waveform (Nd, T); the semantic tokenizer's
                    # StreamingConv1d expects (Nd, C, T) with C=1 (mono), so add the channel dim
                    # for the encode ONLY. `audio_chunk` itself is left intact because it is still
                    # appended to audio_chunks[i] for waveform assembly below (kept 1-D/2-D on purpose).
                    sem_input = audio_chunk
                    if sem_input.dim() == 2:
                        sem_input = sem_input.unsqueeze(1)                # (Nd, T) -> (Nd, 1, T)
                    elif sem_input.dim() == 1:
                        sem_input = sem_input.unsqueeze(0).unsqueeze(0)   # (T,)   -> (1, 1, T)
                    semantic_features = self.model.semantic_tokenizer.encode(
                        sem_input,
                        cache=semantic_cache,
                        sample_indices=diffusion_idx,
                        use_cache=True,
                        debug=False,
                    ).mean  # (Nd, semantic_vae_dim, T_sem); consumed by semantic_connector
                    acoustic_embed = self.model.acoustic_connector(latent)             # (Nd, H)
                    semantic_embed = self.model.semantic_connector(semantic_features)  # (Nd, H)
                    next_inputs_embeds[diffusion_idx] = acoustic_embed + semantic_embed

            # 7) Forward the positive model for the next step.
            last_embeds = next_inputs_embeds
            if not finished.all():
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=device)], dim=-1
                )
                cache_position = torch.tensor([cur_pos], device=device, dtype=torch.long)
                pos_ids = cache_position.unsqueeze(0)
                outputs = self.model.language_model(
                    inputs_embeds=next_inputs_embeds.to(lm_dev),
                    attention_mask=attention_mask,
                    position_ids=pos_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                hidden = outputs.last_hidden_state  # (B, 1, H)
                cur_pos += 1
            else:
                break
            step += 1

            # Progress reporting: one callback per completed AR step.
            if progress_callback is not None:
                progress_callback(step, max_steps)

        # ------------------------------------------------------------------
        # Assemble the per-sample waveform(s).
        # ------------------------------------------------------------------
        final_audio = []
        for chunks in audio_chunks:
            if chunks:
                final_audio.append(torch.cat(chunks, dim=-1))
            else:
                final_audio.append(None)
        speech_outputs = final_audio if return_speech else None

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