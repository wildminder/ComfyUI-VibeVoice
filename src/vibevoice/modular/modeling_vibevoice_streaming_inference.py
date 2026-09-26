from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from tqdm import tqdm
import inspect
import torch
import torch.nn as nn

from transformers.models.auto import AutoModel, AutoModelForCausalLM
from transformers.generation import GenerationMixin, GenerationConfig, LogitsProcessor, LogitsProcessorList, StoppingCriteriaList
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers import modeling_utils
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import logging

from .modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache
from .modular_vibevoice_diffusion_head import VibeVoiceDiffusionHead
from ..schedule.dpm_solver import DPMSolverMultistepScheduler
from .configuration_vibevoice_streaming import VibeVoiceStreamingConfig
from .modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizer, VibeVoiceTextTokenizerFast
from .modeling_vibevoice_streaming import VibeVoiceStreamingPreTrainedModel, VibeVoiceStreamingModel, BinaryClassifier
from .streamer import AudioStreamer, AsyncAudioStreamer

logger = logging.get_logger(__name__)

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

TTS_TEXT_WINDOW_SIZE = 5
TTS_SPEECH_WINDOW_SIZE = 6


def _generation_config_accepts_positional_flag() -> bool:
    """Return whether ``_prepare_generation_config`` takes a positional flag.

    transformers 4.x signature: ``(self, generation_config, is_init, **kwargs)``
    where ``is_init`` is positional-or-keyword. transformers 5.x signature:
    ``(self, generation_config, **kwargs)`` where every extra parameter is
    var-keyword. Detect the flag by looking for a positional parameter that is
    not var-keyword and is not ``self``/``generation_config``.
    """
    from transformers.generation.utils import GenerationMixin

    try:
        parameters = list(
            inspect.signature(
                GenerationMixin._prepare_generation_config
            ).parameters.values()
        )
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False

    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        and parameter.name not in ("self", "generation_config")
    ]
    return bool(positional)


# ============================================================================
# Transformers >= 4.57 / 5.x Compatibility Layer
# The cache system was refactored in transformers 4.57, requiring these helpers.
#
# WHY THIS SHIM EXISTS -- do not "simplify" it away
# -------------------------------------------------
# The official ``.pt`` voice prompts that ship with VibeVoice-Realtime are pickled
# with a **legacy ``DynamicCache``**: their KV lives in two plain Python lists,
# ``cache.key_cache[i]`` / ``cache.value_cache[i]``. From transformers 4.57
# onwards a cache is instead a *container* (``cache.layers``) holding one *layer
# object* per attention head, and 5.x reads ``layer.keys`` / ``layer.values``.
# A legacy pickle therefore arrives with no ``layers`` list at all.
#
# That gap fails **silently, not loudly**, which is the whole reason the shim
# carries both spellings:
#
#   * Container level. ``DynamicCache.get_mask_sizes()`` answers
#     ``(cache_position.shape[0], 0)`` -- kv_length == query_length -- whenever
#     the requested ``layer_idx`` is beyond ``len(self.layers)``. The attention
#     mask is then built as if the 316-token voice prefill did not exist. No
#     exception, no warning, no missing-key report: the model conditions on
#     nothing and emits a fraction of a second of noise. ``_ensure_cache_has_layers``
#     must therefore run *before* the first forward pass.
#   * Layer level. A wrapper implementing *only* the 4.x names
#     (``key_cache`` / ``value_cache``) would still answer
#     ``get_seq_length() == 0`` to 5.x, so the prefill would be invisible in
#     exactly the same silent way. Every modern accessor on ``MockCacheLayer``
#     is therefore backed by the same tensors as the legacy ones.
#
# Measured on the installed 5.3: with the shim, ``kv_length == 321`` for a
# 316-token prefill plus a 5-token query, and the mask delivered to attention is
# ``(1, 1, 5, 321)`` with no prefill position masked out. See
# docs/plans/evidence/2026-09-26-s1.2-decision.md and
# docs/plans/2026-09-26-two-version-matrix.md for the full record.
# ============================================================================

def _tensor_device_and_dtype(*tensors):
    """Device and dtype of the first real tensor among ``tensors``.

    The 5.x layer API carries ``device`` / ``dtype`` attributes (set by
    ``DynamicLayer.lazy_initialization``) because ``offload`` / ``prefetch``
    need to know where the layer is supposed to live. A legacy cache is already
    populated when we wrap it, so those attributes are derived here instead.
    """
    for tensor in tensors:
        if torch.is_tensor(tensor):
            return tensor.device, tensor.dtype
    return torch.device("cpu"), torch.get_default_dtype()


class MockCacheLayer:
    """
    Mock cache layer bridging the pre-4.57 and 5.x ``CacheLayer`` APIs.

    The released ``.pt`` voice prompts are pickled with a ``DynamicCache`` that
    stores its tensors in ``key_cache`` / ``value_cache`` lists. Since
    transformers 5.x a cache is a list of layer objects instead, so a cache
    loaded from those files has to be wrapped to expose the modern interface.

    Both spellings are supported because the two transformers generations read
    different attributes: 4.x reads ``layer.key_cache`` / ``layer.value_cache``,
    while 5.x reads ``layer.keys`` / ``layer.values`` and, crucially, calls
    ``get_seq_length()`` to size the attention mask. A wrapper providing only
    the 4.x names does not fail loudly on 5.x -- the voice prefill is simply
    invisible, the model conditions on nothing, and the output is garbled
    syllables, or an immediate end-of-speech. Every modern accessor below is
    therefore backed by the same tensors as the legacy ones.
    """

    def __init__(self, key_cache, value_cache, parent_cache=None, layer_idx=0):
        self._key_cache = key_cache
        self._value_cache = value_cache
        self._parent_cache = parent_cache
        self._layer_idx = layer_idx
        # ``CacheLayerMixin.prefetch`` (and ``Cache.prefetch``) read
        # ``self.device``; on 5.x it is normally set by
        # ``DynamicLayer.lazy_initialization``, which this cache has already
        # been through by the time we wrap it. Derive it from the tensor we
        # were handed, and refresh it in ``update`` so a cache that migrates
        # devices mid-generation still prefetches onto the right one.
        self.device, self.dtype = _tensor_device_and_dtype(key_cache, value_cache)

    # --- legacy (pre-4.57) attribute names ---------------------------------
    @property
    def key_cache(self):
        return self._key_cache

    @key_cache.setter
    def key_cache(self, value):
        self._key_cache = value

    @property
    def value_cache(self):
        return self._value_cache

    @value_cache.setter
    def value_cache(self, value):
        self._value_cache = value

    # --- transformers 5.x attribute names -----------------------------------
    @property
    def keys(self):
        return self._key_cache

    @keys.setter
    def keys(self, value):
        self._key_cache = value

    @property
    def values(self):
        return self._value_cache

    @values.setter
    def values(self, value):
        self._value_cache = value

    def _store(self, keys, values):
        """Publish new tensors on both APIs, the parent list included.

        Every mutating 5.x accessor (``offload``, ``prefetch``,
        ``reorder_cache``, ``crop``, the batch helpers) assigns through
        ``self.keys`` / ``self.values``. Those setters only rebind the layer's
        own fields, so a parent-backed layer would silently go out of sync
        with ``key_cache`` / ``value_cache`` - and ``update`` reads the parent
        lists, so the next forward would resurrect the stale tensors. Writing
        both keeps the two spellings pointing at one object at all times.

        ``device`` / ``dtype`` are deliberately *not* refreshed here: they
        describe where the layer belongs, which ``offload`` must not redefine
        by moving the data to CPU. ``update`` refreshes them from the freshly
        concatenated states instead.
        """
        self._key_cache = keys
        self._value_cache = values
        parent = self._parent_cache
        if parent is not None and 0 <= self._layer_idx < len(parent.key_cache):
            parent.key_cache[self._layer_idx] = keys
            parent.value_cache[self._layer_idx] = values
        return keys, values

    @property
    def is_initialized(self) -> bool:
        return self._key_cache is not None and self._key_cache.numel() > 0

    @property
    def is_sliding(self) -> bool:
        return False

    @property
    def is_compileable(self) -> bool:
        return False

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not self.is_initialized:
            return 0
        return self._key_cache.shape[-2]

    def get_max_cache_shape(self) -> int:
        return -1

    def get_mask_sizes(self, cache_position):
        """Return KV length and offset for mask creation."""
        seq_length = self.get_seq_length()
        query_length = cache_position.shape[0]
        return seq_length + query_length, 0

    def lazy_initialization(self, key_states, value_states) -> None:
        """5.x initialization hook; this cache is already populated."""
        return None

    def update(self, key_states, value_states, cache_kwargs=None):
        """Update the cache with new key/value states."""
        if self._parent_cache is None:
            return self._key_cache, self._value_cache

        parent = self._parent_cache
        idx = self._layer_idx

        # Extend cache lists if needed
        while len(parent.key_cache) <= idx:
            parent.key_cache.append(None)
            parent.value_cache.append(None)

        # Concatenate or initialize cache
        if parent.key_cache[idx] is not None:
            parent.key_cache[idx] = torch.cat([parent.key_cache[idx], key_states], dim=-2)
            parent.value_cache[idx] = torch.cat([parent.value_cache[idx], value_states], dim=-2)
        else:
            parent.key_cache[idx] = key_states
            parent.value_cache[idx] = value_states

        # Update local references
        self._key_cache = parent.key_cache[idx]
        self._value_cache = parent.value_cache[idx]
        self.device, self.dtype = _tensor_device_and_dtype(
            self._key_cache, self._value_cache
        )
        return self._key_cache, self._value_cache

    def offload(self) -> None:
        """Move this layer's tensors to CPU (mirrors ``CacheLayerMixin.offload``)."""
        if self.is_initialized:
            self._store(
                self._key_cache.to("cpu", non_blocking=True),
                self._value_cache.to("cpu", non_blocking=True),
            )

    def prefetch(self) -> None:
        """Move this layer's tensors back to ``self.device``.

        ``self.device`` is the device the layer was last seen on, derived from
        the key tensor in ``__init__`` and refreshed by every mutation, exactly
        as ``DynamicLayer.lazy_initialization`` would have set it.
        """
        if self.is_initialized and self._key_cache.device != self.device:
            self._store(
                self._key_cache.to(self.device, non_blocking=True),
                self._value_cache.to(self.device, non_blocking=True),
            )

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder this layer's batch dimension for beam search."""
        if self.get_seq_length() > 0:
            beam_idx = beam_idx.to(self._key_cache.device)
            self._store(
                self._key_cache.index_select(0, beam_idx),
                self._value_cache.index_select(0, beam_idx),
            )

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.is_initialized and 0 <= max_length < self.get_seq_length():
            self._store(
                self._key_cache[..., :max_length, :],
                self._value_cache[..., :max_length, :],
            )

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() > 0:
            self._store(
                self._key_cache.repeat_interleave(repeats, dim=0),
                self._value_cache.repeat_interleave(repeats, dim=0),
            )

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() > 0:
            indices = indices.to(self._key_cache.device)
            self._store(self._key_cache[indices, ...], self._value_cache[indices, ...])

    def reset(self) -> None:
        self._store(None, None)


class _LazyPrefetchStream:
    """Copyable stand-in for ``Cache.prefetch_stream``.

    ``Cache.offload``, ``Cache.prefetch`` and ``Cache.update`` use the stream as
    a context manager, and nothing else reads it. A real ``torch.Stream`` cannot
    be deep-copied (it raises ``TypeError: cannot pickle 'torch.Stream' object``)
    and a cached voice prompt is deep-copied once per generation, so a stream
    attached at wrap time would make an otherwise copyable cache uncopyable.
    Materialising on first use keeps the surface identical while leaving a plain
    Python object behind until something actually offloads.
    """

    def __init__(self):
        self._stream = None

    def _materialize(self):
        if self._stream is None:
            self._stream = torch.Stream()
        return self._stream

    def __enter__(self):
        self._materialize().__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._materialize().__exit__(*exc_info)

    def __deepcopy__(self, memo):
        # The stream is recreated on demand, so a copy starts unmaterialized
        # rather than failing outright.
        return _LazyPrefetchStream()


def _ensure_cache_has_layers(cache):
    """
    Ensure the cache has all required attributes for transformers >= 4.57.
    Creates MockCacheLayer wrappers to provide the expected `layers` interface.

    ``Cache.get_mask_sizes`` answers ``(query_length, 0)`` whenever
    ``layer_idx >= len(self.layers)`` (transformers 5.3 ``cache_utils``), i.e. an
    unpopulated ``layers`` list silently tells the mask builder that no voice
    prefill exists. ``layers`` therefore has to be filled *before* the first
    forward, which is what this function is for: it is called from
    ``_update_model_kwargs_for_generation`` ahead of every step, including the
    first.

    The 5.x container also has two stateful requirements that a cache pickled
    by an older transformers does not satisfy:

    * ``Cache.update`` touches ``self.prefetch_stream`` on every step while
      ``self.offloading`` is truthy, but that stream is only ever created when
      ``Cache.__init__`` was called with ``offloading=True``. A legacy pickle
      that carries the flag without the stream raises ``AttributeError`` inside
      the model's first forward, so the flag is cleared when the stream it
      implies is missing.
    * ``Cache.offload`` / ``Cache.prefetch`` are part of the public container
      API and read ``self.prefetch_stream`` unconditionally, so the stream is
      provided for adapted legacy caches even when offloading stays off.
    """
    if cache is None:
        return cache

    # Add required attributes (skip if read-only)
    for attr, default in [('layer_class_to_replicate', None), ('offloading', False), ('is_compileable', False)]:
        if not hasattr(cache, attr):
            try:
                setattr(cache, attr, default)
            except AttributeError:
                pass

    # ``offloading`` is only honoured by 5.x when the prefetch stream that
    # ``Cache.__init__`` would have created alongside it also exists. Keeping a
    # stale ``True`` from an older pickle turns the very first ``Cache.update``
    # into an AttributeError, and the adapted layers carry no per-layer device
    # bookkeeping that would make CPU offloading meaningful anyway.
    if getattr(cache, 'offloading', False) and not hasattr(cache, 'prefetch_stream'):
        logger.warning(
            "VibeVoice realtime: dropping `offloading=True` from a cached voice prompt - "
            "the pickled cache has no prefetch stream, and Cache.update would raise on it."
        )
        try:
            cache.offloading = False
        except AttributeError:
            pass

    # ``Cache.offload``/``Cache.prefetch`` are reachable from the container API
    # regardless of ``offloading``, and both open ``prefetch_stream``.
    if not hasattr(cache, 'prefetch_stream'):
        try:
            cache.prefetch_stream = _LazyPrefetchStream()
        except AttributeError:
            pass

    # Build layers list from key_cache/value_cache
    if hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
        try:
            cache.layers = [
                MockCacheLayer(cache.key_cache[i], cache.value_cache[i], parent_cache=cache, layer_idx=i)
                for i in range(len(cache.key_cache))
            ]
        except AttributeError:
            pass
    elif not hasattr(cache, 'layers'):
        try:
            cache.layers = []
        except AttributeError:
            pass
    
    return cache


def _update_model_kwargs_for_generation(
    outputs: ModelOutput,
    model_kwargs: Dict[str, Any],
    num_new_tokens: int = 1,
) -> Dict[str, Any]:
    """
    Update model_kwargs after adding new tokens (supports multi-token windows).
    
    Updates past_key_values, attention_mask, and cache_position for the next forward pass.
    """
    model_kwargs["past_key_values"] = _ensure_cache_has_layers(outputs.past_key_values)
    
    attention_mask = model_kwargs["attention_mask"]
    model_kwargs["attention_mask"] = torch.cat(
        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], num_new_tokens))], dim=-1
    )
    
    cache_pos = model_kwargs["cache_position"]
    model_kwargs["cache_position"] = torch.arange(
        cache_pos[-1] + 1, cache_pos[-1] + num_new_tokens + 1, device=cache_pos.device
    )
    
    return model_kwargs


@dataclass
class VibeVoiceCausalLMOutputWithPast(BaseModelOutputWithPast):
    logits: Optional[torch.FloatTensor] = None


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
    reach_max_step_sample: Optional[torch.BoolTensor] = None


class VibeVoiceStreamingForConditionalGenerationInference(VibeVoiceStreamingPreTrainedModel, GenerationMixin):

    def __init__(self, config):
        super().__init__(config)
        
        # Initialize the base model
        self.model = VibeVoiceStreamingModel(config)

        # TTS generation EOS classifier
        self.tts_eos_classifier = BinaryClassifier(config.decoder_config.hidden_size)
        
        # inference configuration
        self.ddpm_inference_steps = config.diffusion_head_config.ddpm_num_inference_steps

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def noise_scheduler(self):
        return self.model.noise_scheduler

    @property
    def prediction_head(self):
        return self.model.prediction_head
    
    @property
    def speech_scaling_factor(self):
        return self.model.speech_scaling_factor

    @property
    def speech_bias_factor(self):
        return self.model.speech_bias_factor

    @property
    def acoustic_tokenizer(self):
        return self.model.acoustic_tokenizer
    
    @property
    def acoustic_connector(self):
        return self.model.acoustic_connector
        
    def tie_weights(self, missing_keys=None, recompute_mapping=True, **kwargs):
        """
        Tie the weights between the input embeddings and the output embeddings.

        Accepts missing_keys and recompute_mapping for transformers 5.x
        compatibility (PreTrainedModel.init_weights passes these kwargs).
        """
        # Tie lm_head.weight to language_model.embed_tokens.weight
        if not getattr(self.config, 'tie_word_embeddings', False):
            return
         
        if hasattr(self, 'lm_head') and hasattr(self.model.language_model, 'embed_tokens'):
            self.lm_head.weight = self.model.language_model.embed_tokens.weight
        
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
    
    def get_output_embeddings(self):
        """
        This model does not define an `lm_head` (vocabulary projection).
        """
        return None
    
    def set_output_embeddings(self, new_embeddings):
        """
        No-op because there is no `lm_head`. Provided only to satisfy optional API calls.
        To enable, first create `self.lm_head` then allow assignment.
        """
        raise RuntimeError("Output embeddings (lm_head) are not defined for this model. "
                           "Create one before calling set_output_embeddings if needed.")
    
    def set_speech_tokenizers(self, acoustic_tokenizer=None):
        """Set the speech tokenizers used for encoding and decoding speech."""
        self.model.set_speech_tokenizers(acoustic_tokenizer)
    
    def set_ddpm_inference_steps(self, num_steps=None):
        self.ddpm_inference_steps = num_steps or self.config.diffusion_head_config.ddpm_num_inference_steps

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        """Prepare model inputs for generation (transformers >= 4.57 compatible)."""
        model_inputs = {"cache_position": cache_position}

        # Slice inputs when using cache
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            if inputs_embeds is not None and input_ids.shape[1] == 0:
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0]:]
            elif inputs_embeds is not None or (cache_position is not None and cache_position[-1] >= input_ids.shape[1]):
                input_ids = input_ids[:, -cache_position.shape[0]:]
            elif cache_position is not None and input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]

        # Set input_ids or inputs_embeds
        use_embeds = inputs_embeds is not None and (
            past_key_values is None or (cache_position is not None and len(cache_position) == inputs_embeds.shape[1])
        )
        if use_embeds:
            model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs["input_ids"] = input_ids.clone(memory_format=torch.contiguous_format) if input_ids is not None else None
            model_inputs["inputs_embeds"] = None

        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        # Create position_ids from attention_mask
        if attention_mask is not None and kwargs.get("position_ids") is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs["position_ids"] = position_ids

        # Slice position_ids when using cache
        if kwargs.get("position_ids") is not None:
            if past_key_values is not None:
                seq_len = model_inputs["inputs_embeds"].shape[1] if model_inputs.get("inputs_embeds") is not None else model_inputs["input_ids"].shape[1]
                model_inputs["position_ids"] = kwargs["position_ids"][:, -seq_len:].clone(memory_format=torch.contiguous_format)
            else:
                model_inputs["position_ids"] = kwargs.pop("position_ids").clone(memory_format=torch.contiguous_format)

        # Forward remaining kwargs
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        model_inputs.pop("labels", None)
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder=False,
        num_new_tokens=1,
    ):
        """Override to ensure cache compatibility with transformers >= 4.57."""
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder, num_new_tokens=num_new_tokens
        )
        # ``cache_position`` is the *input* to the next forward: the positions of
        # the tokens about to be fed, nothing else. 4.57.6 honoured that contract
        # (``cache_position[-1:] + num_new_tokens``, i.e. a tensor of exactly
        # ``num_new_tokens`` entries). 5.3.0 changed it to append the new
        # positions to the whole history (``torch.cat((cache_position,
        # next_cache_position))``, transformers/generation/utils.py:938), so the
        # tensor grows by one entry per step and starts at position 0.
        #
        # That is not cosmetic. ``prepare_inputs_for_generation`` below slices
        # the inputs with ``input_ids[:, -cache_position.shape[0]:]``, so the
        # query window silently widens to the entire history: step one re-feeds
        # 6 tokens, step two 7, and so on. The already-cached positions are
        # recomputed and appended again, the KV cache grows super-linearly
        # (measured: 316 -> 321 -> 327 -> 334 -> 342 -> 351 -> 361 -> 372 for
        # the 0.5B checkpoint) and every position after the first window is
        # served a sequence that no longer matches its index. The mask is built
        # correctly for the wrong query, so nothing raises: the model simply
        # never reaches its own end-of-speech and the clip runs to the length
        # budget. S1.1's single-forward mask probe cannot see this, because the
        # first forward is correct on both versions.
        #
        # The base implementation's last ``num_new_tokens`` entries are exactly
        # the 4.x answer, so keeping only those restores the old contract
        # without branching on a version number, and is a no-op on 4.5x. The
        # ``0 <`` guard matters: ``tensor[-0:]`` is the whole tensor, not an
        # empty tail.
        cache_position = model_kwargs.get("cache_position")
        if cache_position is not None and 0 < num_new_tokens < cache_position.numel():
            model_kwargs["cache_position"] = cache_position[-num_new_tokens:]
        if "past_key_values" in model_kwargs:
            model_kwargs["past_key_values"] = _ensure_cache_has_layers(model_kwargs["past_key_values"])
        return model_kwargs

    def _init_cache_for_generation(self, generation_config, model_kwargs, batch_size, max_cache_length, device):
        """
        Initialize cache for generation, handling different transformers versions.
        For transformers >= 4.57, returns None to let the model create the cache dynamically.
        """
        try:
            from transformers.cache_utils import DynamicCache
            sig = inspect.signature(DynamicCache.__init__)
            if 'config' in sig.parameters:
                # transformers >= 4.57: let model handle cache creation
                return None
            else:
                # Older versions: use parent method
                prep_sig = inspect.signature(self._prepare_cache_for_generation)
                if 'device' in prep_sig.parameters:
                    self._prepare_cache_for_generation(generation_config, model_kwargs, None, batch_size, max_cache_length, device)
                else:
                    self._prepare_cache_for_generation(generation_config, model_kwargs, None, batch_size, max_cache_length)
                return model_kwargs.get("past_key_values")
        except Exception:
            return None

    # @can_return_tuple
    def forward_lm(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Single pass of the base text LM.

        - Builds embeddings if `inputs_embeds` not provided.
        - Uses (and returns) `past_key_values` when `use_cache=True`.
        - No loss / no lm_head / no speech logic.

        Args:
            input_ids: (B, S) token ids.
            attention_mask: (B, S) mask.
            past_key_values: cache from previous steps.
            cache_position: positions for cached tokens.
            labels: unsupported (will raise).

        Returns:
            BaseModelOutputWithPast with `last_hidden_state` and `past_key_values`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
                
        if labels is not None:
            raise NotImplementedError("Loss computation is not implemented in this version.")

        return BaseModelOutputWithPast(
            past_key_values=outputs.past_key_values,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
        )

    # @can_return_tuple
    def forward_tts_lm(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        lm_last_hidden_state: Optional[torch.FloatTensor] = None,
        tts_text_masks: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Union[Tuple, VibeVoiceCausalLMOutputWithPast]:
        """
        Single pass of the TTS LM.

        - Overwrites tail embeddings with `lm_last_hidden_state`.
        - Adds type embedding via `tts_text_masks` (1=text, 0=speech).
        - Predicts EOS from last hidden state (binary classifier).
        - No loss / no full acoustic decoding here.

        Args:
            input_ids: (B, S) token ids.
            attention_mask: (B, S) mask.
            lm_last_hidden_state: (B, K, H) hidden states to splice into the tail.
            tts_text_masks: (B, 1) mask marking current position as text(1)/speech(0).
            past_key_values: cache from previous TTS steps.
            cache_position: positions for cached tokens.
            labels: unsupported (will raise).

        Returns:
            VibeVoiceCausalLMOutputWithPast with `logits` (EOS), `last_hidden_state`, `past_key_values`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get embeddings
        if inputs_embeds is None:
            # Will be replaced with lm_last_hidden_state
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # Replace the last part of inputs_embeds with lm_last_hidden_state
        start_idx = inputs_embeds.shape[1] - lm_last_hidden_state.shape[1]
        inputs_embeds[:, start_idx:, :] = lm_last_hidden_state
        
        # Adds type embedding via `tts_text_masks`.
        inputs_embeds = inputs_embeds + self.model.tts_input_types(tts_text_masks.long())

        outputs = self.model.tts_language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        logits = self.tts_eos_classifier(hidden_states[:, -1, :])
                
        if labels is not None:
            raise NotImplementedError("Loss computation is not implemented in this version.")

        return VibeVoiceCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
        )

    def forward(self, *args, **kwargs):
        """
        Unified forward is intentionally disabled.

        Reasons:
          1. The inference pipeline is staged: base text LM, then TTS LM, plus streaming & diffusion handled in `generate`.
          2. A monolithic call would hide required sequencing (prefill, window stepping, speech diffusion sampling).

        Use instead:
          - self.forward_lm(...)       for a base text LM step (prefill or incremental).
          - self.forward_tts_lm(...)   for a single TTS LM step (needs LM hidden states).
          - self.generate(...)         for full streaming (text + speech + diffusion + audio assembly).

        Raises:
            RuntimeError: Always (by design).
        """
        raise RuntimeError(
            "Unified forward is disabled. Use `forward_lm`, `forward_tts_lm`, or `generate` instead."
        )

    def _build_generate_config_model_kwargs(self, generation_config, inputs, tokenizer, return_processors=False, **kwargs):
        if generation_config is None:
            generation_config = GenerationConfig(
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id
            )
        else:
            generation_config = GenerationConfig(
                **generation_config,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id
            )

        # transformers 4.x accepted a positional "is_init" flag here; 5.x
        # narrowed the signature to (generation_config, **kwargs). Inspect the
        # live signature so one vendored file supports both APIs. The custom
        # speech token ids are assigned to the returned config below rather
        # than passed through kwargs, because GenerationConfig no longer
        # tolerates unknown keyword attributes.
        _prepare_args = (
            (generation_config, True)
            if _generation_config_accepts_positional_flag()
            else (generation_config,)
        )

        generation_config, model_kwargs = self._prepare_generation_config(
            *_prepare_args,
            **kwargs
        )
        generation_config.speech_start_id = tokenizer.speech_start_id
        generation_config.speech_end_id = tokenizer.speech_end_id
        generation_config.speech_diffusion_id = tokenizer.speech_diffusion_id

        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, generation_config.bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]
        device = self.device
        
        self._prepare_special_tokens(generation_config, True, device=device)
        generation_config.use_cache = True
        model_kwargs["use_cache"] = generation_config.use_cache
        input_ids = inputs_tensor.to(self.device)

        input_ids_length = input_ids.shape[1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        max_cache_length = generation_config.max_length - 1
        # Handle cache initialization for different transformers versions
        model_kwargs["past_key_values"] = self._init_cache_for_generation(
            generation_config, model_kwargs, batch_size, max_cache_length, device
        )
        model_kwargs['cache_position'] = torch.arange(input_ids_length, device=device, dtype=torch.long)
        for k, v in model_kwargs.items():
            if isinstance(v, torch.Tensor):
                model_kwargs[k] = v.to(device=device)
        
        if return_processors:
            logits_processor = self._get_logits_processor(
                generation_config=generation_config,
                input_ids_seq_length=input_ids_length,
                encoder_input_ids=inputs_tensor,
                prefix_allowed_tokens_fn=None,
                logits_processor=LogitsProcessorList(),
                device=inputs_tensor.device,
                model_kwargs=model_kwargs,
            )

            stopping_criteria = self._get_stopping_criteria(generation_config=generation_config, stopping_criteria=StoppingCriteriaList())
        
            return generation_config, model_kwargs, input_ids, logits_processor, stopping_criteria
        else:
            return generation_config, model_kwargs, input_ids

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        audio_streamer: Optional[Union[AudioStreamer, AsyncAudioStreamer]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        tts_text_ids: Optional[torch.LongTensor] = None,
        return_speech: bool = True,
        cfg_scale: float = 1.0,
        stop_check_fn: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs,
    ) -> Union[torch.LongTensor, VibeVoiceGenerationOutput]:
        """
        Text is fed in small windows (dynamic slicing of `tts_text_ids`), which enables streaming text input: you don’t need the full text upfront. After each text window, a loop samples several speech latents (diffusion). The interleaved text encoding + speech generation enables streaming text input and realtime speech output.
        The function only supports batch size = 1 currently.

        - Windowed text prefill → incremental LM + TTS LM updates.
        - Interleave speech token diffusion sampling (`sample_speech_tokens`).
        - Stops on EOS (binary classifier) or max length / external `stop_check_fn`.
        - Returns final token `sequences` and (optionally) concatenated speech audio.

        Args (selected):
            tts_text_ids: Full text tokens to stream in windows.
            audio_streamer: If provided, emits audio chunks during generation.
            cfg_scale: Classifier-free guidance scale for speech diffusion.
            return_speech: If False, skips audio decode concatenation.
            stop_check_fn: External early-stop hook (returns True to halt).
            progress_callback: Optional framework-agnostic hook invoked as
                ``progress_callback(current, total)`` after each text-window
                prefill and after each generated speech token, where
                ``total == tts_lm_generation_config.max_length``. ``current``
                is monotonic non-decreasing and bounded by ``total``. ``None``
                disables reporting. The callback may raise to interrupt
                generation (the exception propagates out of ``generate()``);
                callers use this for progress UI and cancellation. Orthogonal
                to the console ``tqdm`` bar (``show_progress_bar``).

        Returns:
            VibeVoiceGenerationOutput with:
              - sequences: final token ids
              - speech_outputs: list of concatenated audio tensors (or None)
              - reach_max_step_sample: flags for samples stopped by max length
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        tokenizer = kwargs.pop("tokenizer", None)
        neg_text_input_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        
        tts_lm_input_ids = kwargs.pop("tts_lm_input_ids", None)
        tts_lm_attention_mask = kwargs.pop("tts_lm_attention_mask", None)
        # all_prefilled_outputs: cached prefilled prompt outputs for lm, tts_lm, neg_lm, neg_tts_lm
        all_prefilled_outputs = kwargs.pop("all_prefilled_outputs", None)
        tts_text_ids = tts_text_ids.to(self.device)

        if kwargs.get('max_new_tokens', None) is None:
            kwargs['max_new_tokens'] = self.config.decoder_config.max_position_embeddings - tts_lm_input_ids.shape[-1]

        generation_config, model_kwargs, input_ids, logits_processor, stopping_criteria = self._build_generate_config_model_kwargs(
            generation_config, inputs, tokenizer, return_processors=True, **kwargs
        )
        
        negative_kwargs = {
            'input_ids': torch.full((kwargs['input_ids'].shape[0], 1), neg_text_input_id, dtype=torch.long, device=kwargs['input_ids'].device),
            'attention_mask':  torch.ones((kwargs['input_ids'].shape[0], 1), dtype=torch.long, device=kwargs['input_ids'].device),
            'max_new_tokens': kwargs.get('max_new_tokens', 100) 
        }
        negative_generation_config, negative_model_kwargs, negative_input_ids = self._build_generate_config_model_kwargs(
            None, None, tokenizer, return_processors=False, **negative_kwargs
        )

        tts_lm_kwargs = {
            'input_ids': tts_lm_input_ids,
            'attention_mask': tts_lm_attention_mask,
            'max_new_tokens': kwargs.get('max_new_tokens', 100) 
        }
        tts_lm_generation_config, tts_lm_model_kwargs, tts_lm_input_ids = self._build_generate_config_model_kwargs(
            None, None, tokenizer, return_processors=False, **tts_lm_kwargs
        )

        tts_lm_negative_kwargs = {
            'input_ids': torch.full((kwargs['input_ids'].shape[0], 1), neg_text_input_id, dtype=torch.long, device=kwargs['input_ids'].device),
            'attention_mask':  torch.ones((kwargs['input_ids'].shape[0], 1), dtype=torch.long, device=kwargs['input_ids'].device),
            'max_new_tokens': kwargs.get('max_new_tokens', 100) 
        }
        tts_lm_negative_generation_config, tts_lm_negative_model_kwargs, tts_lm_negative_input_ids = self._build_generate_config_model_kwargs(
            None, None, tokenizer, return_processors=False, **tts_lm_negative_kwargs
        )

        acoustic_cache = VibeVoiceTokenizerStreamingCache()
        batch_size = input_ids.shape[0]
        assert batch_size == 1, "Currently only supports batch size == 1"
        device = input_ids.device
        finished_tags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        verbose = kwargs.get("verbose", False)

        # Initialize audio chunks storage for each sample
        audio_chunks = [[] for _ in range(batch_size)]
        tts_text_window_index = 0
        reach_max_step_sample = torch.zeros(batch_size, dtype=torch.bool, device=device)
        first_text_window_size = TTS_TEXT_WINDOW_SIZE if tts_text_ids.shape[1] >= TTS_TEXT_WINDOW_SIZE else tts_text_ids.shape[1]

        outputs = all_prefilled_outputs["lm"]
        tts_lm_outputs = all_prefilled_outputs["tts_lm"]
        negative_outputs = all_prefilled_outputs["neg_lm"]
        tts_lm_negative_outputs = all_prefilled_outputs["neg_tts_lm"]

        model_kwargs = _update_model_kwargs_for_generation(
            outputs, model_kwargs, num_new_tokens=first_text_window_size,
        )
        tts_lm_model_kwargs = _update_model_kwargs_for_generation(
            tts_lm_outputs, tts_lm_model_kwargs, num_new_tokens=first_text_window_size,
        )
        negative_model_kwargs = self._update_model_kwargs_for_generation(
            negative_outputs, negative_model_kwargs, is_encoder_decoder=False,
        )
        tts_lm_negative_model_kwargs = self._update_model_kwargs_for_generation(
            tts_lm_negative_outputs, tts_lm_negative_model_kwargs, is_encoder_decoder=False,
        )

        step = tts_lm_input_ids.shape[1]
        total_generated_speech_tokens = 0
        total_prefilled_text_tokens = 0
        if kwargs.get("show_progress_bar", True):
            progress_bar = tqdm(
                total=tts_lm_generation_config.max_length,
                desc=f"Prefilled {step} tokens, current step ({step} / {tts_lm_generation_config.max_length})",
                initial=step,
                leave=False
            )
        else:
            progress_bar = None

        while True:
            # Check for external stop signal
            if stop_check_fn is not None and stop_check_fn():
                if verbose:
                    print(f"Generation stopped externally at step {step + 1}")
                # End the audio streamer if it exists
                if audio_streamer is not None:
                    audio_streamer.end()
                break
            
            # # Check if audio_streamer has been ended (stopped externally)
            # if audio_streamer is not None and hasattr(audio_streamer, 'finished_flags'):
            #     if any(audio_streamer.finished_flags):
            #         if verbose:
            #             print(f"Audio generation stopped externally at step {step + 1}")
            #         break
            
            if finished_tags.all():
                if hasattr(progress_bar, 'set_description'):
                    progress_bar.set_description("Generation complete")
                break

            cur_input_tts_text_ids = tts_text_ids[:, tts_text_window_index*TTS_TEXT_WINDOW_SIZE:(tts_text_window_index+1)*TTS_TEXT_WINDOW_SIZE]
            next_text_window_size = tts_text_ids[:, (tts_text_window_index+1)*TTS_TEXT_WINDOW_SIZE:(tts_text_window_index+2)*TTS_TEXT_WINDOW_SIZE].shape[1]
            tts_text_window_index += 1

            if cur_input_tts_text_ids.shape[1] > 0:
                input_ids = torch.cat([input_ids, cur_input_tts_text_ids], dim=-1)
                tts_lm_input_ids = torch.cat([tts_lm_input_ids, cur_input_tts_text_ids], dim=-1)

                if tts_lm_input_ids.shape[1] > tts_lm_generation_config.max_length:
                    if verbose:
                        print(f"Reached maximum generation length {generation_config.max_length}, stopped it.")
                    reached_samples = torch.arange(batch_size, device=device)[~finished_tags]
                    if reached_samples.numel() > 0:
                        reach_max_step_sample[reached_samples] = True
                    break
                
                step += cur_input_tts_text_ids.shape[1]
                total_prefilled_text_tokens += cur_input_tts_text_ids.shape[1]
                if progress_bar is not None:
                    progress_bar.update(cur_input_tts_text_ids.shape[1])
                    progress_bar.set_description(f"Prefilled {total_prefilled_text_tokens} text tokens, generated {total_generated_speech_tokens} speech tokens, current step ({step} / {tts_lm_generation_config.max_length})")
                # Framework-agnostic progress hook (ComfyUI progress bar).
                if progress_callback is not None:
                    progress_callback(step, tts_lm_generation_config.max_length)

                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                # Forward pass through the model
                outputs = self.forward_lm(
                    **model_inputs, return_dict=True, output_attentions=False, output_hidden_states=False,
                )
                model_kwargs = _update_model_kwargs_for_generation(
                    outputs, model_kwargs, num_new_tokens=next_text_window_size,
                )

                tts_lm_model_inputs = self.prepare_inputs_for_generation(tts_lm_input_ids, **tts_lm_model_kwargs)
                tts_lm_additional_inputs = {
                    "tts_text_masks": torch.ones_like(tts_lm_input_ids[:, -1:]),
                    "lm_last_hidden_state": outputs.last_hidden_state,
                }
                # Forward pass through the model
                tts_lm_outputs = self.forward_tts_lm(
                    **tts_lm_model_inputs, **tts_lm_additional_inputs, return_dict=True, output_attentions=False, output_hidden_states=False,
                )
                tts_lm_model_kwargs = self._update_model_kwargs_for_generation(
                    tts_lm_outputs, tts_lm_model_kwargs, is_encoder_decoder=False,
                )

            diffusion_indices = torch.LongTensor([0])
            for cur_speech_index in range(TTS_SPEECH_WINDOW_SIZE):
                positive_condition = tts_lm_outputs.last_hidden_state[diffusion_indices, -1, :]
                negative_condition = tts_lm_negative_outputs.last_hidden_state[diffusion_indices, -1, :]
                
                speech_latent = self.sample_speech_tokens(
                    positive_condition,
                    negative_condition,
                    cfg_scale=cfg_scale,
                ).unsqueeze(1)
                                
                # Decode acoustic latent to audio using acoustic streaming cache
                scaled_latent = speech_latent / self.model.speech_scaling_factor.to(speech_latent.device) - self.model.speech_bias_factor.to(speech_latent.device)
                audio_chunk = self.model.acoustic_tokenizer.decode(
                    scaled_latent.to(self.model.acoustic_tokenizer.device),
                    cache=acoustic_cache,  # Use acoustic-specific cache
                    sample_indices=diffusion_indices.to(self.model.acoustic_tokenizer.device),
                    use_cache=True,
                    debug=False
                )
                
                # Store audio chunks for each sample
                for i, sample_idx in enumerate(diffusion_indices):
                    idx = sample_idx.item()
                    # Only append audio chunk if the sample is not finished
                    if not finished_tags[idx]:
                        audio_chunks[idx].append(audio_chunk[i])

                 # Add streaming support here
                if audio_streamer is not None:
                    # Stream the audio chunks immediately
                    audio_streamer.put(audio_chunk, diffusion_indices)

                acoustic_embed = self.model.acoustic_connector(speech_latent)
                tts_lm_input_ids = torch.cat([tts_lm_input_ids, torch.ones_like(tts_lm_input_ids[:, -1:])], dim=-1)

                if tts_lm_input_ids.shape[1] > tts_lm_generation_config.max_length:
                    break
                
                step += 1
                total_generated_speech_tokens += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_description(f"Prefilled {total_prefilled_text_tokens} text tokens, generated {total_generated_speech_tokens} speech tokens, current step ({step} / {tts_lm_generation_config.max_length})")
                # Framework-agnostic progress hook (ComfyUI progress bar).
                if progress_callback is not None:
                    progress_callback(step, tts_lm_generation_config.max_length)

                tts_lm_model_inputs = self.prepare_inputs_for_generation(tts_lm_input_ids, **tts_lm_model_kwargs)
                tts_lm_additional_inputs = {
                    "tts_text_masks": torch.zeros_like(tts_lm_input_ids[:, -1:]),
                    "lm_last_hidden_state": acoustic_embed,
                }
                # Forward pass through the model
                tts_lm_outputs = self.forward_tts_lm(
                    **tts_lm_model_inputs, **tts_lm_additional_inputs, return_dict=True, output_attentions=False, output_hidden_states=False,
                )
                if cur_speech_index == TTS_SPEECH_WINDOW_SIZE - 1 and next_text_window_size > 0:
                    tts_lm_model_kwargs = _update_model_kwargs_for_generation(
                        tts_lm_outputs, tts_lm_model_kwargs, num_new_tokens=next_text_window_size,
                    )
                else:
                    tts_lm_model_kwargs = self._update_model_kwargs_for_generation(
                        tts_lm_outputs, tts_lm_model_kwargs, is_encoder_decoder=False,
                    )

                tts_lm_negative_input_ids = torch.cat([tts_lm_negative_input_ids, torch.ones_like(tts_lm_input_ids[:, -1:])], dim=-1)
                tts_lm_negative_model_inputs = self.prepare_inputs_for_generation(tts_lm_negative_input_ids, **tts_lm_negative_model_kwargs)
                # Forward negative pass through the model
                tts_lm_negative_additional_inputs = {
                    "tts_text_masks": torch.zeros_like(tts_lm_negative_input_ids[:, -1:]),
                    "lm_last_hidden_state": acoustic_embed,
                }
                tts_lm_negative_outputs = self.forward_tts_lm(
                    **tts_lm_negative_model_inputs, **tts_lm_negative_additional_inputs, return_dict=True, output_attentions=False, output_hidden_states=False,
                )
                tts_lm_negative_model_kwargs = self._update_model_kwargs_for_generation(
                    tts_lm_negative_outputs, tts_lm_negative_model_kwargs, is_encoder_decoder=False,
                )

                tts_eos_logits = torch.sigmoid(self.tts_eos_classifier(tts_lm_outputs.last_hidden_state[diffusion_indices, -1, :]))
                if tts_eos_logits[0].item() > 0.5:
                    # If EOS token is predicted, we can stop generation for this sample
                    finished_tags[diffusion_indices] = True
                    if audio_streamer is not None:
                        audio_streamer.end(diffusion_indices)

            if tts_lm_input_ids.shape[1] > tts_lm_generation_config.max_length:
                if verbose:
                    print(f"Reached maximum generation length {tts_lm_generation_config.max_length}, stopped it.")
                reached_samples = torch.arange(batch_size, device=device)[~finished_tags]
                if reached_samples.numel() > 0:
                    reach_max_step_sample[reached_samples] = True
                break

        if audio_streamer is not None:
            audio_streamer.end()

        # Concatenate audio chunks for each sample
        final_audio_outputs = []
        for sample_chunks in audio_chunks:
            if sample_chunks:
                # Concatenate all chunks along the time dimension (assumed to be the last dimension)
                concatenated_audio = torch.cat(sample_chunks, dim=-1)
                final_audio_outputs.append(concatenated_audio)
            else:
                # If no audio was generated for this sample, append None
                final_audio_outputs.append(None)
        
        if reach_max_step_sample is not None and reach_max_step_sample.any():
            print(f"Reached maximum generation length {tts_lm_generation_config.max_length}, stopped it.")

        return VibeVoiceGenerationOutput(
            sequences=tts_lm_input_ids,
            speech_outputs=final_audio_outputs if return_speech else None,
            reach_max_step_sample=reach_max_step_sample,
        )

    @torch.no_grad()
    def sample_speech_tokens(self, condition, neg_condition, cfg_scale=3.0):
        self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)
        condition = torch.cat([condition, neg_condition], dim=0).to(self.model.prediction_head.device)
        speech = torch.randn(condition.shape[0], self.config.acoustic_vae_dim).to(condition)
        for t in self.model.noise_scheduler.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = self.model.prediction_head(combined, t.repeat(combined.shape[0]).to(combined), condition=condition)
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample
        return speech[: len(speech) // 2]
    

AutoModelForCausalLM.register(VibeVoiceStreamingConfig, VibeVoiceStreamingForConditionalGenerationInference, exist_ok=True)

__all__ = [
    "VibeVoiceStreamingForConditionalGenerationInference",
]
