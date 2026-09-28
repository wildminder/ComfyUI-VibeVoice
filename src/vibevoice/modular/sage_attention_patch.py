# Author: Wildminder
# Desc: SageAttention and patcher
# License: Apache 2.0

import torch
from typing import Optional, Tuple

from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb, repeat_kv
from transformers.cache_utils import Cache
import logging

logger = logging.getLogger(__name__)

try:
    from sageattention.core import (
        sageattn_qk_int8_pv_fp16_cuda,
        sageattn_qk_int8_pv_fp8_cuda,
        sageattn_qk_int8_pv_fp8_cuda_sm90,
    )
    SAGE_ATTENTION_AVAILABLE = True
except ImportError:
    SAGE_ATTENTION_AVAILABLE = False


def get_sage_attention_function_and_params():
    """
    Selects the best available SageAttention CUDA kernel and its parameters
    based on the current GPU architecture.

    Dispatch is by EXACT architecture, not by ">= the newest known". The
    previous threshold chain (`>= 120`, then `>= 90`, then `== 89`, then
    `>= 80`) routed sm100/sm103 (Blackwell datacenter, CC 10.x) into the SM90
    branch. Note that THIS module calls the kernels directly and never goes
    through sage's own `sageattn()`, so nothing here raises sage's
    `ValueError: Unsupported CUDA architecture` for an unknown arch: the sm90
    kernel asserts only that it COMPILED, not that it is running on the right
    silicon, so the pre-fix failure mode was SILENT wrong-kernel selection --
    a Hopper kernel handed to a GPU it was never built for, with no exception
    anywhere downstream. Unknown archs are refused here instead.

    KNOWN DIVERGENCE from sage's own ``sageattn()`` dispatcher, recorded rather
    than silently reconciled. sage (2.2.0, core.py) picks ``pv_accum_dtype``
    per arch *and* per CUDA runtime: for sm89 and sm120 it uses ``fp32+fp16``
    ("SageAttention2++") whenever ``torch.version.cuda >= (12, 8)`` and
    ``fp32+fp32`` below that. This module hardcodes ``fp32+fp32`` on both.
    That is a deliberate, unchanged-from-the-initial-commit choice, not an
    oversight introduced by the arch-alignment work above, and it is NOT
    changed here: measured on this box (sm89, CUDA 13.0) the two settings are
    numerically indistinguishable for a bf16 1x8x512x128 causal attention --
    rel_l2 vs an fp32 SDPA reference 0.03401 (fp32+fp32) vs 0.03415 (fp32+fp16)
    -- so switching would alter published output quality by an amount that
    cannot be justified, in either direction, from a default suite. A change
    here needs a Tier-G measurement on the real checkpoint, exactly like every
    other backend-parity decision in this repo.
    """
    if not SAGE_ATTENTION_AVAILABLE or not torch.cuda.is_available():
        return None, None, None

    major, minor = torch.cuda.get_device_capability()
    arch_code = major * 10 + minor

    attn_func = None
    pv_accum_dtype = "fp32"

    if arch_code in (80, 86):  # Ampere
        pv_accum_dtype = "fp32"
        attn_func = sageattn_qk_int8_pv_fp16_cuda
        logger.info(f"SageAttention: Using SM80+ (Ampere) FP16 kernel with pv_accum_dtype='{pv_accum_dtype}'.")
    elif arch_code == 89:  # Ada Lovelace
        pv_accum_dtype = "fp32+fp32"
        attn_func = sageattn_qk_int8_pv_fp8_cuda
        logger.info(f"SageAttention: Using SM89 (Ada) FP8 kernel with pv_accum_dtype='{pv_accum_dtype}'.")
    elif arch_code == 90:  # Hopper
        pv_accum_dtype = "fp32+fp32"
        attn_func = sageattn_qk_int8_pv_fp8_cuda_sm90
        logger.info(f"SageAttention: Using SM90 (Hopper) FP8 kernel with pv_accum_dtype='{pv_accum_dtype}'.")
    elif arch_code == 120:  # Blackwell
        pv_accum_dtype = "fp32+fp32"
        attn_func = sageattn_qk_int8_pv_fp8_cuda
        logger.info(f"SageAttention: Using SM120 (Blackwell) FP8 kernel with pv_accum_dtype='{pv_accum_dtype}'.")
    else:
        logger.warning(
            f"SageAttention has no kernel for SM{arch_code}; SageAttention "
            f"supports SM80/86/89/90/120."
        )
        return None, None, None

    return attn_func, "per_warp", pv_accum_dtype

SAGE_ATTENTION_FUNCTION, QK_QUANT_GRAN, PV_ACCUM_DTYPE = get_sage_attention_function_and_params()


def resolve_sage_target_dtype(q_proj, hidden_states: torch.Tensor) -> torch.dtype:
    """Compute dtype the projections expect ``hidden_states`` in.

    - bnb 4-bit linears carry uint8 weights + ``quant_state``; their compute
      convention here is bfloat16.
    - Quant-resident linears (GGUF raw blocks / ConvRot INT8) dequantize to the
      ACTIVATION dtype per matmul — no cast needed, pass through unchanged.
    - Plain float linears: match the stored weight dtype (legacy behavior).
    """

    if hasattr(q_proj, 'quant_state'):
        return torch.bfloat16
    if getattr(q_proj, '_quant_resident', False):
        return hidden_states.dtype
    return q_proj.weight.dtype


def sage_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    
    if SAGE_ATTENTION_FUNCTION is None:
        raise RuntimeError("SageAttention was selected but no compatible kernel was found for this GPU.")

    # sageattention 2.2.0's `sageattn` takes no attn_mask argument at all: the
    # additive mask transformers builds can be neither forwarded nor folded
    # into the kernel. The `is_causal` line below would still "use" it — and
    # then drop it — so a padded ASR prefill or any right-padded batch would
    # silently attend to the pad columns. Refuse instead of computing the
    # wrong thing; callers that legitimately have a mask must keep this
    # backend off the path (see
    # modules/attention_utils.ASR_EXCLUDED_ATTENTION_MODES).
    if attention_mask is not None:
        raise ValueError(
            "SageAttention received a non-None attention_mask, which it "
            "cannot honour: sageattn() has no attn_mask parameter, and the "
            "mask would be used only to pick `is_causal` and then discarded, "
            "silently letting every query attend to masked-out positions. "
            "Use sdpa or flash_attention_2 for this input."
        )

    original_dtype = hidden_states.dtype

    target_dtype = resolve_sage_target_dtype(self.q_proj, hidden_states)

    if hidden_states.dtype != target_dtype:
        hidden_states = hidden_states.to(target_dtype)

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    # NOTE: Newer transformers versions removed the `position_ids` kwarg from
    # apply_rotary_pos_emb (the RoPE cos/sin are already sliced to the sequence
    # by the parent Qwen2 layer). Passing it raises TypeError. Call positionally.
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # !! DO NOT repeat K and V heads here. The SageAttention kernel is optimized
    # to handle the broadcasting internally.

    # The mask is guaranteed None by the guard at the top of this function,
    # so causality is decided purely by the query length.
    is_causal = q_len > 1
    
    attn_output = SAGE_ATTENTION_FUNCTION(
        query_states.to(target_dtype),
        key_states.to(target_dtype),
        value_states.to(target_dtype),
        tensor_layout="HND",
        is_causal=is_causal,
        qk_quant_gran=QK_QUANT_GRAN,
        pv_accum_dtype=PV_ACCUM_DTYPE,
    )
    
    if isinstance(attn_output, tuple):
        attn_output = attn_output[0] 

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)
    
    attn_output = self.o_proj(attn_output)

    if attn_output.dtype != original_dtype:
        attn_output = attn_output.to(original_dtype)

    attn_weights = None
    
    return attn_output, attn_weights


def set_sage_attention(model):
    """
    Recursively iterates through the model's modules and monkey-patches the
    forward method of each Qwen2Attention layer.
    """
    if not SAGE_ATTENTION_AVAILABLE:
        raise ImportError("SageAttention library is not installed or failed to load.")
    
    if SAGE_ATTENTION_FUNCTION is None:
        return

    for module in model.modules():
        if isinstance(module, Qwen2Attention):
            module.forward = sage_attention_forward.__get__(module, Qwen2Attention)