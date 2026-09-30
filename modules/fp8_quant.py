"""FP8 rowwise checkpoint runtime via comfy-kitchen.

Executes safetensors checkpoints carrying ``*.comfy_quant`` metadata with a
rowwise fp8 format and a SCALAR per-tensor scale (produced by
comfy-model-tools):

    weight      : torch.float8_e4m3fn / float8_e5m2 [out_features, in_features]
    weight_scale: float32 scalar
    comfy_quant : uint8 JSON {"format": "float8_e4m3fn",
                              "orig_dtype": "torch.bfloat16"}

Weights stay FP8 resident in VRAM (1 byte/element); each forward dequantizes
the weight per-tensor through :func:`comfy_kitchen.dequantize_per_tensor_fp8`
and runs a plain ``F.linear``. LOAD-ONLY: this module executes existing
checkpoints, it does not requantize.

Resident contract identical to
:class:`~modules.convrot_quant.ConvRotInt8Linear` (plan 2026-08-27
fp8-resident + streaming load): the ``_quant_resident`` marker excludes the
fp8 storage + fp32 scale from bulk dtype casts and from the streaming-tree
conversion; ``comfy_cast_weights`` tells core this module tolerates being
offloaded, and the streamed forward pulls the fp8 storage back
dtype-preservingly (never through ``cast_bias_weight``, which would recast
the storage to the activation dtype).
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.nn import functional as F

from .convrot_quant import (
    QUANT_META_SUFFIX,
    ROWWISE_FLOAT_FORMATS,
    UnsupportedQuantFormat,
    resolve_orig_dtype,
)

logger = logging.getLogger(__name__)

FP8_DEQUANT_CAPABILITY = "dequantize_per_tensor_fp8"

# Reverse lookup: torch fp8 dtype -> comfy_quant format name.
_FP8_FORMAT_NAMES = {v: k for k, v in ROWWISE_FLOAT_FORMATS.items()}


def probe_fp8_backend():
    """Return the comfy-kitchen backend providing per-tensor fp8 dequant.

    Soft probe (returns ``None`` instead of raising): fp8-resident execution
    is an optimization — when no backend provides the kernel the loader
    falls back to dequant-at-load (plan 2026-08-27, D3), which stays
    correct at the cost of RAM/VRAM residency.

    A backend qualifies only when it is available, advertises the
    ``dequantize_per_tensor_fp8`` capability, AND the top-level callable
    actually exists — capability strings have shipped without callables
    before (the GGUF plan's D-3 trap), so both are checked.

    Returns:
        ``"triton"`` / ``"cuda"`` / ``"eager"``, or ``None`` when no
        available backend can execute fp8 dequantization.
    """
    try:
        import comfy_kitchen
    except ImportError:
        return None
    if not callable(getattr(comfy_kitchen, FP8_DEQUANT_CAPABILITY, None)):
        return None
    backends = comfy_kitchen.list_backends()
    for name in ("triton", "cuda", "eager"):
        info = backends.get(name)
        if not info or not info.get("available"):
            continue
        if FP8_DEQUANT_CAPABILITY in set(info.get("capabilities") or ()):
            return name
    return None


class FP8Linear(nn.Module):
    """Drop-in nn.Linear replacement executing fp8 storage with a scalar scale.

    The fp8 weight and the fp32 scalar scale stay resident at their storage
    dtypes; dequantization happens per forward call into the activation
    dtype, so a 7B fp8 model occupies ~1 byte/weight in VRAM instead of the
    2 bytes/weight a dequant-at-load bf16 model needs.
    """

    # Dtype-cast filter marker: fp8 weight + fp32 scale must never be recast.
    _quant_resident = True
    # Native comfy streaming: core may offload this module; forward pulls the
    # fp8 weight back dtype-preservingly.
    comfy_cast_weights = True
    weight_function = []
    bias_function = []

    def __init__(self, in_features: int, out_features: int, bias: bool,
                 fp8_dtype: torch.dtype, compute_dtype: torch.dtype = None):
        super().__init__()
        if fp8_dtype not in ROWWISE_FLOAT_FORMATS.values():
            raise ValueError(
                f"FP8Linear requires an fp8 dtype from "
                f"{sorted(_FP8_FORMAT_NAMES.values())}, got {fp8_dtype}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.fp8_dtype = fp8_dtype
        # Dtype callers should cast ACTIVATIONS to before feeding this module
        # (the checkpoint's orig dtype, e.g. bf16). ``.weight.dtype`` is the
        # fp8 STORAGE dtype and must never be used to derive an activation
        # dtype — vendored code that aligns inputs to "the mlp's dtype"
        # reads this attribute first (diffusion-head TimestepEmbedder).
        self.compute_dtype = compute_dtype
        # Meta-safe: parameters start empty-shaped correctly; storage is
        # assigned later by the loader (assign semantics).
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=fp8_dtype),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty((), dtype=torch.float32), requires_grad=False
        )
        # Documentation/core-interop marker: fp8 storage is moved, never
        # recast (our streamed forward handles pulls itself).
        self.weight_comfy_model_dtype = fp8_dtype
        self.bias = (
            nn.Parameter(torch.empty(out_features), requires_grad=False)
            if bias else None
        )
        self.quant_format = _FP8_FORMAT_NAMES[fp8_dtype]

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # comfy_quant is metadata, not a tensor parameter; consume it here so
        # it never surfaces as an unexpected key.
        state_dict.pop(f"{prefix}{QUANT_META_SUFFIX}", None)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, x):
        import comfy_kitchen

        if not x.dtype.is_floating_point:
            raise TypeError(
                f"FP8Linear expects a floating-point activation, got {x.dtype}."
            )
        if x.dtype in _FP8_FORMAT_NAMES:
            # fp8 IS floating point, so the check above passes — guard
            # explicitly: upstream code cast the activation to the quantized
            # WEIGHT storage dtype. Dequantizing into fp8 is unsupported and
            # wrong; activations must stay in the model compute dtype.
            raise TypeError(
                f"FP8Linear received fp8 activations ({x.dtype}). Upstream "
                f"code derived an activation dtype from the quantized weight "
                f"storage; use the module's compute_dtype "
                f"({self.compute_dtype}) instead of weight.dtype."
            )

        wf = getattr(self, "weight_function", None)
        if (self.weight.device != x.device) or (wf and len(wf) > 0):
            return self._forward_streamed(x)

        w = comfy_kitchen.dequantize_per_tensor_fp8(
            self.weight, self.weight_scale, x.dtype
        )
        bias = self.bias
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(x.dtype)
        return F.linear(x, w, bias)

    def _forward_streamed(self, x):
        """Paged path: pull the raw fp8 weight through core's cast machinery.

        2026-09-30: this used to move the weight with a bare
        ``self.weight.to(x.device)`` because ``cast_bias_weight`` "recasts to
        the activation dtype" — true only of its DEFAULT form, where
        ``dtype`` is derived from ``input.dtype`` (comfy/ops.py:344-349).
        Core's own quantized layers pass the weight's dtype explicitly
        (``CastBiasWeightContext(self, device=input.device,
        dtype=weight.dtype, offloadable=True)``, comfy/ops.py:1722), so
        passing ``dtype=self.weight_comfy_model_dtype`` (fp8) keeps the raw
        bytes intact AND, under ModelPatcherDynamic, hands back the vbar
        window so the weight is paged disk->VRAM (comfy/model_patcher.py:1993,
        comfy/memory_management.py:18) instead of being H2D-copied whole on
        every offloaded forward.

        ``bias_dtype`` is passed separately (comfy/ops.py:350-351) so the
        bias keeps the ACTIVATION dtype; defaulting it to the weight dtype
        would cast bf16 to fp8. ``offloadable=True`` + ``uncast_bias_weight``
        is core's documented contract (comfy/ops.py:341-343); the uncast is a
        stream sync / vbar unpin only (comfy/ops.py:444-462) — it does not
        move weights back, so the legacy route does not thrash.
        """
        import comfy.ops
        import comfy_kitchen

        weight, bias, offload_stream = comfy.ops.cast_bias_weight(
            self,
            x,
            device=x.device,
            dtype=self.weight_comfy_model_dtype,
            bias_dtype=x.dtype,
            offloadable=True,
        )
        try:
            scale = (self.weight_scale if self.weight_scale.device == x.device
                     else self.weight_scale.to(x.device))
            if bias is not None and bias.dtype != x.dtype:
                bias = bias.to(x.dtype)
            w = comfy_kitchen.dequantize_per_tensor_fp8(weight, scale, x.dtype)
            return F.linear(x, w, bias)
        finally:
            comfy.ops.uncast_bias_weight(self, weight, bias, offload_stream)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, fp8_dtype={self.fp8_dtype}"
        )


def make_fp8_linear(info):
    """Factory adapter for :func:`replace_linears_for_quant` plans.

    Args:
        info: :class:`~modules.convrot_quant.QuantLayerInfo` whose
            ``rowwise_dtype`` is the checkpoint's fp8 storage dtype.
    """
    fp8_dtype = info.rowwise_dtype
    try:
        compute_dtype = resolve_orig_dtype(info.orig_dtype)
    except UnsupportedQuantFormat:
        compute_dtype = None

    def _factory(in_features: int, out_features: int, has_bias: bool):
        return FP8Linear(in_features, out_features, has_bias, fp8_dtype,
                         compute_dtype)

    return _factory
