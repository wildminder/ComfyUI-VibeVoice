"""ConvRot INT8 checkpoint runtime via comfy-kitchen.

Executes safetensors checkpoints carrying ``*.comfy_quant`` metadata (produced
by comfy-model-tools) through comfy-kitchen's INT8 ConvRot kernels:

    weight    : torch.int8, offline-rotated per group (W_rot = W @ H^T)
    scale     : float32 [out_features, 1] per-output-row scale
    comfy_quant: uint8 JSON {"format": "int8_tensorwise", "convrot": true,
                             "convrot_groupsize": G}

Weights stay INT8 resident in VRAM; activations are rotated and dynamically
row-quantized online by :func:`comfy_kitchen.int8_linear`. LOAD-ONLY: this
module executes existing checkpoints, it does not requantize.

No global state; statistics are per-call.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


QUANT_META_SUFFIX = "comfy_quant"
CONVROT_FORMAT = "int8_tensorwise"

# Rowwise (non-rotated) low-bit formats found in comfy-model-tools
# checkpoints. Executed via dequant-at-load (weight * per-row scale).
ROWWISE_FLOAT_FORMATS = {
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}

_ORIG_DTYPE_NAMES = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
}


class UnsupportedQuantFormat(RuntimeError):
    """Checkpoint uses a quant format this nodepack cannot execute.

    Raised instead of silently loading integer/low-bit weights as floats —
    a silent misload produces garbage output far from the root cause.
    """


def resolve_orig_dtype(name):
    dt = _ORIG_DTYPE_NAMES.get(name)
    if dt is None:
        raise UnsupportedQuantFormat(
            f"quant metadata declares orig_dtype={name!r}; supported: "
            f"{sorted(_ORIG_DTYPE_NAMES)}"
        )
    return dt


def assert_convrot_backend() -> str:
    """Fail fast unless comfy-kitchen exposes the ConvRot INT8 capabilities."""
    try:
        import comfy_kitchen
    except ImportError as e:
        raise RuntimeError(
            "ConvRot INT8 checkpoints require the 'comfy_kitchen' package. "
            "Install/upgrade it to run this model."
        ) from e

    required = ("int8_linear", "dequantize_int8_convrot_weight")
    backends = comfy_kitchen.list_backends()
    for name in ("triton", "cuda", "eager"):
        info = backends.get(name)
        if not info or not info.get("available"):
            continue
        caps = set(info.get("capabilities") or ())
        if all(cap in caps for cap in required):
            return name
    raise RuntimeError(
        "comfy_kitchen is installed but no available backend provides the "
        f"ConvRot INT8 capabilities {required}. Backends: {backends}. "
        "Upgrade comfy_kitchen to run this checkpoint."
    )


@dataclass(frozen=True)
class QuantLayerInfo:
    prefix: str
    group_size: int
    in_features: int = 0
    out_features: int = 0
    has_bias: bool = False
    convrot: bool = True
    orig_dtype: str = ""
    rowwise_dtype: "torch.dtype | None" = None
    resident_fp8: bool = False


class ConvRotInt8Linear(nn.Module):
    """Drop-in nn.Linear replacement executing kitchen's INT8 ConvRot path."""

    _quant_resident = True
    comfy_cast_weights = True
    weight_function = []
    bias_function = []

    def __init__(self, in_features: int, out_features: int, bias: bool,
                 group_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.convrot_groupsize = group_size
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.int8),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(out_features, 1, dtype=torch.float32), requires_grad=False
        )
        self.weight_comfy_model_dtype = torch.int8
        self.bias = (
            nn.Parameter(torch.empty(out_features), requires_grad=False)
            if bias else None
        )
        self.quant_format = CONVROT_FORMAT

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        state_dict.pop(f"{prefix}{QUANT_META_SUFFIX}", None)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, x):
        import comfy_kitchen

        if not x.dtype.is_floating_point:
            raise TypeError(
                f"ConvRotInt8Linear expects a floating-point activation, got "
                f"{x.dtype}."
            )

        wf = getattr(self, "weight_function", None)
        if (self.weight.device != x.device) or (wf and len(wf) > 0):
            return self._forward_streamed(x)

        return comfy_kitchen.int8_linear(
            x.contiguous(),
            self.weight,
            self.weight_scale,
            self.bias,
            out_dtype=x.dtype,
            convrot=True,
            convrot_groupsize=self.convrot_groupsize,
        )

    def _forward_streamed(self, x):
        """Paged path: pull the raw int8 weight through core's cast machinery."""
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
            return comfy_kitchen.int8_linear(
                x.contiguous(),
                weight,
                scale,
                bias,
                out_dtype=x.dtype,
                convrot=True,
                convrot_groupsize=self.convrot_groupsize,
            )
        finally:
            comfy.ops.uncast_bias_weight(self, weight, bias, offload_stream)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"convrot_groupsize={self.convrot_groupsize}"
        )


def validate_group_size(group_size: int, in_features: int) -> None:
    """ConvRot group sizes must be powers of four dividing ``in_features``."""
    if group_size < 4 or group_size & (group_size - 1) != 0 \
            or math.log(group_size, 4) % 1 != 0:
        raise ValueError(
            f"ConvRot group size must be a power of four (4/16/64/256/...), "
            f"got {group_size}"
        )
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features {in_features} is not divisible by "
            f"convrot_groupsize {group_size}"
        )


def scan_checkpoint_quantization(weights_path) -> dict:
    """Read every ``*.comfy_quant`` key from a safetensors file."""
    from safetensors import safe_open
    from .fp8_quant import probe_fp8_backend

    fp8_backend = probe_fp8_backend()

    quant_map: dict = {}
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        return quant_map
    with safe_open(str(weights_path), framework="pt", device="cpu") as f:
        key_names = list(f.keys())
        meta_keys = [k for k in key_names if k.endswith(f".{QUANT_META_SUFFIX}")]
        for key in meta_keys:
            meta = json.loads(f.get_tensor(key).numpy().tobytes())
            fmt = meta.get("format")

            if fmt == CONVROT_FORMAT and meta.get("convrot") is True:
                prefix = key[: -len(f".{QUANT_META_SUFFIX}")]
                w_key = f"{prefix}.weight"
                in_f = int(meta.get("in_features", 0))
                out_f = int(meta.get("out_features", 0))
                if (not in_f or not out_f) and w_key in key_names:
                    shape = f.get_slice(w_key).get_shape()
                    out_f, in_f = int(shape[0]), int(shape[1])
                quant_map[prefix] = QuantLayerInfo(
                    prefix=prefix,
                    group_size=int(meta["convrot_groupsize"]),
                    in_features=in_f,
                    out_features=out_f,
                    has_bias=bool(
                        meta.get("has_bias", f"{prefix}.bias" in key_names)
                    ),
                    convrot=True,
                    orig_dtype=str(meta.get("orig_dtype", "")),
                )
                continue

            if fmt == CONVROT_FORMAT:
                prefix = key[: -len(f".{QUANT_META_SUFFIX}")]
                orig = str(meta.get("orig_dtype", ""))
                resolve_orig_dtype(orig)
                quant_map[prefix] = _rowwise_info(
                    f, key_names, prefix, meta,
                    rowwise_dtype=None,
                    orig_dtype=orig,
                )
                continue

            if fmt == "int8_blockwise":
                prefix = key[: -len(f".{QUANT_META_SUFFIX}")]
                orig = str(meta.get("orig_dtype", ""))
                resolve_orig_dtype(orig)
                quant_map[prefix] = _rowwise_info(
                    f, key_names, prefix, meta,
                    rowwise_dtype=None,
                    orig_dtype=orig,
                    group_size=int(meta["group_size"]),
                )
                continue

            if fmt in ROWWISE_FLOAT_FORMATS:
                prefix = key[: -len(f".{QUANT_META_SUFFIX}")]
                orig = str(meta.get("orig_dtype", ""))
                resolve_orig_dtype(orig)
                s_key = f"{prefix}.weight_scale"
                scalar_scale = False
                if fp8_backend is not None and s_key in key_names:
                    s_shape = f.get_slice(s_key).get_shape()
                    scalar_scale = (
                        len(s_shape) == 0
                        or (len(s_shape) == 1 and s_shape[0] == 1)
                    )
                quant_map[prefix] = _rowwise_info(
                    f, key_names, prefix, meta,
                    rowwise_dtype=ROWWISE_FLOAT_FORMATS[fmt],
                    orig_dtype=orig,
                    resident_fp8=scalar_scale,
                )
                continue

            raise UnsupportedQuantFormat(
                f"{weights_path.name}:{key} uses quant format {meta!r}, "
                f"which this nodepack cannot run. Supported: "
                f"'{CONVROT_FORMAT}' with convrot=true (rotated INT8), "
                f"'{CONVROT_FORMAT}' rowwise (plain INT8), "
                f"'int8_blockwise' with group_size (unrotated block scales), "
                f"or {sorted(ROWWISE_FLOAT_FORMATS)} (rowwise fp8)."
            )
    return quant_map


def _rowwise_info(f, key_names, prefix, meta, *, rowwise_dtype, orig_dtype,
                  group_size: int = 0, resident_fp8: bool = False):
    w_key = f"{prefix}.weight"
    in_f = int(meta.get("in_features", 0))
    out_f = int(meta.get("out_features", 0))
    if (not in_f or not out_f) and w_key in key_names:
        shape = f.get_slice(w_key).get_shape()
        out_f, in_f = int(shape[0]), int(shape[1])
    return QuantLayerInfo(
        prefix=prefix,
        group_size=group_size,
        in_features=in_f,
        out_features=out_f,
        has_bias=bool(meta.get("has_bias", f"{prefix}.bias" in key_names)),
        convrot=False,
        orig_dtype=orig_dtype,
        rowwise_dtype=rowwise_dtype,
        resident_fp8=resident_fp8,
    )


def make_convrot_linear(info: QuantLayerInfo):
    """Factory adapter for :func:`replace_linears_for_quant` plans."""
    def _factory(in_features: int, out_features: int, has_bias: bool):
        validate_group_size(info.group_size, in_features)
        return ConvRotInt8Linear(in_features, out_features, has_bias,
                                 info.group_size)
    return _factory