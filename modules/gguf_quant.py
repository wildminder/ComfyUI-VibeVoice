"""GGUF quant-resident runtime primitives.

Torch-native GGML block-dequantization kernels plus the tensor/key-mapping
plumbing needed to keep GGUF weights in RAW-BLOCK form end-to-end:

    disk (mmap) -> raw uint8 blocks -> resident nn.Parameter (uint8)
    -> H2D transfer of raw bytes -> per-matmul dequantization to x.dtype

This removes the historical load-time RAM spike where every Q8_0/K-quant
block was expanded to a fresh fp32 array (~2x full-float peak) and the whole
model transferred to VRAM at full float size.

Dequantization kernels are ports of ``gguf-quants`` (gguf-py) semantics and
are pinned BITWISE to that package by tests (tests/test_gguf_quant_blocks.py)
— the installed gguf package acts as the parity oracle.

Supported quant matrix (v1): Q8_0, Q4_K, Q5_K, Q6_K.
Float passthrough: F32, F16, BF16 (zero-copy torch views where possible).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

# ====================================================================
# 0. Type matrix
# ====================================================================


def _ggml_types():
    from gguf.constants import GGMLQuantizationType as T

    return T


class UnsupportedGGMLType(TypeError):
    """Raised for GGML tensor types outside the supported execution matrix."""

    def __init__(self, ggml_type, tensor_name: str = ""):
        self.ggml_type = ggml_type
        self.tensor_name = tensor_name
        self.supported = sorted(t.name for t in SUPPORTED_GGML_TYPES)
        super().__init__(
            f"GGUF tensor '{tensor_name}' uses GGML type "
            f"{getattr(ggml_type, 'name', ggml_type)}, which is not supported. "
            f"Supported types: {', '.join(self.supported)}. Re-convert the "
            f"model with one of these quants (e.g. Q8_0 / Q4_K_M) or use the "
            f"safetensors checkpoint."
        )


def _build_matrices():
    T = _ggml_types()
    quant = frozenset({T.Q8_0, T.Q4_K, T.Q5_K, T.Q6_K})
    floats = frozenset({T.F32, T.F16, T.BF16})
    return T, quant, floats


_T, SUPPORTED_GGML_TYPES, FLOAT_GGML_TYPES = _build_matrices()

# GGML_QUANT_SIZES[qtype] = (block_size, type_size)
_BLOCK_SIZES = {}


def _block_size(qtype) -> int:
    from gguf.constants import GGML_QUANT_SIZES

    return GGML_QUANT_SIZES[qtype][0]


def _type_size(qtype) -> int:
    from gguf.constants import GGML_QUANT_SIZES

    return GGML_QUANT_SIZES[qtype][1]


def expected_numel(n_elements: int, ggml_type) -> int:
    """Raw BYTES needed to store ``n_elements`` values as ``ggml_type``.

    Pinned against real GGUFReader byte counts in the groundwork tests.
    """
    bs, ts = _block_size(ggml_type), _type_size(ggml_type)
    n_blocks = n_elements // bs
    if n_elements % bs != 0:
        raise ValueError(
            f"{ggml_type.name} requires the last tensor dimension to be a "
            f"multiple of {bs}, got n_elements={n_elements}"
        )
    return n_blocks * ts


# ====================================================================
# 1. Dequantization kernels (bitwise ports of gguf-py quants.py)
# ====================================================================

# Each kernel receives ``blocks``: a contiguous uint8 CPU/GPU tensor shaped
# (n_blocks, type_size) and returns an fp32/int tensor shaped
# (n_blocks, block_size). All integer math is exact; float math mirrors the
# numpy expression trees op-for-op so results are BITWISE equal.


def _u(tensor: torch.Tensor) -> torch.Tensor:
    """View raw bytes as uint8 (no copy)."""
    return tensor.view(torch.uint8) if tensor.dtype != torch.uint8 else tensor


def _f16_scale(bytes_u8: torch.Tensor) -> torch.Tensor:
    """Interpret a (n, 2) uint8 field as float16, widen to float32."""
    return bytes_u8.contiguous().view(torch.float16).to(torch.float32)


def _dequant_q8_0(blocks: torch.Tensor) -> torch.Tensor:
    d, x = torch.split(blocks, [2, 32], dim=-1)
    d = _f16_scale(d)
    x = x.contiguous().view(torch.int8).to(torch.float32)
    return x * d


def _get_scale_min_k(scales_u8: torch.Tensor):
    """Unpack Q4_K/Q5_K 6-bit scale/min pairs (port of Q4_K.get_scale_min).

    Input: (n_blocks, 12) uint8. Returns (sc, mn): (n_blocks, 8) int tensors.
    """
    n = scales_u8.shape[0]
    s = scales_u8.reshape(n, 3, 4)
    d, m, m_d = torch.split(s, 1, dim=-2)  # each (n, 1, 4)
    sc = torch.cat(
        [d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1
    ).reshape(n, 8)
    mn = torch.cat(
        [m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1
    ).reshape(n, 8)
    return sc, mn


_QK_K = 256


def _dequant_q4_k(blocks: torch.Tensor) -> torch.Tensor:
    n = blocks.shape[0]
    d_b, rest = torch.split(blocks, [2, 142], dim=-1)
    dmin_b, rest = torch.split(rest, [2, 140], dim=-1)
    scales, qs = torch.split(rest, [12, 128], dim=-1)

    d = _f16_scale(d_b)
    dmin = _f16_scale(dmin_b)

    sc, mn = _get_scale_min_k(scales)

    d = (d * sc.to(torch.float32)).reshape(n, -1, 1)
    dm = (dmin * mn.to(torch.float32)).reshape(n, -1, 1)

    # (n, 8, 1, 32) >> [0, 4] -> nibbles
    qs = qs.reshape(n, -1, 1, 32)
    shifts = torch.tensor([0, 4], dtype=torch.uint8, device=blocks.device).reshape(1, 1, 2, 1)
    qs = ((qs >> shifts) & 0x0F).reshape(n, -1, 32).to(torch.float32)

    return (d * qs - dm).reshape(n, _QK_K)


def _dequant_q5_k(blocks: torch.Tensor) -> torch.Tensor:
    n = blocks.shape[0]
    d_b, rest = torch.split(blocks, [2, 174], dim=-1)
    dmin_b, rest = torch.split(rest, [2, 172], dim=-1)
    scales, rest = torch.split(rest, [12, 160], dim=-1)
    qh, qs = torch.split(rest, [32, 128], dim=-1)

    d = _f16_scale(d_b)
    dmin = _f16_scale(dmin_b)

    sc, mn = _get_scale_min_k(scales)

    d = (d * sc.to(torch.float32)).reshape(n, -1, 1)
    dm = (dmin * mn.to(torch.float32)).reshape(n, -1, 1)

    shifts4 = torch.tensor([0, 4], dtype=torch.uint8, device=blocks.device).reshape(1, 1, 2, 1)
    shifts8 = torch.tensor([i for i in range(8)], dtype=torch.uint8, device=blocks.device).reshape(1, 1, 8, 1)

    ql = ((qs.reshape(n, -1, 1, 32) >> shifts4) & 0x0F).reshape(n, -1, 32)
    qh = ((qh.reshape(n, -1, 1, 32) >> shifts8) & 0x01).reshape(n, -1, 32)
    q = (ql | (qh << 4)).to(torch.float32)

    return (d * q - dm).reshape(n, _QK_K)


def _dequant_q6_k(blocks: torch.Tensor) -> torch.Tensor:
    n = blocks.shape[0]
    ql_b, rest = torch.split(blocks, [128, 82], dim=-1)
    qh_b, rest = torch.split(rest, [64, 18], dim=-1)
    scales_b, d_b = torch.split(rest, [16, 2], dim=-1)

    scales = scales_b.contiguous().view(torch.int8).to(torch.float32)
    d = _f16_scale(d_b)
    d = (d * scales).reshape(n, _QK_K // 16, 1)

    shifts2 = torch.tensor([0, 4], dtype=torch.uint8, device=blocks.device).reshape(1, 1, 2, 1)
    shifts4 = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=blocks.device).reshape(1, 1, 4, 1)

    ql = ((ql_b.reshape(n, -1, 1, 64) >> shifts2) & 0x0F).reshape(n, -1, 32)
    qh = ((qh_b.reshape(n, -1, 1, 32) >> shifts4) & 0x03).reshape(n, -1, 32)
    # (ql | qh << 4) is a 6-bit value (0..63); replicate gguf-py's int8
    # arithmetic exactly: reinterpret as int8, subtract 32 (wrapping).
    six = (ql | (qh << 4)).to(torch.uint8).view(torch.int8)
    q = (six - 32).to(torch.int8)
    q = q.reshape(n, _QK_K // 16, -1).to(torch.float32)

    return (d * q).reshape(n, _QK_K)


def _kernel_for(ggml_type):
    T = _T
    if ggml_type == T.Q8_0:
        return _dequant_q8_0
    if ggml_type == T.Q4_K:
        return _dequant_q4_k
    if ggml_type == T.Q5_K:
        return _dequant_q5_k
    if ggml_type == T.Q6_K:
        return _dequant_q6_k
    return None


def dequantize_blocks(raw: torch.Tensor, ggml_type, out_dtype: torch.dtype, shape) -> torch.Tensor:
    """Dequantize raw GGML blocks to ``out_dtype`` with logical ``shape``.

    Args:
        raw: uint8 tensor holding ``n_blocks * type_size`` bytes (any device).
        ggml_type: GGMLQuantizationType member (must be in SUPPORTED_GGML_TYPES).
        out_dtype: Target torch dtype (floating point).
        shape: Logical tensor shape, e.g. ``(out_features, in_features)``.

    Returns:
        Dequantized tensor of ``shape`` in ``out_dtype``.
    """
    kernel = _kernel_for(ggml_type)
    if kernel is None:
        raise UnsupportedGGMLType(ggml_type, tensor_name="<raw blocks>")

    if raw.dtype != torch.uint8:
        raw = raw.view(torch.uint8)
    n_bytes = raw.numel()
    ts = _type_size(ggml_type)
    if n_bytes % ts != 0:
        raise ValueError(
            f"Raw block count {n_bytes} is not a multiple of the "
            f"{ggml_type.name} type size {ts}"
        )
    blocks = raw.reshape(-1, ts)
    out = kernel(blocks)
    return out.reshape(shape).to(out_dtype)


def dequantize_dense(raw_or_view: torch.Tensor, ggml_type) -> torch.Tensor:
    """Materialize a FLOAT ggml tensor (F32/F16/BF16) as a native torch tensor.

    Zero-copy where possible: F32 returns a view; F16/BF16 reinterpret the
    underlying bytes (both are exact bit patterns for their torch dtypes).
    """
    T = _T
    if ggml_type == T.F32:
        t = raw_or_view
        if raw_or_view.dtype == torch.uint8:
            t = raw_or_view.view(torch.float32)
        return t
    if ggml_type == T.F16:
        if raw_or_view.dtype == torch.uint8:
            return raw_or_view.view(torch.float16)
        return raw_or_view.view(torch.float16) if raw_or_view.dtype != torch.float16 else raw_or_view
    if ggml_type == T.BF16:
        src = raw_or_view
        if src.dtype != torch.uint8:
            src = src.contiguous().view(torch.uint8)
        return src.view(torch.bfloat16)
    raise UnsupportedGGMLType(ggml_type, tensor_name="<dense>")


# ====================================================================
# 2. Raw-block tensor container
# ====================================================================


@dataclass
class GGUFTensor:
    """A GGUF tensor kept in RAW BLOCK FORM (no float materialization).

    ``raw`` holds the exact on-disk block bytes as a contiguous uint8 tensor;
    ``shape`` is the logical (dequantized) shape and ``ggml_type`` the format.
    """

    raw: torch.Tensor
    ggml_type: object
    shape: tuple

    @classmethod
    def from_reader_tensor(cls, reader_tensor) -> "GGUFTensor":
        """Build from a ``gguf.GGUFReader`` tensor (zero float materialization).

        The mmap-backed numpy buffer is copied once into owned uint8 storage —
        this copy IS the final residency, so peak RAM stays ~file size.

        NOTE on shapes: GGUF stores dimensions slowest-first (ggml ``ne``
        order), so ``reader_tensor.shape`` is the REVERSE of the torch
        logical shape (e.g. an ``(out, in)`` torch weight reads back as
        ``(in, out)``). The stored ``shape`` here is the TORCH-logical one;
        plan-time validation against the target module catches any deviation.
        """
        import numpy as np

        data = reader_tensor.data
        # Reader data is an mmap-backed READ-ONLY buffer; torch.from_numpy
        # warns on non-writable arrays even when the tensor is immediately
        # cloned afterwards. Materialize owned writable storage up front
        # instead — this single copy IS the final residency.
        arr = np.ascontiguousarray(data)
        if arr.flags.writeable:
            # Writable but still reader-owned (test doubles): clone on the
            # torch side so the tensor never aliases the reader buffer.
            raw = torch.from_numpy(arr).clone()
        else:
            raw = torch.from_numpy(arr.copy())
        if data.dtype != np.uint8:
            raw = raw.view(torch.uint8)
        shape = tuple(int(s) for s in reversed(reader_tensor.shape))
        return cls(raw=raw, ggml_type=reader_tensor.tensor_type, shape=shape)

    def to(self, device, *, dtype=None) -> "GGUFTensor":
        """Move the RAW bytes to ``device`` (dtype changes are rejected)."""
        if dtype is not None:
            raise ValueError(
                "GGUFTensor.to() moves raw bytes only; dtype conversion would "
                "break quant residency. Use dequantize_blocks() instead."
            )
        return GGUFTensor(raw=self.raw.to(device), ggml_type=self.ggml_type, shape=self.shape)

    @property
    def n_bytes(self) -> int:
        return self.raw.numel()

    def dequantize(self, out_dtype: torch.dtype) -> torch.Tensor:
        return dequantize_blocks(self.raw, self.ggml_type, out_dtype, self.shape)


# ====================================================================
# 2b. Resident linear module
# ====================================================================


class GGUFLinear(torch.nn.Module):
    """Drop-in ``nn.Linear`` replacement holding RAW GGML block bytes.

    The weight is stored as a uint8 ``nn.Parameter`` of ``expected_numel``
    bytes; ``forward`` dequantizes it to the activation dtype per call
    (the standard quant-resident tradeoff — residency ~file size instead of
    full float size).

    Meta-safe: safe to construct under a ``torch.device("meta")`` context;
    :meth:`set_raw_weight` installs real storage afterwards.
    """

    # Dtype-cast filter marker: raw uint8 storage must never be recast.
    _quant_resident = True
    # Native comfy streaming (plan 2026-08-26): core may offload this module;
    # forward pulls raw bytes back through cast_bias_weight.
    comfy_cast_weights = True
    weight_function = []
    bias_function = []

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 ggml_type=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ggml_type = ggml_type
        n_bytes = expected_numel(in_features * out_features, ggml_type)
        # Uninitialized storage placeholder; replaced by set_raw_weight().
        self.weight = torch.nn.Parameter(
            torch.empty(n_bytes, dtype=torch.uint8), requires_grad=False
        )
        self.bias = (
            torch.nn.Parameter(torch.empty(out_features), requires_grad=False)
            if bias else None
        )
        # Documentation/core-interop marker: raw uint8 blocks are moved,
        # never recast (our streamed forward handles pulls itself).
        self.weight_comfy_model_dtype = torch.uint8
        self._gguf: GGUFTensor | None = None
        # Reserved seam for future scratch-buffer reuse; intentionally unused.
        self.weight_scratch_cache = None

    def set_raw_weight(self, raw_uint8: torch.Tensor) -> None:
        """Install raw block bytes as the weight parameter storage."""
        if raw_uint8.dtype != torch.uint8:
            raw_uint8 = raw_uint8.view(torch.uint8)
        if raw_uint8.numel() != self.weight.numel():
            raise ValueError(
                f"Raw byte count mismatch for GGUFLinear: got "
                f"{raw_uint8.numel()}, expected {self.weight.numel()} "
                f"(ggml_type={getattr(self.ggml_type, 'name', self.ggml_type)}, "
                f"shape=({self.out_features}, {self.in_features}))"
            )
        self.weight = torch.nn.Parameter(raw_uint8.contiguous(), requires_grad=False)
        self._gguf = GGUFTensor(
            raw=self.weight.data, ggml_type=self.ggml_type,
            shape=(self.out_features, self.in_features),
        )

    def forward(self, x):
        if not x.dtype.is_floating_point:
            # A non-float activation here (e.g. a caller casting hidden states
            # to the raw weight dtype) would silently dequantize INTO byte
            # garbage via out_dtype=x.dtype — fail loudly instead.
            raise TypeError(
                f"GGUFLinear expects a floating-point activation, got "
                f"{x.dtype}. Callers must not cast hidden states to the "
                f"raw weight dtype."
            )

        wf = getattr(self, "weight_function", None)
        if (self.weight.device != x.device) or (wf and len(wf) > 0):
            return self._forward_streamed(x)

        w = dequantize_blocks(
            self.weight, self.ggml_type, x.dtype,
            (self.out_features, self.in_features),
        )
        return torch.nn.functional.linear(x, w, self.bias)

    def _forward_streamed(self, x):
        """Lowvram path: raw blocks were offloaded; pull them back
        dtype-preservingly.

        NOTE: deliberately NOT comfy.ops.cast_bias_weight — it recasts the
        weight to the ACTIVATION dtype whenever they differ
        (ops.py: "if weight_has_function or weight.dtype != dtype:
        weight = weight.to(dtype=dtype)"), which would corrupt raw byte
        blocks. Core's own partial-unload contract attaches LowVramPatch
        callables whose job is exactly "move tensor to device, keep dtype";
        we honor those directly."""
        w_raw = self._pull_to_device(self.weight, x.device)
        bias = self.bias
        if bias is not None and bias.device != x.device:
            bias = bias.to(x.device)
        w = dequantize_blocks(
            w_raw, self.ggml_type, x.dtype,
            (self.out_features, self.in_features),
        )
        return torch.nn.functional.linear(x, w, bias)

    def _pull_to_device(self, tensor, device):
        if tensor.device != device:
            tensor = tensor.to(device)
        # Honor core's LowVramPatch-style callables (move-to-device,
        # dtype-preserving) exactly as cast_bias_weight would.
        for fn in (getattr(self, "weight_function", None) or ()):
            tensor = fn(tensor)
        if tensor.device != device:
            tensor = tensor.to(device)
        return tensor

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, ggml_type="
            f"{getattr(self.ggml_type, 'name', self.ggml_type)}, "
            f"raw_bytes={getattr(self.weight, 'numel', lambda: 0)()}"
        )


def make_gguf_linear(target: torch.nn.Linear, ggml_type) -> GGUFLinear:
    """Build a GGUFLinear matching an existing nn.Linear's shape/bias."""
    return GGUFLinear(
        target.in_features, target.out_features,
        bias=target.bias is not None, ggml_type=ggml_type,
    )


def gguf_linear_factory(ggml_type):
    """Factory adapter for :func:`replace_linears_for_quant` plans."""
    def _factory(in_features: int, out_features: int, has_bias: bool):
        return GGUFLinear(in_features, out_features, bias=has_bias,
                          ggml_type=ggml_type)
    return _factory


# ====================================================================
# 3. Key-scheme detection and mapping
# ====================================================================

class UnmappedKeyError(ValueError):
    """Raised when GGUF tensor names cannot be mapped onto the model tree."""

    def __init__(self, unknown_keys, scheme: str):
        self.unknown_keys = list(unknown_keys)
        preview = ", ".join(self.unknown_keys[:10])
        super().__init__(
            f"{len(self.unknown_keys)} GGUF tensor key(s) could not be mapped "
            f"onto the VibeVoice module tree (scheme='{scheme}'). First "
            f"unknown keys: [{preview}]. Check that the sidecar config "
            f"matches the checkpoint architecture; if the converter used a "
            f"newer naming scheme, extend the mapping table in "
            f"modules/gguf_quant.py."
        )


_LLAMACPP_TO_HF = {
    "tok_embeddings.weight": "model.language_model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.language_model.norm.weight",
}

_LLAMACPP_LAYER_PATTERNS = {
    "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}

import re as _re

import re as _re

_LLAMACPP_LAYER_RE = _re.compile(r"^blk\.(\d+)\.(.+)$")


def detect_key_scheme(keys) -> str:
    """Classify GGUF tensor names as 'hf', 'llamacpp', or 'mixed'.

    'hf': names already match the VibeVoice/HF module tree
    ('model.'/'lm_head.' prefixes, proj/norm suffixes).
    'llamacpp': names follow llama.cpp conversion conventions
    ('blk.N.', 'tok_embeddings.', 'output.').
    """
    hf_re = _re.compile(r"^(model\.|lm_head\.|transformer\.)")
    llamacpp_re = _re.compile(r"^(blk\.\d+\.|tok_embeddings\.|output\.|output_norm\.)")
    votes = {"hf": 0, "llamacpp": 0}
    for k in keys:
        if hf_re.match(k):
            votes["hf"] += 1
        elif llamacpp_re.match(k):
            votes["llamacpp"] += 1
    if votes["llamacpp"] and votes["hf"]:
        return "mixed"
    if votes["llamacpp"]:
        return "llamacpp"
    return "hf"


def _map_llamacpp_key(key: str) -> str:
    m = _LLAMACPP_LAYER_RE.match(key)
    if m:
        idx, rest = m.group(1), m.group(2)
        # Strict suffix whitelist: unknown sub-tensors inside a block must
        # hard-fail (with the key reported) rather than silently map.
        if rest in _LLAMACPP_LAYER_PATTERNS:
            return f"model.language_model.layers.{idx}.{_LLAMACPP_LAYER_PATTERNS[rest]}"
        for suffix, hf_suffix in _LLAMACPP_LAYER_PATTERNS.items():
            if rest.startswith(suffix + "."):
                tail = rest[len(suffix) + 1:]
                if tail in ("weight", "bias"):
                    return (
                        f"model.language_model.layers.{idx}.{hf_suffix}.{tail}"
                    )
        raise KeyError(key)
    if key in _LLAMACPP_TO_HF:
        return _LLAMACPP_TO_HF[key]
    raise KeyError(key)


def map_keys(keys, scheme: str = None) -> dict:
    """Map GGUF tensor keys onto VibeVoice module-tree parameter paths.

    Args:
        keys: Iterable of GGUF tensor names.
        scheme: Force a scheme; auto-detected when omitted. 'mixed' raises.

    Returns:
        dict mapping ORIGINAL key -> module parameter path (identity for hf).

    Raises:
        UnmappedKeyError: On unmapped llamacpp keys or a mixed census.
    """
    scheme = scheme or detect_key_scheme(keys)
    if scheme == "mixed":
        raise UnmappedKeyError(sorted(keys)[:10], scheme)
    mapping = {}
    unknown = []
    for k in keys:
        if scheme == "hf":
            mapping[k] = k
            continue
        try:
            mapping[k] = _map_llamacpp_key(k)
        except KeyError:
            unknown.append(k)
    if unknown:
        raise UnmappedKeyError(unknown, scheme)
    return mapping
