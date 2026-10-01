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
import os
import re
from collections import OrderedDict
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

# Files already announced with the flat-pool recovery warning this session
# (the loader opens one GGUF several times; warn once per file, not per open).
_FLAT_POOL_WARNED = set()

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

def _u(tensor: torch.Tensor) -> torch.Tensor:
    """View raw bytes as uint8 (no copy)."""
    return tensor.view(torch.uint8) if tensor.dtype != torch.uint8 else tensor


_SHIFT_SPECS = {
    "nibble2": (0, 4),
    "bits8": tuple(range(8)),
    "shifts2": (0, 4),
    "shifts4": (0, 2, 4, 6),
}
_SHIFT_CACHE: dict[tuple[str, torch.device], torch.Tensor] = {}


def _shift_const(name: str, device: torch.device) -> torch.Tensor:
    """Broadcastable (1, 1, n, 1) uint8 shift vector, cached per device."""
    key = (name, device)
    cached = _SHIFT_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(
            _SHIFT_SPECS[name], dtype=torch.uint8, device=device
        ).reshape(1, 1, -1, 1)
        _SHIFT_CACHE[key] = cached
    return cached


def _reinterpret(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Reinterpret ``t``'s bytes as ``dtype``, copying only if forced to."""
    if t.stride(-1) != 1:
        t = t.contiguous()
    return t.view(dtype)


def _f16_scale(
    bytes_u8: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Interpret a (n, 2) uint8 field as float16, widen to ``dtype``."""
    return _reinterpret(bytes_u8, torch.float16).to(dtype)


def _dequant_q8_0(
    blocks: torch.Tensor, out_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Q8_0: ``x * d``."""
    d, x = torch.split(blocks, [2, 32], dim=-1)
    d = _f16_scale(d, out_dtype)
    x = _reinterpret(x, torch.int8).to(out_dtype)
    return x * d


def _get_scale_min_k(scales_u8: torch.Tensor):
    """Unpack Q4_K/Q5_K 6-bit scale/min pairs."""
    n = scales_u8.shape[0]
    s = scales_u8.reshape(n, 3, 4)
    d, m, m_d = torch.split(s, 1, dim=-2)
    sc = torch.cat(
        [d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1
    ).reshape(n, 8)
    mn = torch.cat(
        [m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1
    ).reshape(n, 8)
    return sc, mn


_QK_K = 256


def _dequant_q4_k(
    blocks: torch.Tensor, out_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    n = blocks.shape[0]
    d_b, rest = torch.split(blocks, [2, 142], dim=-1)
    dmin_b, rest = torch.split(rest, [2, 140], dim=-1)
    scales, qs = torch.split(rest, [12, 128], dim=-1)

    d = _f16_scale(d_b)
    dmin = _f16_scale(dmin_b)

    sc, mn = _get_scale_min_k(scales)

    d = (d * sc.to(torch.float32)).reshape(n, -1, 1)
    dm = (dmin * mn.to(torch.float32)).reshape(n, -1, 1)

    qs = qs.reshape(n, -1, 1, 32)
    shifts = _shift_const("nibble2", blocks.device)
    qs = ((qs >> shifts) & 0x0F).reshape(n, -1, 32).to(torch.float32)

    return (d * qs - dm).reshape(n, _QK_K)


def _dequant_q5_k(
    blocks: torch.Tensor, out_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
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

    shifts4 = _shift_const("nibble2", blocks.device)
    shifts8 = _shift_const("bits8", blocks.device)

    ql = ((qs.reshape(n, -1, 1, 32) >> shifts4) & 0x0F).reshape(n, -1, 32)
    qh = ((qh.reshape(n, -1, 1, 32) >> shifts8) & 0x01).reshape(n, -1, 32)
    q = (ql | (qh << 4)).to(torch.float32)

    return (d * q - dm).reshape(n, _QK_K)


def _dequant_q6_k(
    blocks: torch.Tensor, out_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    n = blocks.shape[0]
    ql_b, rest = torch.split(blocks, [128, 82], dim=-1)
    qh_b, rest = torch.split(rest, [64, 18], dim=-1)
    scales_b, d_b = torch.split(rest, [16, 2], dim=-1)

    scales = scales_b.contiguous().view(torch.int8).to(torch.float32)
    d = _f16_scale(d_b)
    d = (d * scales).reshape(n, _QK_K // 16, 1)

    shifts2 = _shift_const("shifts2", blocks.device)
    shifts4 = _shift_const("shifts4", blocks.device)

    ql = ((ql_b.reshape(n, -1, 1, 64) >> shifts2) & 0x0F).reshape(n, -1, 32)
    qh = ((qh_b.reshape(n, -1, 1, 32) >> shifts4) & 0x03).reshape(n, -1, 32)
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


_NATIVE_DTYPES: frozenset = frozenset()
_NATIVE_COMPUTE_DTYPES = frozenset({torch.bfloat16, torch.float16})


def _compute_dtype_for(ggml_type, out_dtype: torch.dtype) -> torch.dtype:
    """Dtype the kernel itself should compute in."""
    if ggml_type in _NATIVE_DTYPES and out_dtype in _NATIVE_COMPUTE_DTYPES:
        return out_dtype
    return torch.float32


def dequantize_blocks(raw: torch.Tensor, ggml_type, out_dtype: torch.dtype, shape) -> torch.Tensor:
    """Dequantize raw GGML blocks to ``out_dtype`` with logical ``shape``."""
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
    out = kernel(blocks, _compute_dtype_for(ggml_type, out_dtype))
    return out.reshape(shape).to(out_dtype)


# ====================================================================
# 3. Dequantized-weight cache
# ====================================================================
#
# A quant-resident GGUFLinear dequantizes its ENTIRE weight on every forward
# call. At 7B q8_0 that is 151,382 full dequantizations of ~8.6 GB of raw
# blocks for one prompt, and it is the reason q8_0 infers ~2.2x slower than the
# fp8 checkpoint, which pays the same logical work through a fused CUDA kernel
# (comfy_kitchen) that writes bf16 in one pass. See
# .dev/docs/2026-10-01-gguf-inference-speed/13-gguf-inference-speed-analysis.md
#
# This cache reuses the dequantized result for a layer that is hit again while
# its bytes still fit the VRAM headroom. It is deliberately partial: 8.6 GB of
# raw blocks dequantize to 16.5 GB of bf16 and only ~2.9 GB is cacheable at
# 7B, so eviction is SMALLEST-FIRST rather than least-recently-used. Pure LRU
# would evict the 18944-wide acoustic-decoder Linears -- which are ~10x a
# decoder Linear and are exactly where dequant dominates -- and leave a cache
# of little value.

_DEQUANT_CACHE_ENABLED = True
_cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_cache_bytes: int = 0
_cache_budget: int | None = None
_cache_epoch: int = 0


def _cache_budget_bytes(device=None) -> int:
    """Bytes the cache may hold, or 0 to disable it.

    Sampled once. Half of free VRAM, not all of it: the acoustic decoder's
    activations need headroom on a card that already holds the whole model.
    ``VIBEVOICE_GGUF_DEQUANT_CACHE_MB`` overrides for measurement; a negative
    value disables the cache, 0 means auto.
    """
    global _cache_budget
    if _cache_budget is not None:
        return _cache_budget

    override = os.environ.get("VIBEVOICE_GGUF_DEQUANT_CACHE_MB")
    if override is not None and override.strip():
        mb = int(override)
        if mb < 0:
            _cache_budget = 0
            return 0
        _cache_budget = mb * 1024 ** 2
        return _cache_budget

    free = 0
    try:
        import comfy.model_management as mm

        free = mm.get_free_memory(device) if device is not None else \
            mm.get_free_memory(mm.get_torch_device())
    except Exception:
        free = 0
    _cache_budget = max(0, int(free * 0.5))
    return _cache_budget


def clear_dequant_cache() -> None:
    """Drop every cached dequantized weight and bump the invalidation epoch.

    Called when the model is offloaded: a cached weight is real VRAM that core
    does not know about, and stranding it across a cold offload is how a model
    that loads fine later fails to load a second time.
    """
    global _cache, _cache_bytes, _cache_epoch
    _cache.clear()
    _cache_bytes = 0
    _cache_epoch += 1


def dequant_cache_stats() -> dict:
    """Snapshot for the generate-time log line."""
    return {"entries": len(_cache), "bytes": _cache_bytes,
            "budget": _cache_budget_bytes(None)}


def _cache_insert(key, tensor, budget) -> None:
    """Insert, evicting smallest-first until the budget is respected.

    A tensor larger than the whole budget is simply not cached -- the
    transient dequant already worked, and holding it would starve everything.
    """
    global _cache_bytes
    size = tensor.numel() * tensor.element_size()
    if size > budget:
        return
    _cache[key] = tensor
    _cache_bytes += size
    while _cache and _cache_bytes > budget:
        victim = min(_cache, key=lambda k: _cache[k].numel() * _cache[k].element_size())
        _cache_bytes -= _cache[victim].numel() * _cache[victim].element_size()
        del _cache[victim]


def cached_dequantize_blocks(linear, out_dtype):
    """``dequantize_blocks`` for ``linear``'s weight, reusing a cached result.

    The cached tensor is returned directly, not cloned: ``F.linear`` does not
    mutate its weight, and cloning would defeat the entire point. A hit is
    dtype-exact, because ``out_dtype`` selects the output dtype and a mismatch
    would change numerics silently.
    """
    if not _DEQUANT_CACHE_ENABLED:
        return dequantize_blocks(
            linear.weight, linear.ggml_type, out_dtype,
            (linear.out_features, linear.in_features),
        )

    key = (id(linear.weight), _cache_epoch, out_dtype)
    hit = _cache.get(key)
    if hit is not None:
        _cache.move_to_end(key)
        return hit

    tensor = dequantize_blocks(
        linear.weight, linear.ggml_type, out_dtype,
        (linear.out_features, linear.in_features),
    )
    _cache_insert(key, tensor, _cache_budget_bytes(linear.weight.device))
    return tensor


def dequantize_dense(raw_or_view: torch.Tensor, ggml_type) -> torch.Tensor:
    """Materialize a FLOAT ggml tensor (F32/F16/BF16) as a native torch tensor."""
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
# 1b. Tolerant reader open (flat-block recovery)
# ====================================================================

def _flat_byte_shape(shape, quant_type) -> tuple:
    from gguf.constants import GGML_QUANT_SIZES

    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape and shape[-1] % block_size == 0:
        return (*shape[:-1], shape[-1] // block_size * type_size)
    n_elements = 1
    for d in shape:
        n_elements *= int(d)
    return (n_elements // block_size * type_size,) if shape else ()


def open_gguf_reader(weight_path):
    """Open a GGUF file, recovering flat-block tensors if needed."""
    import os
    import gguf
    import gguf.gguf_reader as _gr

    try:
        return gguf.GGUFReader(weight_path)
    except ValueError as e:
        if "block size" not in str(e):
            raise
        if weight_path not in _FLAT_POOL_WARNED:
            _FLAT_POOL_WARNED.add(weight_path)
            logger.warning(
                "GGUF file '%s' was written by a converter that stores "
                "some data in a nonstandard layout; loading it with automatic "
                "recovery.",
                os.path.basename(weight_path),
            )
        _orig = _gr.quant_shape_to_byte_shape
        _gr.quant_shape_to_byte_shape = _flat_byte_shape
        try:
            return _gr.GGUFReader(weight_path)
        finally:
            _gr.quant_shape_to_byte_shape = _orig


# ====================================================================
# 1c. Dequantize a reader tensor
# ====================================================================

def dequantize_reader_tensor(
    reader_tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Dequantize one reader tensor to a LOGICAL-shaped torch tensor."""
    import numpy as np
    from gguf.constants import GGML_QUANT_SIZES

    t = reader_tensor
    if t.tensor_type in FLOAT_GGML_TYPES:
        data = np.ascontiguousarray(t.data)
        tensor = torch.from_numpy(data.copy())
        if t.tensor_type == _T.BF16 and tensor.dtype == torch.uint8:
            tensor = tensor.view(torch.bfloat16)
        shape = tuple(int(s) for s in reversed(t.shape))
        return tensor.reshape(shape).clone().to(dtype)

    if t.tensor_type not in SUPPORTED_GGML_TYPES:
        raise UnsupportedGGMLType(t.tensor_type, f"{t.name}")

    block_size, type_size = GGML_QUANT_SIZES[t.tensor_type]
    n_elements = 1
    for s in t.shape:
        n_elements *= int(s)
    if n_elements % block_size != 0:
        raise ValueError(
            f"GGUF tensor '{t.name}': element count {n_elements} is not a "
            f"multiple of the {t.tensor_type.name} block size {block_size}."
        )
    raw = torch.from_numpy(np.ascontiguousarray(t.data).copy())
    if raw.dtype != torch.uint8:
        raw = raw.view(torch.uint8)
    shape = tuple(int(s) for s in reversed(t.shape))
    return dequantize_blocks(raw, t.tensor_type, dtype, shape)


@dataclass
class GGUFTensor:
    """A GGUF tensor kept in RAW BLOCK FORM (no float materialization)."""

    raw: torch.Tensor
    ggml_type: object
    shape: tuple

    @classmethod
    def from_reader_tensor(cls, reader_tensor) -> "GGUFTensor":
        import numpy as np

        data = reader_tensor.data
        arr = np.ascontiguousarray(data)
        if arr.flags.writeable:
            raw = torch.from_numpy(arr).clone()
        else:
            raw = torch.from_numpy(arr.copy())
        if data.dtype != np.uint8:
            raw = raw.view(torch.uint8)
        shape = tuple(int(s) for s in reversed(reader_tensor.shape))
        return cls(raw=raw, ggml_type=reader_tensor.tensor_type, shape=shape)

    def to(self, device, *, dtype=None) -> "GGUFTensor":
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

_FORWARD_COUNTERS = {"fast": 0, "streamed": 0}


def gguf_forward_counters() -> dict:
    return dict(_FORWARD_COUNTERS)


def reset_gguf_forward_counters() -> None:
    _FORWARD_COUNTERS["fast"] = 0
    _FORWARD_COUNTERS["streamed"] = 0


def log_gguf_forward_counters(tag: str) -> None:
    counters = gguf_forward_counters()
    if counters["fast"] == 0 and counters["streamed"] == 0:
        return
    cache = dequant_cache_stats()
    logger.info(
        "GGUF forward diagnostics: stage=%s gguf_forward_fast=%d "
        "gguf_forward_streamed=%d dequant_cache_entries=%d "
        "dequant_cache_mb=%.1f dequant_cache_budget_mb=%.1f",
        tag, counters["fast"], counters["streamed"],
        cache["entries"], cache["bytes"] / 1024 ** 2,
        cache["budget"] / 1024 ** 2,
    )


class GGUFLinear(torch.nn.Module):
    """Drop-in ``nn.Linear`` replacement holding RAW GGML block bytes."""

    _quant_resident = True
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
        self.weight = torch.nn.Parameter(
            torch.empty(n_bytes, dtype=torch.uint8), requires_grad=False
        )
        self.bias = (
            torch.nn.Parameter(torch.empty(out_features), requires_grad=False)
            if bias else None
        )
        self.weight_function = []
        self.bias_function = []
        self.weight_comfy_model_dtype = torch.uint8
        self._gguf: GGUFTensor | None = None
        self.weight_scratch_cache = None

    def set_raw_weight(self, raw_uint8: torch.Tensor) -> None:
        if raw_uint8.dtype != torch.uint8:
            raw_uint8 = raw_uint8.view(torch.uint8)
        if raw_uint8.numel() != self.weight.numel():
            raise ValueError(
                f"Raw byte count mismatch for GGUFLinear: got "
                f"{raw_uint8.numel()}, expected {self.weight.numel()} "
                f"(ggml_type={getattr(self.ggml_type, 'name', self.ggml_type)}, "
                f"shape=({self.out_features}, {self.in_features}))"
            )
        clear_dequant_cache()  # the raw blocks changed; any cached dequant is stale
        self.weight = torch.nn.Parameter(raw_uint8.contiguous(), requires_grad=False)
        self._gguf = GGUFTensor(
            raw=self.weight.data, ggml_type=self.ggml_type,
            shape=(self.out_features, self.in_features),
        )

    def forward(self, x):
        if not x.dtype.is_floating_point:
            raise TypeError(
                f"GGUFLinear expects a floating-point activation, got {x.dtype}."
            )

        wf = getattr(self, "weight_function", None)
        if (self.weight.device != x.device) or (wf and len(wf) > 0):
            return self._forward_streamed(x)

        _FORWARD_COUNTERS["fast"] += 1

        # Cached on the fast path ONLY. _forward_streamed below must not cache:
        # it hands the weight to cast_bias_weight(offloadable=True) and unpins
        # it in a finally, so a retained reference would pin host memory.
        w = cached_dequantize_blocks(self, x.dtype)
        return torch.nn.functional.linear(x, w, self.bias)

    def _forward_streamed(self, x):
        """Paged path: pull raw blocks through core's cast machinery."""
        import comfy.ops

        _FORWARD_COUNTERS["streamed"] += 1
        weight, bias, offload_stream = comfy.ops.cast_bias_weight(
            self,
            x,
            device=x.device,
            dtype=self.weight_comfy_model_dtype,
            bias_dtype=x.dtype,
            offloadable=True,
        )
        try:
            if bias is not None and bias.dtype != x.dtype:
                bias = bias.to(x.dtype)
            w = dequantize_blocks(
                weight, self.ggml_type, x.dtype,
                (self.out_features, self.in_features),
            )
            return torch.nn.functional.linear(x, w, bias)
        finally:
            comfy.ops.uncast_bias_weight(self, weight, bias, offload_stream)

    def _pull_to_device(self, tensor, device):
        fns = getattr(self, "weight_function", None)
        if not fns and tensor.device == device:
            return tensor
        if tensor.device != device:
            tensor = tensor.to(device)
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
    return GGUFLinear(
        target.in_features, target.out_features,
        bias=target.bias is not None, ggml_type=ggml_type,
    )


def gguf_linear_factory(ggml_type):
    def _factory(in_features: int, out_features: int, has_bias: bool):
        return GGUFLinear(in_features, out_features, bias=has_bias,
                          ggml_type=ggml_type)
    return _factory


# ====================================================================
# 3. Key-scheme detection and mapping
# ====================================================================

class UnmappedKeyError(ValueError):
    def __init__(self, unknown_keys, scheme: str):
        self.unknown_keys = list(unknown_keys)
        preview = ", ".join(self.unknown_keys[:10])
        super().__init__(
            f"{len(self.unknown_keys)} GGUF tensor key(s) could not be mapped "
            f"onto the VibeVoice module tree (scheme='{scheme}'). First "
            f"unknown keys: [{preview}]."
        )


_LLAMACPP_TO_HF = {
    "tok_embeddings.weight": "model.language_model.embed_tokens.weight",
    "token_embd.weight": "model.language_model.embed_tokens.weight",
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

_LLAMACPP_LAYER_RE = _re.compile(r"^blk\.(\d+)\.(.+)$")


def detect_key_scheme(keys) -> str:
    hf_re = _re.compile(r"^(model\.|lm_head\.|transformer\.)")
    llamacpp_re = _re.compile(
        r"^(blk\.\d+\.|tok_embeddings\.|token_embd\.|output\.|output_norm\.)"
    )
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
    scheme = scheme or detect_key_scheme(keys)
    if scheme == "mixed":
        n = len(list(keys)) if not isinstance(keys, list) else len(keys)
        hf_keys = [k for k in keys if _re.match(r"^(model\.|lm_head\.|transformer\.)", k)]
        n_hf = len(hf_keys)
        if n_hf * 2 == n:
            raise UnmappedKeyError(sorted(keys)[:10], scheme)
        scheme = "hf" if n_hf * 2 > n else "llamacpp"
        logger.info(
            "GGUF file mixes tensor naming conventions (%d HF / %d llamacpp); "
            "resolving with the %s convention and aliasing the rest.",
            n_hf, n - n_hf, scheme,
        )
    mapping = {}
    unknown = []
    for k in keys:
        if scheme == "hf":
            if _re.match(r"^(model\.|lm_head\.|transformer\.)", k):
                mapping[k] = k
                continue
            try:
                mapping[k] = _map_llamacpp_key(k)
            except KeyError:
                unknown.append(k)
            continue
        if _re.match(r"^(model\.|lm_head\.|transformer\.)", k):
            mapping[k] = k
            continue
        try:
            mapping[k] = _map_llamacpp_key(k)
        except KeyError:
            unknown.append(k)
    if unknown:
        raise UnmappedKeyError(unknown, scheme)
    return mapping