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

# Each kernel receives ``blocks``: a contiguous uint8 CPU/GPU tensor shaped
# (n_blocks, type_size) and returns an fp32/int tensor shaped
# (n_blocks, block_size). All integer math is exact; float math mirrors the
# numpy expression trees op-for-op so results are BITWISE equal.


def _u(tensor: torch.Tensor) -> torch.Tensor:
    """View raw bytes as uint8 (no copy)."""
    return tensor.view(torch.uint8) if tensor.dtype != torch.uint8 else tensor


# Shift constants for the k-quant nibble/bit unpacking, memoised PER DEVICE.
# Hygiene only: these used to be re-materialised (and re-uploaded H2D) on
# every single dequant call. They live on the q4_k/q5_k/q6_k path ONLY — the
# q8_0 kernel below never touches them, so this is not part of any Q8_0
# speedup. A plain module-level CPU tensor would break (or silently sync) on
# CUDA blocks, hence the per-device memo.
_SHIFT_SPECS = {
    "nibble2": (0, 4),          # q4_k low nibbles, q5_k ql
    "bits8": tuple(range(8)),   # q5_k high-bit plane
    "shifts2": (0, 4),          # q6_k ql
    "shifts4": (0, 2, 4, 6),    # q6_k qh
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
    """Reinterpret ``t``'s bytes as ``dtype``, copying only if forced to.

    ``view(dtype)`` requires stride 1 in the LAST dimension — nothing more.
    A ``torch.split`` slice of a contiguous block tensor (``strides=(34, 1)``)
    already satisfies that, so the ``.contiguous()`` these call sites used to
    do was a full weight-sized copy that was immediately discarded.

    That matters because ``dequantize_blocks`` runs once per GGUFLinear
    FORWARD, not once per load: 151k times for a 7B q8_0 generate. The copy
    was ~15% of the dequant's time (3.80 -> 3.26 ms across four decoder
    shapes), for nothing.

    The fallback keeps this safe for a caller whose last dim genuinely is not
    stride-1, where ``view`` would raise.
    """
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
    """Q8_0: ``x * d``.

    The int8 payload (-128..127) is exact in fp32, and the fp16 scale widens
    to fp32 losslessly, so computing in fp32 is EXACT — the only rounding is
    the single one the caller applies at the final ``.to(out_dtype)``.

    ``out_dtype`` is accepted (the k-quant kernels take the same signature so
    callers can pass a compute dtype unconditionally) but is deliberately
    ignored here too, for the reason recorded at ``_NATIVE_DTYPES`` below:
    computing the product in a low-precision dtype rounds the scale AND the
    product, and for a model whose utterance length is decided by a bare EOS
    sample that divergence is audible in the output duration.
    """
    d, x = torch.split(blocks, [2, 32], dim=-1)
    d = _f16_scale(d, out_dtype)
    x = _reinterpret(x, torch.int8).to(out_dtype)
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


def _dequant_q4_k(
    blocks: torch.Tensor, out_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    # WHY the unused out_dtype: the k-quant kernels accept the same signature
    # as _dequant_q8_0 so callers can pass a compute dtype unconditionally,
    # but they IGNORE it and keep their fp32 math. Their 6-bit scale/min
    # unpacking (d*q - dm) has no independent accuracy measurement in a
    # low-precision compute dtype, so widening the numerics change is
    # deliberately confined to the measured Q8_0 case.
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
    shifts = _shift_const("nibble2", blocks.device)
    qs = ((qs >> shifts) & 0x0F).reshape(n, -1, 32).to(torch.float32)

    return (d * qs - dm).reshape(n, _QK_K)


def _dequant_q5_k(
    blocks: torch.Tensor, out_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    # out_dtype intentionally ignored — see _dequant_q4_k for the why.
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
    # out_dtype intentionally ignored — see _dequant_q4_k for the why.
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


# GGML types whose dequant math is PROVEN safe to run in the activation dtype.
#
# This was `frozenset({_T.Q8_0})` and was REVERTED to empty on 2026-09-28. It
# bought ~1.5x on the dequant kernel and cost numerical fidelity that turned
# out to be user-visible: computing `x * d` in the OUTPUT dtype rounds twice
# instead of once (the fp16 scale is rounded to bf16 on the way in, then the
# product is rounded again), whereas the fp32 round-trip rounds exactly once
# at the final `.to(out_dtype)`. Measured on a real oracle-quantized 4096x3584
# Q8_0 tensor, comparing the two paths' STORED bf16 weights (which is what the
# model actually consumes): 22.7% of elements differ, mean rel 1.23e-3, p99
# rel 7.5e-3.
#
# Why that matters here specifically: VibeVoice's standard TTS path samples
# from ~5 valid tokens and stops ONLY on a sampled EOS token — there is no EOS
# confidence floor, no minimum duration, no repetition guard. A ~1e-3 logit
# shift therefore does not merely perturb the waveform, it flips draws near the
# end of the utterance and moves the OUTPUT DURATION by tens of seconds. That
# is exactly the 21s -> 24-27s regression reported against the q8_0 7B
# checkpoint. A load-time micro-optimization is not worth that.
#
# The gate is kept (rather than deleted) so re-evaluating it is a one-token
# change — but only with an END-TO-END output check, never a tolerance check.
# A tolerance test cannot see this class of regression: see
# tests/test_gguf_dequant_perf.py, which now pins bit-exactness instead.
#
# COST, measured (RTX 4070 Ti SUPER, q8_0, fp32 compute vs the bf16 path over
# 3584x3584 / 5120x3584 / 11008x3584 / 18944x3584): 2.53 ms -> 3.80 ms, i.e.
# fp32 compute is ~1.5x SLOWER. An earlier version of this comment claimed the
# difference was "paid once at load". THAT WAS WRONG, and the number above is
# why it matters: GGUFLinear.forward dequantizes the whole weight on EVERY
# call, so a 7B q8_0 generate runs this kernel 151382 times. There is no
# cache. The honest framing is that this choice costs 1.5x on the dequant
# kernel and buys bit-exact weights, and that the real fix for the speed is a
# dequant CACHE (which makes the choice a one-time cost) — not picking a
# different dtype here.
_NATIVE_DTYPES: frozenset = frozenset()
_NATIVE_COMPUTE_DTYPES = frozenset({torch.bfloat16, torch.float16})


def _compute_dtype_for(ggml_type, out_dtype: torch.dtype) -> torch.dtype:
    """Dtype the kernel itself should compute in (fp32 unless proven safe)."""
    if ggml_type in _NATIVE_DTYPES and out_dtype in _NATIVE_COMPUTE_DTYPES:
        return out_dtype
    return torch.float32


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
    out = kernel(blocks, _compute_dtype_for(ggml_type, out_dtype))
    # No-op (returns self) when the kernel already produced out_dtype.
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
# 1b. Tolerant reader open (flat-block recovery)
# ====================================================================

# Files seen in the wild (e.g. third-party conversions) quantize small-kernel
# conv weights — whose row (kernel-size) is BELOW the block size — as a flat
# C-order pool of whole blocks. The bytes are self-consistent, but gguf-py's
# GGUFReader validates each row against the block size at OPEN time and
# refuses the whole file. This fallback maps the byte shape as one flat row
# (n_blocks * type_size) so every tensor, quantized or not, still opens; the
# flat pool is dequantized and reshaped to the logical shape by the caller.
def _flat_byte_shape(shape, quant_type) -> tuple:
    """``quant_shape_to_byte_shape`` without the per-row block-size check.

    Returns a FLAT (n_bytes,) shape whenever the last row is smaller than
    the block size; otherwise delegates to the stock mapping (row-safe
    files keep their native per-row byte shape).
    """
    from gguf.constants import GGML_QUANT_SIZES

    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape and shape[-1] % block_size == 0:
        return (*shape[:-1], shape[-1] // block_size * type_size)
    n_elements = 1
    for d in shape:
        n_elements *= int(d)
    return (n_elements // block_size * type_size,) if shape else ()


def open_gguf_reader(weight_path):
    """Open a GGUF file, recovering flat-block (sub-block-row) tensors.

    Tries the stock :class:`gguf.GGUFReader` first; every spec-conformant
    file takes that path unchanged. When the stock reader rejects the file
    because a quantized tensor's row is below its block size (some converters
    write conv kernels as a flat pool of whole blocks), re-opens with a
    byte-shape mapping that flattens ONLY those tensors. Tensor ``.shape``
    still reports the header dims, so key mapping / logical shapes / config
    detection are unaffected; ``.data`` for the affected tensors is a flat
    uint8 block pool.

    The flat-pool warning fires at most once per file per session — the
    loader opens the same file several times (fingerprint, inspection,
    install) and each open would otherwise re-announce the recovery.

    Raises the ORIGINAL error when the open fails for any other reason.
    """
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
                "some data in a nonstandard layout; loading it anyway "
                "with automatic recovery (this file keeps working, but "
                "re-converting it without the conv layers would make it "
                "fully standard).",
                os.path.basename(weight_path),
            )
        # gguf_reader imports quant_shape_to_byte_shape into its own
        # namespace at module load; the fallback must replace it there.
        _orig = _gr.quant_shape_to_byte_shape
        _gr.quant_shape_to_byte_shape = _flat_byte_shape
        try:
            return _gr.GGUFReader(weight_path)
        finally:
            _gr.quant_shape_to_byte_shape = _orig


# ====================================================================
# 1c. Dequantize a reader tensor (dense fallback for non-Linear targets)
# ====================================================================

def dequantize_reader_tensor(
    reader_tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Dequantize one reader tensor to a LOGICAL-shaped torch tensor.

    Used for quantized GGUF tensors that cannot become quant-resident
    ``GGUFLinear`` weights (embeddings, conv heads): they are materialized
    once at load into the dense state dict instead — same philosophy as the
    fp8 path's dequant-at-load fallback for non-Linear modules.

    Handles BOTH byte layouts the reader may hand back:
    * row-mapped (spec-conformant): blocks shaped ``(..., n_blocks*ts)``;
    * flat pool (recovery): a single ``(n_bytes,)`` row whose
      block count follows from the logical element count.

    Returns an owned CPU tensor (the mmap view is copied).

    Args:
        reader_tensor: a tensor from an open ``GGUFReader``.
        dtype: target dtype for the result. fp32 by default. Callers that
            install into a model should pass the destination parameter dtype,
            so the fp32 intermediate is never materialized at all.
    """
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
            f"multiple of the {t.tensor_type.name} block size {block_size}; "
            f"the flat-block pool is truncated and cannot be recovered."
        )
    raw = torch.from_numpy(np.ascontiguousarray(t.data).copy())
    if raw.dtype != torch.uint8:
        raw = raw.view(torch.uint8)
    shape = tuple(int(s) for s in reversed(t.shape))
    return dequantize_blocks(raw, t.tensor_type, dtype, shape)


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
        # Flat-block recovery (non-spec files): a tensor whose rows are
        # below the block size was opened with a FLAT byte shape; the raw
        # pool is exactly what the resident Linear stores either way, and
        # dequantize_blocks() reshapes by element count, so nothing here
        # depends on the byte-shape layout.
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


# Forward-path instrumentation: how many GGUFLinear forwards took the resident
# fast path vs the streamed (hook/offload) path. Zeroed once per external load
# by modules/external_loader.py and reported at the end of a generation run by
# log_gguf_forward_counters(), so a slow run can be attributed from the log.
_FORWARD_COUNTERS = {"fast": 0, "streamed": 0}


def gguf_forward_counters() -> dict:
    """Snapshot of the per-load GGUFLinear forward counters."""
    return dict(_FORWARD_COUNTERS)


def reset_gguf_forward_counters() -> None:
    """Zero both GGUFLinear forward counters (used at load time / between tests)."""
    _FORWARD_COUNTERS["fast"] = 0
    _FORWARD_COUNTERS["streamed"] = 0


def log_gguf_forward_counters(tag: str) -> None:
    """Log the per-load GGUF forward counters AFTER a generation run.

    WHY this is not emitted at load time: the counters only describe something
    once forwards have run, so a readout taken inside
    ``load_external_vibevoice_model`` is structurally always ``fast=0
    streamed=0`` — indistinguishable from the "hook-poisoning ruled out"
    answer, and actively misleading. The loaders now only reset the counters;
    the generation paths report them at the end of the run, where the numbers
    describe the work that was actually done.

    No-ops when no GGUFLinear forward happened, so non-GGUF models do not get
    a meaningless line per generation.
    """
    counters = gguf_forward_counters()
    if counters["fast"] == 0 and counters["streamed"] == 0:
        return
    logger.info(
        "GGUF forward diagnostics: stage=%s gguf_forward_fast=%d "
        "gguf_forward_streamed=%d",
        tag, counters["fast"], counters["streamed"],
    )


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
    # Read-only DEFAULT, kept at class level so core's `hasattr(m,
    # "weight_function")` checks and our own getattr(..., None) keep their
    # current meaning for an instance that never built its own list. Every
    # instance shadows these in __init__ — see the WHY there.
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
        # Per-INSTANCE hook lists (defensive). WHY: core MUTATES this
        # attribute — comfy/model_patcher.py:1022 rebinds it per instance and
        # :1057 / :1227 APPEND to whatever list it finds. A class-level list
        # is shared by every GGUFLinear in the model, so one module's append
        # would force the slow streamed path on ALL of them. This is a latent
        # hazard fix, NOT the measured cause of any user-visible slowdown.
        self.weight_function = []
        self.bias_function = []
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
            # The streamed branch counts itself in _forward_streamed, so the
            # "streamed" total stays exact whether it is reached from here
            # or called directly. Counting it here too would double-count.
            return self._forward_streamed(x)

        _FORWARD_COUNTERS["fast"] += 1

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
        _FORWARD_COUNTERS["streamed"] += 1
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
        # A fully resident tensor with no hooks is already where it needs to
        # be: return the SAME object without touching `.to()`. Any `.to()` on
        # a resident uint8 weight would be a no-op copy at best and a
        # realloc at worst.
        fns = getattr(self, "weight_function", None)
        if not fns and tensor.device == device:
            return tensor
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
    # modern name for the input embedding
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

import re as _re

_LLAMACPP_LAYER_RE = _re.compile(r"^blk\.(\d+)\.(.+)$")


def detect_key_scheme(keys) -> str:
    """Classify GGUF tensor names as 'hf', 'llamacpp', or 'mixed'.

    'hf': names already match the VibeVoice/HF module tree
    ('model.'/'lm_head.' prefixes, proj/norm suffixes).
    'llamacpp': names follow llama.cpp conversion conventions
    ('blk.N.', 'tok_embeddings.', 'output.').
    'mixed': both conventions appear. Real converters do produce these
    (some exporters keep HF names for everything but the lm_head, which they
    renames llama.cpp-style to 'output.weight'); map_keys resolves such
    files with a majority vote instead of rejecting them.
    """
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
        scheme: Force a scheme; auto-detected when omitted. A 'mixed'
            census is resolved by majority vote — the dominant convention
            is kept and the minority keys are mapped through the other
            convention's alias table (some 7B exports are 1204 HF
            keys + llamacpp 'output.weight'). An ambiguous 50/50 mix or
            an unknown minority key still raises.

    Returns:
        dict mapping ORIGINAL key -> module parameter path (identity for hf).

    Raises:
        UnmappedKeyError: On unmapped llamacpp keys or a 50/50 mixed census.
    """
    scheme = scheme or detect_key_scheme(keys)
    if scheme == "mixed":
        # Majority vote resolves genuine converter output (a file that is
        # overwhelmingly one convention plus a handful of aliased names).
        # Only an exact tie stays unmappable.
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
            # HF-majority files may still carry a handful of llamacpp-style
            # aliases (the lm_head is written as 'output.weight').
            if _re.match(r"^(model\.|lm_head\.|transformer\.)", k):
                mapping[k] = k
                continue
            try:
                mapping[k] = _map_llamacpp_key(k)
            except KeyError:
                unknown.append(k)
            continue
        # llamacpp-majority (or forced-llamacpp): HF-shaped keys are passed
        # through so a single stray HF tensor doesn't poison the whole map.
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
