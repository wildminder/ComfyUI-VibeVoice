"""Regression tests: GGUF dequant must round ONCE, and only at the end.

``dequantize_blocks`` used to throw ``out_dtype`` away and materialize fp32
for EVERY ggml type, then cast the 4x-larger result afterwards. A "FIX 2"
then made Q8_0 compute natively in bf16/fp16 for a measured ~1.5-1.9x on the
dequant kernel — and that has been REVERTED (2026-09-28). See the long
comment at ``modules/gguf_quant.py::_NATIVE_DTYPES`` for the full why.

The short version, and the reason this file is shaped the way it is now:
computing ``x * d`` in a low-precision dtype rounds TWICE (the fp16 scale on
the way in, then the product on the way out) where the fp32 round-trip
rounds exactly once. Measured on Q8_0 that moved 98.1% of weight elements.
VibeVoice decides utterance length by sampling an EOS token with no
confidence floor, so a ~1e-3 logit shift does not just alter timbre — it
moves the output DURATION (observed 21s -> 24-27s on the 7B q8_0 checkpoint).

**The lesson encoded in these tests: a tolerance assertion cannot catch that
class of regression.** The original guard asserted ``mean_rel < 5e-3`` and
passed while 98% of the weights changed. The gate below is now BITWISE:

1. the fp32 request path is equal to the gguf-py oracle (this is what keeps
   tests/test_gguf_quant_blocks.py green), and
2. a low-precision request is BITWISE equal to that fp32 result cast once —
   the same operation, rounded the same number of times.

They also cover the three measured/defensive items that shipped with the fix:
per-instance weight hooks, the zero-work resident pull, and the forward-path
counters behind the per-run diagnostics line.

WHAT THE REVERT COST, stated honestly (RTX 4070 Ti SUPER, torch 2.11.0+cu130,
three consecutive runs within +/-0.01x). These are the numbers the removed
native path delivered, kept so they are not re-inflated:

    3584x3584   0.327 -> 0.170 ms   1.92x-1.93x
    5120x3584   0.476 -> 0.305 ms   1.56x
    18944x3584  1.897 -> 1.293 ms   1.46x-1.48x

Two things the earlier "2.47x / 2.18x / 1.98x" claim got wrong: the ratio
DEGRADES MONOTONICALLY as the weight grows (~1.9x small, ~1.5x at the
18944-wide acoustic-decoder head), because the fixed per-call launch overhead
that the fp32 round-trip paid twice is amortized away; and ~1.5-1.9x is what
the KERNEL delivered, NOT the 2.5-3x the original end-to-end report claimed.
That was a LOAD-TIME kernel measurement and could never account for the rest
of that gap. A one-time load cost of a few hundred milliseconds is not worth
changing the weights of the model.
"""

import time

import numpy as np
import pytest
import torch
from gguf.constants import GGMLQuantizationType as T
from gguf.quants import dequantize as oracle_dequantize
from gguf.quants import quantize as oracle_quantize

from ComfyUI_VibeVoice.modules import gguf_quant as G

_KERNEL_NAMES = ("_dequant_q8_0", "_dequant_q4_k", "_dequant_q5_k", "_dequant_q6_k")


def _seeded_raw(shape=(512, 256), seed=0):
    """Oracle-quantized Q8_0 raw blocks (flat uint8 numpy) for ``shape``."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(shape) * 0.05).astype(np.float32)
    raw = oracle_quantize(x, T.Q8_0)
    return np.ascontiguousarray(raw).reshape(-1).view(np.uint8), x


def _handcrafted_blocks(qtype, n_blocks):
    """Deterministic raw bytes for a k-quant, valid enough to unpack."""
    from gguf.constants import GGML_QUANT_SIZES

    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(n_blocks, GGML_QUANT_SIZES[qtype][1]),
                        dtype=np.uint8)


def _count_differing(a: torch.Tensor, b: torch.Tensor) -> int:
    """How many elements of two same-dtype tensors are not bit-identical."""
    return int((a != b).sum())


class TestDequantRoundsExactlyOnce:
    """The dequant gate: compute in fp32, round ONCE at the final cast.

    Every assertion here is BITWISE. That is the whole point — see the module
    docstring. The removed native path passed a ``mean_rel < 5e-3`` tolerance
    while changing 98.1% of the weight elements, and that was enough to move
    VibeVoice's output duration by several seconds.
    """

    def test_q8_0_fp32_request_still_bitwise(self):
        """The fp32 request path must stay BITWISE equal to the oracle.

        This is the single guard for the whole existing bitwise suite
        (tests/test_gguf_quant_blocks.py).
        """
        raw, _ = _seeded_raw()
        t = torch.from_numpy(raw.copy())
        ref = oracle_dequantize(raw, T.Q8_0).reshape(512, 256)
        ours = G.dequantize_blocks(t, T.Q8_0, torch.float32, (512, 256))
        assert ours.dtype is torch.float32
        assert np.array_equal(ref, ours.numpy())

    @pytest.mark.parametrize("lowp", [torch.bfloat16, torch.float16])
    def test_q8_0_lowp_request_is_bitwise_fp32_cast_once(self, lowp):
        """A low-precision request == the fp32 result cast exactly once.

        This is the assertion that would have caught the regression. A
        tolerance bound cannot: any bound loose enough to tolerate legitimate
        reordering is also loose enough to tolerate 98% of the weights
        changing, and this model's utterance length is decided by a bare EOS
        sample with no confidence floor.
        """
        raw, _ = _seeded_raw()
        shape = (512, 256)
        t = torch.from_numpy(raw.copy())

        ref = G.dequantize_blocks(t, T.Q8_0, torch.float32, shape)
        ours = G.dequantize_blocks(t, T.Q8_0, lowp, shape)

        assert ours.dtype is lowp
        assert torch.equal(ours, ref.to(lowp)), (
            f"{lowp} q8_0 dequant diverged from a single fp32->{lowp} cast on "
            f"{_count_differing(ours, ref.to(lowp))}/{ours.numel()} elements"
        )

    @pytest.mark.parametrize("lowp", [torch.bfloat16, torch.float16])
    def test_q8_0_lowp_matches_oracle_cast_once(self, lowp):
        """Same gate, anchored to the ORACLE rather than to our own fp32 path.

        Guards against "both paths are wrong in the same way".
        """
        raw, _ = _seeded_raw()
        shape = (512, 256)
        t = torch.from_numpy(raw.copy())
        ref = torch.from_numpy(
            oracle_dequantize(raw, T.Q8_0).reshape(shape)
        )
        ours = G.dequantize_blocks(t, T.Q8_0, lowp, shape)
        assert torch.equal(ours, ref.to(lowp)), (
            f"{_count_differing(ours, ref.to(lowp))}/{ours.numel()} elements "
            f"differ from the oracle cast once to {lowp}"
        )

    def test_every_kernel_computes_in_fp32(self, monkeypatch):
        """No ggml type may compute in the low-precision output dtype.

        Q8_0 was the one type that used to (and was reverted); the k-quants'
        6-bit scale/min unpacking was never measured in a low-precision
        compute dtype at all. Both are now pinned to fp32 — this is what
        stops a future edit from quietly re-widening either one.
        """
        calls = []
        for name in _KERNEL_NAMES:
            real = getattr(G, name)

            def _spy(blocks, out_dtype=torch.float32, _name=name, _real=real):
                calls.append((_name, out_dtype))
                return _real(blocks, out_dtype)

            monkeypatch.setattr(G, name, _spy)

        raw, _ = _seeded_raw()
        calls.clear()
        out = G.dequantize_blocks(torch.from_numpy(raw.copy()), T.Q8_0,
                                  torch.bfloat16, (512, 256))
        assert calls == [("_dequant_q8_0", torch.float32)], calls
        # ...and the cast happens at the very end, as the output dtype.
        assert out.dtype is torch.bfloat16

        for qtype, name in ((T.Q4_K, "_dequant_q4_k"), (T.Q5_K, "_dequant_q5_k"),
                            (T.Q6_K, "_dequant_q6_k")):
            calls.clear()
            blocks = _handcrafted_blocks(qtype, n_blocks=16)
            out = G.dequantize_blocks(torch.from_numpy(blocks.reshape(-1)),
                                      qtype, torch.bfloat16, (128, 32))
            assert calls == [(name, torch.float32)], (
                f"{qtype.name} must compute in fp32, got {calls}"
            )
            assert out.dtype is torch.bfloat16

    def test_reinterpret_does_not_copy_a_contiguous_last_dim(self):
        """The dequant must not copy a stride-1 slice before ``view(dtype)``.

        ``dequantize_blocks`` runs once per GGUFLinear FORWARD (151k times for
        a 7B q8_0 generate), so a redundant weight-sized copy here is not a
        micro-optimization — it is a measurable share of total runtime. The
        old ``.contiguous().view(...)`` forced exactly that copy and threw it
        away, costing ~15% of the dequant.

        Value-neutrality is already covered by the bitwise oracle tests; this
        pins the OPTIMIZATION (no copy in the common case, still correct when
        a copy really is needed).
        """
        blocks = torch.randint(0, 255, (8, 34), dtype=torch.uint8)
        _, x = torch.split(blocks, [2, 32], dim=-1)

        # The common case: a split of a contiguous block tensor, stride 1 in the
        # last dim. Must reinterpret in place.
        assert x.stride(-1) == 1
        assert G._reinterpret(x, torch.int8).data_ptr() == x.data_ptr()
        assert G._reinterpret(blocks[:, :2], torch.float16).data_ptr() == \
            blocks[:, :2].data_ptr()

        # A genuinely non-stride-1 last dim must still work (falls back to a
        # copy) rather than raising from view(). A transpose is the clean case:
        # last dim is 8 bytes (divisible by 2, so the fp16 view is legal) but
        # its stride is 34, so view() alone would raise. NOTE the result is
        # (34, 4) — viewing 8 uint8 as float16 halves the last dim, exactly
        # like the .contiguous().view() it replaces.
        strided = blocks.t()
        assert strided.stride(-1) != 1 and strided.shape[-1] % 2 == 0
        out = G._reinterpret(strided, torch.float16)
        assert out.shape == (34, 4) and out.dtype is torch.float16
        # NaN-SAFE bitwise compare: arbitrary bytes reinterpreted as float16
        # can produce NaN, and torch.equal is False whenever NaNs are present
        # even for two bit-identical tensors.
        expected = strided.contiguous().view(torch.float16)
        assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))

    def test_q8_0_dequant_matches_oracle_for_every_shape(self):
        """Belt-and-braces: the shipped kernel still equals the gguf-py oracle.

        This is the test that would have caught a bad ``_reinterpret``; it
        runs over several realistic decoder/acoustic shapes rather than one.
        """
        for out_f, in_f in ((64, 32), (512, 256), (1024, 3584)):
            rng = np.random.default_rng(out_f)
            x = (rng.standard_normal((out_f, in_f)) * 0.05).astype(np.float32)
            raw = np.ascontiguousarray(oracle_quantize(x, T.Q8_0))
            flat = raw.reshape(-1).view(np.uint8)
            ours = G.dequantize_blocks(
                torch.from_numpy(flat.copy()), T.Q8_0, torch.float32, (out_f, in_f)
            )
            ref = oracle_dequantize(flat, T.Q8_0).reshape(out_f, in_f)
            assert ours.dtype is torch.float32
            assert np.array_equal(ref, ours.numpy()), f"diverged at {out_f}x{in_f}"

    def test_native_dtype_gate_is_empty(self):
        """The opt-in that re-enables low-precision compute stays OFF.

        Deliberately a separate assertion from the behaviour tests above: if
        someone re-populates ``_NATIVE_DTYPES`` the behaviour tests fail with a
        numeric message, and this one fails with the reason why it must not be
        repopulated without an END-TO-END output check.
        """
        assert G._NATIVE_DTYPES == frozenset(), (
            "re-enabling activation-dtype dequant requires an end-to-end "
            "output-duration check, not a tolerance test"
        )


class TestGGUFLinearHooksAndCounters:
    def _resident(self, out_f=64, in_f=32):
        lin = G.GGUFLinear(in_f, out_f, bias=False, ggml_type=T.Q8_0)
        rng = np.random.default_rng(5)
        raw = oracle_quantize(
            (rng.standard_normal((out_f, in_f)) * 0.05).astype(np.float32), T.Q8_0
        )
        lin.set_raw_weight(
            torch.from_numpy(np.ascontiguousarray(raw)).view(torch.uint8).reshape(-1)
        )
        return lin

    def test_forward_counters_and_per_instance_hooks(self):
        G.reset_gguf_forward_counters()
        a = self._resident()
        b = self._resident()
        x = torch.randn(4, 32, dtype=torch.bfloat16)

        a(x)
        counters = G.gguf_forward_counters()
        assert counters["fast"] == 1
        assert counters["streamed"] == 0
        assert G.gguf_forward_counters() is not counters  # a copy, not the live dict

        # A hook routes THIS instance to the streamed path only.
        a.weight_function = [lambda t: t]
        a(x)
        counters = G.gguf_forward_counters()
        assert counters["fast"] == 1
        assert counters["streamed"] == 1

        # Appending to one instance must not leak into any other instance.
        a.weight_function.append(lambda t: t)
        assert b.weight_function == []
        b(x)
        assert G.gguf_forward_counters()["streamed"] == 1

    def test_pull_to_device_is_zero_work_when_resident(self):
        lin = self._resident()
        t = torch.zeros(8, dtype=torch.uint8)
        assert lin._pull_to_device(t, t.device) is t

        seen = []

        def _hook(tensor):
            seen.append(tensor)
            return tensor.clone()

        lin.weight_function = [_hook]
        pulled = lin._pull_to_device(t, t.device)
        assert seen == [t], "the hook must still be honoured"
        assert pulled is not t, "with a hook the pull is no longer a no-op"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gguf_perf_benchmark_q8_0_dequant_cost():
    """CUDA: informational timing for the single remaining dequant path.

    There is nothing to compare against any more — the bf16 path it used to
    race is gone by design (see the module docstring). What this keeps is the
    ability to SEE the load-time cost of the fp32 round-trip, so a future
    change that reintroduces an accidental extra copy is visible in the log.

    **It deliberately asserts nothing.** This was a `>= 1.05x faster` race
    before, and when it was re-pointed at a wall-clock ceiling it flaked: it
    passes standalone (0.438 / 0.657 / 2.550 ms for the shapes below) and
    fails inside the full suite, where other tests hold the GPU. A benchmark
    that measures contention is worse than no benchmark — the numbers are for
    a human reading them, run with `-s`.

    Reference, RTX 4070 Ti SUPER, q8_0, standalone:
        3584x3584   0.438 ms
        5120x3584   0.657 ms
        18944x3584  2.550 ms

    The removed bf16 path measured 0.170 / 0.305 / 1.293 ms for the same
    shapes. That ~1 ms difference is paid ONCE at load, in exchange for
    bit-exact weights.
    """
    from gguf.constants import GGML_QUANT_SIZES

    block_size, type_size = GGML_QUANT_SIZES[T.Q8_0]
    iters, warmups = 50, 10

    for out_f, in_f in ((3584, 3584), (5120, 3584), (18944, 3584)):
        shape = (out_f, in_f)
        n_bytes = (out_f * in_f // block_size) * type_size
        raw = torch.randint(0, 255, (n_bytes,), dtype=torch.uint8, device="cuda")

        for _ in range(warmups):
            G.dequantize_blocks(raw, T.Q8_0, torch.bfloat16, shape)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            G.dequantize_blocks(raw, T.Q8_0, torch.bfloat16, shape)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0 / iters

        print(f"\n[gguf q8_0 dequant] {out_f}x{in_f}: {ms:.3f} ms")
        del raw
        torch.cuda.empty_cache()
