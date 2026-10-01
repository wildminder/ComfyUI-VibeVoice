"""Bitwise contract for the fused Triton Q8_0 dequant kernel.

The torch chain in `_dequant_q8_0` is pinned bitwise to gguf-quants semantics
by `test_every_kernel_computes_in_fp32`. The fused Triton kernel exists only to
move less memory, so it must be **indistinguishable** — int8 -> fp32 and the
scale -> fp32, multiply in fp32, round exactly once to the output dtype.

A fused kernel that is merely *close* would silently reintroduce the class of
bug reverted on 2026-09-28, where a rounding difference moved 98.1% of weight
elements and shifted VibeVoice's utterance duration from 21s to 24-27s.

These tests need Triton, which is importable under ComfyUI's embedded Python
but NOT under the dev venv. They skip cleanly when unavailable.
"""

import pytest
import torch

from ComfyUI_VibeVoice.modules import gguf_quant as G

# Scales covering the interesting fp16 shapes: sign, subnormal-adjacent,
# ordinary, top of range. NaN/Inf bit patterns are deliberately NOT fixtures —
# torch.equal is False for NaN even against itself, so they would test the
# harness rather than the kernel.
SCALES = [
    ("positive", 0.5),
    ("negative", -1.25),
    ("tiny", 6.1e-5),
    ("max", 65504.0),
    ("zero", 0.0),
]
DTYPES = [torch.bfloat16, torch.float16, torch.float32]


def _blocks(n, scale, seed=0):
    """A (n, 34) uint8 q8_0 buffer: 2 bytes of fp16 scale + 32 bytes of int8."""
    torch.manual_seed(seed)
    b = torch.randint(0, 256, (n, 34), dtype=torch.uint8)
    b[:, :2] = torch.tensor([scale], dtype=torch.float16).view(torch.uint8)
    return b


def _reference(blocks, out_dtype):
    return G.dequantize_blocks(
        blocks, G._T.Q8_0, out_dtype, (blocks.shape[0] * 32,)
    ).reshape(-1)


def _fused(blocks, out_dtype):
    return G._dequant_q8_0_fused(blocks, out_dtype).reshape(-1)


@pytest.fixture
def cuda_required():
    if not (torch.cuda.is_available() and G.triton_q8_0_available()):
        pytest.skip("fused Q8_0 kernel needs CUDA + Triton")


class TestFusedKernelMatchesReference:
    @pytest.mark.parametrize("name,scale", SCALES)
    @pytest.mark.parametrize("out_dtype", DTYPES)
    def test_bitwise_equal(self, cuda_required, name, scale, out_dtype):
        blocks = _blocks(4096, scale).cuda()
        assert torch.equal(_reference(blocks, out_dtype),
                           _fused(blocks, out_dtype)), (
            f"fused kernel differs from the reference for scale={name} "
            f"dtype={out_dtype}"
        )

    def test_signed_payload_not_unsigned(self, cuda_required):
        """The buffer is uint8 but the payload is int8.

        Loading it without a reinterpret gives every value above 127 the wrong
        sign — which is exactly the bug the kernel was first written with.
        """
        blocks = _blocks(4096, 0.5).cuda()
        payload = blocks[:, 2:].view(torch.int8)
        assert (payload < 0).any(), "fixture has no negative values to test"

        fused = _fused(blocks, torch.float32)
        assert torch.equal(fused, _reference(blocks, torch.float32))
        assert (fused < 0).any(), "negative payload produced non-negative output"

    def test_edge_scales_do_not_diverge(self, cuda_required):
        """Subnormal and max-magnitude scales are where an fp32 intermediate
        and a single fused round are most likely to disagree."""
        for i, (_, scale) in enumerate(SCALES):
            blocks = _blocks(2048, scale, seed=i).cuda()
            assert torch.equal(_reference(blocks, torch.bfloat16),
                               _fused(blocks, torch.bfloat16))


class TestCapabilityProbe:
    def test_available_is_a_bool(self):
        assert isinstance(G.triton_q8_0_available(), bool)

    def test_routing_uses_the_fused_kernel_when_enabled(self, cuda_required,
                                                       monkeypatch):
        """Assert on ROUTING, not on values.

        The two paths are bitwise equal, so comparing outputs can never tell
        which one ran — a value-based version of this test passes vacuously and
        reports nothing. Spy on the fused entry point instead.
        """
        called = []
        real = G._dequant_q8_0_fused

        def _spy(blocks, out_dtype):
            called.append(1)
            return real(blocks, out_dtype)

        monkeypatch.setattr(G, "_dequant_q8_0_fused", _spy)
        monkeypatch.setattr(G, "_FUSED_Q8_0_ENABLED", True)
        G.dequantize_blocks(_blocks(512, 0.5).cuda(), G._T.Q8_0,
                            torch.bfloat16, (512 * 32,))
        assert called, "enabled but the fused kernel was never called"

    def test_routing_skips_the_fused_kernel_when_disabled(self, cuda_required,
                                                          monkeypatch):
        """`_FUSED_Q8_0_ENABLED` is False after the live 10x regression."""
        called = []
        real = G._dequant_q8_0_fused

        def _spy(blocks, out_dtype):
            called.append(1)
            return real(blocks, out_dtype)

        monkeypatch.setattr(G, "_dequant_q8_0_fused", _spy)
        monkeypatch.setattr(G, "_FUSED_Q8_0_ENABLED", False)
        out = G.dequantize_blocks(_blocks(512, 0.5).cuda(), G._T.Q8_0,
                                  torch.bfloat16, (512 * 32,))
        assert not called, "disabled but the fused kernel ran anyway"
        # Still correct — the torch chain is the reference.
        assert out.numel() == 512 * 32

    def test_fused_is_disabled_by_default(self):
        """The regression gate, asserted so it cannot be flipped silently."""
        assert G._FUSED_Q8_0_ENABLED is False, (
            "the fused Q8_0 kernel slowed live generation ~10x; enabling it "
            "requires a measured win against test_gguf_dequant_perf.py's "
            "reference numbers, not merely bitwise correctness"
        )

    def test_cpu_tensors_never_take_the_fused_path(self):
        """No CUDA, no Triton kernel — it must fall through to torch."""
        blocks = _blocks(256, 0.5)
        out = G.dequantize_blocks(blocks, G._T.Q8_0, torch.float32, (256 * 32,))
        assert out.device.type == "cpu"
        assert torch.equal(
            out,
            G._dequant_q8_0(blocks.reshape(-1, 34), torch.float32).reshape(-1),
        )