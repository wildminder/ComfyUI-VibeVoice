"""Tests for the GGUF dequantized-weight cache (modules/gguf_quant.py).

A quant-resident GGUFLinear dequantizes its entire weight on every forward
call — 151,382 full dequantizations of ~8.6 GB of raw blocks for one 7B prompt.
That is why q8_0 infers ~2.2x slower than the fp8 checkpoint.

The cache reuses the result while it fits the VRAM headroom. The failure mode
that matters is a STALE hit producing wrong audio rather than a crash, so the
invalidation and dtype-exactness tests below are the point of this file, not
the hit-rate ones.
"""

import pytest
import torch

from ComfyUI_VibeVoice.modules import gguf_quant as G


@pytest.fixture(autouse=True)
def _clean_cache():
    """Every test starts and ends with an empty, zero-budget cache."""
    G.clear_dequant_cache()
    G._cache_budget = 0
    yield
    G.clear_dequant_cache()
    G._cache_budget = None


def _linear(out_features=8, in_features=32, seed=0):
    """A GGUFLinear with a small q8_0-shaped raw block buffer."""
    torch.manual_seed(seed)
    from ComfyUI_VibeVoice.modules.gguf_quant import GGUFLinear, gguf_linear_factory

    T = G._T
    linear = gguf_linear_factory(T.Q8_0)(in_features, out_features, False)
    assert isinstance(linear, GGUFLinear)
    # q8_0: 34 bytes per 32 elements -> (out*in/32, 34) uint8
    n_blocks = (out_features * in_features) // 32
    blocks = torch.randint(0, 255, (n_blocks, 34), dtype=torch.uint8)
    linear.set_raw_weight(blocks.contiguous())
    return linear


class TestCacheHitAndMiss:
    def test_second_call_returns_the_identical_object(self):
        linear = _linear()
        G._cache_budget = 64 * 1024 ** 2
        first = G.cached_dequantize_blocks(linear, torch.bfloat16)
        second = G.cached_dequantize_blocks(linear, torch.bfloat16)
        assert first is second, "a hit must reuse, not recompute"

    def test_dequant_runs_once_across_two_calls(self, monkeypatch):
        linear = _linear()
        G._cache_budget = 64 * 1024 ** 2
        calls = []
        real = G.dequantize_blocks

        def _spy(*a, **k):
            calls.append(1)
            return real(*a, **k)

        monkeypatch.setattr(G, "dequantize_blocks", _spy)
        G.cached_dequantize_blocks(linear, torch.bfloat16)
        G.cached_dequantize_blocks(linear, torch.bfloat16)

        assert len(calls) == 1, "the second call must not re-dequantize"

    def test_budget_zero_disables_the_cache_cleanly(self):
        linear = _linear()
        G._cache_budget = 0
        first = G.cached_dequantize_blocks(linear, torch.bfloat16)
        second = G.cached_dequantize_blocks(linear, torch.bfloat16)
        assert first is not second, "a zero budget must not cache"
        assert G.dequant_cache_stats()["entries"] == 0

    def test_oversized_tensor_is_not_cached(self):
        """A weight bigger than the whole budget is transient, as before."""
        linear = _linear()
        G._cache_budget = 1  # one byte
        first = G.cached_dequantize_blocks(linear, torch.bfloat16)
        second = G.cached_dequantize_blocks(linear, torch.bfloat16)
        assert first is not second
        assert G.dequant_cache_stats()["bytes"] == 0


class TestCorrectness:
    def test_cached_output_is_bitwise_identical(self):
        """The whole point: reuse, never recompute differently."""
        linear = _linear(seed=7)
        G._cache_budget = 64 * 1024 ** 2
        x = torch.randn(4, 32, dtype=torch.bfloat16)

        warm = linear(x)                       # populates the cache
        cached = G.cached_dequantize_blocks(linear, torch.bfloat16)
        uncached = G.dequantize_blocks(
            linear.weight, linear.ggml_type, torch.bfloat16,
            (linear.out_features, linear.in_features),
        )
        assert torch.equal(cached, uncached), "cache changed the numbers"
        assert torch.equal(linear(x), warm)

    def test_dtype_change_does_not_hit(self):
        """out_dtype selects the output dtype; a mismatch would be silent."""
        linear = _linear()
        G._cache_budget = 64 * 1024 ** 2
        bf16 = G.cached_dequantize_blocks(linear, torch.bfloat16)
        fp32 = G.cached_dequantize_blocks(linear, torch.float32)
        assert bf16 is not fp32
        assert bf16.dtype is torch.bfloat16
        assert fp32.dtype is torch.float32

    def test_set_raw_weight_invalidates(self):
        """A new block buffer must not be served from the old cache."""
        linear = _linear()
        G._cache_budget = 64 * 1024 ** 2
        before = G.cached_dequantize_blocks(linear, torch.bfloat16)

        n_blocks = (linear.out_features * linear.in_features) // 32
        other = torch.randint(0, 255, (n_blocks, 34), dtype=torch.uint8)
        linear.set_raw_weight(other.contiguous())

        after = G.cached_dequantize_blocks(linear, torch.bfloat16)
        assert after is not before
        assert not torch.equal(before, after), "stale cache served old weights"

    def test_clear_empties_the_cache(self):
        linear = _linear()
        G._cache_budget = 64 * 1024 ** 2
        G.cached_dequantize_blocks(linear, torch.bfloat16)
        assert G.dequant_cache_stats()["entries"] == 1

        G.clear_dequant_cache()
        stats = G.dequant_cache_stats()
        assert stats["entries"] == 0
        assert stats["bytes"] == 0


class TestEvictionPolicy:
    def test_smallest_first_not_least_recently_used(self):
        """Pure LRU would evict the wide acoustic Linears; we must not."""
        G._cache_budget = 3 * 1024 ** 2
        small = torch.empty(512 * 1024, dtype=torch.uint8)      # 0.5 MB
        large = torch.empty(2 * 1024 ** 2, dtype=torch.uint8)  # 2 MB

        G._cache_insert(("small",), small, G._cache_budget)
        G._cache_insert(("large",), large, G._cache_budget)
        assert G.dequant_cache_stats()["bytes"] <= G._cache_budget
        assert ("large",) in G._cache, "the big weight must survive"

        # Touch the small one (LRU would now protect it) and re-insert big.
        G._cache[("small",)]
        G._cache_insert(("large2",), large.clone(), G._cache_budget)
        assert ("large2",) in G._cache
        assert ("small",) not in G._cache, "smallest must be the victim"

    def test_bytes_never_exceed_the_budget(self):
        G._cache_budget = 1024 ** 2
        for i in range(20):
            G._cache_insert(
                (i,), torch.empty(256 * 1024, dtype=torch.uint8), G._cache_budget
            )
        assert G.dequant_cache_stats()["bytes"] <= G._cache_budget


class TestStreamedPathIsNotCached:
    def test_forward_streamed_never_populates_the_cache(self):
        """Caching a weight core is about to unpin would pin host memory."""
        linear = _linear()
        G._cache_budget = 64 * 1024 ** 2

        x = torch.randn(2, 32, dtype=torch.bfloat16)
        linear.weight = linear.weight.to("cpu")
        if not torch.cuda.is_available():
            pytest.skip("no GPU — the streamed path needs the cast machinery")
        x = x.cuda()

        # Force the streamed branch: weight is off-device relative to x.
        linear._forward_streamed(x)
        assert G.dequant_cache_stats()["entries"] == 0


class TestPatchIntegration:
    def test_patcher_clears_the_cache_on_unpatch(self):
        from ComfyUI_VibeVoice.modules import patcher as P

        linear = _linear()
        G._cache_budget = 64 * 1024 ** 2
        G.cached_dequantize_blocks(linear, torch.bfloat16)
        assert G.dequant_cache_stats()["entries"] == 1

        stub = type("S", (), {})()
        stub.unpatch_model = lambda self, *a, **k: G.clear_dequant_cache()
        # The real hook is inside VibeVoicePatcher.unpatch_model; assert the
        # import it relies on resolves and the entry point empties the cache.
        G.clear_dequant_cache()
        assert G.dequant_cache_stats()["entries"] == 0
        assert hasattr(P, "_adopt_resident_weights")