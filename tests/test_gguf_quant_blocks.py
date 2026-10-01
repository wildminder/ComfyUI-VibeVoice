"""Phase A tests: GGUF block dequantization kernels, bitwise vs gguf-py oracle.

gguf-py can QUANTIZE only F32/F16/BF16/Q8_0 (K-quant quantize_blocks raises
NotImplementedError), so K-quant test data comes from the conftest handcrafted
block builders with controlled fp16 scales; the oracle dequantizes whatever
bytes we craft.
"""

import os

import numpy as np
import pytest
import torch
import gguf
from gguf.constants import GGMLQuantizationType as T
from gguf.quants import dequantize as oracle_dequantize, quantize as oracle_quantize

from ComfyUI_VibeVoice.modules import gguf_quant as G


def _seeded_float(shape, seed=42):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * 0.05).astype(np.float32)


class TestBitwiseParityAgainstOracle:
    """dequantize_blocks must be BITWISE equal to gguf.dequantize."""

    def _check(self, raw_bytes_u8, qtype, logical_shape):
        ref = oracle_dequantize(raw_bytes_u8.view(np.uint8), qtype)
        ours = G.dequantize_blocks(
            torch.from_numpy(raw_bytes_u8.copy()), qtype,
            torch.float32, tuple(ref.shape),
        )
        assert np.array_equal(ref, ours.numpy())

    def _parity(self, raw_bytes, qtype, shape):
        """Assert our dequantizer is bitwise equal to gguf-py's.

        ``raw_bytes`` is whatever a ``GGUFReader`` tensor carries (the mmap is
        read through here, not copied wholesale) and ``shape`` is the
        TORCH-logical shape. The reader reports ggml (reversed) dims, so the
        caller is responsible for having already flipped them.
        """
        ref = oracle_dequantize(raw_bytes, qtype).reshape(shape)
        ours = G.dequantize_blocks(
            torch.from_numpy(np.ascontiguousarray(raw_bytes)).view(torch.uint8),
            qtype, torch.float32, shape,
        )
        assert np.array_equal(ref, ours.numpy())

    def test_q8_0_quantized_by_oracle(self):
        for shape in [(64, 32), (37, 96), (1, 32)]:
            x = _seeded_float(shape)
            raw = oracle_quantize(x, T.Q8_0)
            self._check(raw.reshape(-1), T.Q8_0, None)

    def test_q4_k_crafted(self):
        for nb in (1, 3, 17, 129):
            blocks = craft_for_test("Q4_K", nb, seed=nb)
            self._check(blocks.reshape(-1), T.Q4_K, None)

    def test_q5_k_crafted(self):
        for nb in (1, 9, 65):
            blocks = craft_for_test("Q5_K", nb, seed=nb)
            self._check(blocks.reshape(-1), T.Q5_K, None)

    def test_q6_k_crafted(self):
        for nb in (1, 5, 33):
            blocks = craft_for_test("Q6_K", nb, seed=nb)
            self._check(blocks.reshape(-1), T.Q6_K, None)

    def test_synthetic_q8_0_tensor(self, make_gguf_file):
        """Bitwise parity on a Q8_0 tensor with a realistic block count.

        This is the machine-independent half of the real-checkpoint check: the
        synthetic file carries a 2048x256 Q8_0 weight (16 K blocks, the same
        order as a real projection), so the oracle comparison below runs on
        every machine and in CI instead of only where a 3.2 GB fixture happens
        to live. See ``test_real_file_q8_0_tensor`` for the real checkpoint.
        """
        path = make_gguf_file(
            [("model.language_model.layers.0.self_attn.q_proj.weight",
              "Q8_0", (2048, 256))],
            tag="q8_oracle",
        )
        reader = gguf.GGUFReader(str(path))
        t = max(reader.tensors, key=lambda x: int(np.prod(x.shape)))
        assert t.tensor_type == T.Q8_0
        shape = tuple(int(s) for s in t.shape)[::-1]
        self._parity(t.data, t.tensor_type, shape)

    def test_real_file_q8_0_tensor(self):
        """Bitwise parity on a REAL tensor from the user's VibeVoice GGUF.

        The file is selected by TENSOR TYPE, not by name: this checkpoint
        quantises 378 Q8_0 tensors but names none of them
        `*.self_attn.q_proj.weight` (the vendored tree uses a different module
        layout), so the old name filter raised StopIteration and the oracle
        check never ran. The largest Q8_0 tensor is used so the comparison
        exercises a realistic block count rather than a single block.

        Optional by construction: the path is overridable via
        ``VIBEVOICE_TEST_GGUF`` and the test skips when no real checkpoint is
        present. It must stay optional — the 3.2 GB file is not something a
        unit suite can require, and nothing here may load a real model. The
        always-runs coverage of the same code path is
        ``test_synthetic_q8_0_tensor`` above.
        """
        path = os.environ.get("VIBEVOICE_TEST_GGUF", "")
        if not path:
            pytest.skip(
                "set VIBEVOICE_TEST_GGUF to a real q8_0 VibeVoice GGUF to run "
                "this; it stays optional because the file is 3.2 GB"
            )
        try:
            reader = gguf.GGUFReader(path)
        except Exception:
            pytest.skip("real vibevoice gguf not present")
        candidates = [t for t in reader.tensors if t.tensor_type == T.Q8_0]
        assert candidates, (
            f"{path} has no Q8_0 tensor (types: "
            f"{sorted({t.tensor_type.name for t in reader.tensors})}); the "
            f"fixture is not the q8_0 checkpoint this test is written for."
        )
        t = max(candidates, key=lambda x: int(np.prod(x.shape)))
        shape = tuple(int(s) for s in t.shape)
        self._parity(t.data, t.tensor_type, shape)


def craft_for_test(qtype_name: str, n_blocks: int, seed: int = 0):
    """Thin re-export so this module does not depend on conftest internals."""
    from conftest import craft_kquant_blocks

    block_size, type_size = {
        "Q8_0": (32, 34), "Q4_K": (256, 144),
        "Q5_K": (256, 176), "Q6_K": (256, 210),
    }[qtype_name]
    return craft_kquant_blocks(qtype_name, n_blocks * block_size, seed=seed)


class TestDenseFloatPaths:
    def test_f32_zero_copy_view(self):
        arr = _seeded_float((9, 17))
        t = torch.from_numpy(arr)
        out = G.dequantize_dense(t, T.F32)
        assert out.data_ptr() == t.data_ptr()
        assert torch.equal(out, t)

    def test_f16_view_exact(self):
        x = _seeded_float((7, 33)).astype(np.float16)
        t = torch.from_numpy(np.ascontiguousarray(x))
        out = G.dequantize_dense(t, T.F16)
        assert out.dtype == torch.float16
        assert torch.equal(out, torch.from_numpy(x.copy()))

    def test_bf16_view_matches_oracle_fp32(self):
        x = _seeded_float((4, 8))
        bf_bytes = oracle_quantize(x, T.BF16)  # uint8
        ref = oracle_dequantize(bf_bytes.view(np.uint8), T.BF16).reshape(4, 8)
        ours = G.dequantize_dense(torch.from_numpy(np.ascontiguousarray(bf_bytes)), T.BF16)
        assert torch.equal(torch.from_numpy(ref.copy()), ours.float())


class TestExpectedNumel:
    @pytest.mark.parametrize("kind,n_elem,expected", [
        ("Q8_0", 32, 34),
        ("Q8_0", 96, 102),
        ("Q4_K", 256, 144),
        ("Q5_K", 256, 176),
        ("Q6_K", 256, 210),
    ])
    def test_formulas(self, kind, n_elem, expected):
        assert G.expected_numel(n_elem, getattr(T, kind)) == expected

    def test_rejects_indivisible(self):
        with pytest.raises(ValueError, match="multiple of"):
            G.expected_numel(100, T.Q4_K)


class TestGGUFTensor:
    def test_from_reader_tensor_reverses_shape_and_clones_bytes(self, make_gguf_file):
        spec = [("model.language_model.layers.0.self_attn.q_proj.weight", "Q8_0", (16, 32))]
        path = make_gguf_file(spec, tag="gt")
        reader = gguf.GGUFReader(str(path))
        rt = reader.tensors[0]

        gt = G.GGUFTensor.from_reader_tensor(rt)
        # Reader reports REVERSED dims; GGUFTensor stores TORCH-logical shape.
        assert gt.shape == (16, 32)
        assert gt.ggml_type == T.Q8_0
        assert gt.raw.dtype == torch.uint8
        expected_bytes = int(rt.n_bytes)
        assert gt.n_bytes == expected_bytes
        assert np.array_equal(
            gt.raw.numpy().reshape(-1),
            np.ascontiguousarray(rt.data).reshape(-1),
        )

    def test_to_moves_raw_and_rejects_dtype(self, make_gguf_file):
        spec = [("w", "Q8_0", (16, 32))]
        path = make_gguf_file(spec, tag="gtto")
        gt = G.GGUFTensor.from_reader_tensor(gguf.GGUFReader(str(path)).tensors[0])
        moved = gt.to(torch.device("meta"))
        assert moved.raw.device.type == "meta"
        assert moved.shape == gt.shape
        with pytest.raises(ValueError, match="raw bytes"):
            gt.to(torch.device("cpu"), dtype=torch.float32)

    def test_dequantize_method(self, make_gguf_file):
        spec = [("w", "Q8_0", (16, 32))]
        path = make_gguf_file(spec, tag="gtdq")
        gt = G.GGUFTensor.from_reader_tensor(gguf.GGUFReader(str(path)).tensors[0])
        out = gt.dequantize(torch.float32)
        assert out.shape == (16, 32)


class TestErrorTaxonomy:
    def test_unsupported_type_message_quality(self):
        err = G.UnsupportedGGMLType(T.Q2_K, "blk.3.attn_v.weight")
        msg = str(err)
        assert "blk.3.attn_v.weight" in msg
        assert "Q2_K" in msg
        assert "Q8_0" in msg  # supported list included

    def test_dequantize_blocks_rejects_unsupported(self):
        raw = torch.zeros(210, dtype=torch.uint8)
        with pytest.raises(G.UnsupportedGGMLType):
            G.dequantize_blocks(raw, T.Q2_K, torch.float32, (1, 256))

    def test_dequantize_blocks_rejects_bad_byte_count(self):
        with pytest.raises(ValueError, match="multiple"):
            G.dequantize_blocks(torch.zeros(33, dtype=torch.uint8),
                                T.Q8_0, torch.float32, (1, 32))
