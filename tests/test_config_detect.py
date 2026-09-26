"""Tests for modules/config_detect.py — deterministic architecture detection.

Plan 2026-08-27 (Phase 2). Real file I/O is exercised with tiny synthetic
checkpoints: the family signature table is monkeypatched to a toy family so
tests stay millisecond-fast while running the genuine safetensors/GGUF
reader code paths.
"""

import numpy as np
import pytest
import torch

from ComfyUI_VibeVoice.modules import config_detect as cd
from ComfyUI_VibeVoice.modules.config_detect import (
    classify_embedding_shape,
    detect_config_name,
    fingerprint_gguf_reader,
    fingerprint_safetensors,
)

# Toy family: hidden=8, vocab=6 (real ones are 3584/152064 and 1536/151936 —
# far too big to materialize in unit tests).
_TOY_SIGNATURES = {"Toy-7B": (8, 6), "Toy-1.5B": (4, 5)}


@pytest.fixture
def toy_signatures(monkeypatch):
    monkeypatch.setattr(cd, "_FAMILY_SIGNATURES", dict(_TOY_SIGNATURES))


def _save_safetensors(tmp_path, tensors: dict, name="model.safetensors") -> str:
    from safetensors.torch import save_file

    path = tmp_path / name
    save_file(tensors, str(path))
    return str(path)


def _write_gguf(tmp_path, tensors: dict, name="model.gguf") -> str:
    """Write a minimal real GGUF file with the given named float tensors."""
    import gguf

    path = str(tmp_path / name)
    writer = gguf.GGUFWriter(path, arch="llama")
    for tensor_name, array in tensors.items():
        writer.add_tensor(tensor_name, array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    return path


class TestClassifyEmbeddingShape:
    """Pure classifier: exact-match, orientation-agnostic, never guesses."""

    def test_toy_family_standard_orientation(self, toy_signatures):
        fp = classify_embedding_shape((6, 8))  # [vocab, hidden]
        assert fp is not None
        assert fp.config_name == "Toy-7B"
        assert fp.hidden_size == 8 and fp.vocab_size == 6

    def test_toy_family_reversed_orientation(self, toy_signatures):
        """comfy-kitchen's GGUF writer stores shapes reversed."""
        fp = classify_embedding_shape((8, 6))  # [hidden, vocab]
        assert fp is not None
        assert fp.config_name == "Toy-7B"

    def test_second_toy_family(self, toy_signatures):
        fp = classify_embedding_shape((5, 4))
        assert fp is not None and fp.config_name == "Toy-1.5B"

    def test_contradictory_dims_return_none(self, toy_signatures):
        # hidden from one family, vocab from another → ambiguous.
        assert classify_embedding_shape((5, 8)) is None

    def test_foreign_shape_returns_none(self, toy_signatures):
        assert classify_embedding_shape((4096, 4096)) is None

    def test_non_2d_returns_none(self, toy_signatures):
        assert classify_embedding_shape((8,)) is None
        assert classify_embedding_shape((6, 8, 2)) is None
        assert classify_embedding_shape(None) is None

    def test_real_signatures_without_fixture(self):
        """Pin the production table values (fixture NOT applied)."""
        fp = classify_embedding_shape((152064, 3584))
        assert fp is not None and fp.config_name == "VibeVoice-7B"
        fp = classify_embedding_shape((3584, 152064))
        assert fp is not None and fp.config_name == "VibeVoice-7B"
        fp = classify_embedding_shape((151936, 1536))
        assert fp is not None and fp.config_name == "VibeVoice-1.5B"
        fp = classify_embedding_shape((1536, 151936))
        assert fp is not None and fp.config_name == "VibeVoice-1.5B"
        assert classify_embedding_shape((151936, 3584)) is None


class TestFingerprintSafetensors:
    """Header-only safetensors fingerprinting via real files."""

    def test_hf_named_embedding_detected(self, tmp_path, toy_signatures):
        path = _save_safetensors(
            tmp_path,
            {"model.language_model.embed_tokens.weight": torch.zeros(6, 8)},
        )
        fp = fingerprint_safetensors(path)
        assert fp is not None
        assert fp.config_name == "Toy-7B"
        assert fp.source_key == "model.language_model.embed_tokens.weight"

    def test_reversed_shape_detected(self, tmp_path, toy_signatures):
        path = _save_safetensors(
            tmp_path,
            {"model.language_model.embed_tokens.weight": torch.zeros(8, 6)},
        )
        fp = fingerprint_safetensors(path)
        assert fp is not None and fp.config_name == "Toy-7B"

    def test_llamacpp_named_embedding_detected(self, tmp_path, toy_signatures):
        path = _save_safetensors(
            tmp_path, {"tok_embeddings.weight": torch.zeros(5, 4)}
        )
        fp = fingerprint_safetensors(path)
        assert fp is not None and fp.config_name == "Toy-1.5B"

    def test_no_candidate_keys_returns_none(self, tmp_path, toy_signatures):
        path = _save_safetensors(tmp_path, {"some.other.weight": torch.zeros(6, 8)})
        assert fingerprint_safetensors(path) is None

    def test_foreign_embedding_shape_returns_none(self, tmp_path, toy_signatures):
        path = _save_safetensors(
            tmp_path,
            {"model.language_model.embed_tokens.weight": torch.zeros(9, 9)},
        )
        assert fingerprint_safetensors(path) is None

    def test_missing_file_returns_none(self, toy_signatures):
        assert fingerprint_safetensors("/nonexistent/model.safetensors") is None

    def test_corrupt_file_returns_none(self, tmp_path, toy_signatures):
        bad = tmp_path / "bad.safetensors"
        bad.write_bytes(b"this is not a safetensors file")
        assert fingerprint_safetensors(str(bad)) is None


class TestFingerprintGgufReader:
    """GGUF fingerprinting from real minimal GGUF files (metadata only)."""

    def test_llamacpp_named_tensor_detected(self, tmp_path, toy_signatures):
        path = _write_gguf(
            tmp_path, {"tok_embeddings.weight": np.zeros((5, 4), dtype=np.float32)}
        )
        import gguf

        reader = gguf.GGUFReader(path)
        fp = fingerprint_gguf_reader(reader)
        assert fp is not None and fp.config_name == "Toy-1.5B"
        assert fp.source_key == "tok_embeddings.weight"

    def test_hf_named_tensor_detected_reversed(self, tmp_path, toy_signatures):
        path = _write_gguf(
            tmp_path,
            {"model.language_model.embed_tokens.weight": np.zeros((8, 6), dtype=np.float32)},
        )
        import gguf

        reader = gguf.GGUFReader(path)
        fp = fingerprint_gguf_reader(reader)
        assert fp is not None and fp.config_name == "Toy-7B"

    def test_unrelated_tensors_return_none(self, tmp_path, toy_signatures):
        path = _write_gguf(
            tmp_path, {"blk.0.attn_q.weight": np.zeros((8, 8), dtype=np.float32)}
        )
        import gguf

        reader = gguf.GGUFReader(path)
        assert fingerprint_gguf_reader(reader) is None

    def test_modern_llamacpp_token_embd_detected(self, tmp_path, toy_signatures):
        """Modern llama.cpp 'token_embd.weight' name fingerprints too
        (quantui-rs 'new' exports name the LM this way)."""
        path = _write_gguf(
            tmp_path, {"token_embd.weight": np.zeros((5, 4), dtype=np.float32)}
        )
        import gguf

        reader = gguf.GGUFReader(path)
        fp = fingerprint_gguf_reader(reader)
        assert fp is not None and fp.config_name == "Toy-1.5B"
        assert fp.source_key == "token_embd.weight"

    def test_foreign_embedding_shape_returns_none(self, tmp_path, toy_signatures):
        path = _write_gguf(
            tmp_path, {"tok_embeddings.weight": np.zeros((9, 9), dtype=np.float32)}
        )
        import gguf

        reader = gguf.GGUFReader(path)
        assert fingerprint_gguf_reader(reader) is None


class TestDetectConfigNameDispatch:
    """detect_config_name(): extension-based routing + fail-safe Nones."""

    def test_safetensors_dispatch(self, tmp_path, toy_signatures):
        path = _save_safetensors(
            tmp_path,
            {"model.language_model.embed_tokens.weight": torch.zeros(6, 8)},
        )
        assert detect_config_name(path) == "Toy-7B"

    def test_gguf_dispatch_with_provided_reader(self, tmp_path, toy_signatures):
        path = _write_gguf(
            tmp_path, {"tok_embeddings.weight": np.zeros((6, 8), dtype=np.float32)}
        )
        import gguf

        reader = gguf.GGUFReader(path)
        assert detect_config_name(path, gguf_reader=reader) == "Toy-7B"

    def test_gguf_dispatch_opens_defensively_without_reader(
        self, tmp_path, toy_signatures
    ):
        path = _write_gguf(
            tmp_path, {"tok_embeddings.weight": np.zeros((6, 8), dtype=np.float32)}
        )
        assert detect_config_name(path) == "Toy-7B"

    def test_bin_returns_none_without_touching_file(self, tmp_path, toy_signatures):
        # No file created on purpose: .bin must short-circuit before I/O.
        assert detect_config_name(str(tmp_path / "model.bin")) is None
        assert detect_config_name(str(tmp_path / "model.pt")) is None

    def test_unknown_extension_returns_none(self, tmp_path, toy_signatures):
        assert detect_config_name(str(tmp_path / "model.onnx")) is None

    def test_unfingerprintable_safetensors_returns_none(
        self, tmp_path, toy_signatures
    ):
        path = _save_safetensors(tmp_path, {"unrelated.weight": torch.zeros(2, 2)})
        assert detect_config_name(path) is None
