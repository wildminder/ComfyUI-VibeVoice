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

# Toy realtime row: hidden=3, vocab=5. It shares the VOCAB with Toy-1.5B
# exactly as the real realtime model shares 151936 with the real 1.5B, so the
# toy table reproduces the real near-miss (same vocab, different hidden) that
# a single-dimension lookup would get wrong.
_TOY_SIGNATURES_WITH_REALTIME = dict(_TOY_SIGNATURES, **{"Toy-Realtime": (3, 5)})

# Toy ASR family. The real one is (3584, 152064) — the same pair as Toy-7B's
# real counterpart, which is exactly the collision the ASR gate exists for.
_TOY_ASR_SIGNATURES = {"Toy-ASR": (8, 6)}


@pytest.fixture
def toy_signatures(monkeypatch):
    monkeypatch.setattr(cd, "_FAMILY_SIGNATURES", dict(_TOY_SIGNATURES))


@pytest.fixture
def toy_signatures_with_realtime(monkeypatch):
    monkeypatch.setattr(
        cd, "_FAMILY_SIGNATURES", dict(_TOY_SIGNATURES_WITH_REALTIME)
    )


@pytest.fixture
def toy_asr_signatures(monkeypatch):
    monkeypatch.setattr(cd, "_ASR_FAMILY_SIGNATURES", dict(_TOY_ASR_SIGNATURES))


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


class TestRealtimeSignature:
    """VibeVoice-Realtime-0.5B is fingerprinted by (896, 151936).

    The pair is the streaming decoder_config of the packaged asset — the same
    (hidden, vocab) ``config_fingerprint`` reads out of
    ``default_VibeVoice-Realtime-0.5B_config.json``. These tests pin the
    detector side of that contract and, above all, the collisions the new row
    must NOT create.
    """

    def test_real_signature_matches_the_packaged_asset(self):
        """Production table, no fixture: the shape the asset declares."""
        from ComfyUI_VibeVoice.modules.config_detect import config_fingerprint

        assert cd._FAMILY_SIGNATURES["VibeVoice-Realtime-0.5B"] == (896, 151936)
        # The same value the packaged asset's decoder_config yields, so the
        # fingerprint a realtime file produces names the packaged config.
        assert config_fingerprint(
            {"decoder_config": {"hidden_size": 896, "vocab_size": 151936}}
        ) == cd._FAMILY_SIGNATURES["VibeVoice-Realtime-0.5B"]

    def test_real_signature_classifies_both_orientations(self):
        """(896, 151936) and its reverse both name the realtime family."""
        for shape in ((151936, 896), (896, 151936)):
            fp = classify_embedding_shape(shape)
            assert fp is not None, shape
            assert fp.config_name == "VibeVoice-Realtime-0.5B", shape
            assert fp.is_asr is False
            assert (fp.hidden_size, fp.vocab_size) == (896, 151936)

    def test_realtime_never_steals_the_1p5b_signature(self):
        """The mandatory collision guard: 1.5B and realtime share the VOCAB.

        1.5B is (1536, 151936), realtime is (896, 151936). Classification is
        a full-tuple comparison, so the shared vocab dimension alone must not
        decide either row. Both orientations of both pairs are asserted.
        """
        for shape, expected in (
            ((1536, 151936), "VibeVoice-1.5B"),
            ((151936, 1536), "VibeVoice-1.5B"),
            ((896, 151936), "VibeVoice-Realtime-0.5B"),
            ((151936, 896), "VibeVoice-Realtime-0.5B"),
        ):
            fp = classify_embedding_shape(shape)
            assert fp is not None, shape
            assert fp.config_name == expected, shape

    def test_realtime_never_steals_the_7b_signature(self):
        """7B (3584, 152064) is untouched by the new row."""
        for shape in ((152064, 3584), (3584, 152064)):
            fp = classify_embedding_shape(shape)
            assert fp is not None
            assert fp.config_name == "VibeVoice-7B"

    def test_mixed_dimension_pairs_still_return_none(self):
        """A hidden from one row and a vocab from another is not a family.

        This is the case a single-dimension lookup would wrongly accept, so
        it is the load-bearing guard for the shared-vocab collision.
        """
        for shape in (
            (896, 1536),    # realtime hidden + 1.5B vocab
            (1536, 896),    # 1.5B hidden + realtime vocab
            (896, 152064),  # realtime hidden + 7B vocab
            (3584, 151936), # 7B hidden + 1.5B/realtime vocab
            (151936, 3584), # the pre-existing 7B/1.5B near-miss
        ):
            assert classify_embedding_shape(shape) is None, shape

    def test_realtime_shape_is_never_asr(self):
        """The streaming family must not leak into the ASR branch either."""
        fp = cd.classify_asr_embedding_shape((151936, 896))
        assert fp is None
        fp = cd.classify_asr_embedding_shape((896, 151936))
        assert fp is None

    def test_realtime_file_classifies_end_to_end(
        self, tmp_path, toy_signatures_with_realtime
    ):
        """Header-only safetensors path, with the streaming key namespace.

        The realtime state dict carries the HF embedding key name plus the
        streaming acoustic tower's own ``model.``-prefixed keys. Those must
        reach the TTS table (not the ASR gate) and resolve to the realtime row
        while the toy table is in force.
        """
        path = _save_safetensors(
            tmp_path,
            {
                "model.language_model.embed_tokens.weight": torch.zeros(5, 3),
                # The streaming namespace: singular ``acoustic_tokenizer``,
                # always under ``model.`` — none of the ASR-only prefixes.
                "model.acoustic_tokenizer.decoder.head.conv.conv.weight": torch.zeros(2, 2),
                "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(3, 3),
            },
        )
        fp = fingerprint_safetensors(path)
        assert fp is not None
        assert fp.is_asr is False
        assert fp.config_name == "Toy-Realtime"
        assert fp.source_key == "model.language_model.embed_tokens.weight"
        assert detect_config_name(path) == "Toy-Realtime"

    def test_streaming_keys_do_not_trip_the_asr_gate(
        self, toy_signatures_with_realtime, toy_asr_signatures
    ):
        """``model.acoustic_tokenizer.*`` must never be read as ASR-only.

        The ASR gate is a plain ``str.startswith`` over module names. The
        streaming tower's name is close to an ASR-only one but is neither of
        them, and it carries a ``model.`` prefix — if it ever matched, the
        fingerprint would resolve through the ASR branch and return None.
        """
        assert cd._has_asr_only_keys([
            "model.acoustic_tokenizer.encoder.downsample_layers.0.weight",
            "model.acoustic_tokenizer.decoder.head.conv.conv.weight",
        ]) is False

    def test_shared_vocab_is_not_enough_to_classify(
        self, toy_signatures_with_realtime
    ):
        """Toy analogue of the real 1.5B/realtime vocab collision.

        Toy-1.5B is (4, 5) and Toy-Realtime is (3, 5): the same vocab, a
        different hidden. A vocab-only lookup would map (5, 4) and (5, 3) onto
        whichever row it saw first; the full-tuple compare keeps them apart.
        """
        assert classify_embedding_shape((5, 4)).config_name == "Toy-1.5B"
        assert classify_embedding_shape((5, 3)).config_name == "Toy-Realtime"
        assert classify_embedding_shape((4, 5)).config_name == "Toy-1.5B"
        assert classify_embedding_shape((3, 5)).config_name == "Toy-Realtime"
        # hidden from one row, vocab from the other → unrecognised
        assert classify_embedding_shape((4, 3)) is None
        assert classify_embedding_shape((3, 4)) is None


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


class TestAsrAutoDetect:
    """ASR detection is gated on ASR-only key prefixes, never on shape alone.

    The ASR embedding shape (152064, 3584) is IDENTICAL to the VibeVoice-7B
    TTS signature, so shape alone cannot separate them; only the module names
    a TTS checkpoint does not have can.
    """

    def test_asr_prefixes_route_to_asr_family(self, tmp_path, toy_asr_signatures):
        path = _save_safetensors(
            tmp_path,
            {
                "language_model.model.embed_tokens.weight": torch.zeros(6, 8),
                "multi_modal_projector.weight": torch.zeros(2, 2),
            },
        )
        fp = fingerprint_safetensors(path)
        assert fp is not None
        assert fp.is_asr is True
        assert fp.config_name == "Toy-ASR"
        assert fp.source_key == "language_model.model.embed_tokens.weight"
        assert detect_config_name(path) == "Toy-ASR"

    @pytest.mark.parametrize(
        "asr_prefix",
        [
            "multi_modal_projector.",
            "acoustic_tokenizer_encoder.",
            "semantic_tokenizer_encoder.",
        ],
    )
    def test_each_asr_prefix_fires_the_gate(
        self, tmp_path, toy_asr_signatures, asr_prefix
    ):
        path = _save_safetensors(
            tmp_path,
            {
                "language_model.model.embed_tokens.weight": torch.zeros(6, 8),
                f"{asr_prefix}layer.weight": torch.zeros(2, 2),
            },
        )
        fp = fingerprint_safetensors(path)
        assert fp is not None and fp.config_name == "Toy-ASR"

    def test_reversed_asr_shape_detected(self, tmp_path, toy_asr_signatures):
        path = _save_safetensors(
            tmp_path,
            {
                "language_model.model.embed_tokens.weight": torch.zeros(8, 6),
                "acoustic_tokenizer_encoder.weight": torch.zeros(2, 2),
            },
        )
        fp = fingerprint_safetensors(path)
        assert fp is not None and fp.config_name == "Toy-ASR"

    def test_shared_shape_without_asr_prefixes_never_becomes_asr(
        self, tmp_path, toy_signatures, toy_asr_signatures
    ):
        """The mandatory no-misroute case.

        A TTS file whose embedding shape IS an ASR signature must resolve
        through _FAMILY_SIGNATURES and NEVER to "VibeVoice-ASR". Both toy
        families carry the same (8, 6) pair here, which is precisely the
        collision the real tables have (3584, 152064).
        """
        path = _save_safetensors(
            tmp_path, {"model.language_model.embed_tokens.weight": torch.zeros(6, 8)}
        )
        fp = fingerprint_safetensors(path)
        assert fp is not None
        assert fp.is_asr is False
        assert fp.config_name == "Toy-7B"
        assert detect_config_name(path) == "Toy-7B"

    def test_real_measured_pair_classifies_by_flag_alone(self):
        """Pin the real collision: (152064, 3584) is BOTH 'VibeVoice-7B' and
        'VibeVoice-ASR'. The ASR flag is the only discriminator, so no code
        path may pick an ASR name from a shape alone.

        Production tables (no fixture): 152064x3584 weights are far too large
        to materialize, so this is exercised through the classifiers.
        """
        tts_fp = cd.classify_embedding_shape((152064, 3584))
        assert tts_fp is not None
        assert tts_fp.is_asr is False
        assert tts_fp.config_name == "VibeVoice-7B"
        assert tts_fp.config_name != "VibeVoice-ASR"

        asr_fp = cd.classify_asr_embedding_shape((152064, 3584))
        assert asr_fp is not None
        assert asr_fp.is_asr is True
        assert asr_fp.config_name == "VibeVoice-ASR"

    def test_asr_key_without_asr_prefix_is_not_asr(self, tmp_path):
        """The ASR embedding key name alone proves nothing — without an
        ASR-only prefix the gate must not fire."""
        path = _save_safetensors(
            tmp_path,
            {
                "language_model.model.embed_tokens.weight": torch.zeros(6, 8),
                "language_model.model.layers.0.q.weight": torch.zeros(2, 2),
            },
        )
        fp = fingerprint_safetensors(path)
        # (6, 8) matches no production TTS signature → None, but never ASR.
        assert fp is None or fp.config_name != "VibeVoice-ASR"

    def test_asr_prefix_without_asr_embedding_key_never_resolves_as_tts(self, tmp_path):
        """ASR-only prefixes with a differently-named embedding key must NOT
        fall through to the TTS loop.

        _FAMILY_SIGNATURES' only 7B entry is (3584, 152064) — the ASR shape —
        so an ASR variant spelling its embedding ``tok_embeddings.weight``
        would otherwise be classified "VibeVoice-7B" and misrouted to the TTS
        loader. The ASR branch is terminal: unrecognised ASR → None, which the
        loader turns into the actionable "Auto-detect" error.
        """
        path = _save_safetensors(
            tmp_path,
            {
                # ASR-only prefixes present...
                "multi_modal_projector.acoustic_linear_1.weight": torch.zeros(4, 4),
                "acoustic_tokenizer_encoder.conv_layers.0.weight": torch.zeros(2, 2, 2),
                # ...but the embedding uses a TTS-style key name and the
                # EXACT VibeVoice-7B signature shape.
                "tok_embeddings.weight": torch.zeros(152064, 3584),
            },
        )
        assert fingerprint_safetensors(path) is None
        assert detect_config_name(path) is None

    def test_file_with_neither_returns_none(self, tmp_path):
        """Neither an ASR prefix nor a known TTS shape → None, so the
        loader's actionable ValueError (external_loader.
        resolve_auto_config_name) is preserved instead of a guess."""

        path = _save_safetensors(tmp_path, {"unrelated.weight": torch.zeros(2, 2)})
        assert fingerprint_safetensors(path) is None
        assert detect_config_name(path) is None

    def test_unfingerprintable_file_still_raises_actionable_error(self, tmp_path):
        """That None must reach the user as the actionable ValueError."""
        from ComfyUI_VibeVoice.modules.external_loader import (
            resolve_auto_config_name,
        )

        path = _save_safetensors(tmp_path, {"unrelated.weight": torch.zeros(2, 2)})
        with pytest.raises(ValueError) as exc:
            resolve_auto_config_name(path)
        assert "Auto-detect" in str(exc.value)

    def test_real_asr_signature_table_without_fixture(self):
        """Pin the production ASR table values (fixture NOT applied)."""
        for shape in ((152064, 3584), (3584, 152064)):
            fp = cd.classify_asr_embedding_shape(shape)
            assert fp is not None
            assert fp.is_asr is True
            assert fp.config_name == "VibeVoice-ASR"
        assert cd.classify_asr_embedding_shape((151936, 1536)) is None
        assert cd.classify_asr_embedding_shape((6,)) is None
        assert cd.classify_asr_embedding_shape(None) is None

    def test_gguf_never_produces_asr(
        self, tmp_path, toy_signatures, toy_asr_signatures
    ):
        """GGUF ASR quant files are out of scope here; the reader path must
        keep answering from the TTS table alone."""
        path = _write_gguf(
            tmp_path, {"tok_embeddings.weight": np.zeros((5, 4), dtype=np.float32)}
        )
        import gguf

        reader = gguf.GGUFReader(path)
        fp = fingerprint_gguf_reader(reader)
        assert fp is not None
        assert fp.is_asr is False
        assert fp.config_name == "Toy-1.5B"
        assert fp.config_name != "Toy-ASR"


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
