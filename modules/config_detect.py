"""Deterministic architecture fingerprinting for external weight files.

Plan 2026-08-27 (Phase 2, D3/D4): the LLM embedding tensor of a VibeVoice
checkpoint is a perfect fingerprint of its architecture family —

    model.language_model.embed_tokens.weight  shape [vocab, hidden]
        VibeVoice-7B    (152064, 3584)
        VibeVoice-1.5B  (151936, 1536)

    language_model.model.embed_tokens.weight  shape [vocab, hidden]
        VibeVoice-ASR   (152064, 3584)

The ASR shape COLLIDES with the 7B one, so ASR files are recognized first by
module names only they have (``multi_modal_projector.*`` and the two
``*_tokenizer_encoder.*`` towers) and only then classified against their own
signature table. A TTS file never carries those prefixes, so the two
branches cannot be confused.

This module reads that shape WITHOUT materializing any tensor data:

* safetensors → ``safe_open`` reads the JSON header only (instant on
  multi-GB files);
* GGUF → metadata from an already-open ``gguf.GGUFReader`` (mmap; the
  loader opens one anyway and passes it in — never re-opened here);
* .bin/.pt → no cheap header exists → ``None`` (callers fall back to the
  user-selected config).

Classification is exact-match only (never guess): the hidden size must be a
known signature value in EITHER dimension (comfy-kitchen's GGUF writer stores
shapes reversed), and the other dimension must equal that family's vocab.
Contradictory or foreign shapes → ``None``.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional, Sequence


# Candidate embedding tensor names, first found wins.
_EMBEDDING_KEY_CANDIDATES = (
    "model.language_model.embed_tokens.weight",
    "tok_embeddings.weight",
    "token_embd.weight",
)

# config_name -> (hidden_size, vocab_size) of the LLM embedding.
_FAMILY_SIGNATURES = {
    "VibeVoice-7B": (3584, 152064),
    "VibeVoice-1.5B": (1536, 151936),
}

_ASR_ONLY_KEY_PREFIXES = (
    "multi_modal_projector.",
    "acoustic_tokenizer_encoder.",
    "semantic_tokenizer_encoder.",
)

_ASR_EMBEDDING_CANDIDATES = (
    "language_model.model.embed_tokens.weight",
)

# config_name -> (hidden_size, vocab_size) for the ASR family.
_ASR_FAMILY_SIGNATURES = {
    "VibeVoice-ASR": (3584, 152064),
}


@dataclass(frozen=True)
class WeightsFingerprint:
    """Architecture fingerprint extracted from a weight file's header."""

    hidden_size: int
    vocab_size: int
    source_key: str
    is_asr: bool = False

    @property
    def config_name(self) -> str:
        """The config_name whose signature matches this fingerprint."""
        table = _ASR_FAMILY_SIGNATURES if self.is_asr else _FAMILY_SIGNATURES
        for name, (hidden, vocab) in table.items():
            if (hidden, vocab) == (self.hidden_size, self.vocab_size):
                return name
        return ""


def classify_embedding_shape(shape: Sequence[int]) -> Optional[WeightsFingerprint]:
    """Classify a 2-D embedding shape onto a known family, orientation-agnostic."""
    if shape is None or len(shape) != 2:
        return None
    d0, d1 = int(shape[0]), int(shape[1])

    for name, (hidden, vocab) in _FAMILY_SIGNATURES.items():
        if (d0, d1) == (vocab, hidden) or (d0, d1) == (hidden, vocab):
            return WeightsFingerprint(
                hidden_size=hidden, vocab_size=vocab, source_key=""
            )
    return None


def classify_asr_embedding_shape(shape: Sequence[int]) -> Optional[WeightsFingerprint]:
    """Classify a 2-D embedding shape onto the ASR family, orientation-agnostic."""
    if shape is None or len(shape) != 2:
        return None
    d0, d1 = int(shape[0]), int(shape[1])

    for name, (hidden, vocab) in _ASR_FAMILY_SIGNATURES.items():
        if (d0, d1) == (vocab, hidden) or (d0, d1) == (hidden, vocab):
            return WeightsFingerprint(
                hidden_size=hidden, vocab_size=vocab, source_key="", is_asr=True
            )
    return None


def _has_asr_only_keys(keys) -> bool:
    """True when the state dict carries a module only an ASR checkpoint has."""
    for key in keys:
        for prefix in _ASR_ONLY_KEY_PREFIXES:
            if key.startswith(prefix):
                return True
    return False


def fingerprint_safetensors(path: str) -> Optional[WeightsFingerprint]:
    """Fingerprint a ``.safetensors`` file from its header only."""
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            keys = set(f.keys())

            if _has_asr_only_keys(keys):
                for candidate in _ASR_EMBEDDING_CANDIDATES:
                    if candidate in keys:
                        shape = f.get_slice(candidate).get_shape()
                        fp = classify_asr_embedding_shape(shape)
                        if fp is not None:
                            return WeightsFingerprint(
                                hidden_size=fp.hidden_size,
                                vocab_size=fp.vocab_size,
                                source_key=candidate,
                                is_asr=True,
                            )
                        logging.debug(
                            "[VibeVoice TTS] ASR embedding key '%s' in '%s' has foreign shape %s",
                            candidate, os.path.basename(path), shape,
                        )
                        return None
                logging.debug(
                    "[VibeVoice TTS] '%s' carries ASR-only prefixes but none of the ASR "
                    "embedding keys %s; refusing to classify it as a TTS family.",
                    os.path.basename(path), list(_ASR_EMBEDDING_CANDIDATES),
                )
                return None

            for candidate in _EMBEDDING_KEY_CANDIDATES:
                if candidate in keys:
                    shape = f.get_slice(candidate).get_shape()
                    fp = classify_embedding_shape(shape)
                    if fp is not None:
                        return WeightsFingerprint(
                            hidden_size=fp.hidden_size,
                            vocab_size=fp.vocab_size,
                            source_key=candidate,
                        )
                    logging.debug(
                        "[VibeVoice TTS] Embedding key '%s' in '%s' has foreign shape %s",
                        candidate, os.path.basename(path), shape,
                    )
                    return None
    except Exception as e:
        logging.debug("[VibeVoice TTS] Safetensors fingerprint failed for '%s': %s", path, e)
    return None


def fingerprint_gguf_reader(reader) -> Optional[WeightsFingerprint]:
    """Fingerprint GGUF weights from an ALREADY-OPEN ``gguf.GGUFReader``."""
    try:
        by_name = {t.name: t for t in reader.tensors}
        for candidate in _EMBEDDING_KEY_CANDIDATES:
            tensor = by_name.get(candidate)
            if tensor is not None:
                fp = classify_embedding_shape(list(tensor.shape))
                if fp is not None:
                    return WeightsFingerprint(
                        hidden_size=fp.hidden_size,
                        vocab_size=fp.vocab_size,
                        source_key=candidate,
                    )
                logging.debug(
                    "[VibeVoice TTS] GGUF tensor '%s' has foreign shape %s",
                    candidate, list(tensor.shape),
                )
                return None
    except Exception as e:
        logging.debug("[VibeVoice TTS] GGUF fingerprint failed: %s", e)
    return None


def fingerprint_weights(weight_path: str, gguf_reader=None) -> Optional[WeightsFingerprint]:
    """Fingerprint the architecture of a weight file (header-only)."""
    lower = weight_path.lower()

    if lower.endswith(".gguf"):
        if gguf_reader is not None:
            return fingerprint_gguf_reader(gguf_reader)
        try:
            from .gguf_quant import open_gguf_reader

            reader = open_gguf_reader(weight_path)
            return fingerprint_gguf_reader(reader)
        except Exception as e:
            logging.debug("[VibeVoice TTS] GGUF reader open failed for '%s': %s", weight_path, e)
            return None

    if lower.endswith(".safetensors"):
        return fingerprint_safetensors(weight_path)

    return None


def detect_config_name(weight_path: str, gguf_reader=None) -> Optional[str]:
    """Detect the architecture family of a weight file."""
    fp = fingerprint_weights(weight_path, gguf_reader=gguf_reader)
    return fp.config_name if fp is not None else None


def config_fingerprint(config) -> Optional[tuple]:
    """Extract ``(hidden_size, vocab_size)`` from a loaded VibeVoice config."""
    decoder = getattr(config, "decoder_config", None)
    if decoder is None and isinstance(config, dict):
        decoder = config.get("decoder_config")
    if decoder is None:
        return None

    hidden = getattr(decoder, "hidden_size", None)
    vocab = getattr(decoder, "vocab_size", None)
    if hidden is None and isinstance(decoder, dict):
        hidden, vocab = decoder.get("hidden_size"), decoder.get("vocab_size")
    if hidden is None or vocab is None:
        return None
    return (int(hidden), int(vocab))