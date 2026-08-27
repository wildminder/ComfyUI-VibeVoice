"""Deterministic architecture fingerprinting for external weight files.

Plan 2026-08-27 (Phase 2, D3/D4): the LLM embedding tensor of a VibeVoice
checkpoint is a perfect fingerprint of its architecture family —

    model.language_model.embed_tokens.weight  shape [vocab, hidden]
        VibeVoice-7B    (152064, 3584)
        VibeVoice-1.5B  (151936, 1536)

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

logger = logging.getLogger(__name__)

# Candidate embedding tensor names, first found wins. HF-style checkpoints
# carry the full name; llama.cpp-converted GGUF files use the short name
# (see modules/gguf_quant.py:_LLAMACPP_TO_HF). The lm head is NOT used:
# 1.5B checkpoints tie word embeddings and may omit it entirely.
_EMBEDDING_KEY_CANDIDATES = (
    "model.language_model.embed_tokens.weight",
    "tok_embeddings.weight",
)

# config_name -> (hidden_size, vocab_size) of the LLM embedding. Extend this
# table to teach detection new families; nothing else needs to change.
_FAMILY_SIGNATURES = {
    "VibeVoice-7B": (3584, 152064),
    "VibeVoice-1.5B": (1536, 151936),
}


@dataclass(frozen=True)
class WeightsFingerprint:
    """Architecture fingerprint extracted from a weight file's header."""

    hidden_size: int
    vocab_size: int
    source_key: str

    @property
    def config_name(self) -> str:
        """The config_name whose signature matches this fingerprint."""
        for name, (hidden, vocab) in _FAMILY_SIGNATURES.items():
            if (hidden, vocab) == (self.hidden_size, self.vocab_size):
                return name
        return ""


def classify_embedding_shape(shape: Sequence[int]) -> Optional[WeightsFingerprint]:
    """Classify a 2-D embedding shape onto a known family, orientation-agnostic.

    Args:
        shape: The embedding tensor's shape, ``[vocab, hidden]`` or reversed.

    Returns:
        A :class:`WeightsFingerprint` when the shape matches exactly one
        known family, else ``None`` (foreign/contradictory shapes never
        guess).
    """
    if shape is None or len(shape) != 2:
        return None
    d0, d1 = int(shape[0]), int(shape[1])

    for name, (hidden, vocab) in _FAMILY_SIGNATURES.items():
        if (d0, d1) == (vocab, hidden) or (d0, d1) == (hidden, vocab):
            return WeightsFingerprint(
                hidden_size=hidden, vocab_size=vocab, source_key=""
            )
    return None


def fingerprint_safetensors(path: str) -> Optional[WeightsFingerprint]:
    """Fingerprint a ``.safetensors`` file from its header only.

    No tensor data is read or materialized.

    Args:
        path: Absolute path to the safetensors file.

    Returns:
        Fingerprint or ``None`` (no candidate key / unreadable / foreign).
    """
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            keys = set(f.keys())
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
                    logger.debug(
                        "Embedding key '%s' in '%s' has foreign shape %s",
                        candidate, os.path.basename(path), shape,
                    )
                    return None
    except Exception as e:
        logger.debug("Safetensors fingerprint failed for '%s': %s", path, e)
    return None


def fingerprint_gguf_reader(reader) -> Optional[WeightsFingerprint]:
    """Fingerprint GGUF weights from an ALREADY-OPEN ``gguf.GGUFReader``.

    Uses tensor metadata only (names + shapes); no dequantization. Accepts
    the reader the loader already opened so the file is never re-opened.

    Args:
        reader: An open ``gguf.GGUFReader`` instance.

    Returns:
        Fingerprint or ``None`` (no candidate tensor / foreign shape).
    """
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
                logger.debug(
                    "GGUF tensor '%s' has foreign shape %s",
                    candidate, list(tensor.shape),
                )
                return None
    except Exception as e:
        logger.debug("GGUF fingerprint failed: %s", e)
    return None


def fingerprint_weights(weight_path: str, gguf_reader=None) -> Optional[WeightsFingerprint]:
    """Fingerprint the architecture of a weight file (header-only).

    Dispatch facade (plan 2026-08-27, Step 2.4):

    * ``.gguf`` → :func:`fingerprint_gguf_reader` (reuses ``gguf_reader``
      when provided; opens one defensively otherwise);
    * ``.safetensors`` → :func:`fingerprint_safetensors` (header-only);
    * anything else → ``None`` (no cheap header for .bin/.pt).

    Args:
        weight_path: Absolute path to the weight file.
        gguf_reader: Optional open ``gguf.GGUFReader`` for ``.gguf`` files.

    Returns:
        A :class:`WeightsFingerprint` or ``None`` when the architecture
        cannot be determined deterministically.
    """
    lower = weight_path.lower()

    if lower.endswith(".gguf"):
        if gguf_reader is not None:
            return fingerprint_gguf_reader(gguf_reader)
        try:
            import gguf

            # GGUFReader mmaps the file and has no close() in current
            # gguf versions; leaving it to GC matches the loader's own
            # usage pattern.
            reader = gguf.GGUFReader(weight_path)
            return fingerprint_gguf_reader(reader)
        except Exception as e:
            logger.debug("GGUF reader open failed for '%s': %s", weight_path, e)
            return None

    if lower.endswith(".safetensors"):
        return fingerprint_safetensors(weight_path)

    return None


def detect_config_name(weight_path: str, gguf_reader=None) -> Optional[str]:
    """Detect the architecture family of a weight file.

    Thin wrapper over :func:`fingerprint_weights` returning the config_name
    string (or ``None``).

    Args:
        weight_path: Absolute path to the weight file.
        gguf_reader: Optional open ``gguf.GGUFReader`` for ``.gguf`` files.

    Returns:
        The detected config_name (e.g. ``"VibeVoice-7B"``) or ``None``.
    """
    fp = fingerprint_weights(weight_path, gguf_reader=gguf_reader)
    return fp.config_name if fp is not None else None


def config_fingerprint(config) -> Optional[tuple]:
    """Extract ``(hidden_size, vocab_size)`` from a loaded VibeVoice config.

    Works on the config OBJECT (already parsed by the loader), reading the
    nested LLM (``decoder_config``) section that carries the qwen2
    hidden/vocab values.

    Args:
        config: A loaded ``VibeVoiceConfig``-like object, or a raw dict.

    Returns:
        ``(hidden_size, vocab_size)`` or ``None`` when not derivable.
    """
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
