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

logger = logging.getLogger(__name__)

# Candidate embedding tensor names, first found wins. HF-style checkpoints
# carry the full name; llama.cpp-converted GGUF files use the short name
# (see modules/gguf_quant.py:_LLAMACPP_TO_HF). The lm head is NOT used:
# 1.5B checkpoints tie word embeddings and may omit it entirely.
_EMBEDDING_KEY_CANDIDATES = (
    "model.language_model.embed_tokens.weight",
    "tok_embeddings.weight",
    # modern name for the input embedding
    "token_embd.weight",
)

# config_name -> (hidden_size, vocab_size) of the LLM embedding. Extend this
# table to teach detection new families; nothing else needs to change.
_FAMILY_SIGNATURES = {
    "VibeVoice-7B": (3584, 152064),
    "VibeVoice-1.5B": (1536, 151936),
}

# ASR checkpoints are a DIFFERENT family that happens to share the 7B
# embedding shape ((3584, 152064) is the VibeVoice-7B TTS signature), so they
# must never be resolved through _FAMILY_SIGNATURES: that table is scanned
# first-match and would answer "VibeVoice-7B", routing an ASR checkpoint into
# the TTS loader. The gate below keys on module names that exist ONLY in an ASR
# state dict, and this table is scanned on its own.
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


def classify_asr_embedding_shape(shape: Sequence[int]) -> Optional[WeightsFingerprint]:
    """Classify a 2-D embedding shape onto the ASR family, orientation-agnostic.

    Separate from :func:`classify_embedding_shape` on purpose: the ASR
    signature overlaps the 7B TTS one, and only the ASR key prefixes (checked
    by the caller) justify consulting this table.

    Args:
        shape: The embedding tensor's shape, ``[vocab, hidden]`` or reversed.

    Returns:
        A fingerprint with ``is_asr=True`` when the shape matches exactly one
        known ASR family, else ``None``.
    """
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

            # ASR gate FIRST: an ASR checkpoint shares the 7B embedding shape,
            # so classifying it with _FAMILY_SIGNATURES would return
            # "VibeVoice-7B" and send it to the TTS loader.
            #
            # This branch is TERMINAL in both directions. A state dict carrying
            # an ASR-only prefix is an ASR checkpoint by construction, so if
            # the ASR embedding key is missing or its shape is foreign we must
            # NOT fall through to the TTS loop below: a hypothetical ASR
            # variant that spells its embedding differently (e.g.
            # ``tok_embeddings.weight``) would otherwise be classified with
            # _FAMILY_SIGNATURES, whose 7B entry is the very same (3584,
            # 152064) shape, and would be misrouted to the TTS loader as
            # "VibeVoice-7B". Returning None makes the caller raise the
            # actionable "could not auto-detect, pick a config_name" error
            # instead of silently loading the wrong family.
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
                        logger.debug(
                            "ASR embedding key '%s' in '%s' has foreign shape %s",
                            candidate, os.path.basename(path), shape,
                        )
                        return None
                logger.debug(
                    "'%s' carries ASR-only prefixes but none of the ASR "
                    "embedding keys %s; refusing to classify it as a TTS "
                    "family (the ASR and 7B-TTS signatures collide)",
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
            from .gguf_quant import open_gguf_reader

            # Tolerant open: files with sub-block-row conv
            # tensors crash the stock reader; auto-detect must still see
            # the embedding fingerprint (header-only either way).
            reader = open_gguf_reader(weight_path)
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
