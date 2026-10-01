"""Registration of VibeVoice-vendored + HF norm classes as streamable.

Kept separate from :mod:`modules.comfy_stream` so the heavy vendored /
transformers imports happen only when a conversion pass actually runs
(`comfy_stream.convert_tree_for_streaming` calls this lazily).
"""

from .comfy_stream import (
    _compute_convlayernorm,
    _compute_convrmsnorm,
    _compute_qwen2rmsnorm,
    _compute_rmsnorm,
)


def register_vendored_types(register) -> None:
    """Teach ``convert_tree_for_streaming`` about vendored norm classes.

    Each registration is isolated: one unavailable/broken class must never
    prevent the others (or the builtin kinds) from streaming.
    """
    entries = []

    try:
        from ..src.vibevoice.modular.modular_vibevoice_tokenizer import (
            ConvLayerNorm,
            ConvRMSNorm,
            RMSNorm as TokenizerRMSNorm,
        )
        from ..src.vibevoice.modular.modular_vibevoice_diffusion_head import (
            RMSNorm as DiffusionRMSNorm,
        )
        # Tokenizer and diffusion-head RMSNorm share identical math.
        entries += [
            (TokenizerRMSNorm, _compute_rmsnorm),
            (DiffusionRMSNorm, _compute_rmsnorm),
            (ConvRMSNorm, _compute_convrmsnorm),
            (ConvLayerNorm, _compute_convlayernorm),
        ]
    except Exception as e:  # pragma: no cover - heavy deps optional here
        _log_skip("vendored norms", e)

    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

        entries.append((Qwen2RMSNorm, _compute_qwen2rmsnorm))
    except Exception as e:  # pragma: no cover
        _log_skip("Qwen2RMSNorm", e)

    # transformers-native VibeVoice-ASR norm classes (checkpoint
    # microsoft/VibeVoice-ASR-HF). Both are T5-style RMSNorm — fp32 stats,
    # scale AFTER cast-back, `variance_epsilon` attr — identical math to
    # Qwen2RMSNorm.
    import importlib

    for mod_path, cls_name in (
        ("transformers.models.vibevoice_asr.modeling_vibevoice_asr", "VibeVoiceAsrRMSNorm"),
        ("transformers.models.vibevoice_acoustic_tokenizer.modeling_vibevoice_acoustic_tokenizer",
         "VibeVoiceAcousticTokenizerRMSNorm"),
    ):
        try:
            entries.append((getattr(importlib.import_module(mod_path), cls_name),
                            _compute_qwen2rmsnorm))
        except Exception as e:  # pragma: no cover - older transformers lack these
            _log_skip(cls_name, e)

    import logging

    for base_cls, compute_fn in entries:
        if not isinstance(base_cls, type):
            logging.debug("[ComfyUI-VibeVoice] Skipping non-type streaming entry %r", base_cls)
            continue
        register(base_cls, compute_fn)


def _log_skip(what, e):
    import logging

    logging.debug(
        "[ComfyUI-VibeVoice] Streaming registration skipped %s: %s", what, e
    )