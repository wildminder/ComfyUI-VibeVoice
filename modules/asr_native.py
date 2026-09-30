"""Native transformers 5.3.0 support for the external VibeVoice-ASR branch.

The published ``VibeVoice-ASR-HF`` checkpoint ships ``model_type
"vibevoice_asr"`` and is built by transformers itself, not by the vendored
``src/vibevoice`` classes. Everything native lives here so
:mod:`modules.external_loader` only gains thin dispatchers.

Packaged assets are addressed by PATH, never by importing
``src.vibevoice.configs``: the test suite mocks that module wholesale
(``conftest.py``), so an import-based lookup would resolve to a MagicMock in
tests while working in production. Paths come from
``external_loader._packaged_configs_dir()`` so there is a single definition of
where the node's own assets live.
"""

import json
import logging
import os
import contextlib

import torch

from .dtype_utils import set_config_dtype

logger = logging.getLogger(__name__)

# model_type of the published ASR-HF checkpoints.
NATIVE_ASR_MODEL_TYPE = "vibevoice_asr"

# Minimum transformers that ships VibeVoiceAsrForConditionalGeneration.
MIN_TRANSFORMERS_VERSION = "5.3.0"

# Composite sub-configs that may ONLY run eager attention. The acoustic and
# semantic tokenizer encoders are ConvNext-based; an explicit sdpa/flash
# request raises "does not support ... scaled_dot_product_attention" inside
# their forward.
_LM_ONLY = ("acoustic_tokenizer_encoder_config", "semantic_tokenizer_encoder_config")

_EAGER_ONLY = "eager"

# ------------------------------------------------------------------
# Packaged assets
# ------------------------------------------------------------------

PACKAGED_ASR_CONFIG_FILE = "default_VibeVoice-ASR_config.json"

PACKAGED_TOKENIZER_FILE = "tokenizer.json"
PACKAGED_TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
PACKAGED_PROCESSOR_CONFIG_FILE = "processor_config.json"
PACKAGED_CHAT_TEMPLATE_FILE = "chat_template.jinja"

PROCESSOR_ASSET_FILES = (
    PACKAGED_TOKENIZER_CONFIG_FILE,
    PACKAGED_PROCESSOR_CONFIG_FILE,
    PACKAGED_CHAT_TEMPLATE_FILE,
)


def _packaged_configs_dir() -> str:
    from . import external_loader

    return external_loader._packaged_configs_dir()


def packaged_asset_path(filename: str) -> str:
    """Return the absolute path of a packaged asset."""
    return os.path.normpath(os.path.join(_packaged_configs_dir(), filename))


def packaged_processor_assets() -> dict:
    """Return ``{filename: path}`` for the three small processor assets."""
    missing = [name for name in PROCESSOR_ASSET_FILES if not os.path.exists(packaged_asset_path(name))]
    if missing:
        raise FileNotFoundError(
            f"Packaged VibeVoice-ASR processor assets missing from "
            f"'{_packaged_configs_dir()}': {', '.join(missing)}"
        )
    return {name: packaged_asset_path(name) for name in PROCESSOR_ASSET_FILES}


def resolve_asset_file(filename: str, asset_dir: str) -> str:
    """Resolve an asset file next to the weight file, else from the node folder."""
    if asset_dir:
        local = os.path.join(asset_dir, filename)
        if os.path.exists(local):
            return local

    packaged = packaged_asset_path(filename)
    if os.path.exists(packaged):
        return packaged

    raise FileNotFoundError(
        f"No '{filename}' found next to the weight file ('{asset_dir}') or in "
        f"the packaged configs directory ('{_packaged_configs_dir()}')."
    )


def read_json_asset(path: str) -> dict:
    """Read a packaged/user JSON asset, returning ``{}`` for an empty path."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text_asset(path: str) -> str:
    """Read a packaged/user text asset (the Jinja chat template)."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ------------------------------------------------------------------
# Native processor
# ------------------------------------------------------------------


def import_native_processor_classes():
    """Import the transformers classes a native ASR processor is built from."""
    try:
        import transformers
        from transformers import Qwen2TokenizerFast, VibeVoiceAsrProcessor
        from transformers.models.vibevoice_acoustic_tokenizer.feature_extraction_vibevoice_acoustic_tokenizer import (
            VibeVoiceAcousticTokenizerFeatureExtractor,
        )
    except ImportError as e:
        raise RuntimeError(
            f"Native VibeVoice-ASR checkpoints require transformers >= "
            f"{MIN_TRANSFORMERS_VERSION} (VibeVoiceAsrProcessor); "
            f"installed: {e}"
        ) from e
    return (
        VibeVoiceAsrProcessor,
        VibeVoiceAcousticTokenizerFeatureExtractor,
        Qwen2TokenizerFast,
    )


def _resolve_feature_extractor_kwargs(preprocessor_path: str) -> dict:
    """Return the feature-extractor kwargs: packaged defaults + sidecar overlay."""
    packaged = read_json_asset(
        packaged_asset_path(PACKAGED_PROCESSOR_CONFIG_FILE)
    ).get("feature_extractor")
    if not packaged:
        raise FileNotFoundError(
            f"Packaged '{PACKAGED_PROCESSOR_CONFIG_FILE}' has no "
            f"'feature_extractor' section ('{packaged_asset_path(PACKAGED_PROCESSOR_CONFIG_FILE)}')."
        )
    kwargs = dict(packaged)
    kwargs.update(read_json_asset(preprocessor_path))
    return kwargs


def build_native_asr_processor(tokenizer_dir: str, preprocessor_path: str = ""):
    """Build a native ``transformers.VibeVoiceAsrProcessor`` from components."""
    (
        processor_cls,
        feature_extractor_cls,
        tokenizer_cls,
    ) = import_native_processor_classes()

    tokenizer_file = resolve_asset_file(PACKAGED_TOKENIZER_FILE, tokenizer_dir)
    tokenizer_config = read_json_asset(
        resolve_asset_file(PACKAGED_TOKENIZER_CONFIG_FILE, tokenizer_dir)
    )
    tokenizer_config.pop("tokenizer_file", None)

    processor_config = read_json_asset(
        packaged_asset_path(PACKAGED_PROCESSOR_CONFIG_FILE)
    )
    chat_template = read_text_asset(
        resolve_asset_file(PACKAGED_CHAT_TEMPLATE_FILE, tokenizer_dir)
    )

    logger.debug(
        f"Building native VibeVoice-ASR processor from tokenizer "
        f"'{tokenizer_file}' (preprocessor overlay: "
        f"'{preprocessor_path or 'packaged defaults'}')"
    )

    processor = processor_cls(
        feature_extractor=feature_extractor_cls(
            **_resolve_feature_extractor_kwargs(preprocessor_path)
        ),
        tokenizer=tokenizer_cls(tokenizer_file=tokenizer_file, **tokenizer_config),
        chat_template=chat_template,
        audio_token=processor_config["audio_token"],
        audio_bos_token=processor_config["audio_bos_token"],
        audio_eos_token=processor_config["audio_eos_token"],
        audio_duration_token=processor_config["audio_duration_token"],
    )

    if not isinstance(processor, processor_cls):
        raise RuntimeError(
            f"Native VibeVoice-ASR processor construction returned "
            f"{type(processor).__name__} instead of {processor_cls.__name__}."
        )

    return processor


# ------------------------------------------------------------------
# Model class selection
# ------------------------------------------------------------------


def import_native_asr_classes():
    """Import the transformers classes a native ASR checkpoint needs."""
    try:
        from transformers import AutoConfig, VibeVoiceAsrForConditionalGeneration
    except ImportError as e:
        raise RuntimeError(
            f"Native VibeVoice-ASR checkpoints require transformers >= "
            f"{MIN_TRANSFORMERS_VERSION} (native VibeVoice-ASR support); "
            f"installed: {e}"
        ) from e
    return AutoConfig, VibeVoiceAsrForConditionalGeneration


def native_asr_available() -> bool:
    """Return True when the installed transformers can build native ASR models."""
    try:
        import_native_asr_classes()
    except RuntimeError:
        return False
    return True


def is_native_asr_config_path(path) -> bool:
    """Return True when ``path`` holds a native config."""
    if not path:
        return False
    config_file = os.path.join(path, "config.json") if os.path.isdir(path) else path
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    return data.get("model_type") == NATIVE_ASR_MODEL_TYPE


def load_native_asr_config(path):
    """Load a native ASR config through ``AutoConfig``."""
    AutoConfig, _ = import_native_asr_classes()
    return AutoConfig.from_pretrained(path)


def apply_native_attn_implementation(config, mode: str) -> None:
    """Route ``mode`` to the language model and pin the encoders to eager."""
    config._attn_implementation = mode
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        text_config._attn_implementation = mode
    for sub_name in _LM_ONLY:
        sub_config = getattr(config, sub_name, None)
        if sub_config is not None:
            sub_config._attn_implementation = _EAGER_ONLY


def instantiate_native_asr_model(
    config,
    attn_implementation: str,
    final_load_dtype: torch.dtype,
    use_meta: bool = True,
):
    """Build a ``VibeVoiceAsrForConditionalGeneration`` from a native config."""
    _, model_cls = import_native_asr_classes()

    apply_native_attn_implementation(config, attn_implementation)

    set_config_dtype(config, final_load_dtype)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        set_config_dtype(text_config, final_load_dtype)

    ctx = torch.device("meta") if use_meta else contextlib.nullcontext()
    with ctx:
        return model_cls(config)