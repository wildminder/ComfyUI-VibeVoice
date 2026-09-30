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
# their forward. Same rule as the from_pretrained kwargs in
# ``modules/asr_loader.py``, expressed on config OBJECTS here: the native
# ctor takes no ``attn_implementation`` and VibeVoiceAsrConfig has no
# ``set_attn_implementation``, but the ctor builds its submodules from these
# very config objects (``AutoModelForCausalLM.from_config(text_config)`` /
# ``AutoModel.from_config(acoustic_tokenizer_encoder_config)``), so writing
# them before construction is what makes the routing survive.
_LM_ONLY = ("acoustic_tokenizer_encoder_config", "semantic_tokenizer_encoder_config")

# The only attention implementation the ConvNext encoders accept.
_EAGER_ONLY = "eager"

# ------------------------------------------------------------------
# Packaged assets
# ------------------------------------------------------------------

# Architecture config (the same file _PACKAGED_CONFIG_FILES points at).
PACKAGED_ASR_CONFIG_FILE = "default_VibeVoice-ASR_config.json"

# The small processor assets. A native VibeVoiceAsrProcessor cannot be built
# from a partial set: with only a tokenizer AutoProcessor silently returns a
# TokenizersBackend, and without the chat template
# apply_transcription_request() raises. All four ship together.
PACKAGED_TOKENIZER_FILE = "tokenizer.json"
PACKAGED_TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
PACKAGED_PROCESSOR_CONFIG_FILE = "processor_config.json"
PACKAGED_CHAT_TEMPLATE_FILE = "chat_template.jinja"

# The processor assets that are read verbatim from the packaged directory
# (everything except the ~7 MB tokenizer, which is resolved separately so a
# user-supplied tokenizer next to the weight file still wins).
PROCESSOR_ASSET_FILES = (
    PACKAGED_TOKENIZER_CONFIG_FILE,
    PACKAGED_PROCESSOR_CONFIG_FILE,
    PACKAGED_CHAT_TEMPLATE_FILE,
)


def _packaged_configs_dir() -> str:
    """Return the packaged configs directory.

    Imported lazily: ``external_loader`` dispatches into this module, so a
    module-level import would close an import cycle.
    """
    from . import external_loader

    return external_loader._packaged_configs_dir()


def packaged_asset_path(filename: str) -> str:
    """Return the absolute path of a packaged asset (whether or not it exists)."""
    return os.path.normpath(os.path.join(_packaged_configs_dir(), filename))


def packaged_processor_assets() -> dict:
    """Return ``{filename: path}`` for the three small processor assets.

    Raises:
        FileNotFoundError: If any of them is missing from the node folder, so
            a broken install fails with the file name instead of a downstream
            processor construction error.
    """
    missing = [name for name in PROCESSOR_ASSET_FILES if not os.path.exists(packaged_asset_path(name))]
    if missing:
        raise FileNotFoundError(
            f"Packaged VibeVoice-ASR processor assets missing from "
            f"'{_packaged_configs_dir()}': {', '.join(missing)}"
        )
    return {name: packaged_asset_path(name) for name in PROCESSOR_ASSET_FILES}


def resolve_asset_file(filename: str, asset_dir: str) -> str:
    """Resolve an asset file next to the weight file, else from the node folder.

    Mirrors the sidecar-first rule of
    :func:`external_loader.resolve_sidecar_config`: a user-supplied file in
    ``asset_dir`` always beats the packaged copy, and the packaged copy is
    read in place (never copied into the user's model directory).

    Args:
        filename: Asset file name, e.g. ``"tokenizer.json"``.
        asset_dir: Directory next to the weight file (may be empty).

    Returns:
        Absolute path to the file to read.

    Raises:
        FileNotFoundError: If the asset is in neither location.
    """
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
    """Import the transformers classes a native ASR processor is built from.

    Returns:
        Tuple of ``(VibeVoiceAsrProcessor,
        VibeVoiceAcousticTokenizerFeatureExtractor, Qwen2TokenizerFast)``.

    Raises:
        RuntimeError: If the installed transformers predates native
            VibeVoice-ASR support (the ``ImportError`` is chained so the
            original message stays readable).
    """
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
    """Return the feature-extractor kwargs: packaged defaults + sidecar overlay.

    The packaged ``processor_config.json["feature_extractor"]`` is the base
    (sampling_rate 24000, normalize_audio true, target_dB_FS -25, eps 1e-6 —
    the values the checkpoint was trained with); the optional sidecar that
    :func:`external_loader.resolve_sidecar_preprocessor` returned is overlaid
    on top key by key. An empty path means "no overlay", never a crash.

    Args:
        preprocessor_path: Resolved preprocessor sidecar path (may be empty).

    Returns:
        Keyword arguments for the feature extractor.

    Raises:
        FileNotFoundError: If the packaged processor config carries no
            ``feature_extractor`` section (broken install).
    """
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
    """Build a native ``transformers.VibeVoiceAsrProcessor`` from components.

    ``AutoProcessor.from_pretrained`` is deliberately NOT used. It is
    directory-based, and a weight directory that carries no processor assets
    fails SILENTLY in shape — measured on transformers 5.3.0, reading a
    directory holding only ``tokenizer.json`` returns a ``TokenizersBackend``
    (not a processor at all); adding ``tokenizer_config.json`` raises
    ``OSError: Can't load feature extractor``; adding ``processor_config.json``
    yields a processor whose ``apply_transcription_request`` then raises
    ``ValueError: Cannot use apply_chat_template because this processor does
    not have a chat template`` until ``chat_template.jinja`` is present too.
    A native model also cannot be paired with the vendored processor: the
    vendored one emits ``{input_ids, acoustic_input_mask, speech,
    vae_tok_len}`` while the native forward takes ``(input_ids,
    attention_mask, input_values, padding_mask)``.

    So the same four files are read directly — sidecar-first, packaged as
    fallback — and the objects are constructed by hand. Nothing is ever
    written into the user's model directory.

    Args:
        tokenizer_dir: Directory searched first for the tokenizer assets (the
            weight file's own directory; may be empty).
        preprocessor_path: Path returned by
            :func:`external_loader.resolve_sidecar_preprocessor`. Overlays the
            packaged feature-extractor settings; empty means no overlay.

    Returns:
        A ``transformers.VibeVoiceAsrProcessor``.

    Raises:
        RuntimeError: If the installed transformers has no native ASR
            processor, or the constructed object is not one.
        FileNotFoundError: If a required asset is in neither location.
    """
    (
        processor_cls,
        feature_extractor_cls,
        tokenizer_cls,
    ) = import_native_processor_classes()

    tokenizer_file = resolve_asset_file(PACKAGED_TOKENIZER_FILE, tokenizer_dir)
    tokenizer_config = read_json_asset(
        resolve_asset_file(PACKAGED_TOKENIZER_CONFIG_FILE, tokenizer_dir)
    )
    # tokenizer_file is ours to set; a sidecar config that also names one must
    # not collide with that keyword argument.
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

    # asr_generation._asr_processor_kind classifies by class module, so a
    # silently wrong object type routes transcription into the vendored
    # branch and only fails deep inside generation. Fail at load time.
    if not isinstance(processor, processor_cls):
        raise RuntimeError(
            f"Native VibeVoice-ASR processor construction returned "
            f"{type(processor).__name__} instead of {processor_cls.__name__}."
        )

    return processor


# ------------------------------------------------------------------
# Model class selection
#
# Two ASR families share the node's "VibeVoice-ASR" branch and need
# DIFFERENT classes, so the checkpoint's own ``model_type`` decides:
#
#   native  ("vibevoice_asr")  -> transformers' VibeVoiceAsrConfig /
#                                 VibeVoiceAsrForConditionalGeneration
#                                 (checkpoint keys ``language_model.model.*``)
#   vendored ("vibevoice")     -> src/vibevoice's VibeVoiceASRConfig /
#                                 VibeVoiceASRForConditionalGeneration
#                                 (checkpoint keys ``model.language_model.*``)
#
# Guessing wrong is silent in both directions: the vendored config's
# ``__init__`` reads ``decoder_config`` / ``acoustic_tokenizer_config`` keys
# that a native config does not carry (they land in ``**kwargs`` and the Qwen2
# default hidden size survives), and the vendored model nests everything under
# ``self.model``, producing ``model.language_model.*`` keys for a checkpoint
# that stores ``language_model.model.*``.
#
# ``transformers`` is imported LAZILY, inside the functions below: a
# module-scope import would add a multi-second cost to every
# ``import external_loader`` and would make the version guard impossible to
# exercise without uninstalling transformers.
# ------------------------------------------------------------------


def import_native_asr_classes():
    """Import the transformers classes a native ASR checkpoint needs.

    Returns:
        Tuple of ``(AutoConfig, VibeVoiceAsrForConditionalGeneration)``.

    Raises:
        RuntimeError: If the installed transformers predates native
            VibeVoice-ASR support (the ``ImportError`` is chained so the
            original message stays readable).
    """
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
    """Return True when ``path`` holds a native (``model_type "vibevoice_asr"``) config.

    Reads only the JSON — no transformers import, no model build — so the
    loader can pick a class before doing any heavy work.

    Args:
        path: Path to a ``config.json`` file, or to a directory holding one.

    Returns:
        True on a readable config declaring the native model_type; False on
        any other model_type, a missing file, or unreadable/invalid JSON.
    """
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
    """Load a native ASR config through ``AutoConfig``.

    Args:
        path: Path to a ``config.json`` file, or to a directory holding one.

    Returns:
        A ``VibeVoiceAsrConfig`` instance.

    Raises:
        RuntimeError: If the installed transformers is too old.
    """
    AutoConfig, _ = import_native_asr_classes()
    return AutoConfig.from_pretrained(path)


def apply_native_attn_implementation(config, mode: str) -> None:
    """Route ``mode`` to the language model and pin the encoders to eager.

    Writes ``_attn_implementation`` on the root config and on ``text_config``
    (the language model), and ``"eager"`` on every ConvNext encoder sub-config
    present.

    Args:
        config: A native ASR config (the root config, not a sub-config).
        mode: Requested attention implementation, e.g. ``"sdpa"``.
    """
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
    """Build a ``VibeVoiceAsrForConditionalGeneration`` from a native config.

    Native counterpart of the vendored instantiation in
    :mod:`modules.external_loader`: applies the attention routing and the
    dtype to the config objects, then constructs the class directly
    (bypassing ``from_pretrained``) so the state dict can be bound
    afterwards. By default construction runs under a ``torch.device("meta")``
    context — zero RAM, zero random init; the weights are bound afterwards by
    the shared assign-loading path.

    Args:
        config: A native ASR config.
        attn_implementation: Requested attention implementation.
        final_load_dtype: torch.dtype for the model.
        use_meta: Construct under a meta device context (default True).

    Returns:
        Model instance (weights not yet loaded).

    Raises:
        RuntimeError: If the installed transformers is too old.
    """
    _, model_cls = import_native_asr_classes()

    apply_native_attn_implementation(config, attn_implementation)

    # Dtype on the root AND on text_config: the submodules are built FROM
    # text_config, so a dtype recorded only on the root is not inherited.
    set_config_dtype(config, final_load_dtype)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        set_config_dtype(text_config, final_load_dtype)

    ctx = torch.device("meta") if use_meta else contextlib.nullcontext()
    with ctx:
        return model_cls(config)
