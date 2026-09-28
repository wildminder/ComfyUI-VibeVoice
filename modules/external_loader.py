"""External model loading for VibeVoice.

Loads a VibeVoice model from a standalone weight file (safetensors / .bin /
.gguf) placed in ComfyUI's ``diffusion_models`` folder, binding the required
config / tokenizer / preprocessor JSONs via sidecar files or packaged defaults.

This module bypasses ComfyUI's ``model_detection.detect_unet_config()`` entirely
(VibeVoice's state-dict keys are not recognized by it) and instead reuses the
existing :class:`~modules.loader.VibeVoiceLoader` internals for config,
tokenizer, processor, and model instantiation.

Sidecar file convention (place next to the weight file):
    - ``<weight>.config.json``          → architecture config (preferred)
    - ``config.json``                   → architecture config (same directory)
    - ``<weight>.preprocessor.json``    → audio preprocessor config (preferred)
    - ``preprocessor_config.json``      → audio preprocessor config (same dir)
    - ``tokenizer.json``                → Qwen2.5 text tokenizer (same dir)
"""

import os
import json
import logging
import contextlib
import gc

import torch

import comfy.utils
import comfy.model_management as model_management

from transformers import BitsAndBytesConfig

from ..src.vibevoice.modular.configuration_vibevoice import VibeVoiceConfig, VibeVoiceASRConfig
from ..src.vibevoice.modular.configuration_vibevoice_streaming import VibeVoiceStreamingConfig
from ..src.vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from ..src.vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceASRTextTokenizerFast
from ..src.vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from ..src.vibevoice.processor.vibevoice_tokenizer_processor import VibeVoiceTokenizerProcessor

from .loader import VibeVoiceLoader
from .base_loader import iter_safetensors_tensors
from .attention_utils import (
    SAGE_ATTENTION_AVAILABLE,
    resolve_attention_mode,
    resolve_realtime_attention_mode,
    resolve_asr_attention_mode,
    get_attn_implementation_for_load,
    check_sage_attention_compatible,
)
from .dtype_utils import resolve_dtype, cast_model_to_dtype_if_needed, set_config_dtype
from .convrot_quant import UnsupportedQuantFormat
from .quant_common import validate_weight_plan

if SAGE_ATTENTION_AVAILABLE:
    from ..src.vibevoice.modular.sage_attention_patch import set_sage_attention

logger = logging.getLogger(__name__)

# Packaged default config filenames keyed by config_name.
# Only models with a packaged default are listed here; all other config_name
# values require a sidecar config file next to the weight file.
_PACKAGED_CONFIG_FILES = {
    "VibeVoice-1.5B": "default_VibeVoice-1.5B_config.json",
    "VibeVoice-7B": "default_VibeVoice-Large_config.json",
}

# Sentinel config_name that resolves the architecture from the weight file's
# embedding fingerprint at load time (plan 2026-08-27, D6).
AUTO_CONFIG_NAME = "Auto-detect"

# All config_name values accepted by the loader node dropdown.
EXTERNAL_CONFIG_OPTIONS = [
    AUTO_CONFIG_NAME,
    "VibeVoice-1.5B",
    "VibeVoice-7B",
    "VibeVoice-Realtime-0.5B",
    "VibeVoice-ASR",
]

# Legacy config_name values removed from the dropdown but still honored so
# saved workflows keep loading (plan 2026-08-27, D1/D2). Keys are compared
# case-insensitively.
_LEGACY_CONFIG_ALIASES = {
    "vibevoice-large": "VibeVoice-7B",
}


def normalize_config_name(config_name: str) -> str:
    """Map legacy/alias config_name values onto their canonical option.

    Exact options pass through unchanged; case-insensitive alias hits return
    the canonical name; anything unknown is returned as-is (callers decide
    validity).

    Args:
        config_name: Raw config_name value (may come from a saved workflow).

    Returns:
        Canonical config_name when an alias applies, else the input value.
    """
    if not config_name:
        return config_name
    if config_name in EXTERNAL_CONFIG_OPTIONS:
        return config_name
    return _LEGACY_CONFIG_ALIASES.get(config_name.lower(), config_name)


def resolve_auto_config_name(weight_path: str, gguf_reader=None, weights_fp=None) -> str:
    """Resolve ``AUTO_CONFIG_NAME`` to a concrete family from the weights.

    Plan 2026-08-27 (D6): Auto-detect reads the weight file's architecture
    fingerprint (header-only). A conclusive fingerprint becomes the selected
    config_name; an inconclusive one raises an actionable ``ValueError``
    BEFORE any heavy load work. Shared by the node (which must resolve the
    name before computing the cache identity) and the loader (defense in
    depth for direct callers) so both fail fast with the same message.

    Args:
        weight_path: Absolute path to the external weight file.
        gguf_reader: Optional open ``gguf.GGUFReader`` for ``.gguf`` files
            (avoids re-opening when the loader already holds one).
        weights_fp: Optional precomputed ``WeightsFingerprint``; when omitted
            the fingerprint is computed here (header-only).

    Returns:
        The detected config_name (e.g. ``"VibeVoice-7B"``).

    Raises:
        ValueError: When the architecture cannot be determined.
    """
    if weights_fp is None:
        from .config_detect import fingerprint_weights

        weights_fp = fingerprint_weights(weight_path, gguf_reader=gguf_reader)

    detected = weights_fp.config_name if weights_fp is not None else None
    if not detected:
        explicit = [o for o in EXTERNAL_CONFIG_OPTIONS if o != AUTO_CONFIG_NAME]
        if weights_fp is None:
            observed = "Observed dimensions: unavailable (no embedding fingerprint)."
        else:
            observed = (
                "Observed dimensions: "
                f"hidden={weights_fp.hidden_size}, vocab={weights_fp.vocab_size}."
            )
        raise ValueError(
            f"config_name '{AUTO_CONFIG_NAME}': could not determine the "
            f"VibeVoice architecture from "
            f"'{os.path.basename(weight_path)}'. {observed} Auto-detect reads "
            f"the checkpoint's embedding fingerprint; unmatched dimensions "
            f"are not guessed. Select config_name explicitly (one of: "
            f"{', '.join(explicit)}) or place a sidecar config.json next to "
            f"the weight file."
        )
    logger.debug(
        f"Auto-detected architecture '{detected}' from "
        f"'{os.path.basename(weight_path)}'"
    )
    return detected


def reconcile_config(selected_name: str, resolved_config, weights_fp):
    """Reconcile the selected config against the weights' fingerprint.

    Plan 2026-08-27 (D5): the checkpoint's embedding shape is a perfect
    architecture fingerprint. When it CONTRADICTS the resolved config family,
    the fingerprint wins and the detected family's config is substituted —
    a mismatching config (dropdown or sidecar) is a guaranteed size-mismatch
    crash otherwise. When the fingerprint is inconclusive (``None``) or
    agrees, the selection stands.

    Args:
        selected_name: The (normalized) config_name in effect.
        resolved_config: The config object loaded for ``selected_name``.
        weights_fp: ``WeightsFingerprint`` from the weight file, or ``None``.

    Returns:
        ``(effective_name, changed)`` — the config_name to use and whether
        it differs from ``selected_name``.
    """
    from .config_detect import config_fingerprint

    if weights_fp is None:
        return selected_name, False

    cfg_fp = config_fingerprint(resolved_config)
    if cfg_fp is None or cfg_fp == (weights_fp.hidden_size, weights_fp.vocab_size):
        return selected_name, False

    detected = weights_fp.config_name
    if not detected or detected == selected_name:
        return selected_name, False

    logger.warning(
        f"Config mismatch: weights contain '{weights_fp.source_key}' with "
        f"shape (vocab={weights_fp.vocab_size}, hidden={weights_fp.hidden_size}) "
        f"-> {detected}, but config '{selected_name}' "
        f"(hidden={cfg_fp[0]}, vocab={cfg_fp[1]}) was selected. "
        f"Using '{detected}'."
    )
    return detected, True


# ====================================================================
# Reconciliation memo
# ====================================================================
#
# The loader node's unload-before-load gate must run BEFORE the model is
# built (it is the ~1x-model-size peak-RAM guard), so the node cannot know
# what reconcile_config() will decide without opening the weight file itself.
# This memo records the (file identity, requested name) -> effective name
# mapping learned from the PREVIOUS load, so the node can compute the same
# config name the consumer will key the built patcher off.
#
# Keying on the file's stat (mtime_ns + size) means the memo is invalidated
# whenever the checkpoint changes; the first run after such a change falls
# back to the REQUESTED name, which is the pre-fix behaviour. That is the
# safe direction: a genuinely different model is still evicted, at the cost
# of one extra churn on the run that detects the change.

# (abspath, mtime_ns, size, selected_name) -> effective config_name.
_RECONCILIATION_MEMO: dict = {}


def _reconciliation_memo_key(weight_path: str, selected_name: str):
    """Build the stat-guarded memo key for a weight file / config selection.

    Mirrors the guarded ``os.stat`` pattern of
    :func:`modules.model_registry.identity_for_external`: an unreadable or
    missing file degrades to placeholder stat values instead of raising, so
    a hand-built/mocked path can never break the loader node.

    Args:
        weight_path: Absolute path to the weight file (may not exist).
        selected_name: The requested config_name, post alias/auto resolution.

    Returns:
        Hashable memo key tuple.
    """
    abspath = os.path.abspath(weight_path) if weight_path else ""
    mtime_ns = 0
    size = 0
    try:
        stat = os.stat(abspath)
        mtime_ns = stat.st_mtime_ns
        size = stat.st_size
    except OSError:
        # Unreadable/missing file: degrade to placeholders (deterministic).
        pass
    return (abspath, mtime_ns, size, selected_name)


def reconciled_config_name(weight_path: str, selected_name: str) -> str:
    """Return the config name the loader will use for this exact request.

    Args:
        weight_path: Absolute path to the weight file.
        selected_name: The requested config_name (post alias/auto resolution).

    Returns:
        The remembered effective config_name, or ``selected_name`` on a memo
        miss (cold memo, changed weight file, or no reconciliation applied).
    """
    return _RECONCILIATION_MEMO.get(
        _reconciliation_memo_key(weight_path, selected_name), selected_name
    )


def remember_reconciled_config(
    weight_path: str, selected_name: str, effective_name: str
) -> None:
    """Record the config name a completed load actually used.

    Args:
        weight_path: Absolute path to the weight file.
        selected_name: The config_name that was requested from the loader.
        effective_name: The config_name the produced bundle records (post
            reconciliation). Empty values are ignored.
    """
    if not effective_name:
        return
    _RECONCILIATION_MEMO[
        _reconciliation_memo_key(weight_path, selected_name)
    ] = effective_name


def clear_reconciliation_memo() -> None:
    """Forget all reconciliation bookkeeping (test isolation helper)."""
    _RECONCILIATION_MEMO.clear()


# ====================================================================
# Path resolution helpers
# ====================================================================

def _packaged_configs_dir() -> str:
    """Return the absolute path to the packaged configs directory."""
    return os.path.join(
        os.path.dirname(__file__), "..", "src", "vibevoice", "configs"
    )


def _get_packaged_config_path(config_name: str) -> str:
    """Return the packaged default config path for ``config_name``.

    Args:
        config_name: One of the keys in ``_PACKAGED_CONFIG_FILES``.

    Returns:
        Absolute path to the packaged config file, or empty string if no
        packaged default exists for this config_name.
    """
    filename = _PACKAGED_CONFIG_FILES.get(config_name)
    if filename is None:
        return ""
    return os.path.normpath(os.path.join(_packaged_configs_dir(), filename))


def resolve_sidecar_config(weight_path: str, config_name: str) -> str:
    """Resolve the architecture config JSON path for an external weight file.

    Resolution priority:
        1. ``<weight_path>.config.json`` (sidecar, preferred)
        2. ``config.json`` in the same directory as the weight file
        3. Packaged default selected by ``config_name``

    Args:
        weight_path: Absolute path to the external weight file.
        config_name: Config selector used for the packaged-default fallback.

    Returns:
        Absolute path to the config JSON to use.

    Raises:
        FileNotFoundError: If no config can be resolved.
    """
    # 1. Sidecar: <weight_path>.config.json
    sidecar_path = weight_path + ".config.json"
    if os.path.exists(sidecar_path):
        logger.debug(f"Using sidecar config: {sidecar_path}")
        return sidecar_path

    # 2. Sidecar: config.json in the same directory
    dir_sidecar = os.path.join(os.path.dirname(weight_path), "config.json")
    if os.path.exists(dir_sidecar):
        logger.debug(f"Using directory sidecar config: {dir_sidecar}")
        return dir_sidecar

    # 3. Packaged default based on config_name
    packaged = _get_packaged_config_path(config_name)
    if packaged and os.path.exists(packaged):
        logger.debug(f"Using packaged default config for '{config_name}': {packaged}")
        return packaged

    raise FileNotFoundError(
        f"No architecture config found for external model "
        f"'{os.path.basename(weight_path)}'.\n"
        f"Place a sidecar config next to the weight file:\n"
        f"  - '{sidecar_path}'  (preferred), or\n"
        f"  - 'config.json' in '{os.path.dirname(weight_path)}'\n"
        f"Alternatively select a config_name with a packaged default "
        f"({', '.join(_PACKAGED_CONFIG_FILES.keys())})."
    )


def resolve_sidecar_preprocessor(weight_path: str) -> str:
    """Resolve the audio preprocessor config JSON path for an external weight file.

    Resolution priority:
        1. ``<weight_path>.preprocessor.json`` (sidecar, preferred)
        2. ``preprocessor_config.json`` in the same directory as the weight file

    Args:
        weight_path: Absolute path to the external weight file.

    Returns:
        Absolute path to the preprocessor config, or empty string if not found
        (the processor will fall back to built-in defaults).
    """
    sidecar_path = weight_path + ".preprocessor.json"
    if os.path.exists(sidecar_path):
        logger.debug(f"Using sidecar preprocessor config: {sidecar_path}")
        return sidecar_path

    dir_sidecar = os.path.join(os.path.dirname(weight_path), "preprocessor_config.json")
    if os.path.exists(dir_sidecar):
        logger.debug(f"Using directory sidecar preprocessor config: {dir_sidecar}")
        return dir_sidecar

    return ""


def resolve_sidecar_tokenizer_dir(weight_path: str) -> str:
    """Return the directory to search for ``tokenizer.json``.

    The tokenizer is expected to live in the same directory as the weight file.
    If absent, :meth:`VibeVoiceLoader._load_tokenizer` falls back to the
    packaged tokenizer or a HuggingFace download.

    Args:
        weight_path: Absolute path to the external weight file.

    Returns:
        The directory containing the weight file.
    """
    return os.path.dirname(weight_path)


# ====================================================================
# ASR-specific helpers
# ====================================================================

# config_name values that select the ASR loading branch.
ASR_CONFIG_NAMES = {"VibeVoice-ASR"}


def is_asr_config_name(config_name: str) -> bool:
    """Return True if ``config_name`` selects the ASR loading branch."""
    return config_name in ASR_CONFIG_NAMES


def _load_asr_config(config_path: str) -> "VibeVoiceASRConfig":
    """Load a :class:`VibeVoiceASRConfig` from a sidecar config JSON.

    ASR configs share ``model_type == "vibevoice"`` with TTS configs, so the
    caller must select this loader explicitly (via the ``config_name`` dropdown).

    Args:
        config_path: Path to the ASR config.json.

    Returns:
        VibeVoiceASRConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"ASR config not found: {config_path}")
    return VibeVoiceASRConfig.from_pretrained(config_path)


def _load_asr_tokenizer(tokenizer_dir: str) -> "VibeVoiceASRTextTokenizerFast":
    """Load the ASR text tokenizer from ``tokenizer.json``.

    Resolution priority:
        1. ``tokenizer.json`` in ``tokenizer_dir`` (next to the weight file)
        2. The packaged ``tokenizer.json`` fallback

    Args:
        tokenizer_dir: Directory to search for ``tokenizer.json``.

    Returns:
        VibeVoiceASRTextTokenizerFast instance.

    Raises:
        FileNotFoundError: If no tokenizer.json can be found.
    """
    tokenizer_file_path = os.path.join(tokenizer_dir, "tokenizer.json")

    if not os.path.exists(tokenizer_file_path):
        packaged_tokenizer_path = os.path.join(
            _packaged_configs_dir(), "tokenizer.json"
        )
        if os.path.exists(packaged_tokenizer_path):
            logger.debug(
                f"Using packaged tokenizer.json fallback for ASR: {packaged_tokenizer_path}"
            )
            tokenizer_file_path = packaged_tokenizer_path
        else:
            raise FileNotFoundError(
                f"No 'tokenizer.json' found for external ASR model. "
                f"Place a 'tokenizer.json' next to the weight file in "
                f"'{tokenizer_dir}'."
            )

    return VibeVoiceASRTextTokenizerFast(tokenizer_file=tokenizer_file_path)


def _load_asr_processor(
    tokenizer,
    preprocessor_config_path: str,
) -> "VibeVoiceASRProcessor":
    """Build a :class:`VibeVoiceASRProcessor` from a tokenizer + preprocessor config.

    Mirrors :meth:`VibeVoiceASRProcessor.from_pretrained` but accepts
    already-resolved local paths instead of a model directory / HF repo.

    Args:
        tokenizer: VibeVoiceASRTextTokenizerFast instance.
        preprocessor_config_path: Path to preprocessor_config.json (may be empty).

    Returns:
        VibeVoiceASRProcessor instance.
    """
    config = {}
    if preprocessor_config_path and os.path.exists(preprocessor_config_path):
        with open(preprocessor_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    speech_tok_compress_ratio = config.get("speech_tok_compress_ratio", 3200)
    target_sample_rate = config.get("target_sample_rate", 24000)
    normalize_audio = config.get("normalize_audio", True)

    audio_processor = VibeVoiceTokenizerProcessor(
        sampling_rate=target_sample_rate,
        normalize_audio=normalize_audio,
        target_dB_FS=config.get("target_dB_FS", -25),
        eps=config.get("eps", 1e-6),
    )

    return VibeVoiceASRProcessor(
        tokenizer=tokenizer,
        audio_processor=audio_processor,
        speech_tok_compress_ratio=speech_tok_compress_ratio,
        target_sample_rate=target_sample_rate,
        normalize_audio=normalize_audio,
    )


def _instantiate_asr_model(
    config,
    attn_implementation: str,
    final_load_dtype: torch.dtype,
    use_meta: bool = True,
):
    """Instantiate a :class:`VibeVoiceASRForConditionalGeneration` directly.

    ASR counterpart of :meth:`VibeVoiceLoader._instantiate_model`. Instantiates
    the model class directly (bypassing ``from_pretrained``) so the state dict
    can be loaded in-memory afterwards. By default construction runs under a
    ``torch.device("meta")`` context (plan 2026-08-18, D1) — zero RAM, zero
    random init; weights are bound afterwards by ``_apply_state_dict``.

    Args:
        config: VibeVoiceASRConfig instance.
        attn_implementation: Attention implementation string.
        final_load_dtype: torch.dtype for the model.
        use_meta: Construct under a meta device context (default True).

    Returns:
        Model instance (weights not yet loaded).
    """
    # Set attention implementation on the decoder config
    if hasattr(config, "decoder_config"):
        config.decoder_config._attn_implementation = attn_implementation

    # Set dtype on config (version-safe: transformers v5 deprecated torch_dtype)
    set_config_dtype(config, final_load_dtype)
    if hasattr(config, "decoder_config"):
        set_config_dtype(config.decoder_config, final_load_dtype)

    ctx = torch.device("meta") if use_meta else contextlib.nullcontext()
    with ctx:
        return VibeVoiceASRForConditionalGeneration(config)


# ====================================================================
# Weight file state-dict loading (safetensors / bin / gguf dispatch)
# ====================================================================

def _load_gguf_state_dict(weight_path: str, device=None) -> dict:
    """Load a state dict from a ``.gguf`` file via the ``gguf`` package.

    ComfyUI's :func:`comfy.utils.load_torch_file` does not understand the GGUF
    container (it would route ``.gguf`` to ``torch.load`` and fail), so GGUF
    weights are parsed here with :class:`gguf.GGUFReader` and dequantized
    tensor-by-tensor.

    Args:
        weight_path: Absolute path to the ``.gguf`` file.
        device: Target torch device for the tensors (defaults to CPU). The
            patcher owns the single host-to-device transfer, so CPU is the
            normal choice.

    Returns:
        dict mapping tensor name -> torch.Tensor (dequantized, on ``device``).

    Raises:
        RuntimeError: If the ``gguf`` package is not installed.
    """
    if device is None:
        device = torch.device("cpu")

    try:
        import gguf
    except ImportError as e:
        raise RuntimeError(
            "Loading .gguf weights requires the 'gguf' Python package. "
            "Install it with: pip install gguf"
        ) from e

    logger.debug(f"Loading GGUF state dict from: {weight_path}")
    from .gguf_quant import dequantize_reader_tensor, open_gguf_reader

    reader = open_gguf_reader(weight_path)

    state_dict = {}
    for tensor in reader.tensors:
        # Dequantizes to the tensor's LOGICAL shape in both byte layouts
        # (spec-conformant row-mapped and the flat-block pool recovery);
        # the returned tensor is owned (the mmap view is copied), so it is
        # writable and safe for torch.
        dequantized = dequantize_reader_tensor(tensor)
        state_dict[tensor.name] = dequantized.to(device)

    logger.debug(f"Loaded {len(state_dict)} tensors from GGUF file")
    return state_dict


def _load_weight_state_dict(weight_path: str, device) -> dict:
    """Load a state dict from a weight file, dispatching GGUF to the gguf parser.

    Args:
        weight_path: Absolute path to the weight file
            (``.safetensors`` / ``.bin`` / ``.pt`` / ``.gguf``).
        device: Target torch device for the tensors.

    Returns:
        dict mapping tensor name -> torch.Tensor.
    """
    if weight_path.lower().endswith(".gguf"):
        return _load_gguf_state_dict(weight_path, device=device)
    return comfy.utils.load_torch_file(weight_path, device=device)


# ====================================================================
# Quant-resident loading (GGUF raw-block + ConvRot INT8)
# ====================================================================

def _open_gguf_reader(weight_path: str):
    """Open a ``.gguf`` file, recovering flat-block (sub-block-row) tensors.

    Delegates to :func:`modules.gguf_quant.open_gguf_reader` — stock reader
    first, tolerant flat-block reopen only when a quantized tensor's row is
    below its block size (non-spec conv conversions).

    Raises:
        RuntimeError: If the ``gguf`` package is not installed.
    """
    try:
        import gguf  # noqa: F401  (availability probe; see below)
    except ImportError as e:
        raise RuntimeError(
            "Loading .gguf weights requires the 'gguf' Python package. "
            "Install it with: pip install gguf"
        ) from e
    from .gguf_quant import open_gguf_reader

    return open_gguf_reader(weight_path)


def _gguf_kquant_present(reader) -> bool:
    """True when any tensor uses a K-quant format (Q4_K/Q5_K/Q6_K)."""
    from .gguf_quant import _T

    kquants = {_T.Q4_K, _T.Q5_K, _T.Q6_K}
    return any(t.tensor_type in kquants for t in reader.tensors)


# Quant-storage dtypes that must NEVER reach the dense float loader.
_QUANT_STORAGE_DTYPES = frozenset({
    torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2,
})


def _plan_quantized_safetensors_load(quant_map: dict):
    """Build the module-replacement plan for a scanned comfy_quant map.

    Pure planning — no state dict access, so both the batch path
    (``_prepare_quantized_safetensors_load``) and the streaming path
    (``_stream_apply_safetensors``) share one plan definition.

    Returns:
        ``(layer_plan, n_rowwise, n_fp8_resident)``.
    """
    from .convrot_quant import make_convrot_linear
    from .fp8_quant import make_fp8_linear

    layer_plan = {}
    n_rowwise = 0
    n_fp8_resident = 0
    for prefix, info in quant_map.items():
        if info.convrot:
            layer_plan[prefix] = make_convrot_linear(info)
        elif info.resident_fp8:
            layer_plan[prefix] = make_fp8_linear(info)
            n_fp8_resident += 1
        else:
            n_rowwise += 1
    return layer_plan, n_rowwise, n_fp8_resident


def _dequantize_rowwise_weight(prefix: str, info, w, s):
    """Validate + dequant ONE rowwise/blockwise weight (batch + streaming).

    Single-sourced so the batch dict-surgery path and the per-tensor
    streaming path can never drift apart mathematically.

    Returns:
        The dequantized weight in the layer's declared orig dtype.

    Raises:
        QuantTargetMismatch: On storage-dtype / weight-shape / scale-shape
            disagreement with the comfy_quant metadata.
    """
    from .convrot_quant import resolve_orig_dtype
    from .quant_common import QuantTargetMismatch

    storage_dtype = info.rowwise_dtype or torch.int8
    if w.dtype != storage_dtype and w.dtype != torch.uint8:
        raise QuantTargetMismatch(
            f"Rowwise layer '{prefix}': expected {storage_dtype} storage, "
            f"checkpoint has {w.dtype}"
        )
    if w.dim() != 2 or w.shape[0] != info.out_features \
            or (info.in_features and w.shape[1] != info.in_features):
        raise QuantTargetMismatch(
            f"Rowwise layer '{prefix}': weight shape {tuple(w.shape)} "
            f"disagrees with metadata [out={info.out_features}, "
            f"in={info.in_features}]"
        )
    orig_dtype = resolve_orig_dtype(info.orig_dtype)

    if info.group_size:
        # BLOCKWISE scales: shape [out / gs, in / gs], one scale per
        # gs x gs block (int8-blockwise checkpoints).
        gs = int(info.group_size)
        if w.shape[0] % gs or w.shape[1] % gs:
            raise QuantTargetMismatch(
                f"Blockwise layer '{prefix}': weight {tuple(w.shape)} "
                f"not divisible by group_size {gs} on both dims"
            )
        og, ig = w.shape[0] // gs, w.shape[1] // gs
        if tuple(s.shape) != (og, ig):
            raise QuantTargetMismatch(
                f"Blockwise layer '{prefix}': scale shape "
                f"{tuple(s.shape)} disagrees with weight "
                f"{tuple(w.shape)} at group_size {gs} (expected "
                f"({og}, {ig}))"
            )
        # fp32 intermediate: exact rescale before the final cast.
        return (
            (
                w.to(torch.float32).view(og, gs, ig, gs)
                * s.to(torch.float32).view(og, 1, ig, 1)
            ).reshape(w.shape).to(orig_dtype)
        )

    # SCALAR (per-tensor, fp8) or PER-ROW [out, 1] scales; both broadcast.
    if not (s.dim() == 0 or (s.dim() == 2 and s.shape[1] == 1
                             and s.shape[0] in (w.shape[0], 1))):
        raise QuantTargetMismatch(
            f"Rowwise layer '{prefix}': scale shape {tuple(s.shape)} "
            f"is neither a scalar nor [out,1] (weight "
            f"{tuple(w.shape)})"
        )
    return (w.to(torch.float32) * s.to(torch.float32)).to(orig_dtype)


def _prepare_quantized_safetensors_load(state_dict: dict, quant_map: dict):
    """Split a scanned comfy_quant map into execution strategies (in place).

    - ``convrot=True`` layers  -> module-replacement plan (int8 resident +
      comfy-kitchen kernels); their int8/scale tensors stay in the state dict
      for assign.
    - ``resident_fp8=True`` layers -> module-replacement plan (fp8 resident +
      per-matmul kitchen dequant); their fp8/scale tensors stay in the state
      dict for assign UNTOUCHED (no load-time dequant, plan 2026-08-27 D1).
    - remaining ``convrot=False`` layers -> dequant-at-load: ``weight = q *
      scale`` materialized back to the declared orig dtype inside
      ``state_dict``; scale + metadata keys removed.

    Returns:
        ``(layer_plan, n_rowwise, n_fp8_resident)``.

    Raises:
        QuantTargetMismatch: On missing/malformed rowwise tensor pairs or an
            undeclared orig dtype.
    """
    from .convrot_quant import QUANT_META_SUFFIX
    from .quant_common import QuantTargetMismatch

    layer_plan, n_rowwise, n_fp8_resident = _plan_quantized_safetensors_load(
        quant_map
    )
    for prefix, info in quant_map.items():
        if info.convrot:
            continue

        w_key = f"{prefix}.weight"
        s_key = f"{prefix}.weight_scale"
        w = state_dict.get(w_key)
        s = state_dict.get(s_key)
        if w is None or s is None:
            raise QuantTargetMismatch(
                f"Rowwise-quantized layer '{prefix}' is missing its "
                f"'{w_key}' / '{s_key}' tensors in the checkpoint"
            )

        if info.resident_fp8:
            # FP8 RESIDENT: storage + scalar scale stay untouched in the dict
            # for assign into FP8Linear; dequant happens per matmul. The scan
            # only sets resident_fp8 for scalar scales, but verify again —
            # assign replaces the parameter wholesale, so a wrong-shape scale
            # would silently corrupt the module.
            if w.dtype != info.rowwise_dtype:
                raise QuantTargetMismatch(
                    f"Resident fp8 layer '{prefix}': expected "
                    f"{info.rowwise_dtype} storage, checkpoint has {w.dtype}"
                )
            if not (s.dim() == 0 or (s.dim() == 1 and s.numel() == 1)):
                raise QuantTargetMismatch(
                    f"Resident fp8 layer '{prefix}': scale shape "
                    f"{tuple(s.shape)} is not a per-tensor scalar"
                )
            continue

        state_dict[w_key] = _dequantize_rowwise_weight(prefix, info, w, s)
        state_dict.pop(s_key, None)

    for prefix in quant_map:
        state_dict.pop(f"{prefix}.{QUANT_META_SUFFIX}", None)
    return layer_plan, n_rowwise, n_fp8_resident


# ====================================================================
# Streaming safetensors load (plan 2026-08-27, Phase 3 — RAM-spike kill)
# ====================================================================

# iter_safetensors_tensors lives in base_loader (shared with the standard
# loader's sharded/dense streaming path); imported at module top.


def _read_safetensors_tensor(weight_path: str, key: str):
    """Read a single named tensor from a safetensors file."""
    from safetensors import safe_open

    with safe_open(str(weight_path), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _stream_apply_safetensors(model, weight_path: str, quant_map: dict):
    """Assign a quantized safetensors checkpoint per-tensor (no full dict).

    The RAM-spike killer of plan 2026-08-27 (Phase 3): replaces the batch
    chain (``load_torch_file`` full dict -> in-dict dequant ->
    ``load_state_dict``) with a two-pass streaming read, so peak host RAM
    is bounded by the MODEL plus one tensor in flight instead of ~2x the
    dequantized checkpoint.

    Pass 1 reads only the tiny scale tensors of dequant-at-load layers
    (they may follow their weight in the file); pass 2 streams every tensor
    exactly once:

    - ``*.comfy_quant`` metadata: skipped (the resident modules consumed it
      at construction).
    - resident layers (convrot int8 / fp8): raw storage assigned untouched,
      dtype-checked against the swapped-in parameter.
    - dequant-at-load layers: dequantized ONE tensor at a time via the
      shared ``_dequantize_rowwise_weight`` (fp32 scratch freed each step).
    - everything else: assigned at file dtype — the final dtype cast still
      runs afterwards, unchanged.

    Post-assign fixups (re-tie, meta stragglers, sentinel buffers, RoPE,
    streaming conversion, missing/unexpected reporting) are shared with the
    batch path via ``VibeVoiceLoader._post_assign_fixups``.

    Args:
        model: Instantiated model with resident linears ALREADY swapped in
            (``replace_linears_for_quant`` ran first).
        weight_path: Path to the quantized ``.safetensors`` checkpoint.
        quant_map: Scanned ``comfy_quant`` map from
            ``scan_checkpoint_quantization``.

    Returns:
        Tuple of (missing_keys, unexpected_keys).

    Raises:
        QuantTargetMismatch: Missing scale tensors, storage/shape
            disagreement with the comfy_quant metadata.
        ValueError: Shape mismatch vs the model (same friendly message as
            the batch pre-check) or an unplanned quant-storage tensor.
    """
    from .convrot_quant import QUANT_META_SUFFIX
    from .quant_common import QuantTargetMismatch
    from .loader import VibeVoiceLoader, _shape_mismatch_error

    resident_prefixes = {
        p for p, info in quant_map.items()
        if info.convrot or info.resident_fp8
    }
    dequant_infos = {
        p: info for p, info in quant_map.items()
        if not info.convrot and not info.resident_fp8
    }

    # Pass 1 — scales for dequant-at-load layers are tiny; read them up
    # front so the weight stream never has to look back (file order is not
    # guaranteed to put the scale before its weight).
    scales = {}
    for prefix in dequant_infos:
        s_key = f"{prefix}.weight_scale"
        try:
            scales[prefix] = _read_safetensors_tensor(weight_path, s_key)
        except Exception as e:
            raise QuantTargetMismatch(
                f"Rowwise-quantized layer '{prefix}' is missing its "
                f"'{s_key}' tensor in the checkpoint"
            ) from e

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    unexpected = []
    meta_suffix = f".{QUANT_META_SUFFIX}"

    def _assign(key, tensor):
        target = params.get(key)
        if target is not None:
            if tuple(tensor.shape) != tuple(target.shape):
                raise _shape_mismatch_error(
                    model, [(key, tuple(tensor.shape), tuple(target.shape))]
                )
            comfy.utils.set_attr_param(model, key, tensor)
            return True
        target_buf = buffers.get(key)
        if target_buf is not None:
            if tuple(tensor.shape) != tuple(target_buf.shape):
                raise _shape_mismatch_error(
                    model, [(key, tuple(tensor.shape), tuple(target_buf.shape))]
                )
            comfy.utils.set_attr_buffer(model, key, tensor)
            return True
        return False

    # Pass 2 — stream every tensor exactly once.
    assigned = set()
    for key, tensor in iter_safetensors_tensors(weight_path):
        if key.endswith(meta_suffix):
            continue  # metadata consumed by the resident modules themselves
        prefix, _, leaf = key.rpartition(".")
        if prefix in dequant_infos and leaf == "weight_scale":
            continue  # consumed by pass 1
        if prefix in dequant_infos and leaf == "weight":
            # Dequantization builds a fresh tensor that already owns its
            # memory — no file mapping to sever.
            tensor = _dequantize_rowwise_weight(
                prefix, dequant_infos[prefix], tensor, scales[prefix]
            )
        else:
            if tensor.dtype in _QUANT_STORAGE_DTYPES:
                # Planned residents assign their raw storage into the matching
                # raw-storage parameter; anything else is an unplanned quant
                # weight that must never reach a float parameter.
                if prefix not in resident_prefixes \
                        or leaf not in ("weight", "weight_scale"):
                    raise ValueError(
                        f"Checkpoint contains quantized-weight tensors "
                        f"({key}: {tensor.dtype}) that no comfy_quant metadata "
                        f"declares. This node only loads quant formats it can "
                        f"execute; re-export the checkpoint or use a dense "
                        f"(bf16/fp16) file."
                    )
                target = params.get(key)
                if target is not None and tensor.dtype != target.dtype:
                    raise QuantTargetMismatch(
                        f"Resident layer '{prefix}': expected {target.dtype} "
                        f"storage for '{leaf}', checkpoint has {tensor.dtype}"
                    )
            # Sever the checkpoint file mapping: everything assigned raw from
            # the stream is a zero-copy mmap view, and a view retained by an
            # offloaded parameter would pin the whole file in the process
            # working set (same ghost-RAM / pin hazard as the dense sharded
            # path in VibeVoiceLoader._stream_apply_dense).
            tensor = tensor.clone()
        if _assign(key, tensor):
            assigned.add(key)
        else:
            unexpected.append(key)

    # Missing keys = expected model keys the file never delivered. Mirrors
    # torch's own missing-key semantics (persistent state only; tied names
    # appear individually, so a tied lm_head surfaces and is filtered by the
    # shared fixups' tied hint).
    expected = set(model.state_dict().keys())
    missing_keys = [k for k in expected if k not in assigned]
    return VibeVoiceLoader._post_assign_fixups(
        model, missing_keys, unexpected
    )


def _config_ties_word_embeddings(config) -> bool:
    """True when the config marks the lm_head as tied to input embeddings.

    Only REAL boolean flags count: test doubles (MagicMock configs) must not
    read as tied, so non-bool values are ignored.
    """
    decoder_config = getattr(config, "decoder_config", None)
    for cfg in (decoder_config, config):
        if cfg is None:
            continue
        flag = getattr(cfg, "tie_word_embeddings", False)
        if isinstance(flag, bool) and flag:
            return True
    return False


def _assert_lm_head_not_tied(config, quant_map: dict) -> None:
    """Reject checkpoints that quantize a TIED lm_head.

    A tied lm_head shares storage with the input embedding; a checkpoint
    that still carries ``lm_head.comfy_quant`` (quantized or not) contradicts
    the tied config — either the config or the export is wrong, and any load
    path would silently discard the quantized weights at re-tie time.
    """
    from .quant_common import QuantTargetMismatch

    if "lm_head" in quant_map and _config_ties_word_embeddings(config):
        raise QuantTargetMismatch(
            "Checkpoint quantizes 'lm_head' but the config sets "
            "tie_word_embeddings=True: a tied lm_head shares the input "
            "embedding's storage and cannot carry quantized weights. The "
            "sidecar config.json likely does not match this checkpoint — "
            "fix the config or re-export without quantizing lm_head."
        )


def _assert_gguf_lm_head_not_tied(config, reader) -> None:
    """GGUF twin of :func:`_assert_lm_head_not_tied`.

    A quantized ``lm_head.weight`` tensor in a tied config is the same
    contradiction as a quantized ``lm_head.comfy_quant``: the resident install
    would swap lm_head for a GGUFLinear and ``_post_assign_fixups``'s
    ``tie_weights()`` would then silently overwrite its raw-block Parameter
    with the embedding's float one, discarding the quantized weights.
    """
    from .gguf_quant import FLOAT_GGML_TYPES

    if not _config_ties_word_embeddings(config):
        return
    for t in reader.tensors:
        # 'output.weight' is llamacpp naming for lm_head and appears in
        # exports that otherwise keep HF names (VibeVoice-7B).
        if t.name in ("lm_head.weight", "output.weight") and t.tensor_type not in FLOAT_GGML_TYPES:
            from .quant_common import QuantTargetMismatch

            raise QuantTargetMismatch(
                "GGUF file carries a quantized "
                f"'{t.name}' (lm_head) but the "
                "config sets tie_word_embeddings=True: a tied lm_head shares "
                "the input embedding's storage and cannot carry quantized "
                "weights. The sidecar config.json likely does not match "
                "this checkpoint — fix the config or re-export without "
                "quantizing lm_head."
            )


def _demote_nonlinear_fp8_residents(model, quant_map: dict) -> dict:
    """Demote resident-fp8 layers whose target module is not an nn.Linear.

    Real fp8 checkpoints also quantize non-Linear modules — e.g.
    ``model.language_model.embed_tokens`` is an nn.Embedding. ``FP8Linear``
    can only replace nn.Linear, but fp8 dequant-at-load (``weight = q *
    scale``) is correct for ANY 2-D weight, so those layers fall back to the
    dequant-at-load strategy instead of failing the whole load.

    ConvRot layers are NOT demoted: rotated weights cannot be dequantized
    without undoing the rotation, so a non-Linear convrot target keeps the
    hard QuantTargetMismatch from ``replace_linears_for_quant`` (re-export
    is the correct answer there). Missing modules are left untouched so
    the replacement step raises its precise "not found" error.

    Returns:
        A new quant_map with affected infos replaced (QuantLayerInfo is
        frozen); the input map is not mutated.
    """
    from dataclasses import replace as dc_replace

    from .quant_common import resolve_module

    resolved = {}
    for prefix, info in quant_map.items():
        if info.resident_fp8:
            try:
                target = resolve_module(model, prefix)
            except KeyError:
                target = None
            if target is not None and not isinstance(target, torch.nn.Linear):
                logger.debug(
                    f"fp8-resident layer '{prefix}' targets "
                    f"{type(target).__name__}, not nn.Linear — falling back "
                    f"to dequant-at-load"
                )
                info = dc_replace(info, resident_fp8=False)
        resolved[prefix] = info
    return resolved


def _assert_dense_loadable(state_dict: dict) -> None:
    """Hard-fail when unplanned quantized weights reach the dense loader.

    Assigning int8/uint8/fp8 storages into float parameters either crashes
    cryptically or (fp8) silently misloads with scales ignored. Both are
    worse than a clear error at load time.
    """
    bad = [
        (k, str(v.dtype))
        for k, v in state_dict.items()
        if v.dtype in _QUANT_STORAGE_DTYPES
    ]
    if bad:
        sample = ", ".join(f"{k} [{d}]" for k, d in bad[:5])
        raise ValueError(
            "Checkpoint contains quantized-weight tensors but carries no "
            f"executable quantization metadata ({len(bad)} tensors, e.g. "
            f"{sample}). Loading them as floats would corrupt the model. "
            "Re-export it with *.comfy_quant metadata (comfy-model-tools) "
            "or use the dense/BF16 checkpoint."
        )


def _install_gguf_weights(model, reader) -> dict:
    """Install GGUF weights onto a meta-initialized model WITHOUT float
    materialization (plan 2026-08-24, D1 — the RAM-spike kill).

    Pipeline:
    1. Metadata pass: map every reader tensor onto the model tree (HF
       pass-through or llamacpp naming) and decide its disposition. No weight
       bytes are read here.
    2. Quantized tensors whose target is an ``nn.Linear`` become RESIDENTS:
       the Linear is swapped for :class:`~modules.gguf_quant.GGUFLinear` and
       the RAW BLOCK BYTES are installed as its uint8 parameter (the one
       unavoidable copy — it IS the residency).
    3. Quantized tensors whose target is NOT a Linear but has a weight
       parameter of matching shape (embeddings, conv heads) DEQUANTIZE AT
       LOAD directly into that parameter's dtype — the same fallback the fp8
       path uses for non-Linear residents.
    4. Float tensors (F32/F16/BF16) keep their NATIVE dtype.
    5. Install pass: residents are assigned in place, then the dense tensors
       stream through :meth:`VibeVoiceLoader._stream_apply_dense` one at a
       time (assign semantics, re-tie, sentinel/RoPE fixes preserved).

    Peak host RAM ≈ the installed model plus one tensor in flight, instead of
    the whole checkpoint buffered in a dict alongside it.

    Returns:
        Stats dict ``{"n_resident_layers", "raw_bytes", "n_dequant_load",
        "dequant_load_bytes", "weight_family"}``.
    """
    import numpy as np

    from .gguf_quant import (
        FLOAT_GGML_TYPES,
        GGUFTensor,
        SUPPORTED_GGML_TYPES,
        UnsupportedGGMLType,
        dequantize_reader_tensor,
        gguf_linear_factory,
        map_keys,
    )
    from .quant_common import (
        QuantTargetMismatch,
        replace_linears_for_quant,
        resolve_module,
    )

    tensors = list(reader.tensors)
    mapping = map_keys([t.name for t in tensors])
    modules_by_name = dict(model.named_modules())

    # Pass 1 touches metadata only: it decides the resident plan and the
    # dense assignment order without reading a single weight byte, so the
    # swap below can happen before any tensor is materialized.
    resident_plan = {}
    plan = []  # (tensor, "resident"|"dense", target, dtype, nbytes)
    unsupported = []
    dequant_load = []
    dequant_bytes = 0

    for t in tensors:
        target_key = mapping[t.name]
        logical_shape = tuple(int(s) for s in reversed(t.shape))
        tt = t.tensor_type

        if tt in FLOAT_GGML_TYPES:
            plan.append((t, "dense", target_key, None, 0))
            continue

        if tt not in SUPPORTED_GGML_TYPES:
            unsupported.append((t.name, tt))
            continue

        if not target_key.endswith(".weight"):
            raise QuantTargetMismatch(
                f"Quantized GGUF tensor '{t.name}' maps to '{target_key}', "
                f"which is not a Linear weight; this checkpoint cannot be "
                f"executed with quant-resident linears."
            )
        module_path = target_key[: -len(".weight")]
        target_module = modules_by_name.get(module_path)
        if target_module is None:
            raise QuantTargetMismatch(
                f"GGUF-quantized tensor '{t.name}' maps to '{module_path}' "
                f"(missing), expected an nn.Linear. The sidecar config may "
                f"not match this checkpoint."
            )
        if isinstance(target_module, torch.nn.Linear):
            expected_shape = tuple(target_module.weight.shape)
            if logical_shape != expected_shape:
                raise QuantTargetMismatch(
                    f"GGUF tensor '{t.name}' shape {logical_shape} disagrees "
                    f"with model {type(target_module).__name__} shape "
                    f"{expected_shape}"
                )
            plan.append((t, "resident", module_path, None, 0))
            resident_plan[module_path] = gguf_linear_factory(tt)
            continue
        target_param = getattr(target_module, "weight", None)
        if isinstance(target_param, torch.Tensor):
            expected_shape = tuple(target_param.shape)
            if logical_shape != expected_shape:
                raise QuantTargetMismatch(
                    f"GGUF tensor '{t.name}' shape {logical_shape} disagrees "
                    f"with model {type(target_module).__name__} shape "
                    f"{expected_shape}"
                )
            # Materialized once, straight into the destination dtype: the
            # fp32 intermediate is never allocated.
            dst = (
                target_param.dtype
                if target_param.dtype.is_floating_point
                else torch.float32
            )
            nbytes = 1
            for s in expected_shape:
                nbytes *= int(s)
            plan.append((t, "dense", target_key, dst, nbytes * dst.itemsize))
            dequant_load.append(target_key)
            dequant_bytes += nbytes * dst.itemsize
            continue
        kind = type(target_module).__name__
        raise QuantTargetMismatch(
            f"GGUF-quantized tensor '{t.name}' maps to '{module_path}' "
            f"({kind}), expected an nn.Linear. The sidecar config may "
            f"not match this checkpoint."
        )

    if unsupported:
        name, tt = unsupported[0]
        raise UnsupportedGGMLType(tt, f"{name}" + (
            f" (+{len(unsupported) - 1} more)" if len(unsupported) > 1 else ""
        ))

    replaced = replace_linears_for_quant(model, resident_plan)
    resident_set = set(replaced)

    # Pass 2 installs one tensor at a time; the previous tensor is released
    # before the next is read, so host RAM is bounded by the installed model
    # plus one tensor in flight rather than by the whole checkpoint.
    total_raw = 0
    for t, kind, target, _dst, _nbytes in plan:
        if kind != "resident" or target not in resident_set:
            continue
        module = resolve_module(model, target)
        module.set_raw_weight(GGUFTensor.from_reader_tensor(t).raw)
        total_raw += module.weight.numel()

    def _dense_pairs():
        for t, kind, target, dst, _nbytes in plan:
            if kind != "dense":
                continue
            if dst is None:
                arr = np.ascontiguousarray(t.data)
                # read-only mmap: copy so torch never warns
                tensor = torch.from_numpy(
                    arr if arr.flags.writeable else arr.copy()
                ).clone()
                if t.data.dtype == np.uint8:  # BF16 arrives as raw bytes
                    tensor = tensor.view(torch.bfloat16)
                shape = tuple(int(s) for s in reversed(t.shape))
                yield target, tensor.reshape(shape)
            else:
                yield target, dequantize_reader_tensor(t, dst)

    known_missing = {f"{p}.weight" for p in replaced}
    VibeVoiceLoader._stream_apply_dense(
        model, _dense_pairs(), known_missing=known_missing
    )

    if dequant_load:
        logger.debug(
            f"Dequantized {len(dequant_load)} non-Linear GGUF weight(s) at "
            f"load (embeddings/conv heads): "
            f"{', '.join(dequant_load[:5])}"
            + (f" (+{len(dequant_load) - 5} more)" if len(dequant_load) > 5 else "")
        )

    return {
        "n_resident_layers": len(replaced),
        "raw_bytes": total_raw,
        "n_dequant_load": len(dequant_load),
        "dequant_load_bytes": int(dequant_bytes),
        "weight_family": "gguf_block",
    }


# ====================================================================
# Low-bit / naive quantization detection (defensive warning)
# ====================================================================

# safetensors dtypes that indicate integer (quantized) storage.
_SAFETENSORS_INT_DTYPES = {
    "I8", "U8", "I16", "U16", "I32", "U32", "I64", "U64", "I4", "U4",
}

# Tensor-name substrings that indicate proper dequantization metadata is present
# (a real quantized checkpoint stores per-channel scales / zero-points).
_QUANT_SCALE_NAME_HINTS = ("scale", "zero_point", "qzero", "absmax")

# Fraction of integer/low-bit tensors above which we consider the file
# "mostly quantized" (as opposed to a couple of incidental index tensors).
_LOWBIT_FRACTION_THRESHOLD = 0.10


def _inspect_safetensors_quantization(weight_path: str):
    """Parse only the safetensors header (no tensor data) to detect naive int casts.

    A *proper* quantized checkpoint stores dequantization metadata (per-channel
    scale / zero-point tensors). A *naive* ``int8`` cast stores raw integer values
    with no scales, which is unusable (loads as garbage weights).

    Args:
        weight_path: Absolute path to a ``.safetensors`` file.

    Returns:
        dict ``{"total", "int_count", "has_scale_meta"}`` or ``None`` if the
        header could not be parsed.
    """
    import struct

    try:
        with open(weight_path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
    except Exception:
        return None

    header.pop("__metadata__", None)
    total = len(header)
    int_count = 0
    has_scale_meta = False
    for name, info in header.items():
        if info.get("dtype", "") in _SAFETENSORS_INT_DTYPES:
            int_count += 1
        low = name.lower()
        if any(hint in low for hint in _QUANT_SCALE_NAME_HINTS):
            has_scale_meta = True

    return {"total": total, "int_count": int_count, "has_scale_meta": has_scale_meta}


def _inspect_gguf_quantization(weight_path: str):
    """Inspect the GGUF tensor table (no dequantization) to count sub-4-bit I-quants.

    Args:
        weight_path: Absolute path to a ``.gguf`` file.

    Returns:
        dict ``{"total", "low_bit_count"}`` or ``None`` if the file could not be
        read or the ``gguf`` package is unavailable.
    """
    try:
        import gguf  # noqa: F401  (availability probe)
        from gguf.constants import GGMLQuantizationType
    except ImportError:
        return None

    try:
        from .gguf_quant import open_gguf_reader

        reader = open_gguf_reader(weight_path)
    except Exception:
        return None

    # Aggressive sub-4-bit importance-matrix quants. These are faithful but very
    # lossy; on a small (~1.5B) TTS LM they commonly degrade below the threshold
    # needed for text-conditioned generation.
    low_bit_types = {
        GGMLQuantizationType.IQ1_S,
        GGMLQuantizationType.IQ1_M,
        GGMLQuantizationType.IQ2_XXS,
        GGMLQuantizationType.IQ2_XS,
        GGMLQuantizationType.IQ2_S,
        GGMLQuantizationType.IQ3_XXS,
        GGMLQuantizationType.IQ3_S,
    }
    total = len(reader.tensors)
    low_bit_count = sum(1 for t in reader.tensors if t.tensor_type in low_bit_types)
    return {"total": total, "low_bit_count": low_bit_count}


def warn_if_lowbit_quantization(weight_path: str) -> None:
    """Log a warning if ``weight_path`` looks over-quantized or naively cast.

    This is a defensive, load-time heuristic. It does not block loading; it only
    surfaces a clear, actionable warning so the user knows a garbled / silent
    result is likely caused by the checkpoint rather than the node.

    Two failure modes are detected:
      * safetensors with many raw integer tensors but NO scale/zero-point metadata
        -> naive int cast (unusable).
      * GGUF with many sub-4-bit I-quant tensors -> aggressive quant that may
        degrade TTS quality (garbled output / reference echo).

    Args:
        weight_path: Absolute path to the external weight file.
    """
    lower = weight_path.lower()

    if lower.endswith(".safetensors"):
        info = _inspect_safetensors_quantization(weight_path)
        if info is None or info["total"] == 0:
            return
        frac = info["int_count"] / info["total"]
        if frac >= _LOWBIT_FRACTION_THRESHOLD and not info["has_scale_meta"]:
            logger.warning(
                f"[low-bit check] '{os.path.basename(weight_path)}' stores "
                f"{info['int_count']}/{info['total']} tensors as raw integers with "
                f"NO dequantization scale/zero-point metadata. This looks like a "
                f"naive int cast, which loads as garbage weights and will produce "
                f"silent or broken audio. Use a proper quantized or full-precision "
                f"(BF16/FP16) checkpoint instead."
            )
        return

    if lower.endswith(".gguf"):
        info = _inspect_gguf_quantization(weight_path)
        if info is None or info["total"] == 0:
            return
        frac = info["low_bit_count"] / info["total"]
        if frac >= _LOWBIT_FRACTION_THRESHOLD:
            logger.warning(
                f"[low-bit check] '{os.path.basename(weight_path)}' contains "
                f"{info['low_bit_count']}/{info['total']} tensors at sub-4-bit "
                f"I-quant precision. This is very aggressive for a small TTS LM and "
                f"may degrade below the threshold for text-conditioned generation "
                f"(symptoms: garbled syllables, reference-audio echo, never "
                f"terminating). Consider a higher-quality quant (Q4_K_M / Q5_K_M / "
                f"Q8_0 / BF16)."
            )
        return


# ====================================================================
# In-memory state dict loading
# ====================================================================

def _load_state_dict_into_model_from_memory(model, state_dict: dict):
    """Load an in-memory state dict into an already-instantiated model.

    Mirrors :meth:`VibeVoiceLoader._load_state_dict_into_model` but accepts a
    state dict that is already in memory (loaded via
    ``comfy.utils.load_torch_file``) instead of resolving a checkpoint path.

    Delegates to :meth:`VibeVoiceLoader._apply_state_dict` (plan 2026-08-18,
    D2/D3): assign semantics (no copy pass), tied-weight re-tie, and
    zero-materialization of meta stragglers.

    Args:
        model: Instantiated model (weights not yet loaded).
        state_dict: State dict mapping (on CPU).

    Returns:
        The model with the state dict loaded (still on CPU).
    """
    # Sever file-backed storage before assign: comfy.utils.load_torch_file
    # returns zero-copy mmap views for safetensors checkpoints, and any view
    # that survives into an offloaded CPU parameter keeps the whole file
    # mapping resident in the process working set (ghost RAM on top of the
    # VRAM copy + unstable cudaHostRegister pins). Clone in place so peak
    # RAM stays at dict + one tensor. Already-owned tensors (e.g. freshly
    # dequantized ones) pay one extra copy — harmless.
    for key in list(state_dict.keys()):
        tensor = state_dict[key]
        if isinstance(tensor, torch.Tensor):
            state_dict[key] = tensor.clone()
    VibeVoiceLoader._apply_state_dict(model, state_dict)
    return model


# ====================================================================
# Core external loading function
# ====================================================================

def _log_load_diagnostics(
    *,
    config_name: str,
    requested_attention_mode: str,
    resolved_attention_mode: str,
    weight_family: str,
    load_device,
) -> None:
    """Emit the ONE line per load that attributes a slow run from the log.

    WHY the GGUF forward counters are NOT here: they only move once forwards
    run, so a readout taken during a load is structurally always
    ``fast=0 streamed=0`` — the very value a reader would take as "the hook
    path is not poisoning this run", i.e. a wrong answer produced by the
    instrument. The loaders zero the counters instead
    (:func:`~modules.gguf_quant.reset_gguf_forward_counters`) and the
    generation paths report them afterwards under
    "GGUF forward diagnostics:" — paste THAT line back for a slow run.
    """
    logger.info(
        "Load diagnostics: model='%s' family=%s requested_attention=%s "
        "resolved_attention=%s device=%s",
        config_name, weight_family, requested_attention_mode,
        resolved_attention_mode, load_device,
    )


def _reset_gguf_forward_counters() -> None:
    """Zero the GGUF forward counters at load (diagnostics must never fail a load)."""
    try:
        from .gguf_quant import reset_gguf_forward_counters

        reset_gguf_forward_counters()
    except Exception:  # pragma: no cover - diagnostics must never break a load
        pass


def load_external_vibevoice_model(
    weight_path: str,
    config_name: str,
    attention_mode: str = "eager",
    use_llm_4bit: bool = False,
    dtype_str: str = "auto",
    device=None,
) -> dict:
    """Load a VibeVoice model from an external weight file.

    The model is built entirely on CPU. The single host-to-device transfer is
    owned by :class:`~modules.patcher.VibeVoicePatcher` after ComfyUI's VRAM
    arbitration (same contract as the standard loader path).

    Args:
        weight_path: Absolute path to the external weight file
            (safetensors / .bin / .gguf).
        config_name: Architecture config selector. Used for the packaged-default
            fallback when no sidecar config is present. One of
            ``EXTERNAL_CONFIG_OPTIONS``; ``AUTO_CONFIG_NAME`` resolves the
            family from the weights' embedding fingerprint.
        attention_mode: Attention implementation
            ("eager", "sdpa", "flash_attention_2", "sage").
        use_llm_4bit: Whether to quantize the LLM to 4-bit NF4.
        dtype_str: Dtype string ("auto", "bf16", "fp16", "fp32").
        device: Optional target device hint for dtype-auto resolution.

    Returns:
        The ``VIBEVOICE_MODEL`` bundle dict::

            {
                "state_dict": dict[str, torch.Tensor],
                "config": VibeVoiceConfig | VibeVoiceStreamingConfig,
                "processor": VibeVoiceProcessor | VibeVoiceStreamingProcessor,
                "model": torch.nn.Module,
                "model_name": str,
                "source_path": str,
                "is_streaming": bool,
            }

    Raises:
        FileNotFoundError: If the weight file or config cannot be found.
        ValueError: If ``AUTO_CONFIG_NAME`` is selected but the architecture
            cannot be determined from the weight file.
        RuntimeError: If model instantiation or loading fails.
    """
    # Legacy alias normalization (plan 2026-08-27, D1): saved workflows may
    # still carry removed option values (e.g. "VibeVoice-Large").
    config_name = normalize_config_name(config_name)
    # Per-load counter scope: the "GGUF forward diagnostics" line printed
    # after generation describes THIS model, not the whole process history.
    _reset_gguf_forward_counters()

    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"External weight file not found: {weight_path}")

    # File-identity stamp for the cache key (plan 2026-08-20, B1): captured
    # once at entry so the bundle records exactly which bytes were loaded.
    try:
        _stat = os.stat(weight_path)
        source_mtime_ns = _stat.st_mtime_ns
        source_size = _stat.st_size
    except OSError:
        source_mtime_ns = 0
        source_size = 0

    # Defensive: warn early if the checkpoint looks over-quantized / naively cast.
    warn_if_lowbit_quantization(weight_path)

    # ASR models use a distinct loading branch (different config / tokenizer /
    # processor / model classes, and no 4-bit quantization).
    if is_asr_config_name(config_name):
        return load_external_vibevoice_asr_model(
            weight_path=weight_path,
            config_name=config_name,
            attention_mode=attention_mode,
            dtype_str=dtype_str,
            device=device,
        )

    # Resolve attention mode with fallback logic (same as standard loader).
    # Keep the REQUESTED value: the resolved one can silently differ (sage ->
    # sdpa on unsupported hardware, eager -> sdpa under 4-bit), and a silent
    # substitution is exactly what a slow run must be able to rule out.
    requested_attention_mode = attention_mode
    attention_mode = resolve_attention_mode(attention_mode, use_llm_4bit)

    # Weight-plan validation + lazy source opening (plan 2026-08-24, D1/D2/D4).
    # GGUF keeps an mmap READER (no float materialization); plain safetensors
    # are scanned for ConvRot *.comfy_quant metadata BEFORE any load.
    lower_path = weight_path.lower()
    is_gguf_file = lower_path.endswith(".gguf")
    gguf_reader = _open_gguf_reader(weight_path) if is_gguf_file else None
    convrot_quant_map = {}
    if gguf_reader is None and lower_path.endswith(".safetensors"):
        # Scan-only pass. UNREADABLE headers degrade to "no quant metadata"
        # (a genuinely corrupt file still fails at the actual load), but an
        # UNSUPPORTED QUANT FORMAT must propagate: silently falling through
        # to the dense loader misloads int8/fp8 weights as floats.
        try:
            from .convrot_quant import scan_checkpoint_quantization

            convrot_quant_map = scan_checkpoint_quantization(weight_path)
        except UnsupportedQuantFormat:
            raise
        except Exception as e:
            logger.debug(f"ConvRot scan skipped for {weight_path}: {e}")
            convrot_quant_map = {}
    validate_weight_plan(
        is_gguf_file=is_gguf_file,
        convrot_quant_map=convrot_quant_map,
        use_llm_4bit=use_llm_4bit,
        attention_mode=attention_mode,
        gguf_kquant_present=_gguf_kquant_present(gguf_reader) if gguf_reader else False,
    )

    # Architecture fingerprint (plan 2026-08-27, D3): header-only for
    # safetensors, reuses the open GGUF reader. Computed BEFORE the heavy
    # state-dict load so Auto-detect can fail fast (D6) and reconciliation
    # (D5) reuses it below without re-reading the file.
    from .config_detect import fingerprint_weights

    weights_fp = fingerprint_weights(weight_path, gguf_reader=gguf_reader)

    # Auto-detect (D6): resolve the architecture from the fingerprint BEFORE
    # any heavy work. Conclusive -> adopt the detected family; inconclusive
    # -> actionable ValueError (fail-fast, no partial load). Explicit
    # selections skip this and self-heal via reconciliation below (D5).
    if config_name == AUTO_CONFIG_NAME:
        config_name = resolve_auto_config_name(weight_path, weights_fp=weights_fp)

    cpu_device = torch.device("cpu")
    state_dict = None
    # Quantized safetensors load per-tensor AFTER model instantiation
    # (plan 2026-08-27, Phase 3): the full-file state dict never
    # materializes, so peak RAM is bounded by the model + one tensor in
    # flight instead of ~2x the dequantized checkpoint.
    stream_quant_load = (
        bool(convrot_quant_map)
        and weight_path.lower().endswith(".safetensors")
    )
    if gguf_reader is not None:
        logger.debug(
            f"Opening external VibeVoice GGUF weights (raw-block residency): "
            f"{weight_path}"
        )
    elif stream_quant_load:
        logger.debug(
            f"Streaming external VibeVoice quant weights from: {weight_path}"
        )
    else:
        logger.debug(f"Loading external VibeVoice weights from: {weight_path}")
        # Step 1: Load the state dict onto CPU (safetensors/bin via ComfyUI's
        # loader). Always CPU — the patcher owns the single H2D transfer.
        state_dict = _load_weight_state_dict(weight_path, cpu_device)

    # Step 2: Resolve and load the architecture config, reconciled against
    # the weights' architecture fingerprint (plan 2026-08-27, D5). When the
    # fingerprint contradicts the selected config family, the detected
    # family's packaged config is substituted with a WARNING.
    config_path = resolve_sidecar_config(weight_path, config_name)
    config = VibeVoiceLoader._load_config(config_path, config_name)
    effective_name, config_changed = reconcile_config(config_name, config, weights_fp)
    if config_changed:
        config_name = effective_name
        config_path = resolve_sidecar_config(weight_path, config_name)
        config = VibeVoiceLoader._load_config(config_path, config_name)

    # Step 3: Detect streaming from the loaded config.
    is_streaming = isinstance(config, VibeVoiceStreamingConfig)
    if is_streaming:
        logger.debug(f"External model '{config_name}' detected as streaming model")
        # Backends measured to diverge on the realtime path are dropped before
        # anything keys off the mode, so the bundle records what will really be
        # used (plan 2026-09-26, step S3.2).
        attention_mode = resolve_realtime_attention_mode(attention_mode)

    # Step 4: Resolve and load the tokenizer.
    tokenizer_dir = resolve_sidecar_tokenizer_dir(weight_path)
    vibevoice_tokenizer = VibeVoiceLoader._load_tokenizer(tokenizer_dir, config_name)

    # Step 5: Resolve and load the processor.
    preprocessor_path = resolve_sidecar_preprocessor(weight_path)
    processor = VibeVoiceLoader._load_processor(
        vibevoice_tokenizer, preprocessor_path, is_streaming=is_streaming
    )

    # Step 6: Resolve dtype + attention implementation.
    load_device = (
        model_management.get_torch_device()
        if not isinstance(device, torch.device)
        else device
    )
    model_dtype = resolve_dtype(dtype_str, load_device)

    quant_config = None
    final_load_dtype = model_dtype
    if use_llm_4bit:
        bnb_compute_dtype = model_dtype
        if attention_mode == "sage":
            bnb_compute_dtype, final_load_dtype = torch.float32, torch.float32
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bnb_compute_dtype,
        )

    attn_implementation_for_load = get_attn_implementation_for_load(attention_mode)

    try:
        logger.debug(
            f"Instantiating external VibeVoice model '{config_name}' with "
            f"dtype={final_load_dtype}, attention='{attn_implementation_for_load}'"
        )

        # Step 7: Instantiate the model class directly (no meta device init).
        model = VibeVoiceLoader._instantiate_model(
            config=config,
            is_streaming=is_streaming,
            attn_implementation=attn_implementation_for_load,
            final_load_dtype=final_load_dtype,
        )

        # Step 8: Bind the weights — dispatch by weight family:
        # - gguf_block: raw-block residency install (no float materialization)
        # - convrot_int8: linears already swapped; int8/scale assign directly
        # - dense: legacy in-memory assign of the full float state dict
        weight_family = "dense"
        quant_stats = {}
        if gguf_reader is not None:
            # A quantized lm_head under a tied config would be silently
            # discarded by tie_weights() — reject before any install work.
            _assert_gguf_lm_head_not_tied(config, gguf_reader)
            quant_stats = _install_gguf_weights(model, gguf_reader)
            weight_family = "gguf_block"
            del gguf_reader
            gguf_reader = None
            # Unmap the file and drop install scratch before the patcher H2D:
            # the mmap otherwise keeps the whole checkpoint in the working
            # set, on top of the VRAM copy.
            gc.collect()
        elif convrot_quant_map:
            from .quant_common import replace_linears_for_quant

            _assert_lm_head_not_tied(config, convrot_quant_map)
            # fp8 checkpoints may quantize non-Linear modules (embed_tokens);
            # those cannot become FP8Linear and fall back to dequant-at-load.
            convrot_quant_map = _demote_nonlinear_fp8_residents(
                model, convrot_quant_map
            )
            if stream_quant_load:
                # Streaming path: plan only, swap modules, then assign the
                # checkpoint per-tensor (no full state dict ever exists).
                layer_plan, n_rowwise, n_fp8_resident = (
                    _plan_quantized_safetensors_load(convrot_quant_map)
                )
                replaced = replace_linears_for_quant(model, layer_plan)
                _stream_apply_safetensors(model, weight_path, convrot_quant_map)
            else:
                layer_plan, n_rowwise, n_fp8_resident = (
                    _prepare_quantized_safetensors_load(state_dict, convrot_quant_map)
                )
                replaced = replace_linears_for_quant(model, layer_plan)
                model = _load_state_dict_into_model_from_memory(model, state_dict)
                del state_dict
                state_dict = None
            # Family label: convrot dominates; pure fp8-resident files get
            # their own label; pure dequant-at-load keeps the legacy one.
            weight_family = (
                "fp8_resident"
                if n_fp8_resident and len(replaced) == n_fp8_resident
                else "convrot_int8"
            )
            quant_stats = {
                "n_resident_layers": len(replaced),
                "n_rowwise_layers": n_rowwise,
                "n_fp8_resident_layers": n_fp8_resident,
            }
        else:
            # Defensive net: unplanned int8/uint8/fp8 weights must never
            # reach the float loader (crash or silent corruption).
            _assert_dense_loadable(state_dict)
            model = _load_state_dict_into_model_from_memory(model, state_dict)

            # Free the state dict immediately — the model now owns the tensors
            # (assign semantics), so the dict is dead weight (plan D7/RC-4).
            del state_dict
            state_dict = None

        # Step 9: Apply the final dtype ON CPU (single H2D owned by patcher) —
        # only where needed (plan 2026-08-18, D4/RC-3).
        cast_model_to_dtype_if_needed(model, final_load_dtype)

        # Step 10: Apply 4-bit quantization if requested (post-load).
        if quant_config is not None:
            try:
                from transformers.integrations.bitsandbytes import replace_with_bnb_linear
            except ImportError:
                from transformers.utils.bitsandbytes import replace_with_bnb_linear
            replace_with_bnb_linear(
                model,
                quantization_config=quant_config,
                modules_to_not_convert=None,
            )

        # Step 11: Apply SageAttention post-load if requested.
        if attention_mode == "sage":
            if check_sage_attention_compatible():
                set_sage_attention(model)
            else:
                raise RuntimeError("Incompatible hardware/setup for SageAttention.")

        # Step 12: Eval mode.
        model.eval()
        setattr(model, "_llm_4bit", bool(quant_config))

        logger.info(
            f"Successfully loaded external VibeVoice model '{config_name}' "
            f"from {weight_path}"
        )
        _log_load_diagnostics(
            config_name=config_name,
            requested_attention_mode=requested_attention_mode,
            resolved_attention_mode=attention_mode,
            weight_family=weight_family,
            load_device=load_device,
        )

        # Step 13: Assemble the bundle.
        # NOTE (plan 2026-08-18, D7/RC-4): the state dict is intentionally NOT
        # included — the model owns the tensors after assign-loading, and
        # retaining a second ~model-size copy in the node output is dead RAM.
        #
        # Plan 2026-08-20 (B1/P1): the bundle also records the full build
        # identity — file stat, RESOLVED attention mode, 4-bit flag, and the
        # requested dtype string — so consumers can derive a cache key that is
        # byte-for-byte equal to the loader-node's pre-build request identity.
        return {
            "config": config,
            "processor": processor,
            "model": model,
            "model_name": config_name,
            "source_path": weight_path,
            "source_mtime_ns": source_mtime_ns,
            "source_size": source_size,
            "attention_mode": attention_mode,
            "use_llm_4bit": bool(use_llm_4bit),
            "dtype_str": dtype_str,
            "is_streaming": is_streaming,
            "is_asr": False,
            "weight_family": weight_family,
            "quant_stats": quant_stats,
        }

    except Exception as e:
        logger.error(
            f"Failed to load external VibeVoice model '{config_name}' "
            f"from {weight_path}: {e}"
        )
        raise RuntimeError(
            f"Failed to load external VibeVoice model '{config_name}': {e}"
        )


# ====================================================================
# ASR external loading branch
# ====================================================================

def load_external_vibevoice_asr_model(
    weight_path: str,
    config_name: str,
    attention_mode: str = "sdpa",
    dtype_str: str = "auto",
    device=None,
) -> dict:
    """Load a VibeVoice ASR model from an external weight file.

    ASR counterpart of :func:`load_external_vibevoice_model`. ASR models use a
    distinct config (:class:`VibeVoiceASRConfig`), tokenizer
    (:class:`VibeVoiceASRTextTokenizerFast`), processor
    (:class:`VibeVoiceASRProcessor`), and model class
    (:class:`VibeVoiceASRForConditionalGeneration`), and do not support 4-bit
    quantization.

    The model is built entirely on CPU. The single host-to-device transfer is
    owned by :class:`~modules.patcher.VibeVoiceASRPatcher` after ComfyUI's VRAM
    arbitration (same contract as the standard ASR loader path).

    Args:
        weight_path: Absolute path to the external weight file
            (safetensors / .bin / .gguf).
        config_name: Architecture config selector (e.g. ``"VibeVoice-ASR"``).
        attention_mode: Attention implementation
            ("eager", "sdpa", "flash_attention_2", "sage").
        dtype_str: Dtype string ("auto", "bf16", "fp16", "fp32").
        device: Optional target device hint for dtype-auto resolution.

    Returns:
        The ``VIBEVOICE_MODEL`` bundle dict::

            {
                "state_dict": dict[str, torch.Tensor],
                "config": VibeVoiceASRConfig,
                "processor": VibeVoiceASRProcessor,
                "model": VibeVoiceASRForConditionalGeneration,
                "model_name": str,
                "source_path": str,
                "is_streaming": False,
                "is_asr": True,
            }

    Raises:
        FileNotFoundError: If the weight file, config, or tokenizer cannot be found.
        RuntimeError: If model instantiation or loading fails.
    """
    # Legacy alias normalization (plan 2026-08-27, D1) — mirrors the TTS branch.
    config_name = normalize_config_name(config_name)
    # Per-load counter scope — mirrors the TTS branch.
    _reset_gguf_forward_counters()

    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"External weight file not found: {weight_path}")

    # File-identity stamp (plan 2026-08-20, B1) — mirrors the TTS branch.
    try:
        _stat = os.stat(weight_path)
        source_mtime_ns = _stat.st_mtime_ns
        source_size = _stat.st_size
    except OSError:
        source_mtime_ns = 0
        source_size = 0

    # Defensive: warn early if the checkpoint looks over-quantized / naively cast.
    warn_if_lowbit_quantization(weight_path)

    # Resolve attention mode with fallback logic (no 4-bit for ASR), then drop
    # the ASR exclusions — this branch builds an ASR model, whose prefill
    # always carries a left-pad mask the sage kernel cannot honour.
    requested_attention_mode = attention_mode
    attention_mode = resolve_asr_attention_mode(
        resolve_attention_mode(attention_mode, quantize_4bit=False)
    )

    # Weight-plan validation + lazy source opening (mirrors the TTS branch).
    lower_path = weight_path.lower()
    is_gguf_file = lower_path.endswith(".gguf")
    gguf_reader = _open_gguf_reader(weight_path) if is_gguf_file else None
    convrot_quant_map = {}
    if gguf_reader is None and lower_path.endswith(".safetensors"):
        # Scan-only pass. UNREADABLE headers degrade to "no quant metadata"
        # (a genuinely corrupt file still fails at the actual load), but an
        # UNSUPPORTED QUANT FORMAT must propagate: silently falling through
        # to the dense loader misloads int8/fp8 weights as floats.
        try:
            from .convrot_quant import scan_checkpoint_quantization

            convrot_quant_map = scan_checkpoint_quantization(weight_path)
        except UnsupportedQuantFormat:
            raise
        except Exception as e:
            logger.debug(f"ConvRot scan skipped for {weight_path}: {e}")
            convrot_quant_map = {}
    validate_weight_plan(
        is_gguf_file=is_gguf_file,
        convrot_quant_map=convrot_quant_map,
        use_llm_4bit=False,
        attention_mode=attention_mode,
        gguf_kquant_present=_gguf_kquant_present(gguf_reader) if gguf_reader else False,
    )

    # Step 1: Load the state dict onto CPU (safetensors/bin via ComfyUI's
    # loader). GGUF keeps a lazy mmap READER instead (raw-block residency).
    # Quantized safetensors stream per-tensor after instantiation instead
    # (plan 2026-08-27, Phase 3 — see the TTS branch for the full contract).
    cpu_device = torch.device("cpu")
    state_dict = None
    stream_quant_load = (
        bool(convrot_quant_map)
        and weight_path.lower().endswith(".safetensors")
    )
    if gguf_reader is not None:
        logger.debug(
            f"Opening external VibeVoice ASR GGUF weights (raw-block "
            f"residency): {weight_path}"
        )
    elif stream_quant_load:
        logger.debug(
            f"Streaming external VibeVoice ASR quant weights from: "
            f"{weight_path}"
        )
    else:
        logger.debug(f"Loading external VibeVoice ASR weights from: {weight_path}")
        state_dict = _load_weight_state_dict(weight_path, cpu_device)

    # Step 2: Resolve and load the ASR architecture config.
    config_path = resolve_sidecar_config(weight_path, config_name)
    config = _load_asr_config(config_path)

    # Step 3: Resolve and load the ASR tokenizer.
    tokenizer_dir = resolve_sidecar_tokenizer_dir(weight_path)
    asr_tokenizer = _load_asr_tokenizer(tokenizer_dir)

    # Step 4: Resolve and load the ASR processor.
    preprocessor_path = resolve_sidecar_preprocessor(weight_path)
    processor = _load_asr_processor(asr_tokenizer, preprocessor_path)

    # Step 5: Resolve dtype + attention implementation.
    load_device = (
        model_management.get_torch_device()
        if not isinstance(device, torch.device)
        else device
    )
    model_dtype = resolve_dtype(dtype_str, load_device)
    attn_implementation_for_load = get_attn_implementation_for_load(attention_mode)

    try:
        logger.debug(
            f"Instantiating external VibeVoice ASR model '{config_name}' with "
            f"dtype={model_dtype}, attention='{attn_implementation_for_load}'"
        )

        # Step 6: Instantiate the ASR model class directly (no meta device init).
        model = _instantiate_asr_model(
            config=config,
            attn_implementation=attn_implementation_for_load,
            final_load_dtype=model_dtype,
        )

        # Step 7: Bind the weights (same dispatch as the TTS branch).
        weight_family = "dense"
        quant_stats = {}
        if gguf_reader is not None:
            # A quantized lm_head under a tied config would be silently
            # discarded by tie_weights() — reject before any install work.
            _assert_gguf_lm_head_not_tied(config, gguf_reader)
            quant_stats = _install_gguf_weights(model, gguf_reader)
            weight_family = "gguf_block"
            del gguf_reader
            gguf_reader = None
            # Unmap the file and drop install scratch before the patcher H2D:
            # the mmap otherwise keeps the whole checkpoint in the working
            # set, on top of the VRAM copy.
            gc.collect()
        elif convrot_quant_map:
            from .quant_common import replace_linears_for_quant

            _assert_lm_head_not_tied(config, convrot_quant_map)
            # fp8 checkpoints may quantize non-Linear modules (embed_tokens);
            # those cannot become FP8Linear and fall back to dequant-at-load.
            convrot_quant_map = _demote_nonlinear_fp8_residents(
                model, convrot_quant_map
            )
            if stream_quant_load:
                # Streaming path: plan only, swap modules, then assign the
                # checkpoint per-tensor (no full state dict ever exists).
                layer_plan, n_rowwise, n_fp8_resident = (
                    _plan_quantized_safetensors_load(convrot_quant_map)
                )
                replaced = replace_linears_for_quant(model, layer_plan)
                _stream_apply_safetensors(model, weight_path, convrot_quant_map)
            else:
                layer_plan, n_rowwise, n_fp8_resident = (
                    _prepare_quantized_safetensors_load(state_dict, convrot_quant_map)
                )
                replaced = replace_linears_for_quant(model, layer_plan)
                model = _load_state_dict_into_model_from_memory(model, state_dict)
                del state_dict
                state_dict = None
            # Family label: convrot dominates; pure fp8-resident files get
            # their own label; pure dequant-at-load keeps the legacy one.
            weight_family = (
                "fp8_resident"
                if n_fp8_resident and len(replaced) == n_fp8_resident
                else "convrot_int8"
            )
            quant_stats = {
                "n_resident_layers": len(replaced),
                "n_rowwise_layers": n_rowwise,
                "n_fp8_resident_layers": n_fp8_resident,
            }
        else:
            # Defensive net: unplanned int8/uint8/fp8 weights must never
            # reach the float loader (crash or silent corruption).
            _assert_dense_loadable(state_dict)
            model = _load_state_dict_into_model_from_memory(model, state_dict)

            # Free the state dict immediately — the model now owns the tensors
            # (assign semantics), so the dict is dead weight (plan D7/RC-4).
            del state_dict
            state_dict = None

        # Step 8: Apply the final dtype ON CPU (single H2D owned by patcher) —
        # only where needed (plan 2026-08-18, D4/RC-3).
        cast_model_to_dtype_if_needed(model, model_dtype)

        # Step 9: Apply SageAttention post-load if requested.
        if attention_mode == "sage":
            if check_sage_attention_compatible():
                set_sage_attention(model)
            else:
                raise RuntimeError("Incompatible hardware/setup for SageAttention.")

        # Step 10: Eval mode.
        model.eval()

        logger.info(
            f"Successfully loaded external VibeVoice ASR model '{config_name}' "
            f"from {weight_path}"
        )
        _log_load_diagnostics(
            config_name=config_name,
            requested_attention_mode=requested_attention_mode,
            resolved_attention_mode=attention_mode,
            weight_family=weight_family,
            load_device=load_device,
        )

        # Step 11: Assemble the bundle (identity fields mirror the TTS bundle,
        # plan 2026-08-20 B1/P1).
        # NOTE (plan 2026-08-18, D7/RC-4): the state dict is intentionally NOT
        # included — the model owns the tensors after assign-loading, and
        # retaining a second ~model-size copy in the node output is dead RAM.
        return {
            "config": config,
            "processor": processor,
            "model": model,
            "model_name": config_name,
            "source_path": weight_path,
            "source_mtime_ns": source_mtime_ns,
            "source_size": source_size,
            "attention_mode": attention_mode,
            "use_llm_4bit": False,
            "dtype_str": dtype_str,
            "is_streaming": False,
            "is_asr": True,
            "weight_family": weight_family,
            "quant_stats": quant_stats,
        }

    except Exception as e:
        logger.error(
            f"Failed to load external VibeVoice ASR model '{config_name}' "
            f"from {weight_path}: {e}"
        )
        raise RuntimeError(
            f"Failed to load external VibeVoice ASR model '{config_name}': {e}"
        )
