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
from .attention_utils import (
    SAGE_ATTENTION_AVAILABLE,
    resolve_attention_mode,
    get_attn_implementation_for_load,
    check_sage_attention_compatible,
)
from .dtype_utils import resolve_dtype, cast_model_to_dtype_if_needed
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
    "VibeVoice-Large": "default_VibeVoice-Large_config.json",
}

# All config_name values accepted by the loader node dropdown.
EXTERNAL_CONFIG_OPTIONS = [
    "VibeVoice-1.5B",
    "VibeVoice-Large",
    "VibeVoice-Realtime-0.5B",
    "VibeVoice-ASR",
]


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
        logger.info(f"Using sidecar config: {sidecar_path}")
        return sidecar_path

    # 2. Sidecar: config.json in the same directory
    dir_sidecar = os.path.join(os.path.dirname(weight_path), "config.json")
    if os.path.exists(dir_sidecar):
        logger.info(f"Using directory sidecar config: {dir_sidecar}")
        return dir_sidecar

    # 3. Packaged default based on config_name
    packaged = _get_packaged_config_path(config_name)
    if packaged and os.path.exists(packaged):
        logger.info(f"Using packaged default config for '{config_name}': {packaged}")
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
        logger.info(f"Using sidecar preprocessor config: {sidecar_path}")
        return sidecar_path

    dir_sidecar = os.path.join(os.path.dirname(weight_path), "preprocessor_config.json")
    if os.path.exists(dir_sidecar):
        logger.info(f"Using directory sidecar preprocessor config: {dir_sidecar}")
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
            logger.info(
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

    # Set dtype on config
    config.torch_dtype = final_load_dtype
    if hasattr(config, "decoder_config"):
        config.decoder_config.torch_dtype = final_load_dtype

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

    logger.info(f"Loading GGUF state dict from: {weight_path}")
    reader = gguf.GGUFReader(weight_path)

    state_dict = {}
    for tensor in reader.tensors:
        # gguf.dequantize returns a correctly-shaped numpy array for all
        # quantization types (F32/F16 pass through unchanged). The array may be
        # read-only (mmap-backed), so copy it to make it writable for torch.
        dequantized = gguf.dequantize(tensor.data, tensor.tensor_type)
        state_dict[tensor.name] = torch.from_numpy(dequantized.copy()).to(device)

    logger.info(f"Loaded {len(state_dict)} tensors from GGUF file")
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
    """Open a ``.gguf`` file with :class:`gguf.GGUFReader`.

    Raises:
        RuntimeError: If the ``gguf`` package is not installed.
    """
    try:
        import gguf
    except ImportError as e:
        raise RuntimeError(
            "Loading .gguf weights requires the 'gguf' Python package. "
            "Install it with: pip install gguf"
        ) from e
    return gguf.GGUFReader(weight_path)


def _gguf_kquant_present(reader) -> bool:
    """True when any tensor uses a K-quant format (Q4_K/Q5_K/Q6_K)."""
    from .gguf_quant import _T

    kquants = {_T.Q4_K, _T.Q5_K, _T.Q6_K}
    return any(t.tensor_type in kquants for t in reader.tensors)


# Quant-storage dtypes that must NEVER reach the dense float loader.
_QUANT_STORAGE_DTYPES = frozenset({
    torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2,
})


def _prepare_quantized_safetensors_load(state_dict: dict, quant_map: dict):
    """Split a scanned comfy_quant map into execution strategies (in place).

    - ``convrot=True`` layers  -> module-replacement plan (int8 resident +
      comfy-kitchen kernels); their int8/scale tensors stay in the state dict
      for assign.
    - ``convrot=False`` layers -> dequant-at-load: ``weight = q * per-row
      scale`` materialized back to the declared orig dtype inside
      ``state_dict``; scale + metadata keys removed.

    Returns:
        ``(layer_plan, n_rowwise)``.

    Raises:
        QuantTargetMismatch: On missing/malformed rowwise tensor pairs or an
            undeclared orig dtype.
    """
    from .convrot_quant import (
        QUANT_META_SUFFIX,
        make_convrot_linear,
        resolve_orig_dtype,
    )
    from .quant_common import QuantTargetMismatch

    layer_plan = {}
    n_rowwise = 0
    for prefix, info in quant_map.items():
        if info.convrot:
            layer_plan[prefix] = make_convrot_linear(info)
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
            state_dict[w_key] = (
                (
                    w.to(torch.float32).view(og, gs, ig, gs)
                    * s.to(torch.float32).view(og, 1, ig, 1)
                ).reshape(w.shape).to(orig_dtype)
            )
        else:
            # SCALAR (per-tensor, fp8) or PER-ROW [out, 1] scales; both
            # broadcast.
            if not (s.dim() == 0 or (s.dim() == 2 and s.shape[1] == 1
                                     and s.shape[0] in (w.shape[0], 1))):
                raise QuantTargetMismatch(
                    f"Rowwise layer '{prefix}': scale shape {tuple(s.shape)} "
                    f"is neither a scalar nor [out,1] (weight "
                    f"{tuple(w.shape)})"
                )
            # fp32 intermediate: exact rescale before the final cast.
            state_dict[w_key] = (
                (w.to(torch.float32) * s.to(torch.float32)).to(orig_dtype)
            )
        state_dict.pop(s_key, None)
        n_rowwise += 1

    for prefix in quant_map:
        state_dict.pop(f"{prefix}.{QUANT_META_SUFFIX}", None)
    return layer_plan, n_rowwise


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
    1. Map every reader tensor onto the model tree (HF pass-through or
       llamacpp naming).
    2. Quantized tensors whose target is an ``nn.Linear`` become RESIDENTS:
       the Linear is swapped for :class:`~modules.gguf_quant.GGUFLinear` and
       the RAW BLOCK BYTES are installed as its uint8 parameter (the one
       unavoidable copy — it IS the residency).
    3. Float tensors (F32/F16/BF16) take zero-copy views at their NATIVE
       dtype into a filtered dense state dict applied through
       :meth:`VibeVoiceLoader._apply_state_dict` (assign semantics, re-tie,
       sentinel/RoPE fixes preserved).

    Peak RAM ≈ raw file size instead of ~2x full-float size.

    Returns:
        Stats dict ``{"n_resident_layers", "raw_bytes", "weight_family"}``.
    """
    import numpy as np

    from .gguf_quant import (
        FLOAT_GGML_TYPES,
        GGUFTensor,
        SUPPORTED_GGML_TYPES,
        UnsupportedGGMLType,
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

    resident_plan = {}
    resident_tensors = {}
    dense_state = {}
    unsupported = []

    for t in tensors:
        target_key = mapping[t.name]
        logical_shape = tuple(int(s) for s in reversed(t.shape))
        tt = t.tensor_type

        if tt in FLOAT_GGML_TYPES:
            data = t.data
            if data.dtype == np.float32:
                tensor = torch.from_numpy(data)
            elif data.dtype == np.float16:
                tensor = torch.from_numpy(data)
            else:  # BF16 arrives as raw uint8 bytes
                tensor = torch.from_numpy(data).view(torch.bfloat16)
            dense_state[target_key] = tensor.reshape(logical_shape)
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
        if target_module is None or not isinstance(target_module, torch.nn.Linear):
            kind = type(target_module).__name__ if target_module is not None else "missing"
            raise QuantTargetMismatch(
                f"GGUF-quantized tensor '{t.name}' maps to '{module_path}' "
                f"({kind}), expected an nn.Linear. The sidecar config may not "
                f"match this checkpoint."
            )
        expected_shape = tuple(target_module.weight.shape)
        if logical_shape != expected_shape:
            raise QuantTargetMismatch(
                f"GGUF tensor '{t.name}' shape {logical_shape} disagrees with "
                f"model {type(target_module).__name__} shape {expected_shape}"
            )
        # Copy the raw block bytes off the mmap now (this copy IS the final
        # residency); float materialization never happens.
        resident_tensors[module_path] = GGUFTensor.from_reader_tensor(t)
        resident_plan[module_path] = gguf_linear_factory(tt)

    if unsupported:
        name, tt = unsupported[0]
        raise UnsupportedGGMLType(tt, f"{name}" + (
            f" (+{len(unsupported) - 1} more)" if len(unsupported) > 1 else ""
        ))

    replaced = replace_linears_for_quant(model, resident_plan)
    total_raw = 0
    for module_path in replaced:
        module = resolve_module(model, module_path)
        gtensor = resident_tensors[module_path]
        module.set_raw_weight(gtensor.raw)
        total_raw += gtensor.raw.numel()
        resident_tensors[module_path] = None

    known_missing = {f"{p}.weight" for p in replaced}
    VibeVoiceLoader._apply_state_dict(model, dense_state, known_missing=known_missing)

    return {
        "n_resident_layers": len(replaced),
        "raw_bytes": total_raw,
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
        import gguf
        from gguf.constants import GGMLQuantizationType
    except ImportError:
        return None

    try:
        reader = gguf.GGUFReader(weight_path)
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
    VibeVoiceLoader._apply_state_dict(model, state_dict)
    return model


# ====================================================================
# Core external loading function
# ====================================================================

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
            ``EXTERNAL_CONFIG_OPTIONS``.
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
        RuntimeError: If model instantiation or loading fails.
    """
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

    # Resolve attention mode with fallback logic (same as standard loader)
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

    cpu_device = torch.device("cpu")
    state_dict = None
    if gguf_reader is not None:
        logger.info(
            f"Opening external VibeVoice GGUF weights (raw-block residency): "
            f"{weight_path}"
        )
    else:
        logger.info(f"Loading external VibeVoice weights from: {weight_path}")
        # Step 1: Load the state dict onto CPU (safetensors/bin via ComfyUI's
        # loader). Always CPU — the patcher owns the single H2D transfer.
        state_dict = _load_weight_state_dict(weight_path, cpu_device)

    # Step 2: Resolve and load the architecture config.
    config_path = resolve_sidecar_config(weight_path, config_name)
    config = VibeVoiceLoader._load_config(config_path, config_name)

    # Step 3: Detect streaming from the loaded config.
    is_streaming = isinstance(config, VibeVoiceStreamingConfig)
    if is_streaming:
        logger.info(f"External model '{config_name}' detected as streaming model")

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
        logger.info(
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
            quant_stats = _install_gguf_weights(model, gguf_reader)
            weight_family = "gguf_block"
            del gguf_reader
            gguf_reader = None
        elif convrot_quant_map:
            from .quant_common import replace_linears_for_quant

            layer_plan, n_rowwise = _prepare_quantized_safetensors_load(
                state_dict, convrot_quant_map
            )
            replaced = replace_linears_for_quant(model, layer_plan)
            model = _load_state_dict_into_model_from_memory(model, state_dict)
            del state_dict
            state_dict = None
            weight_family = "convrot_int8"
            quant_stats = {
                "n_resident_layers": len(replaced),
                "n_rowwise_layers": n_rowwise,
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

    # Resolve attention mode with fallback logic (no 4-bit for ASR).
    attention_mode = resolve_attention_mode(attention_mode, quantize_4bit=False)

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
    cpu_device = torch.device("cpu")
    state_dict = None
    if gguf_reader is not None:
        logger.info(
            f"Opening external VibeVoice ASR GGUF weights (raw-block "
            f"residency): {weight_path}"
        )
    else:
        logger.info(f"Loading external VibeVoice ASR weights from: {weight_path}")
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
        logger.info(
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
            quant_stats = _install_gguf_weights(model, gguf_reader)
            weight_family = "gguf_block"
            del gguf_reader
            gguf_reader = None
        elif convrot_quant_map:
            from .quant_common import replace_linears_for_quant

            layer_plan, n_rowwise = _prepare_quantized_safetensors_load(
                state_dict, convrot_quant_map
            )
            replaced = replace_linears_for_quant(model, layer_plan)
            model = _load_state_dict_into_model_from_memory(model, state_dict)
            del state_dict
            state_dict = None
            weight_family = "convrot_int8"
            quant_stats = {
                "n_resident_layers": len(replaced),
                "n_rowwise_layers": n_rowwise,
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
