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

The ``config_name`` dropdown selects the loading BRANCH (TTS vs ASR). Inside
the ASR branch the checkpoint's own ``model_type`` selects the model FAMILY,
because two incompatible ASR families are loadable there and the config JSON
is the only thing that can tell them apart:

    - ``"vibevoice_asr"`` → the transformers >= 5.3.0 classes
      (``AutoConfig`` / ``VibeVoiceAsrForConditionalGeneration``), which build
      the ``language_model.model.*`` tree the published ``VibeVoice-ASR-HF``
      checkpoints store. See :mod:`modules.asr_native`.
    - ``"vibevoice"`` → the vendored ``src/vibevoice`` classes
      (``VibeVoiceASRConfig`` / ``VibeVoiceASRForConditionalGeneration``),
      which build the ``model.language_model.*`` tree of the original
      ``microsoft/VibeVoice-ASR`` checkpoints.
"""

import os
import json
import struct
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

from .loader import VibeVoiceLoader, QUANT_STORAGE_DTYPES
from .base_loader import iter_safetensors_tensors, place_tensor_on_device
from .asr_native import (
    NATIVE_ASR_MODEL_TYPE,
    build_native_asr_processor,
    is_native_asr_config_path,
    load_native_asr_config,
    instantiate_native_asr_model,
)
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
from .diagnostics import diagnostics_enabled
from pathlib import Path

from .patcher import (
    VibeVoiceASRPatcher,
    VibeVoicePatcher,
    dynamic_vram_available,
    resolve_core_patcher_class,
    select_patcher_class,
)
from .memory_census import measured_load, report_census

if SAGE_ATTENTION_AVAILABLE:
    from ..src.vibevoice.modular.sage_attention_patch import set_sage_attention

logger = logging.getLogger(__name__)

_PACKAGED_CONFIG_FILES = {
    "VibeVoice-1.5B": "default_VibeVoice-1.5B_config.json",
    "VibeVoice-7B": "default_VibeVoice-Large_config.json",
    "VibeVoice-ASR": "default_VibeVoice-ASR_config.json",
}

AUTO_CONFIG_NAME = "Auto-detect"

EXTERNAL_CONFIG_OPTIONS = [
    AUTO_CONFIG_NAME,
    "VibeVoice-1.5B",
    "VibeVoice-7B",
    "VibeVoice-Realtime-0.5B",
    "VibeVoice-ASR",
]

_LEGACY_CONFIG_ALIASES = {
    "vibevoice-large": "VibeVoice-7B",
}


def normalize_config_name(config_name: str) -> str:
    """Map legacy/alias config_name values onto their canonical option."""
    if not config_name:
        return config_name
    if config_name in EXTERNAL_CONFIG_OPTIONS:
        return config_name
    return _LEGACY_CONFIG_ALIASES.get(config_name.lower(), config_name)


def resolve_auto_config_name(weight_path: str, gguf_reader=None, weights_fp=None) -> str:
    """Resolve ``AUTO_CONFIG_NAME`` to a concrete family from the weights."""
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
    """Reconcile the selected config against the weights' fingerprint."""
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

_RECONCILIATION_MEMO: dict = {}


def _reconciliation_memo_key(weight_path: str, selected_name: str):
    abspath = os.path.abspath(weight_path) if weight_path else ""
    mtime_ns = 0
    size = 0
    try:
        stat = os.stat(abspath)
        mtime_ns = stat.st_mtime_ns
        size = stat.st_size
    except OSError:
        pass
    return (abspath, mtime_ns, size, selected_name)


def reconciled_config_name(weight_path: str, selected_name: str) -> str:
    return _RECONCILIATION_MEMO.get(
        _reconciliation_memo_key(weight_path, selected_name), selected_name
    )


def remember_reconciled_config(
    weight_path: str, selected_name: str, effective_name: str
) -> None:
    if not effective_name:
        return
    _RECONCILIATION_MEMO[
        _reconciliation_memo_key(weight_path, selected_name)
    ] = effective_name


def clear_reconciliation_memo() -> None:
    _RECONCILIATION_MEMO.clear()


# ====================================================================
# Path resolution helpers
# ====================================================================

def _packaged_configs_dir() -> str:
    return os.path.join(
        os.path.dirname(__file__), "..", "src", "vibevoice", "configs"
    )


def _get_packaged_config_path(config_name: str) -> str:
    filename = _PACKAGED_CONFIG_FILES.get(config_name)
    if filename is None:
        return ""
    return os.path.normpath(os.path.join(_packaged_configs_dir(), filename))


def resolve_sidecar_config(weight_path: str, config_name: str) -> str:
    sidecar_path = weight_path + ".config.json"
    if os.path.exists(sidecar_path):
        logger.debug(f"Using sidecar config: {sidecar_path}")
        return sidecar_path

    dir_sidecar = os.path.join(os.path.dirname(weight_path), "config.json")
    if os.path.exists(dir_sidecar):
        logger.debug(f"Using directory sidecar config: {dir_sidecar}")
        return dir_sidecar

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
    return os.path.dirname(weight_path)


# ====================================================================
# ASR-specific helpers
# ====================================================================

ASR_CONFIG_NAMES = {"VibeVoice-ASR"}


def is_asr_config_name(config_name: str) -> bool:
    return config_name in ASR_CONFIG_NAMES


def _load_asr_config(config_path: str) -> "VibeVoiceASRConfig":
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"ASR config not found: {config_path}")
    if is_native_asr_config_path(config_path):
        return load_native_asr_config(config_path)
    return VibeVoiceASRConfig.from_pretrained(config_path)


def _load_asr_tokenizer(tokenizer_dir: str) -> "VibeVoiceASRTextTokenizerFast":
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
    if getattr(config, "model_type", None) == NATIVE_ASR_MODEL_TYPE:
        return instantiate_native_asr_model(
            config,
            attn_implementation=attn_implementation,
            final_load_dtype=final_load_dtype,
            use_meta=use_meta,
        )

    if hasattr(config, "decoder_config"):
        config.decoder_config._attn_implementation = attn_implementation

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
        dequantized = dequantize_reader_tensor(tensor)
        state_dict[tensor.name] = dequantized.to(device)

    logger.debug(f"Loaded {len(state_dict)} tensors from GGUF file")
    return state_dict


def _load_weight_state_dict(weight_path: str, device) -> dict:
    if weight_path.lower().endswith(".gguf"):
        return _load_gguf_state_dict(weight_path, device=device)
    return comfy.utils.load_torch_file(weight_path, device=device)


# ====================================================================
# Quant-resident loading (GGUF raw-block + ConvRot INT8)
# ====================================================================

def _open_gguf_reader(weight_path: str):
    try:
        import gguf  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Loading .gguf weights requires the 'gguf' Python package. "
            "Install it with: pip install gguf"
        ) from e
    from .gguf_quant import open_gguf_reader

    return open_gguf_reader(weight_path)


def _gguf_kquant_present(reader) -> bool:
    from .gguf_quant import _T

    kquants = {_T.Q4_K, _T.Q5_K, _T.Q6_K}
    return any(t.tensor_type in kquants for t in reader.tensors)


_QUANT_STORAGE_DTYPES = QUANT_STORAGE_DTYPES


def _plan_quantized_safetensors_load(quant_map: dict):
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
        return (
            (
                w.to(torch.float32).view(og, gs, ig, gs)
                * s.to(torch.float32).view(og, 1, ig, 1)
            ).reshape(w.shape).to(orig_dtype)
        )

    if not (s.dim() == 0 or (s.dim() == 2 and s.shape[1] == 1
                             and s.shape[0] in (w.shape[0], 1))):
        raise QuantTargetMismatch(
            f"Rowwise layer '{prefix}': scale shape {tuple(s.shape)} "
            f"is neither a scalar nor [out,1] (weight "
            f"{tuple(w.shape)})"
        )
    return (w.to(torch.float32) * s.to(torch.float32)).to(orig_dtype)


def _prepare_quantized_safetensors_load(state_dict: dict, quant_map: dict):
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
# Streaming safetensors load
# ====================================================================

_SAFETENSORS_DTYPE_TO_TORCH = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}


def read_safetensors_tensors_by_name(weight_path: str, names) -> dict:
    """Read the named tensors out of the header's byte ranges, one at a time.

    ``safe_open`` maps the entire file, and on Windows the first read through
    such a mapping commits roughly one file size as private, untouched memory
    that stays pinned for the lifetime of the returned tensor — charging a
    9.5 GB checkpoint to host RAM to look up a few 4-byte scales. The header
    already records each tensor's ``data_offsets``, so reading exactly those
    byte ranges costs only the bytes asked for and touches no mapping.
    """
    wanted = list(dict.fromkeys(names))
    if not wanted:
        return {}
    remaining = set(wanted)
    found = {}
    with open(weight_path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        for name, info in header.items():
            if name not in remaining:
                continue
            dtype = _SAFETENSORS_DTYPE_TO_TORCH.get(info.get("dtype"))
            if dtype is None:
                raise ValueError(
                    f"Checkpoint tensor '{name}' has unsupported safetensors "
                    f"dtype {info.get('dtype')!r}"
                )
            begin, end = info["data_offsets"]
            buf = bytearray(end - begin)
            f.seek(data_start + begin)
            view = memoryview(buf)
            while view:
                read = f.readinto(view)
                if not read:
                    raise ValueError(
                        f"Checkpoint tensor '{name}' is truncated: wanted "
                        f"{end - begin} bytes at offset {data_start + begin}, "
                        f"got {end - begin - len(view)}"
                    )
                view = view[read:]
            found[name] = torch.frombuffer(buf, dtype=dtype).reshape(
                tuple(info["shape"])
            )
            remaining.discard(name)
    return found


def _stream_apply_safetensors(
    model, weight_path: str, quant_map: dict, target_device=None,
):
    """Assign a quantized safetensors checkpoint per-tensor.

    Reads only the required scale tensors in pass 1, straight from their byte
    ranges, and places every tensor onto target_device as it is assigned.
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

    # Pass 1: read ONLY the required scale keys, straight from their byte
    # ranges. safe_open here would commit the whole file as private RAM.
    scales = {}
    if dequant_infos:
        scale_keys = {f"{prefix}.weight_scale" for prefix in dequant_infos}
        read = read_safetensors_tensors_by_name(weight_path, scale_keys)
        for prefix in dequant_infos:
            s_key = f"{prefix}.weight_scale"
            if s_key in read:
                scales[prefix] = read[s_key]
            else:
                raise QuantTargetMismatch(
                    f"Rowwise-quantized layer '{prefix}' is missing its "
                    f"'{s_key}' tensor in the checkpoint"
                )

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
            tensor = place_tensor_on_device(tensor, target_device)
            comfy.utils.set_attr_param(model, key, tensor)
            return True
        target_buf = buffers.get(key)
        if target_buf is not None:
            if tuple(tensor.shape) != tuple(target_buf.shape):
                raise _shape_mismatch_error(
                    model, [(key, tuple(tensor.shape), tuple(target_buf.shape))]
                )
            tensor = place_tensor_on_device(tensor, target_device)
            comfy.utils.set_attr_buffer(model, key, tensor)
            return True
        return False

    with measured_load("stream-apply-safetensors") as _rss:
        _rss.mark("stream-begin")
        assigned = set()
        for key, tensor in iter_safetensors_tensors(weight_path):
            if not assigned:
                _rss.mark("first-view")
            if key.endswith(meta_suffix):
                continue
            prefix, _, leaf = key.rpartition(".")
            if prefix in dequant_infos and leaf == "weight_scale":
                continue
            if prefix in dequant_infos and leaf == "weight":
                tensor = _dequantize_rowwise_weight(
                    prefix, dequant_infos[prefix], tensor, scales[prefix]
                )
            else:
                if tensor.dtype in _QUANT_STORAGE_DTYPES:
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
            if _assign(key, tensor):
                assigned.add(key)
            else:
                unexpected.append(key)

        _rss.mark("stream-end")
        report_census(model, phase=f"stream-end:{Path(weight_path).stem}")

    expected = set(model.state_dict().keys())
    missing_keys = [k for k in expected if k not in assigned]
    return VibeVoiceLoader._post_assign_fixups(
        model, missing_keys, unexpected
    )


def _config_ties_word_embeddings(config) -> bool:
    decoder_config = getattr(config, "decoder_config", None)
    for cfg in (decoder_config, config):
        if cfg is None:
            continue
        flag = getattr(cfg, "tie_word_embeddings", False)
        if isinstance(flag, bool) and flag:
            return True
    return False


def _assert_lm_head_not_tied(config, quant_map: dict) -> None:
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
    from .gguf_quant import FLOAT_GGML_TYPES

    if not _config_ties_word_embeddings(config):
        return
    for t in reader.tensors:
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


def _stream_apply_dense_safetensors(model, weight_path: str, target_device=None):
    """Assign a dense safetensors checkpoint per-tensor straight onto ``target_device``.

    This is the dense counterpart of :func:`_stream_apply_safetensors` and the
    single dense entry point for both the TTS and ASR routes: the same
    zero-copy iterator, the same per-tensor placement, the same post-assign
    fixups. Nothing ever materialises the whole checkpoint as a CPU model, so a
    BF16/FP16 file reaches VRAM the same way an FP8 one does.
    """
    from .loader import VibeVoiceLoader

    with measured_load("dense-stream-apply") as _rss:
        _rss.mark("stream-begin")
        VibeVoiceLoader._stream_apply_dense(
            model,
            iter_safetensors_tensors(weight_path),
            preserve_file_views=True,
            target_device=target_device,
        )
        _rss.mark("stream-end")
        report_census(model, phase=f"stream-end:{Path(weight_path).stem}")


def _install_gguf_weights(model, reader, target_device=None) -> dict:
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

    resident_plan = {}
    plan = []
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

    total_raw = 0
    for t, kind, target, _dst, _nbytes in plan:
        if kind != "resident" or target not in resident_set:
            continue
        module = resolve_module(model, target)
        raw = GGUFTensor.from_reader_tensor(t).raw
        # The raw blocks ARE the model's storage for a quant-resident Linear,
        # so they belong where the rest of the model belongs. Left on the host
        # they cost ~8 GB of private RAM and every forward takes the paged
        # `cast_bias_weight` path; in VRAM they cost the same bytes, load
        # flat, and the forward dequantizes in place. Dequantizing to float at
        # load instead would cost twice the VRAM and lose the point of a GGUF
        # checkpoint.
        module.set_raw_weight(place_tensor_on_device(raw, target_device))
        total_raw += module.weight.numel()

    def _dense_pairs():
        for t, kind, target, dst, _nbytes in plan:
            if kind != "dense":
                continue
            if dst is None:
                arr = np.ascontiguousarray(t.data)
                tensor = torch.from_numpy(
                    arr if arr.flags.writeable else arr.copy()
                ).clone()
                if t.data.dtype == np.uint8:
                    tensor = tensor.view(torch.bfloat16)
                shape = tuple(int(s) for s in reversed(t.shape))
                yield target, tensor.reshape(shape)
            else:
                yield target, dequantize_reader_tensor(t, dst)

    known_missing = {f"{p}.weight" for p in replaced}
    VibeVoiceLoader._stream_apply_dense(
        model, _dense_pairs(), known_missing=known_missing,
        target_device=target_device,
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

_SAFETENSORS_INT_DTYPES = {
    "I8", "U8", "I16", "U16", "I32", "U32", "I64", "U64", "I4", "U4",
}

_QUANT_SCALE_NAME_HINTS = ("scale", "zero_point", "qzero", "absmax")
_LOWBIT_FRACTION_THRESHOLD = 0.10


def _inspect_safetensors_quantization(weight_path: str):
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
    try:
        import gguf  # noqa: F401
        from gguf.constants import GGMLQuantizationType
    except ImportError:
        return None

    try:
        from .gguf_quant import open_gguf_reader

        reader = open_gguf_reader(weight_path)
    except Exception:
        return None

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
                f"NO dequantization scale/zero-point metadata. Use a proper "
                f"quantized or full-precision (BF16/FP16) checkpoint instead."
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
                f"I-quant precision. Consider a higher-quality quant "
                f"(Q4_K_M / Q5_K_M / Q8_0 / BF16)."
            )
        return


# ====================================================================
# In-memory state dict loading
# ====================================================================

def _load_state_dict_into_model_from_memory(
    model, state_dict: dict, preserve_file_views: bool = True, target_device=None,
):
    """Load an in-memory state dict into an already-instantiated model.

    Preserves zero-copy memory-mapped file views directly in parameters.
    When target_device is specified (CUDA), transfers parameters directly
    to GPU VRAM without keeping an intermediate copy in system RAM.
    """
    if not preserve_file_views:
        for key in list(state_dict.keys()):
            tensor = state_dict[key]
            if isinstance(tensor, torch.Tensor):
                state_dict[key] = tensor.clone()

    VibeVoiceLoader._apply_state_dict(model, state_dict)

    if target_device is not None and target_device.type == "cuda":
        model.to(target_device)

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
    if not diagnostics_enabled():
        return
    logger.info(
        "Load diagnostics: model='%s' family=%s requested_attention=%s "
        "resolved_attention=%s device=%s",
        config_name, weight_family, requested_attention_mode,
        resolved_attention_mode, load_device,
    )


def _reset_gguf_forward_counters() -> None:
    try:
        from .gguf_quant import reset_gguf_forward_counters

        reset_gguf_forward_counters()
    except Exception:
        pass


def load_external_vibevoice_model(
    weight_path: str,
    config_name: str,
    attention_mode: str = "eager",
    use_llm_4bit: bool = False,
    dtype_str: str = "auto",
    device=None,
) -> dict:
    """Load a VibeVoice model from an external weight file directly into VRAM."""
    config_name = normalize_config_name(config_name)
    _reset_gguf_forward_counters()

    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"External weight file not found: {weight_path}")

    try:
        _stat = os.stat(weight_path)
        source_mtime_ns = _stat.st_mtime_ns
        source_size = _stat.st_size
    except OSError:
        source_mtime_ns = 0
        source_size = 0

    warn_if_lowbit_quantization(weight_path)

    if is_asr_config_name(config_name):
        return load_external_vibevoice_asr_model(
            weight_path=weight_path,
            config_name=config_name,
            attention_mode=attention_mode,
            dtype_str=dtype_str,
            device=device,
        )

    requested_attention_mode = attention_mode
    attention_mode = resolve_attention_mode(attention_mode, use_llm_4bit)

    lower_path = weight_path.lower()
    is_gguf_file = lower_path.endswith(".gguf")
    gguf_reader = _open_gguf_reader(weight_path) if is_gguf_file else None
    convrot_quant_map = {}
    if gguf_reader is None and lower_path.endswith(".safetensors"):
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

    from .config_detect import fingerprint_weights

    weights_fp = fingerprint_weights(weight_path, gguf_reader=gguf_reader)

    if config_name == AUTO_CONFIG_NAME:
        config_name = resolve_auto_config_name(weight_path, weights_fp=weights_fp)

    config_path = resolve_sidecar_config(weight_path, config_name)
    config = VibeVoiceLoader._load_config(config_path, config_name)
    effective_name, config_changed = reconcile_config(config_name, config, weights_fp)
    if config_changed:
        config_name = effective_name
        config_path = resolve_sidecar_config(weight_path, config_name)
        config = VibeVoiceLoader._load_config(config_path, config_name)

    is_streaming = isinstance(config, VibeVoiceStreamingConfig)
    if is_streaming:
        logger.debug(f"External model '{config_name}' detected as streaming model")
        attention_mode = resolve_realtime_attention_mode(attention_mode)

    tokenizer_dir = resolve_sidecar_tokenizer_dir(weight_path)
    vibevoice_tokenizer = VibeVoiceLoader._load_tokenizer(tokenizer_dir, config_name)

    preprocessor_path = resolve_sidecar_preprocessor(weight_path)
    processor = VibeVoiceLoader._load_processor(
        vibevoice_tokenizer, preprocessor_path, is_streaming=is_streaming
    )

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

        model = VibeVoiceLoader._instantiate_model(
            config=config,
            is_streaming=is_streaming,
            attn_implementation=attn_implementation_for_load,
            final_load_dtype=final_load_dtype,
        )

        weight_family = "dense"
        quant_stats = {}
        # convrot_quant_map is only ever populated from a .safetensors scan, so
        # this is bool(convrot_quant_map). The batch fallback below is kept
        # only because its guard helpers are still exercised directly by
        # tests; every real load goes through the streaming assign.
        stream_quant_load = (
            bool(convrot_quant_map)
            and weight_path.lower().endswith(".safetensors")
        )

        if gguf_reader is not None:
            _assert_gguf_lm_head_not_tied(config, gguf_reader)
            quant_stats = _install_gguf_weights(
                model, gguf_reader, target_device=load_device
            )
            weight_family = "gguf_block"
            del gguf_reader
            gguf_reader = None
            gc.collect()
        elif convrot_quant_map:
            from .quant_common import replace_linears_for_quant

            _assert_lm_head_not_tied(config, convrot_quant_map)
            convrot_quant_map = _demote_nonlinear_fp8_residents(
                model, convrot_quant_map
            )
            if stream_quant_load:
                layer_plan, n_rowwise, n_fp8_resident = (
                    _plan_quantized_safetensors_load(convrot_quant_map)
                )
                replaced = replace_linears_for_quant(model, layer_plan)
                _stream_apply_safetensors(
                    model,
                    weight_path,
                    convrot_quant_map,
                    target_device=load_device,
                )
            else:
                with measured_load("dense-read-state-dict"):
                    state_dict = _load_weight_state_dict(weight_path, torch.device("cpu"))
                layer_plan, n_rowwise, n_fp8_resident = (
                    _prepare_quantized_safetensors_load(state_dict, convrot_quant_map)
                )
                replaced = replace_linears_for_quant(model, layer_plan)
                model = _load_state_dict_into_model_from_memory(
                    model,
                    state_dict,
                    preserve_file_views=True,
                    target_device=load_device,
                )
                del state_dict
                state_dict = None
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
            _stream_apply_dense_safetensors(
                model, weight_path, target_device=load_device
            )

        cast_model_to_dtype_if_needed(model, final_load_dtype)

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

        if attention_mode == "sage":
            if check_sage_attention_compatible():
                set_sage_attention(model)
            else:
                raise RuntimeError("Incompatible hardware/setup for SageAttention.")

        model.eval()
        setattr(model, "_llm_4bit", bool(quant_config))

        report_census(model, phase=f"pre-h2d:{config_name}")

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
            "dynamic_vram_route": False,
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
    config_name = normalize_config_name(config_name)
    _reset_gguf_forward_counters()

    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"External weight file not found: {weight_path}")

    try:
        _stat = os.stat(weight_path)
        source_mtime_ns = _stat.st_mtime_ns
        source_size = _stat.st_size
    except OSError:
        source_mtime_ns = 0
        source_size = 0

    warn_if_lowbit_quantization(weight_path)

    requested_attention_mode = attention_mode
    attention_mode = resolve_asr_attention_mode(
        resolve_attention_mode(attention_mode, quantize_4bit=False)
    )

    lower_path = weight_path.lower()
    is_gguf_file = lower_path.endswith(".gguf")
    gguf_reader = _open_gguf_reader(weight_path) if is_gguf_file else None
    convrot_quant_map = {}
    if gguf_reader is None and lower_path.endswith(".safetensors"):
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

    config_path = resolve_sidecar_config(weight_path, config_name)
    config = _load_asr_config(config_path)

    tokenizer_dir = resolve_sidecar_tokenizer_dir(weight_path)
    preprocessor_path = resolve_sidecar_preprocessor(weight_path)
    if is_native_asr_config_path(config_path):
        processor = build_native_asr_processor(tokenizer_dir, preprocessor_path)
    else:
        processor = _load_asr_processor(
            _load_asr_tokenizer(tokenizer_dir), preprocessor_path
        )

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

        model = _instantiate_asr_model(
            config=config,
            attn_implementation=attn_implementation_for_load,
            final_load_dtype=model_dtype,
        )

        weight_family = "dense"
        quant_stats = {}
        # convrot_quant_map is only ever populated from a .safetensors scan, so
        # this is bool(convrot_quant_map). The batch fallback below is kept
        # only because its guard helpers are still exercised directly by
        # tests; every real load goes through the streaming assign.
        stream_quant_load = (
            bool(convrot_quant_map)
            and weight_path.lower().endswith(".safetensors")
        )

        if gguf_reader is not None:
            _assert_gguf_lm_head_not_tied(config, gguf_reader)
            quant_stats = _install_gguf_weights(
                model, gguf_reader, target_device=load_device
            )
            weight_family = "gguf_block"
            del gguf_reader
            gguf_reader = None
            gc.collect()
        elif convrot_quant_map:
            from .quant_common import replace_linears_for_quant

            _assert_lm_head_not_tied(config, convrot_quant_map)
            convrot_quant_map = _demote_nonlinear_fp8_residents(
                model, convrot_quant_map
            )
            if stream_quant_load:
                layer_plan, n_rowwise, n_fp8_resident = (
                    _plan_quantized_safetensors_load(convrot_quant_map)
                )
                replaced = replace_linears_for_quant(model, layer_plan)
                _stream_apply_safetensors(
                    model,
                    weight_path,
                    convrot_quant_map,
                    target_device=load_device,
                )
            else:
                with measured_load("dense-read-state-dict"):
                    state_dict = _load_weight_state_dict(weight_path, torch.device("cpu"))
                layer_plan, n_rowwise, n_fp8_resident = (
                    _prepare_quantized_safetensors_load(state_dict, convrot_quant_map)
                )
                replaced = replace_linears_for_quant(model, layer_plan)
                model = _load_state_dict_into_model_from_memory(
                    model,
                    state_dict,
                    preserve_file_views=True,
                    target_device=load_device,
                )
                del state_dict
                state_dict = None
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
            _stream_apply_dense_safetensors(
                model, weight_path, target_device=load_device
            )

        cast_model_to_dtype_if_needed(model, model_dtype)

        if attention_mode == "sage":
            if check_sage_attention_compatible():
                set_sage_attention(model)
            else:
                raise RuntimeError("Incompatible hardware/setup for SageAttention.")

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
            "dynamic_vram_route": False,
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