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
from .dtype_utils import resolve_dtype

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
):
    """Instantiate a :class:`VibeVoiceASRForConditionalGeneration` directly.

    ASR counterpart of :meth:`VibeVoiceLoader._instantiate_model`. Instantiates
    the model class directly (no ``from_pretrained`` meta-device context) so the
    state dict can be loaded in-memory afterwards.

    Args:
        config: VibeVoiceASRConfig instance.
        attn_implementation: Attention implementation string.
        final_load_dtype: torch.dtype for the model.

    Returns:
        Model instance (not yet loaded with weights).
    """
    # Set attention implementation on the decoder config
    if hasattr(config, "decoder_config"):
        config.decoder_config._attn_implementation = attn_implementation

    # Set dtype on config
    config.torch_dtype = final_load_dtype
    if hasattr(config, "decoder_config"):
        config.decoder_config.torch_dtype = final_load_dtype

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
# In-memory state dict loading
# ====================================================================

def _load_state_dict_into_model_from_memory(model, state_dict: dict):
    """Load an in-memory state dict into an already-instantiated model.

    Mirrors :meth:`VibeVoiceLoader._load_state_dict_into_model` but accepts a
    state dict that is already in memory (loaded via
    ``comfy.utils.load_torch_file``) instead of resolving a checkpoint path.

    Args:
        model: Instantiated model (weights not yet loaded).
        state_dict: State dict mapping (on CPU).

    Returns:
        The model with the state dict loaded (still on CPU).
    """
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        logger.warning(f"Missing keys when loading external state dict: {len(missing_keys)} keys")
        if len(missing_keys) < 20:
            logger.warning(f"Missing keys: {missing_keys}")
        else:
            logger.warning(f"First 10 missing keys: {missing_keys[:10]}")

    if unexpected_keys:
        logger.warning(f"Unexpected keys when loading external state dict: {len(unexpected_keys)} keys")
        if len(unexpected_keys) < 20:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
        else:
            logger.warning(f"First 10 unexpected keys: {unexpected_keys[:10]}")

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

    # Step 1: Load the state dict onto CPU (safetensors/bin via ComfyUI's
    # loader, .gguf via the gguf package). Always CPU — the patcher owns the
    # single H2D transfer.
    cpu_device = torch.device("cpu")
    logger.info(f"Loading external VibeVoice weights from: {weight_path}")
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

        # Step 8: Load the in-memory state dict into the model.
        model = _load_state_dict_into_model_from_memory(model, state_dict)

        # Step 9: Apply the final dtype ON CPU (single H2D owned by patcher).
        model = model.to(dtype=final_load_dtype)

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
        return {
            "state_dict": state_dict,
            "config": config,
            "processor": processor,
            "model": model,
            "model_name": config_name,
            "source_path": weight_path,
            "is_streaming": is_streaming,
            "is_asr": False,
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

    # Resolve attention mode with fallback logic (no 4-bit for ASR).
    attention_mode = resolve_attention_mode(attention_mode, quantize_4bit=False)

    # Step 1: Load the state dict onto CPU (safetensors/bin via ComfyUI's
    # loader, .gguf via the gguf package). Always CPU — the patcher owns the
    # single H2D transfer.
    cpu_device = torch.device("cpu")
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

        # Step 7: Load the in-memory state dict into the model.
        model = _load_state_dict_into_model_from_memory(model, state_dict)

        # Step 8: Apply the final dtype ON CPU (single H2D owned by patcher).
        model = model.to(dtype=model_dtype)

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

        # Step 11: Assemble the bundle.
        return {
            "state_dict": state_dict,
            "config": config,
            "processor": processor,
            "model": model,
            "model_name": config_name,
            "source_path": weight_path,
            "is_streaming": False,
            "is_asr": True,
        }

    except Exception as e:
        logger.error(
            f"Failed to load external VibeVoice ASR model '{config_name}' "
            f"from {weight_path}: {e}"
        )
        raise RuntimeError(
            f"Failed to load external VibeVoice ASR model '{config_name}': {e}"
        )
