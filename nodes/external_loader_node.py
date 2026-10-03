"""VibeVoice External Model Loader Node (V3 Schema).

Loads a VibeVoice model from a standalone weight file (safetensors / .bin /
.gguf) placed in ComfyUI's ``diffusion_models`` folder, binding the required
config / tokenizer / preprocessor JSONs via sidecar files or packaged defaults.

This node bypasses ComfyUI's ``model_detection.detect_unet_config()`` (which
does not recognize VibeVoice's state-dict keys) and outputs a
``VIBEVOICE_MODEL`` custom type consumed by the TTS / Realtime / ASR nodes via
their optional ``external_model`` input.
"""

import logging

import folder_paths
from comfy_api.latest import io

from ..modules.custom_types import VibeVoiceModel
from ..modules.external_loader import (
    load_external_vibevoice_model,
    AUTO_CONFIG_NAME,
    EXTERNAL_CONFIG_OPTIONS,
    is_asr_config_name,
    normalize_config_name,
    reconciled_config_name,
    remember_reconciled_config,
    resolve_auto_config_name,
)
from ..modules.attention_utils import (
    get_available_attention_modes,
    resolve_attention_mode,
    check_dtype_attention_compatible,
)
from ..modules.dtype_utils import get_dtype_options, DTYPE_AUTO
from ..modules.model_registry import (
    FAMILY_ASR,
    FAMILY_TTS,
    evict_if_changed,
    get_live_bundle,
    identity_for_external,
)
from ..modules.utils import VIBEVOICE_ASR_PATCHER_CACHE, VIBEVOICE_PATCHER_CACHE


def list_external_model_files() -> list:
    """Build the selectable weight-file list, including ``.gguf`` files."""
    files = set()

    try:
        files.update(folder_paths.get_filename_list("diffusion_models"))
    except Exception:
        pass

    try:
        for folder in folder_paths.get_folder_paths("diffusion_models"):
            found, _ = folder_paths.recursive_search(folder)
            for name in found:
                if name.lower().endswith(".gguf"):
                    files.add(name)
    except Exception:
        pass

    try:
        files.update(folder_paths.get_filename_list("unet_gguf"))
    except Exception:
        pass

    return sorted(files)


def resolve_weight_path(model_file: str) -> str:
    """Resolve a selected weight-file name to an absolute path."""
    for folder_name in ("diffusion_models", "unet_gguf"):
        try:
            return folder_paths.get_full_path_or_raise(folder_name, model_file)
        except Exception:
            continue
    raise FileNotFoundError(
        f"Weight file '{model_file}' not found in diffusion_models or unet_gguf folders."
    )


class VibeVoiceExternalLoaderNode(io.ComfyNode):
    """Load a VibeVoice model from an external weight file.

    The weight file is selected from ComfyUI's ``diffusion_models`` folder.
    The architecture config / tokenizer / preprocessor JSONs are bound via
    sidecar files next to the weight file, or fall back to packaged defaults
    selected by the ``config_name`` dropdown.
    """

    CATEGORY = "WMNodes/sound/tts"

    @classmethod
    def define_schema(cls) -> io.Schema:
        model_files = list_external_model_files()
        if not model_files:
            model_files = ["No model files found in diffusion_models"]

        config_options = list(EXTERNAL_CONFIG_OPTIONS)
        attention_modes = get_available_attention_modes()
        dtype_options = get_dtype_options()

        return io.Schema(
            node_id="VibeVoiceLoadExternalModel",
            display_name="Load VibeVoice Model",
            category=cls.CATEGORY,
            description=(
                "Load a VibeVoice model from an external weight file "
                "(safetensors/bin/gguf) in the diffusion_models folder. "
                "Bind sidecar config/tokenizer JSONs or use packaged defaults."
            ),
            inputs=[
                io.Combo.Input(
                    "model_file",
                    options=model_files,
                    default=model_files[0],
                    tooltip=(
                        "Weight file in the diffusion_models folder. Place the "
                        "VibeVoice .safetensors (or .bin/.gguf) file there."
                    ),
                ),
                io.Combo.Input(
                    "config_name",
                    options=config_options,
                    default=AUTO_CONFIG_NAME,
                    tooltip=(
                        "Architecture config to use. Auto-detect reads the "
                        "weight file's embedding fingerprint and selects the "
                        "matching family; an explicit selection that "
                        "contradicts the weights is auto-corrected with a "
                        "warning. A sidecar config next to the weight file "
                        "takes priority over the packaged default fallback."
                    ),
                ),
                io.Combo.Input(
                    "attention_mode",
                    options=attention_modes,
                    default="sdpa",
                    tooltip="Attention implementation: Eager (safest), SDPA (balanced), Flash Attention 2 (fastest), Sage (quantized)",
                ),
                io.Boolean.Input(
                    "quantize_llm_4bit",
                    default=False,
                    label_on="Q4 (LLM only)",
                    label_off="Full precision",
                    tooltip="Quantize the Qwen2.5 LLM to 4-bit NF4 via bitsandbytes.",
                ),
                io.Combo.Input(
                    "dtype",
                    options=dtype_options,
                    default=DTYPE_AUTO,
                    tooltip="Data type for model precision. 'auto' selects optimal type for device.",
                ),
            ],
            outputs=[
                VibeVoiceModel.Output(display_name="VibeVoice Model"),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        """Accept current options AND removed legacy aliases."""
        cfg = kwargs.get("config_name")
        if cfg not in EXTERNAL_CONFIG_OPTIONS and \
                normalize_config_name(cfg) not in EXTERNAL_CONFIG_OPTIONS:
            return (
                f"config_name '{cfg}' is not valid. "
                f"Choose one of: {', '.join(EXTERNAL_CONFIG_OPTIONS)}"
            )

        message = check_dtype_attention_compatible(
            kwargs.get("dtype"),
            kwargs.get("attention_mode"),
            bool(kwargs.get("quantize_llm_4bit")),
        )
        if message is not None:
            return message
        return True

    @classmethod
    def execute(
        cls,
        model_file: str,
        config_name: str,
        attention_mode: str,
        quantize_llm_4bit: bool,
        dtype: str,
    ) -> io.NodeOutput:
        config_name = normalize_config_name(config_name)

        weight_path = resolve_weight_path(model_file)

        if config_name == AUTO_CONFIG_NAME:
            config_name = resolve_auto_config_name(weight_path)

        requested_config_name = config_name

        is_asr = is_asr_config_name(config_name)
        family = FAMILY_ASR if is_asr else FAMILY_TTS
        use_llm_4bit = False if is_asr else bool(quantize_llm_4bit)
        resolved_attn = resolve_attention_mode(attention_mode, use_llm_4bit)
        config_name = reconciled_config_name(
            weight_path, requested_config_name
        )

        request_key = identity_for_external(
            weight_path,
            config_name,
            resolved_attn,
            use_llm_4bit=use_llm_4bit,
            dtype_str=dtype,
            prefix="asr_external" if is_asr else "external",
        )
        evict_if_changed(
            family,
            request_key,
            (VIBEVOICE_ASR_PATCHER_CACHE if is_asr else VIBEVOICE_PATCHER_CACHE,),
        )

        cached_bundle = get_live_bundle(request_key)
        if cached_bundle is not None:
            logging.debug(
                f"[VibeVoice TTS] Reusing the resident model for {request_key!r} "
                f"(identical re-execution; skipping the rebuild)"
            )
            return io.NodeOutput(cached_bundle)

        model_bundle = load_external_vibevoice_model(
            weight_path=weight_path,
            config_name=config_name,
            attention_mode=attention_mode,
            use_llm_4bit=quantize_llm_4bit,
            dtype_str=dtype,
        )

        try:
            remember_reconciled_config(
                weight_path, requested_config_name, model_bundle["model_name"]
            )
        except Exception as e:
            logging.debug(f"[VibeVoice TTS] Could not memoize reconciled config name: {e}")

        return io.NodeOutput(model_bundle)