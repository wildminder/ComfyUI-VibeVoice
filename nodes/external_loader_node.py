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
    EXTERNAL_CONFIG_OPTIONS,
)
from ..modules.attention_utils import get_available_attention_modes
from ..modules.dtype_utils import get_dtype_options, DTYPE_AUTO

logger = logging.getLogger(__name__)


def list_external_model_files() -> list:
    """Build the selectable weight-file list, including ``.gguf`` files.

    ComfyUI's ``get_filename_list("diffusion_models")`` filters by
    ``supported_pt_extensions``, which excludes ``.gguf``. This helper merges
    that list with ``.gguf`` files found in the same folders, plus the
    ``unet_gguf`` folder if it is registered (e.g. by ComfyUI-GGUF).

    Returns:
        Sorted, de-duplicated list of weight-file names (relative paths).
    """
    files = set()

    # Standard diffusion_models list (safetensors / bin / pt / ...).
    try:
        files.update(folder_paths.get_filename_list("diffusion_models"))
    except Exception:
        pass

    # .gguf files are excluded from supported_pt_extensions, so scan the
    # diffusion_models folders directly for them.
    try:
        for folder in folder_paths.get_folder_paths("diffusion_models"):
            found, _ = folder_paths.recursive_search(folder)
            for name in found:
                if name.lower().endswith(".gguf"):
                    files.add(name)
    except Exception:
        pass

    # ComfyUI-GGUF registers a dedicated "unet_gguf" folder; include it if present.
    try:
        files.update(folder_paths.get_filename_list("unet_gguf"))
    except Exception:
        pass

    return sorted(files)


def resolve_weight_path(model_file: str) -> str:
    """Resolve a selected weight-file name to an absolute path.

    Tries the ``diffusion_models`` folder first, then the ``unet_gguf`` folder
    (registered by ComfyUI-GGUF). Raises ``FileNotFoundError`` if the file is
    not found in either.

    Args:
        model_file: Relative weight-file name from the dropdown.

    Returns:
        Absolute path to the weight file.
    """
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

    CATEGORY = "audio/tts"

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
                    default="VibeVoice-1.5B",
                    tooltip=(
                        "Architecture config to use. A sidecar config next to the "
                        "weight file takes priority; this selects the packaged "
                        "default fallback when no sidecar is present."
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
    def execute(
        cls,
        model_file: str,
        config_name: str,
        attention_mode: str,
        quantize_llm_4bit: bool,
        dtype: str,
    ) -> io.NodeOutput:
        weight_path = resolve_weight_path(model_file)

        model_bundle = load_external_vibevoice_model(
            weight_path=weight_path,
            config_name=config_name,
            attention_mode=attention_mode,
            use_llm_4bit=quantize_llm_4bit,
            dtype_str=dtype,
        )

        return io.NodeOutput(model_bundle)
