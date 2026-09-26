"""Deprecated VibeVoice Realtime TTS node - forwarding compatibility shim.

The canonical implementation is :class:`~nodes.tts_node.VibeVoiceTTSNode`. This
class exists only so saved workflows that reference the old ``VibeVoiceRealtime``
node ID keep loading and running. It contains no model loading, preprocessing,
or generation logic; every call is delegated to the canonical node.

The class is targeted for removal in the next major release.
"""

import logging
from typing import Optional

from comfy_api.latest import io

from ..modules.attention_utils import get_available_attention_modes
from ..modules.device_utils import get_device_options
from ..modules.dtype_utils import get_dtype_options, DTYPE_AUTO
from ..modules.custom_types import VibeVoiceModel
from ..modules.voice_presets import PRESET_NONE, list_voice_presets
from .tts_node import VibeVoiceTTSNode, _EXTERNAL_UNSET

logger = logging.getLogger(__name__)

_DEPRECATION_LOGGED = False


def _log_deprecation_once() -> None:
    """Emit one process-level deprecation warning for the legacy node."""
    global _DEPRECATION_LOGGED
    if _DEPRECATION_LOGGED:
        return
    _DEPRECATION_LOGGED = True
    logger.warning(
        "The 'VibeVoiceRealtime' node is deprecated and forwards to "
        "'VibeVoice TTS'. It will be removed in the next major release; "
        "please rebuild the workflow with the canonical VibeVoice TTS node."
    )


class VibeVoiceRealtimeNode(io.ComfyNode):
    """Deprecated delegate for the canonical ``VibeVoice TTS`` node.

    The old input prefix, including the legacy no-op ``stream`` widget, is
    preserved so saved workflow widget values keep their positions. The
    ``max_new_tokens`` and ``voice_preset`` controls are appended at the end.
    """

    CATEGORY = "audio/tts"

    @classmethod
    def define_schema(cls) -> io.Schema:
        # Only expose streaming TTS models here (non-streaming live in the
        # canonical VibeVoice TTS node).
        from ..modules.model_info import get_streaming_tts_models

        model_names = list(get_streaming_tts_models().keys())
        if not model_names:
            model_names.append("No streaming models found in models/tts/VibeVoice")

        try:
            voice_preset_options = [PRESET_NONE, *list_voice_presets().keys()]
        except Exception as exc:  # Asset discovery must not break schema creation.
            logger.warning("Could not discover realtime voice presets: %s", exc)
            voice_preset_options = [PRESET_NONE]

        available_devices = get_device_options()
        default_device = available_devices[0]

        dtype_options = get_dtype_options()
        attention_modes = get_available_attention_modes()

        return io.Schema(
            node_id="VibeVoiceRealtime",
            display_name="VibeVoice Realtime TTS (deprecated - use VibeVoice TTS)",
            category=cls.CATEGORY,
            description=(
                "Deprecated compatibility node. Use the canonical 'VibeVoice TTS' "
                "node with a realtime model and an official cached .pt voice preset."
            ),
            is_deprecated=True,
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=model_names,
                    default=model_names[0],
                    tooltip="Select the VibeVoice streaming model to use. Streaming models only.",
                ),
                io.String.Input(
                    "text",
                    multiline=True,
                    default=(
                        "[1] Hello, this is a cloned voice.\n"
                        "[2] And this is a generated voice, how cool is that?"
                    ),
                    tooltip=(
                        "The script for generation. Use '[1]' or 'Speaker 1:' for speakers. "
                        "Each speaker must be anchored to at least one reference voice; speakers "
                        "without their own reference are cloned from the provided reference(s)."
                    ),
                ),
                io.Boolean.Input(
                    "quantize_llm_4bit",
                    default=False,
                    label_on="Q4 (LLM only)",
                    label_off="Full precision",
                    tooltip="Quantize the Qwen2.5 LLM to 4-bit NF4 via bitsandbytes. Diffusion head stays BF16/FP32.",
                ),
                io.Combo.Input(
                    "attention_mode",
                    options=attention_modes,
                    default="sdpa",
                    tooltip="Attention implementation: Eager (safest), SDPA (balanced), Flash Attention 2 (fastest), Sage (quantized)",
                ),
                io.Float.Input(
                    "cfg_scale",
                    default=1.3,
                    min=0.1,
                    max=50.0,
                    step=0.05,
                    tooltip="Classifier-Free Guidance scale for speech diffusion. Recommended: 1.3",
                ),
                io.Int.Input(
                    "inference_steps",
                    default=10,
                    min=1,
                    max=500,
                    tooltip="Number of diffusion steps for audio generation. Recommended: 10",
                ),
                io.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip="Seed for reproducibility. Set to 0 for a random seed on each run.",
                ),
                io.Boolean.Input(
                    "do_sample",
                    default=True,
                    label_on="Enabled (Sampling)",
                    label_off="Disabled (Greedy)",
                    tooltip="Enable sampling methods for more varied output. Disable for deterministic decoding.",
                ),
                io.Float.Input(
                    "temperature",
                    default=0.95,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="Controls randomness. Active only if 'do_sample' is enabled.",
                ),
                io.Float.Input(
                    "top_p",
                    default=0.95,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Nucleus sampling (Top-P). Active only if 'do_sample' is enabled.",
                ),
                io.Int.Input(
                    "top_k",
                    default=0,
                    min=0,
                    max=500,
                    step=1,
                    tooltip="Top-K sampling. Set to 0 to disable. Active only if 'do_sample' is enabled.",
                ),
                io.Boolean.Input(
                    "stream",
                    default=False,
                    label_on="Streaming",
                    label_off="Full generation",
                    tooltip="Deprecated no-op kept for saved-workflow compatibility. The node always returns a completed AUDIO object.",
                ),
                io.Boolean.Input(
                    "force_offload",
                    default=False,
                    label_on="Force Offload",
                    label_off="Keep in VRAM",
                    tooltip="Force model to be offloaded from VRAM after generation. Useful to free memory between generations.",
                ),
                io.Combo.Input(
                    "device",
                    options=available_devices,
                    default=default_device,
                    tooltip="Device to run inference on. 'auto' follows ComfyUI's default compute device.",
                ),
                io.Combo.Input(
                    "dtype",
                    options=dtype_options,
                    default=DTYPE_AUTO,
                    tooltip="Data type for model precision. 'auto' selects optimal type for device.",
                ),
                # Optional external model input
                VibeVoiceModel.Input(
                    "external_model",
                    optional=True,
                    tooltip=(
                        "Optional externally-loaded VibeVoice streaming model "
                        "(from the 'Load VibeVoice Model' node). When connected, "
                        "this overrides the model_name dropdown."
                    ),
                ),
                io.Audio.Input("speaker_1_voice", optional=True, tooltip="Reference audio for 'Speaker 1' or '[1]' in the script."),
                io.Audio.Input("speaker_2_voice", optional=True, tooltip="Reference audio for 'Speaker 2' or '[2]' in the script."),
                io.Audio.Input("speaker_3_voice", optional=True, tooltip="Reference audio for 'Speaker 3' or '[3]' in the script."),
                io.Audio.Input("speaker_4_voice", optional=True, tooltip="Reference audio for 'Speaker 4' or '[4]' in the script."),
                io.Int.Input(
                    "max_new_tokens",
                    default=0,
                    min=0,
                    max=8192,
                    step=1,
                    tooltip="Maximum generated realtime sequence length. 0 = model default.",
                ),
                io.Combo.Input(
                    "voice_preset",
                    options=voice_preset_options,
                    default=PRESET_NONE,
                    tooltip="Official cached .pt voice prompt for realtime generation.",
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="Audio"),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        model_name: Optional[str] = None,
        voice_preset: str = PRESET_NONE,
        external_model=_EXTERNAL_UNSET,
    ) -> bool | str:
        """Delegate validation to the canonical VibeVoice TTS node.

        The signature is as narrow as the canonical node's: ComfyUI core
        splats only the declared inputs into the validator, so the legacy
        ``stream`` widget can never reach it and the error fan-out is bounded
        to the inspected inputs.
        """
        forwarded: dict = {"model_name": model_name}
        if voice_preset != PRESET_NONE:
            forwarded["voice_preset"] = voice_preset
        if external_model is not _EXTERNAL_UNSET:
            forwarded["external_model"] = external_model
        result = VibeVoiceTTSNode.validate_inputs(**forwarded)
        if result is True and external_model is _EXTERNAL_UNSET:
            from ..modules.model_info import is_model_type

            if model_name and not is_model_type(model_name, "streaming_tts"):
                logger.warning(
                    "The deprecated 'VibeVoiceRealtime' node received the "
                    "standard model '%s'. It will be generated by the canonical "
                    "standard TTS path; rebuild the workflow with the "
                    "'VibeVoice TTS' node.",
                    model_name,
                )
        return result

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        """Strip the legacy ``stream`` key and delegate to the canonical node."""
        _log_deprecation_once()
        kwargs.pop("stream", None)
        return VibeVoiceTTSNode.execute(**kwargs)


__all__ = ["VibeVoiceRealtimeNode"]
