"""VibeVoice ASR Node - Speech-to-Text transcription (V3 Schema).

This node provides automatic speech recognition using the VibeVoice ASR model.
Supports speaker diarization, timestamps, hotwords, and 50+ languages.
"""

import json
import logging
from typing import Optional

import comfy.model_management as model_management
from comfy_api.latest import io

from ..modules.model_info import (
    AVAILABLE_VIBEVOICE_MODELS,
    get_asr_models,
    is_model_type,
    MODEL_CONFIGS,
)
from ..modules.asr_generation import (
    load_asr_model_patched,
    load_asr_from_external,
    transcribe_audio,
    force_offload_asr_model,
)
from ..modules.custom_types import VibeVoiceModel
from ..modules.device_utils import get_available_devices
from ..modules.dtype_utils import get_dtype_options, DTYPE_AUTO
from ..modules.attention_utils import get_available_attention_modes

logger = logging.getLogger(__name__)


class VibeVoiceASRNode(io.ComfyNode):
    """VibeVoice ASR node for speech-to-text transcription.

    Features:
    - Transcribes audio to text with speaker diarization
    - Timestamps for each speaker segment
    - Custom hotwords/context for improved accuracy
    - Supports 50+ languages
    - Handles up to 60 minutes of audio in a single pass
    """

    CATEGORY = "audio/asr"

    @classmethod
    def define_schema(cls) -> io.Schema:
        # Get ASR models
        asr_models = list(get_asr_models().keys())
        if not asr_models:
            # Fallback: show all models if no ASR-specific ones found
            asr_models = [name for name in AVAILABLE_VIBEVOICE_MODELS.keys()]
        if not asr_models:
            asr_models.append("No ASR models found")

        available_devices = get_available_devices()
        default_device = available_devices[0]
        dtype_options = get_dtype_options()
        attention_modes = get_available_attention_modes()

        return io.Schema(
            node_id="VibeVoiceASR",
            display_name="VibeVoice ASR",
            category=cls.CATEGORY,
            description=(
                "Transcribe audio to text using VibeVoice ASR. "
                "Supports speaker diarization, timestamps, hotwords, and 50+ languages. "
                "Handles up to 60 minutes of audio in a single pass."
            ),
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=asr_models,
                    default=asr_models[0],
                    tooltip="Select the VibeVoice ASR model to use.",
                ),
                io.Audio.Input(
                    "audio",
                    tooltip="Audio to transcribe. Supports up to 60 minutes of audio.",
                ),
                io.String.Input(
                    "context_info",
                    default="",
                    tooltip="Optional hotwords or context info to improve transcription accuracy (e.g., 'Tea Brew, Aiden Host').",
                ),
                io.Int.Input(
                    "max_new_tokens",
                    default=32768,
                    min=256,
                    max=131072,
                    tooltip="Maximum number of tokens to generate. Increase for longer audio.",
                ),
                io.Float.Input(
                    "temperature",
                    default=0.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="Temperature for sampling. 0 = greedy (deterministic).",
                ),
                io.Float.Input(
                    "top_p",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Top-p for nucleus sampling. Active only if temperature > 0.",
                ),
                io.Boolean.Input(
                    "do_sample",
                    default=False,
                    label_on="Sampling",
                    label_off="Greedy",
                    tooltip="Enable sampling for more varied output. Disable for deterministic transcription.",
                ),
                io.Int.Input(
                    "num_beams",
                    default=1,
                    min=1,
                    max=10,
                    tooltip="Number of beams for beam search. 1 = no beam search. Higher values may improve quality but are slower.",
                ),
                io.Combo.Input(
                    "device",
                    options=available_devices,
                    default=default_device,
                    tooltip="Device to run inference on.",
                ),
                io.Combo.Input(
                    "dtype",
                    options=dtype_options,
                    default=DTYPE_AUTO,
                    tooltip="Data type for model precision. 'auto' selects optimal type for device.",
                ),
                io.Combo.Input(
                    "attention_mode",
                    options=attention_modes,
                    default="sdpa",
                    tooltip="Attention implementation: Eager (safest), SDPA (balanced), Flash Attention 2 (fastest).",
                ),
                # Optional external model input
                VibeVoiceModel.Input(
                    "external_model",
                    optional=True,
                    tooltip=(
                        "Optional externally-loaded VibeVoice ASR model (from the "
                        "'Load VibeVoice Model' node). When connected, this "
                        "overrides the model_name dropdown."
                    ),
                ),
                io.Boolean.Input(
                    "force_offload",
                    default=False,
                    label_on="Force Offload",
                    label_off="Keep in VRAM",
                    tooltip="Force model to be offloaded from VRAM after transcription.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="Transcription"),
                io.String.Output(display_name="Segments (JSON)"),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        """Validate inputs, allowing dynamically-discovered custom ASR models."""
        # An externally-loaded model bypasses the model_name dropdown entirely.
        # NOTE: During prompt validation ComfyUI resolves *linked* inputs to
        # None (no execution cache exists yet — see execution.get_input_data /
        # mark_missing), so the value cannot be inspected here. We therefore
        # detect that the external_model input is *connected* by its presence
        # in kwargs: a linked input is always present (resolved to None), while
        # an unconnected optional input is absent from the prompt entirely.
        if "external_model" in kwargs:
            return True

        model_name = kwargs.get("model_name")
        if model_name is not None and model_name != "No ASR models found":
            if model_name not in AVAILABLE_VIBEVOICE_MODELS:
                available = list(AVAILABLE_VIBEVOICE_MODELS.keys())
                return f"Model '{model_name}' not found. Available models: {available}"
            # Reject non-ASR types before they reach the ASR loader.
            if not is_model_type(model_name, "asr"):
                cfg_type = MODEL_CONFIGS.get(model_name, {}).get("model_type")
                return (
                    f"Model '{model_name}' is type '{cfg_type}'; "
                    f"use the VibeVoice TTS node for TTS models."
                )
        return True

    @classmethod
    def execute(
        cls,
        model_name: str,
        audio: dict,
        context_info: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        num_beams: int,
        device: str,
        dtype: str,
        attention_mode: str,
        force_offload: bool,
        external_model: Optional[dict] = None,
    ) -> io.NodeOutput:
        """Execute VibeVoice ASR transcription."""

        # Load ASR model — external bundle overrides the model_name dropdown.
        if external_model is not None:
            # Guard: streaming (realtime) models cannot transcribe audio.
            if external_model.get("is_streaming"):
                raise ValueError(
                    "The provided external model is a streaming (realtime) model. "
                    "Use the 'VibeVoice Realtime TTS' node for streaming models; "
                    "the ASR node requires a VibeVoice ASR model."
                )
            # Guard: TTS models cannot transcribe audio. Bundles explicitly
            # marked is_asr=False are rejected; bundles without the flag are
            # accepted for backward compatibility.
            if external_model.get("is_asr") is False:
                raise ValueError(
                    "The provided external model is a TTS (text-to-speech) model. "
                    "Use the 'VibeVoice TTS' node for TTS models; "
                    "the ASR node requires a VibeVoice ASR model."
                )
            patcher, model, processor = load_asr_from_external(
                external_model,
                device=device,
                dtype=dtype,
                attention_mode=attention_mode,
            )
            # Use the bundle's model name for offload cache keying.
            model_name = external_model.get("model_name", model_name)
        else:
            # Load ASR model via the patcher/VRAM system (CRIT-001 fix).
            patcher, model, processor = load_asr_model_patched(
                model_name=model_name,
                device=device,
                dtype=dtype,
                attention_mode=attention_mode,
            )

        try:
            # Transcribe audio
            raw_text, segments = transcribe_audio(
                model=model,
                processor=processor,
                audio_input=audio,
                context_info=context_info if context_info.strip() else None,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                num_beams=num_beams,
            )

            # Format segments as JSON string
            segments_json = json.dumps(segments, indent=2, ensure_ascii=False)

            logger.info(f"ASR transcription complete. {len(segments)} segments.")

            if force_offload:
                force_offload_asr_model(model_name, patcher)

            return io.NodeOutput(raw_text, segments_json)

        except model_management.InterruptProcessingException:
            logger.info("VibeVoice ASR transcription was cancelled")
            return io.NodeOutput("", "[]")

        except Exception as e:
            logger.error(f"Error during VibeVoice ASR transcription: {e}")
            raise
