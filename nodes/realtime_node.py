"""VibeVoice Realtime TTS Node - Streaming TTS for VibeVoice-Realtime-0.5B (V3 Schema).

Exposes the ``streaming_tts``-type model (``VibeVoice-Realtime-0.5B``) through the
same patcher / attention / VRAM machinery as the standard TTS node, but routes
generation through the streaming inference path (``model.generate`` with
``tts_text_ids`` / ``all_prefilled_outputs``).

Voice cloning via reference audio (at least one reference is required), mirroring
the non-streaming node. A ``stream`` toggle is provided for forward-compatible
incremental streaming output.
"""

import torch
import logging
from typing import Optional

import comfy.model_management as model_management
from comfy_api.latest import io, ui

from ..modules.model_info import (
    AVAILABLE_VIBEVOICE_MODELS,
    get_streaming_tts_models,
    is_model_type,
    MODEL_CONFIGS,
)
from ..modules.generation import (
    load_vibevoice_model,
    load_vibevoice_from_external,
    generate_streaming_audio,
    force_offload_model,
)
from ..modules.attention_utils import ATTENTION_MODES, get_available_attention_modes
from ..modules.device_utils import get_available_devices
from ..modules.dtype_utils import get_dtype_options, DTYPE_AUTO
from ..modules.custom_types import VibeVoiceModel

logger = logging.getLogger(__name__)


class VibeVoiceRealtimeNode(io.ComfyNode):
    """VibeVoice realtime / streaming TTS node.

    Features:
    - Streaming multi-speaker TTS via ``VibeVoice-Realtime-0.5B``
    - Voice cloning via reference audio inputs (at least one reference required)
    - Same attention modes and 4-bit LLM quantization as the standard TTS node
    - ``stream`` toggle for forward-compatible incremental streaming output
    """

    CATEGORY = "audio/tts"

    @classmethod
    def define_schema(cls) -> io.Schema:
        # Only expose streaming TTS models here (non-streaming live in VibeVoiceTTSNode).
        model_names = list(get_streaming_tts_models().keys())
        if not model_names:
            model_names.append("No streaming models found in models/tts/VibeVoice")

        available_devices = get_available_devices()
        default_device = available_devices[0]

        dtype_options = get_dtype_options()
        attention_modes = get_available_attention_modes()

        return io.Schema(
            node_id="VibeVoiceRealtime",
            display_name="VibeVoice Realtime TTS",
            category=cls.CATEGORY,
            description=(
                "Generate expressive, low-latency, multi-speaker conversational audio "
                "using the VibeVoice streaming model (e.g. VibeVoice-Realtime-0.5B). "
                "Supports voice cloning via reference audio."
            ),
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
                    tooltip="Reserved for incremental streaming output. Currently returns the assembled waveform; kept for forward compatibility.",
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
                    tooltip="Device to run inference on.",
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
            ],
            outputs=[
                io.Audio.Output(display_name="Audio"),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        """Validate inputs; only ``streaming_tts`` models are accepted here."""
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
        if model_name is not None:
            if model_name not in AVAILABLE_VIBEVOICE_MODELS:
                available = list(AVAILABLE_VIBEVOICE_MODELS.keys())
                return f"Model '{model_name}' not found. Available models: {available}"
            if not is_model_type(model_name, "streaming_tts"):
                cfg_type = MODEL_CONFIGS.get(model_name, {}).get("model_type")
                return (
                    f"Model '{model_name}' is type '{cfg_type}'; "
                    f"this node only supports streaming TTS models "
                    f"(use the VibeVoice TTS or ASR node for other types)."
                )
        return True

    @classmethod
    def execute(
        cls,
        model_name: str,
        text: str,
        quantize_llm_4bit: bool,
        attention_mode: str,
        cfg_scale: float,
        inference_steps: int,
        seed: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        stream: bool,
        force_offload: bool,
        device: str,
        dtype: str,
        speaker_1_voice: Optional[dict] = None,
        speaker_2_voice: Optional[dict] = None,
        speaker_3_voice: Optional[dict] = None,
        speaker_4_voice: Optional[dict] = None,
        external_model: Optional[dict] = None,
    ) -> io.NodeOutput:
        """Execute VibeVoice streaming TTS generation."""

        # Load model — external bundle overrides the model_name dropdown.
        if external_model is not None:
            # Guard: this node requires a streaming (realtime) model.
            if not external_model.get("is_streaming"):
                raise ValueError(
                    "The provided external model is not a streaming model. "
                    "The 'VibeVoice Realtime TTS' node requires a streaming "
                    "(realtime) model; use the 'VibeVoice TTS' node for "
                    "non-streaming models."
                )
            patcher, model, processor = load_vibevoice_from_external(
                external_model,
                device=device,
                dtype=dtype,
                attention_mode=attention_mode,
            )
            # Use the bundle's model name for offload cache keying.
            model_name = external_model.get("model_name", model_name)
        else:
            # Load model through the shared patcher / VRAM system.
            patcher, model, processor = load_vibevoice_model(
                model_name=model_name,
                device=device,
                dtype=dtype,
                attention_mode=attention_mode,
                quantize_4bit=quantize_llm_4bit,
            )

        # Collect speaker voice samples.
        speaker_inputs = {
            1: speaker_1_voice,
            2: speaker_2_voice,
            3: speaker_3_voice,
            4: speaker_4_voice,
        }

        from ..modules.audio_utils import parse_script_1_based
        _, speaker_ids_1_based = parse_script_1_based(text)
        voice_samples = [speaker_inputs.get(sid) for sid in speaker_ids_1_based]

        try:
            output_waveform, sample_rate = generate_streaming_audio(
                model=model,
                processor=processor,
                text=text,
                voice_samples=voice_samples,
                speaker_ids=speaker_ids_1_based,
                cfg_scale=cfg_scale,
                inference_steps=inference_steps,
                seed=seed,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stream=stream,
            )

            output_audio = {
                "waveform": output_waveform,
                "sample_rate": sample_rate,
            }

            logger.info(f"Realtime TTS generation complete. Sample rate: {sample_rate}Hz")

            if force_offload:
                # NTH-004 warm re-attach: keep tensors on the intermediate device.
                force_offload_model(patcher, model_name, warm=True)

            return io.NodeOutput(output_audio, ui=ui.PreviewAudio(output_audio, cls=cls))

        except model_management.InterruptProcessingException:
            logger.info("VibeVoice Realtime TTS generation was cancelled")
            return io.NodeOutput(
                {"waveform": torch.zeros((1, 1, 24000), dtype=torch.float32), "sample_rate": 24000}
            )

        except Exception as e:
            logger.error(f"Error during VibeVoice Realtime generation with {attention_mode} attention: {e}")
            if "interrupt" in str(e).lower() or "cancel" in str(e).lower():
                logger.info("Generation was interrupted")
                return io.NodeOutput(
                    {"waveform": torch.zeros((1, 1, 24000), dtype=torch.float32), "sample_rate": 24000}
                )
            raise
