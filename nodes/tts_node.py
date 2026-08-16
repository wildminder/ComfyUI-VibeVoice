"""VibeVoice TTS Node - Main Text-to-Speech Node (V3 Schema).

This node provides multi-speaker conversational TTS using the VibeVoice model.
Supports voice cloning via reference audio (at least one reference is required).
"""

import torch
import logging
from typing import Optional

import comfy.model_management as model_management
from comfy_api.latest import io, ui

from ..modules.model_info import (
    AVAILABLE_VIBEVOICE_MODELS,
    get_tts_models,
    is_model_type,
    MODEL_CONFIGS,
)
from ..modules.generation import (
    load_vibevoice_model,
    load_vibevoice_from_external,
    generate_audio,
    force_offload_model,
)
from ..modules.attention_utils import ATTENTION_MODES, get_available_attention_modes
from ..modules.device_utils import get_available_devices
from ..modules.dtype_utils import get_dtype_options, DTYPE_AUTO
from ..modules.custom_types import VibeVoiceModel

logger = logging.getLogger(__name__)


class VibeVoiceTTSNode(io.ComfyNode):
    """VibeVoice TTS node for expressive, long-form, multi-speaker conversational audio.

    Features:
    - Multi-speaker script parsing ([1], [2], etc. or Speaker 1:, Speaker 2:)
    - Voice cloning via reference audio inputs (at least one reference required)
    - Multiple attention modes (eager, sdpa, flash_attention_2, sage)
    - 4-bit LLM quantization for memory savings
    - Configurable generation parameters (CFG, steps, sampling)
    """

    CATEGORY = "audio/tts"

    @classmethod
    def define_schema(cls) -> io.Schema:
        # Only expose non-streaming TTS models. ASR models are handled by the
        # dedicated ASR node, and streaming (realtime) models by the dedicated
        # VibeVoice Realtime TTS node — the streaming processor/model require
        # the prefill + windowed generation path (generate_streaming_audio),
        # which this node does not use.
        model_names = list(get_tts_models().keys())
        if not model_names:
            model_names.append("No models found in models/tts/VibeVoice")

        available_devices = get_available_devices()
        default_device = available_devices[0]

        dtype_options = get_dtype_options()
        attention_modes = get_available_attention_modes()

        return io.Schema(
            node_id="VibeVoiceTTS",
            display_name="VibeVoice TTS",
            category=cls.CATEGORY,
            description=(
                "Generate expressive, long-form, multi-speaker conversational audio "
                "using VibeVoice TTS. Supports voice cloning via reference audio."
            ),
            inputs=[
                # Model selection
                io.Combo.Input(
                    "model_name",
                    options=model_names,
                    default=model_names[0],
                    tooltip="Select the VibeVoice model to use. Official models will be downloaded automatically.",
                ),
                # Text input
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
                # Quantization
                io.Boolean.Input(
                    "quantize_llm_4bit",
                    default=False,
                    label_on="Q4 (LLM only)",
                    label_off="Full precision",
                    tooltip="Quantize the Qwen2.5 LLM to 4-bit NF4 via bitsandbytes. Diffusion head stays BF16/FP32.",
                ),
                # Attention mode
                io.Combo.Input(
                    "attention_mode",
                    options=attention_modes,
                    default="sdpa",
                    tooltip="Attention implementation: Eager (safest), SDPA (balanced), Flash Attention 2 (fastest), Sage (quantized)",
                ),
                # Generation parameters
                io.Float.Input(
                    "cfg_scale",
                    default=1.3,
                    min=0.1,
                    max=50.0,
                    step=0.05,
                    tooltip="Classifier-Free Guidance scale. Higher values increase adherence to the voice prompt but may reduce naturalness. Recommended: 1.3",
                ),
                io.Int.Input(
                    "inference_steps",
                    default=10,
                    min=1,
                    max=500,
                    tooltip="Number of diffusion steps for audio generation. More steps can improve quality but take longer. Recommended: 10",
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
                    tooltip="Enable to use sampling methods (like temperature and top_p) for more varied output. Disable for deterministic (greedy) decoding.",
                ),
                io.Float.Input(
                    "temperature",
                    default=0.95,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="Controls randomness. Higher values make the output more random and creative, while lower values make it more focused and deterministic. Active only if 'do_sample' is enabled.",
                ),
                io.Float.Input(
                    "top_p",
                    default=0.95,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Nucleus sampling (Top-P). The model samples from the smallest set of tokens whose cumulative probability exceeds this value. Active only if 'do_sample' is enabled.",
                ),
                io.Int.Input(
                    "top_k",
                    default=0,
                    min=0,
                    max=500,
                    step=1,
                    tooltip="Top-K sampling. Restricts sampling to the K most likely next tokens. Set to 0 to disable. Active only if 'do_sample' is enabled.",
                ),
                io.Int.Input(
                    "max_new_tokens",
                    default=0,
                    min=0,
                    max=8192,
                    step=1,
                    tooltip="Max generated speech tokens (utterance length budget). 0 = auto (~30x the prompt length). If the voice model exposes a speech-end token, generation stops earlier; raise this if output is cut off, lower it if output is too long.",
                ),
                # System parameters
                io.Boolean.Input(
                    "force_offload",
                    default=False,
                    label_on="Force Offload",
                    label_off="Keep in VRAM",
                    tooltip="Force model to be offloaded from VRAM after generation. Useful to free up memory between generations but may slow down subsequent runs.",
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
                        "Optional externally-loaded VibeVoice model (from the "
                        "'Load VibeVoice Model' node). When connected, this "
                        "overrides the model_name dropdown."
                    ),
                ),
                # Optional speaker voice inputs
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
        """Validate inputs, allowing dynamically-discovered custom TTS models."""
        # An externally-loaded model bypasses the model_name dropdown entirely.
        if kwargs.get("external_model") is not None:
            return True

        model_name = kwargs.get("model_name")
        if model_name is not None:
            if model_name not in AVAILABLE_VIBEVOICE_MODELS:
                available = list(AVAILABLE_VIBEVOICE_MODELS.keys())
                return f"Model '{model_name}' not found. Available models: {available}"
            # Reject streaming (realtime) models: they require the streaming
            # generation path (generate_streaming_audio) used by the dedicated
            # VibeVoice Realtime TTS node. Routing them through this node hits
            # VibeVoiceStreamingProcessor.__call__ with unsupported kwargs.
            if is_model_type(model_name, "streaming_tts"):
                return (
                    f"Model '{model_name}' is a streaming (realtime) model; "
                    f"use the 'VibeVoice Realtime TTS' node for streaming models."
                )
            # Reject ASR / other non-TTS types before they reach the loader.
            if not is_model_type(model_name, "tts"):
                cfg_type = MODEL_CONFIGS.get(model_name, {}).get("model_type")
                return (
                    f"Model '{model_name}' is type '{cfg_type}'; "
                    f"use the VibeVoice ASR node for ASR models."
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
        force_offload: bool,
        device: str,
        dtype: str,
        max_new_tokens: int = 0,
        speaker_1_voice: Optional[dict] = None,
        speaker_2_voice: Optional[dict] = None,
        speaker_3_voice: Optional[dict] = None,
        speaker_4_voice: Optional[dict] = None,
        external_model: Optional[dict] = None,
    ) -> io.NodeOutput:
        """Execute VibeVoice TTS generation."""

        # Load model — external bundle overrides the model_name dropdown.
        if external_model is not None:
            # Guard: streaming (realtime) models must use the Realtime node.
            if external_model.get("is_streaming"):
                raise ValueError(
                    "The provided external model is a streaming (realtime) model. "
                    "Use the 'VibeVoice Realtime TTS' node for streaming models."
                )
            # Guard: ASR models cannot synthesize speech.
            if external_model.get("is_asr"):
                raise ValueError(
                    "The provided external model is an ASR (speech-to-text) model. "
                    "Use the 'VibeVoice ASR' node for ASR models."
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
            patcher, model, processor = load_vibevoice_model(
                model_name=model_name,
                device=device,
                dtype=dtype,
                attention_mode=attention_mode,
                quantize_4bit=quantize_llm_4bit,
            )

        # Collect speaker voice samples
        speaker_inputs = {
            1: speaker_1_voice,
            2: speaker_2_voice,
            3: speaker_3_voice,
            4: speaker_4_voice,
        }

        # Parse script to get speaker IDs
        from ..modules.audio_utils import parse_script_1_based
        _, speaker_ids_1_based = parse_script_1_based(text)

        # Build voice samples list in order of speaker IDs
        voice_samples = [speaker_inputs.get(sid) for sid in speaker_ids_1_based]

        try:
            # Generate audio
            output_waveform, sample_rate = generate_audio(
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
                max_new_tokens=max_new_tokens if max_new_tokens else None,
            )

            output_audio = {
                "waveform": output_waveform,
                "sample_rate": sample_rate,
            }

            logger.info(f"Audio generation complete. Sample rate: {sample_rate}Hz")

            if force_offload:
                # NTH-004: warm re-attach — keep tensors on the intermediate device
                # so a subsequent run re-attaches from memory instead of reloading.
                force_offload_model(patcher, model_name, warm=True)

            return io.NodeOutput(output_audio, ui=ui.PreviewAudio(output_audio, cls=cls))

        except model_management.InterruptProcessingException:
            logger.info("VibeVoice TTS generation was cancelled")
            return io.NodeOutput(
                {"waveform": torch.zeros((1, 1, 24000), dtype=torch.float32), "sample_rate": 24000}
            )

        except Exception as e:
            logger.error(f"Error during VibeVoice generation with {attention_mode} attention: {e}")
            if "interrupt" in str(e).lower() or "cancel" in str(e).lower():
                logger.info("Generation was interrupted")
                return io.NodeOutput(
                    {"waveform": torch.zeros((1, 1, 24000), dtype=torch.float32), "sample_rate": 24000}
                )
            raise
