"""VibeVoice TTS Node - canonical standard and realtime TTS node (V3 Schema).

This is the single user-facing TTS node. It routes standard VibeVoice
checkpoints through the multi-speaker reference-audio path and
``VibeVoice-Realtime`` checkpoints through the official cached-voice-prompt
windowed inference path. Both families return one completed ComfyUI ``AUDIO``
object; no live PCM transport is provided.
"""

import torch
import logging
from typing import Optional

import comfy.model_management as model_management
from comfy_api.latest import io, ui

from ..modules.model_info import (
    AVAILABLE_VIBEVOICE_MODELS,
    get_tts_family_models,
)
from ..modules.generation import (
    load_vibevoice_model,
    load_vibevoice_from_external,
    generate_audio,
    force_offload_model,
    resolve_generation_family,
)
from ..modules.realtime_generation import (
    classify_loaded_tts_pair,
    generate_realtime_audio,
)
from ..modules.voice_presets import (
    PRESET_NONE,
    get_cached_voice_preset,
    list_voice_presets,
)
from ..modules.attention_utils import get_available_attention_modes
from ..modules.device_utils import get_device_options
from ..modules.dtype_utils import get_dtype_options, DTYPE_AUTO
from ..modules.custom_types import VibeVoiceModel

logger = logging.getLogger(__name__)

VOICE_PRESET_HELP = (
    "Official cached .pt voice prompt required by realtime models. Place "
    "prompts in models/tts/VibeVoice/voices (or another registered "
    "vibevoice_voices root). Ignored by standard models."
)
_MISSING_PRESET_MESSAGE = (
    "Realtime models require a voice preset. Select a value for "
    "'voice_preset' from models/tts/VibeVoice/voices."
)
# Sentinel default for the optional external_model input. None is not usable:
# a *linked* input resolves to None during prompt validation, while an
# unconnected optional input is absent from the prompt entirely.
_EXTERNAL_UNSET = object()


class VibeVoiceTTSNode(io.ComfyNode):
    """VibeVoice TTS node for standard and realtime speech generation.

    Features:
    - Standard family: multi-speaker script parsing and reference-audio cloning
    - Realtime family: single-speaker cached ``.pt`` voice prompts
    - Shared attention modes, 4-bit LLM quantization, patcher, and offload path
    - Independent diffusion steps and generated-length controls
    """

    CATEGORY = "WMNodes/sound/tts"

    @classmethod
    def define_schema(cls) -> io.Schema:
        # Standard TTS models first, then realtime models; ASR is excluded.
        model_names = list(get_tts_family_models().keys())
        if not model_names:
            model_names.append("No models found in models/tts/VibeVoice")

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
            node_id="VibeVoiceTTS",
            display_name="VibeVoice TTS",
            category=cls.CATEGORY,
            description=(
                "Generate expressive audio with VibeVoice. Standard models use "
                "multi-speaker reference-audio voice cloning; realtime models "
                "use official cached .pt voice prompts."
            ),
            inputs=[
                # Model selection
                io.Combo.Input(
                    "model_name",
                    options=model_names,
                    default=model_names[0],
                    tooltip="Select the VibeVoice model to use. Standard and realtime TTS models are listed here; official models are downloaded automatically.",
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
                        "without their own reference are cloned from the provided reference(s). "
                        "Realtime models are single-speaker and ignore speaker labels."
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
                    tooltip=(
                        "Classifier-Free Guidance scale. Higher values increase adherence to the voice prompt but may reduce naturalness. "
                        "Recommended: 1.3 for standard models, 1.5-1.8 for realtime models — realtime values below 1.5 repeat syllables instead of speaking the script, and are raised to 1.5 automatically."
                    ),
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
                    tooltip="Enable to use sampling methods (like temperature and top_p) for more varied output. Disable for deterministic (greedy) decoding. Not used by realtime models.",
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
                    tooltip="Max generated speech tokens (utterance length budget). 0 = auto: standard models use ~30x the prompt length; realtime models size the budget from the script (~2.5 latents per text token, capped at 1024) so a short prompt cannot run into a multi-minute decode. If the voice model exposes a speech-end token, generation stops earlier; raise this if output is cut off, lower it if output is too long.",
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
                # Appended: realtime cached voice prompt (standard models ignore it)
                io.Combo.Input(
                    "voice_preset",
                    options=voice_preset_options,
                    default=PRESET_NONE,
                    tooltip=VOICE_PRESET_HELP,
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
        """Validate inputs, allowing dynamically-discovered custom TTS models.

        The signature declares only the inputs this rule inspects. ComfyUI
        core (execution.py) calls the validator once per input present in the
        prompt and emits one error per failing call, so a ``**kwargs``
        signature repeats a single message once per widget.
        """
        # An externally-loaded model bypasses the model_name dropdown entirely.
        # NOTE: During prompt validation ComfyUI resolves *linked* inputs to
        # None (no execution cache exists yet — see execution.get_input_data /
        # mark_missing), so the value cannot be inspected here. We therefore
        # detect that the external_model input is *connected* by it being
        # passed at all: a linked input is always passed (resolved to None),
        # while an unconnected optional input is absent from the prompt
        # entirely and leaves the sentinel default in place.
        if external_model is not _EXTERNAL_UNSET:
            return True

        if model_name is None:
            return True

        if model_name not in AVAILABLE_VIBEVOICE_MODELS:
            available = list(AVAILABLE_VIBEVOICE_MODELS.keys())
            return f"Model '{model_name}' not found. Available models: {available}"

        try:
            family = resolve_generation_family(model_name)
        except ValueError as exc:
            return str(exc)

        if family == "streaming_tts":
            # Old saved prompts have no voice_preset key; the default covers it.
            if not voice_preset or voice_preset == PRESET_NONE:
                return _MISSING_PRESET_MESSAGE
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
        voice_preset: str = PRESET_NONE,
    ) -> io.NodeOutput:
        """Execute standard or realtime VibeVoice TTS generation."""

        # Resolve the family before loading so ASR / unknown inputs fail fast.
        family = resolve_generation_family(model_name, external_model)

        # Load model — external bundle overrides the model_name dropdown.
        if external_model is not None:
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

        # Loaded-class safety net: renamed local checkpoints can disagree with
        # name-based classification. Only tts -> streaming_tts is repaired.
        loaded_classification = classify_loaded_tts_pair(model, processor)
        if loaded_classification == "mismatch":
            raise ValueError(
                "Loaded VibeVoice model and processor classes are inconsistent. "
                "Realtime checkpoints must load the VibeVoice realtime model "
                "with the VibeVoice realtime processor."
            )
        if family == "tts" and loaded_classification == "realtime":
            logger.warning(
                "Model '%s' was classified as standard TTS by name but loaded "
                "realtime classes; routing through the realtime path.",
                model_name,
            )
            family = "streaming_tts"
        elif family == "streaming_tts" and loaded_classification == "standard":
            raise ValueError(
                "Model '%s' was classified as a realtime model but loaded "
                "standard VibeVoice classes. Realtime models require the "
                "VibeVoice realtime model and processor."
                % model_name
            )

        try:
            if family == "streaming_tts":
                output_waveform, sample_rate = cls._generate_realtime(
                    model=model,
                    processor=processor,
                    text=text,
                    voice_preset=voice_preset,
                    cfg_scale=cfg_scale,
                    inference_steps=inference_steps,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    speaker_1_voice=speaker_1_voice,
                    speaker_2_voice=speaker_2_voice,
                    speaker_3_voice=speaker_3_voice,
                    speaker_4_voice=speaker_4_voice,
                )
            else:
                output_waveform, sample_rate = cls._generate_standard(
                    model=model,
                    processor=processor,
                    text=text,
                    speaker_1_voice=speaker_1_voice,
                    speaker_2_voice=speaker_2_voice,
                    speaker_3_voice=speaker_3_voice,
                    speaker_4_voice=speaker_4_voice,
                    cfg_scale=cfg_scale,
                    inference_steps=inference_steps,
                    seed=seed,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
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
            return io.NodeOutput(cls._silent_output())

        except Exception as e:
            logger.error(f"Error during VibeVoice generation with {attention_mode} attention: {e}")
            if "interrupt" in str(e).lower() or "cancel" in str(e).lower():
                logger.info("Generation was interrupted")
                return io.NodeOutput(cls._silent_output())
            raise

    @classmethod
    def _silent_output(cls) -> dict:
        """Return the shared one-second silent fallback for cancellation."""
        return {
            "waveform": torch.zeros((1, 1, 24000), dtype=torch.float32),
            "sample_rate": 24000,
        }

    @classmethod
    def _generate_standard(
        cls,
        model,
        processor,
        text: str,
        speaker_1_voice: Optional[dict],
        speaker_2_voice: Optional[dict],
        speaker_3_voice: Optional[dict],
        speaker_4_voice: Optional[dict],
        cfg_scale: float,
        inference_steps: int,
        seed: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
    ):
        """Run the standard multi-speaker reference-audio generation path."""
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

        return generate_audio(
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

    @classmethod
    def _generate_realtime(
        cls,
        model,
        processor,
        text: str,
        voice_preset: str,
        cfg_scale: float,
        inference_steps: int,
        max_new_tokens: int,
        seed: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        speaker_1_voice: Optional[dict],
        speaker_2_voice: Optional[dict],
        speaker_3_voice: Optional[dict],
        speaker_4_voice: Optional[dict],
    ):
        """Run the official realtime cached-voice-prompt generation path."""
        if not voice_preset or voice_preset == PRESET_NONE:
            raise ValueError(_MISSING_PRESET_MESSAGE)

        if any(
            voice is not None
            for voice in (
                speaker_1_voice,
                speaker_2_voice,
                speaker_3_voice,
                speaker_4_voice,
            )
        ):
            logger.warning(
                "Speaker reference audio is ignored for realtime models: the "
                "VibeVoice realtime architecture is single-speaker and uses "
                "the selected 'voice_preset' cached prompt."
            )

        if do_sample or temperature != 0.95 or top_p != 0.95 or top_k != 0:
            logger.warning(
                "Sampling controls (do_sample/temperature/top_p/top_k) are not "
                "used by the current realtime generation loop; model defaults "
                "are used instead."
            )

        cached_voice_preset = get_cached_voice_preset(
            voice_preset,
            model_management.get_torch_device(),
        )
        return generate_realtime_audio(
            model=model,
            processor=processor,
            text=text,
            voice_preset=cached_voice_preset,
            cfg_scale=cfg_scale,
            diffusion_steps=inference_steps,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )


__all__ = ["VibeVoiceTTSNode"]
