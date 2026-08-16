"""Custom ComfyUI types for VibeVoice nodes.

This module defines custom ComfyUI types for passing structured data
between nodes in a modular architecture.

Types:
- VibeVoiceModel: A pre-loaded VibeVoice model bundle (state dict + config +
  processor + instantiated model) produced by the "Load VibeVoice Model" node
  and consumed by the TTS / Realtime / ASR nodes via their optional
  ``external_model`` input.

The runtime value passed through the wire is a dict containing:

    {
        "state_dict": dict[str, torch.Tensor],   # loaded weights (CPU)
        "config": VibeVoiceConfig | VibeVoiceStreamingConfig,
        "processor": VibeVoiceProcessor | VibeVoiceStreamingProcessor,
        "model": torch.nn.Module,                 # instantiated model (CPU)
        "model_name": str,                        # display name / cache key seed
        "source_path": str,                       # original file path (for logging)
        "is_streaming": bool,                     # streaming model flag
    }

This follows the same pattern as VoxCPM's ``VoiceCloningConfig`` custom type
(via ``io.Custom()``).
"""

from comfy_api.latest import io

# Create the custom type using io.Custom
VibeVoiceModel = io.Custom("VIBEVOICE_MODEL")

__all__ = ["VibeVoiceModel"]
