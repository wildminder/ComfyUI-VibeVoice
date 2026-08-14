"""Shared utilities for VibeVoice nodes.

Contains the global patcher cache and re-exports commonly used helpers
to avoid circular imports.
"""

import torch

# Global cache for model patchers (shared across all nodes)
VIBEVOICE_PATCHER_CACHE = {}

# Separate global cache for ASR model patchers so ASR and TTS memory
# orchestration remain independent (mirrors the split used by the
# legacy direct-load caches LOADED_MODELS_CACHE / LOADED_ASR_MODELS_CACHE).
VIBEVOICE_ASR_PATCHER_CACHE = {}

# Re-export set_seed for convenience
from .audio_utils import set_seed  # noqa: F401, E402
