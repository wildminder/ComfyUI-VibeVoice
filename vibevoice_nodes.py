"""VibeVoice TTS Nodes for ComfyUI.

This module provides the main entry point for VibeVoice nodes.
Nodes are organized in the 'nodes' package for easier maintenance.

Main Node:
- VibeVoiceTTSNode: Multi-speaker TTS synthesis with voice cloning
"""

import logging
from typing import List

from comfy_api.latest import ComfyExtension, io

try:
    from .nodes import VibeVoiceTTSNode, VibeVoiceASRNode, VibeVoiceRealtimeNode
except ImportError:
    from nodes import VibeVoiceTTSNode, VibeVoiceASRNode, VibeVoiceRealtimeNode

logger = logging.getLogger(__name__)


class VibeVoiceExtension(ComfyExtension):
    """ComfyUI extension providing VibeVoice TTS nodes."""

    async def get_node_list(self) -> List[type[io.ComfyNode]]:
        return [
            VibeVoiceTTSNode,
            VibeVoiceASRNode,
            VibeVoiceRealtimeNode,
        ]


async def comfy_entrypoint() -> VibeVoiceExtension:
    """Entry point for ComfyUI extension loading."""
    return VibeVoiceExtension()
