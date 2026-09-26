"""Tests for vibevoice_nodes.py - Extension and entrypoint."""

import asyncio
import pytest

from ComfyUI_VibeVoice.vibevoice_nodes import VibeVoiceExtension, comfy_entrypoint
from ComfyUI_VibeVoice.nodes.tts_node import VibeVoiceTTSNode
from ComfyUI_VibeVoice.nodes.realtime_node import VibeVoiceRealtimeNode
from ComfyUI_VibeVoice.nodes.external_loader_node import VibeVoiceExternalLoaderNode


# Compatibility snapshot captured before canonical-node changes. The extension
# must retain these IDs until the explicit migration/major-release boundary.
LEGACY_EXTENSION_NODE_IDS = (
    "VibeVoiceTTS",
    "VibeVoiceASR",
    "VibeVoiceRealtime",
    "VibeVoiceLoadExternalModel",
)


class TestVibeVoiceExtension:
    """Test VibeVoiceExtension class."""

    def test_extension_is_comfy_extension(self):
        from comfy_api.latest import ComfyExtension
        assert isinstance(VibeVoiceExtension(), ComfyExtension)

    def test_get_node_list_returns_nodes(self):
        ext = VibeVoiceExtension()
        node_list = asyncio.get_event_loop().run_until_complete(ext.get_node_list())
        assert isinstance(node_list, list)
        assert len(node_list) > 0

    def test_node_ids_match_legacy_snapshot(self):
        ext = VibeVoiceExtension()
        node_list = asyncio.get_event_loop().run_until_complete(ext.get_node_list())
        assert tuple(node.define_schema().node_id for node in node_list) == (
            LEGACY_EXTENSION_NODE_IDS
        )

    def test_node_list_contains_tts_node(self):
        ext = VibeVoiceExtension()
        node_list = asyncio.get_event_loop().run_until_complete(ext.get_node_list())
        assert VibeVoiceTTSNode in node_list

    def test_node_list_contains_realtime_node(self):
        ext = VibeVoiceExtension()
        node_list = asyncio.get_event_loop().run_until_complete(ext.get_node_list())
        assert VibeVoiceRealtimeNode in node_list

    def test_node_list_contains_external_loader_node(self):
        ext = VibeVoiceExtension()
        node_list = asyncio.get_event_loop().run_until_complete(ext.get_node_list())
        assert VibeVoiceExternalLoaderNode in node_list

    def test_node_list_not_empty(self):
        ext = VibeVoiceExtension()
        node_list = asyncio.get_event_loop().run_until_complete(ext.get_node_list())
        assert len(node_list) >= 1


class TestComfyEntrypoint:
    """Test comfy_entrypoint function."""

    def test_entrypoint_returns_extension(self):
        ext = asyncio.get_event_loop().run_until_complete(comfy_entrypoint())
        assert isinstance(ext, VibeVoiceExtension)

    def test_entrypoint_returns_comfy_extension(self):
        from comfy_api.latest import ComfyExtension
        ext = asyncio.get_event_loop().run_until_complete(comfy_entrypoint())
        assert isinstance(ext, ComfyExtension)
