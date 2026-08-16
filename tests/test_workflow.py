"""Tests for example workflow JSON."""

import os
import json
import pytest


class TestWorkflow:
    """Test the example workflow JSON."""

    def _get_workflow_path(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(root, "example_workflows", "VibeVoice_example.json")

    def test_workflow_json_valid(self):
        """The workflow JSON should be valid JSON."""
        path = self._get_workflow_path()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data is not None

    def test_workflow_has_nodes(self):
        """The workflow should contain nodes."""
        path = self._get_workflow_path()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Workflow format varies; check for common keys
        if "nodes" in data:
            assert len(data["nodes"]) > 0
        elif "prompt" in data:
            assert len(data["prompt"]) > 0


class TestExternalModelWorkflow:
    """Phase 6.2: the external-model example workflow must be valid."""

    def _get_workflow_path(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(
            root, "example_workflows", "VibeVoice_external_model_example.json"
        )

    def test_external_model_workflow_json_is_valid(self):
        """The external-model workflow JSON parses and contains nodes."""
        path = self._get_workflow_path()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data is not None
        assert "nodes" in data
        assert len(data["nodes"]) > 0

    def test_external_model_workflow_has_loader_node(self):
        """The workflow contains the Load VibeVoice Model node."""
        path = self._get_workflow_path()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        node_types = [n.get("type") for n in data["nodes"]]
        assert "VibeVoiceLoadExternalModel" in node_types

    def test_external_model_workflow_has_tts_node(self):
        """The workflow contains the VibeVoice TTS node."""
        path = self._get_workflow_path()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        node_types = [n.get("type") for n in data["nodes"]]
        assert "VibeVoiceTTS" in node_types

    def test_external_model_workflow_links_loader_to_tts(self):
        """A VIBEVOICE_MODEL link connects the loader node to the TTS node."""
        path = self._get_workflow_path()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Map node id → type.
        node_by_id = {n["id"]: n.get("type") for n in data["nodes"]}

        # links format: [link_id, from_node, from_slot, to_node, to_slot, type]
        vibevoice_model_links = [
            link for link in data.get("links", [])
            if link[5] == "VIBEVOICE_MODEL"
        ]
        assert len(vibevoice_model_links) >= 1
        link = vibevoice_model_links[0]
        assert node_by_id.get(link[1]) == "VibeVoiceLoadExternalModel"
        assert node_by_id.get(link[3]) == "VibeVoiceTTS"

    def test_external_model_workflow_tts_has_external_model_input(self):
        """The TTS node declares an external_model input wired to the link."""
        path = self._get_workflow_path()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tts_node = next(
            n for n in data["nodes"] if n.get("type") == "VibeVoiceTTS"
        )
        ext_inputs = [
            i for i in tts_node.get("inputs", [])
            if i.get("name") == "external_model"
        ]
        assert len(ext_inputs) == 1
        assert ext_inputs[0].get("type") == "VIBEVOICE_MODEL"
        assert ext_inputs[0].get("link") is not None
