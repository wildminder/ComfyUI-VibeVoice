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
