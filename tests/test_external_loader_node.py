"""Tests for nodes/external_loader_node.py - Load VibeVoice Model node."""

import pytest
from unittest.mock import MagicMock, patch

from comfy_api.latest import io

from ComfyUI_VibeVoice.nodes.external_loader_node import VibeVoiceExternalLoaderNode


class TestExternalLoaderNodeSchema:
    """Test the VibeVoiceExternalLoaderNode schema."""

    def test_node_schema_id(self):
        """define_schema().node_id == 'VibeVoiceLoadExternalModel'."""
        schema = VibeVoiceExternalLoaderNode.define_schema()
        assert schema.node_id == "VibeVoiceLoadExternalModel"

    def test_node_display_name(self):
        """Display name is 'Load VibeVoice Model'."""
        schema = VibeVoiceExternalLoaderNode.define_schema()
        assert schema.display_name == "Load VibeVoice Model"

    def test_node_has_model_file_input(self):
        """'model_file' is in the input ids."""
        schema = VibeVoiceExternalLoaderNode.define_schema()
        input_ids = [inp.id for inp in schema.inputs]
        assert "model_file" in input_ids

    def test_node_has_config_name_input(self):
        """'config_name' is in the input ids."""
        schema = VibeVoiceExternalLoaderNode.define_schema()
        input_ids = [inp.id for inp in schema.inputs]
        assert "config_name" in input_ids

    def test_node_has_attention_mode_input(self):
        """'attention_mode' is in the input ids."""
        schema = VibeVoiceExternalLoaderNode.define_schema()
        input_ids = [inp.id for inp in schema.inputs]
        assert "attention_mode" in input_ids

    def test_node_has_quantize_input(self):
        """'quantize_llm_4bit' is in the input ids."""
        schema = VibeVoiceExternalLoaderNode.define_schema()
        input_ids = [inp.id for inp in schema.inputs]
        assert "quantize_llm_4bit" in input_ids

    def test_node_has_dtype_input(self):
        """'dtype' is in the input ids."""
        schema = VibeVoiceExternalLoaderNode.define_schema()
        input_ids = [inp.id for inp in schema.inputs]
        assert "dtype" in input_ids

    def test_node_output_is_vibevoice_model(self):
        """Output type string is 'VIBEVOICE_MODEL'."""
        schema = VibeVoiceExternalLoaderNode.define_schema()
        assert len(schema.outputs) == 1
        assert schema.outputs[0].io_type == "VIBEVOICE_MODEL"

    def test_node_category(self):
        """Node category is 'WMNodes/sound/tts'."""
        assert VibeVoiceExternalLoaderNode.CATEGORY == "WMNodes/sound/tts"


class TestExternalLoaderNodeExecute:
    """Test the VibeVoiceExternalLoaderNode.execute() method."""

    def test_node_execute_calls_load_external(self):
        """execute() calls load_external_vibevoice_model with correct kwargs."""
        fake_bundle = {"model": MagicMock(), "processor": MagicMock()}

        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.load_external_vibevoice_model",
            return_value=fake_bundle,
        ) as mock_load, patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_full_path_or_raise",
            return_value="/fake/path/model.safetensors",
        ):
            result = VibeVoiceExternalLoaderNode.execute(
                model_file="model.safetensors",
                config_name="VibeVoice-1.5B",
                attention_mode="sdpa",
                quantize_llm_4bit=False,
                dtype="auto",
            )

        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["weight_path"] == "/fake/path/model.safetensors"
        assert call_kwargs["config_name"] == "VibeVoice-1.5B"
        assert call_kwargs["attention_mode"] == "sdpa"
        assert call_kwargs["use_llm_4bit"] is False
        assert call_kwargs["dtype_str"] == "auto"

    def test_node_execute_resolves_path_via_folder_paths(self):
        """execute() resolves the path via folder_paths.get_full_path_or_raise."""
        fake_bundle = {"model": MagicMock()}

        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.load_external_vibevoice_model",
            return_value=fake_bundle,
        ), patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_full_path_or_raise",
            return_value="/fake/path/model.safetensors",
        ) as mock_resolve:
            VibeVoiceExternalLoaderNode.execute(
                model_file="model.safetensors",
                config_name="VibeVoice-1.5B",
                attention_mode="sdpa",
                quantize_llm_4bit=False,
                dtype="auto",
            )

        mock_resolve.assert_called_once_with("diffusion_models", "model.safetensors")

    def test_node_execute_returns_node_output(self):
        """execute() returns an io.NodeOutput wrapping the bundle dict."""
        fake_bundle = {"model": MagicMock(), "processor": MagicMock()}

        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.load_external_vibevoice_model",
            return_value=fake_bundle,
        ), patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_full_path_or_raise",
            return_value="/fake/path/model.safetensors",
        ):
            result = VibeVoiceExternalLoaderNode.execute(
                model_file="model.safetensors",
                config_name="VibeVoice-1.5B",
                attention_mode="sdpa",
                quantize_llm_4bit=False,
                dtype="auto",
            )

        assert isinstance(result, io.NodeOutput)
        # The bundle should be the first output value
        assert result[0] is fake_bundle

    def test_node_execute_passes_quantize_flag(self):
        """execute() passes quantize_llm_4bit=True through to the loader."""
        fake_bundle = {"model": MagicMock()}

        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.load_external_vibevoice_model",
            return_value=fake_bundle,
        ) as mock_load, patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_full_path_or_raise",
            return_value="/fake/path/model.safetensors",
        ):
            VibeVoiceExternalLoaderNode.execute(
                model_file="model.safetensors",
                config_name="VibeVoice-7B",
                attention_mode="eager",
                quantize_llm_4bit=True,
                dtype="bf16",
            )

        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["use_llm_4bit"] is True
        assert call_kwargs["dtype_str"] == "bf16"
        assert call_kwargs["config_name"] == "VibeVoice-7B"
