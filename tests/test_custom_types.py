"""Tests for modules/custom_types.py - VIBEVOICE_MODEL custom type."""

import pytest

from ComfyUI_VibeVoice.modules.custom_types import VibeVoiceModel


class TestVibeVoiceModelType:
    """Test the VibeVoiceModel custom type definition."""

    def test_vibevoice_model_type_exists(self):
        """VibeVoiceModel is importable and has the correct io_type."""
        assert VibeVoiceModel is not None
        assert VibeVoiceModel.io_type == "VIBEVOICE_MODEL"

    def test_vibevoice_model_input_creates_socket(self):
        """VibeVoiceModel.Input produces an input with the correct type string."""
        inp = VibeVoiceModel.Input("external_model", optional=True)
        assert inp is not None
        assert inp.id == "external_model"
        assert inp.get_io_type() == "VIBEVOICE_MODEL"

    def test_vibevoice_model_input_optional_flag(self):
        """VibeVoiceModel.Input with optional=True marks the input optional."""
        inp = VibeVoiceModel.Input("external_model", optional=True)
        assert inp.optional is True

    def test_vibevoice_model_output_creates_socket(self):
        """VibeVoiceModel.Output produces an output with the correct type string."""
        out = VibeVoiceModel.Output(display_name="VibeVoice Model")
        assert out is not None
        assert out.io_type == "VIBEVOICE_MODEL"

    def test_vibevoice_model_input_with_tooltip(self):
        """VibeVoiceModel.Input accepts a tooltip argument."""
        inp = VibeVoiceModel.Input(
            "external_model",
            optional=True,
            tooltip="Externally loaded VibeVoice model",
        )
        assert inp.tooltip == "Externally loaded VibeVoice model"
