"""Exhaustive pure-policy tests for generation-family resolution."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ComfyUI_VibeVoice.modules.generation import resolve_generation_family


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("VibeVoice-1.5B", "tts"),
        ("ordinary-local-copy", "tts"),
        ("VibeVoice-Realtime-0.5B", "streaming_tts"),
        ("realtime-0.5b-local", "streaming_tts"),
        ("my-streaming-copy", "streaming_tts"),
    ],
)
def test_named_model_truth_table(name, expected):
    assert resolve_generation_family(name) == expected


@pytest.mark.parametrize(
    ("bundle", "expected"),
    [
        ({"is_asr": False, "is_streaming": False}, "tts"),
        ({"is_asr": False, "is_streaming": True}, "streaming_tts"),
    ],
)
def test_external_bundle_truth_table(bundle, expected):
    assert resolve_generation_family("ignored", bundle) == expected


def test_external_asr_rejected_with_asr_node_guidance():
    with pytest.raises(ValueError, match="ASR.*VibeVoice ASR"):
        resolve_generation_family("ignored", {"is_asr": True})


@pytest.mark.parametrize("name", ["VibeVoice-ASR-HF", "VibeVoice-ASR-Streaming-7B"])
def test_named_asr_rejected_with_asr_node_guidance(name):
    with pytest.raises(ValueError, match="ASR.*VibeVoice ASR"):
        resolve_generation_family(name)


def test_empty_model_rejected():
    with pytest.raises(ValueError, match="No VibeVoice model"):
        resolve_generation_family("")


def test_policy_does_not_load_models_touch_files_or_create_tensors():
    with patch("ComfyUI_VibeVoice.modules.generation.load_vibevoice_model") as load, patch(
        "ComfyUI_VibeVoice.modules.model_info.AVAILABLE_VIBEVOICE_MODELS", {}
    ), patch("builtins.open") as open_file:
        assert resolve_generation_family("VibeVoice-1.5B") == "tts"
    load.assert_not_called()
    open_file.assert_not_called()


def test_source_module_keeps_resolver_pure_and_standard():
    source = __import__("inspect").getsource(resolve_generation_family)
    assert "torch" not in source
    assert "load_vibevoice_model" not in source
    assert "os." not in source


def test_retired_streaming_symbols_are_absent_from_production():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    production = "\n".join(
        (root / "modules" / name).read_text(encoding="utf-8")
        for name in ("generation.py", "realtime_generation.py")
    )
    assert "prefill_voice_prompt" not in production
    assert "generate_streaming_audio" not in production
