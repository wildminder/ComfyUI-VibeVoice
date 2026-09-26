"""Isolated real vendored processor and windowed-loop contract tests.

The test-local package namespace is restored in a fixture finalizer so the
conftest MagicMock bootstrap cannot leak into later tests.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from ComfyUI_VibeVoice.modules.voice_presets import (
    PRESET_CACHE_KEYS,
    load_voice_preset,
    validate_voice_preset,
)


class _ContractAudioNormalizer:
    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture
def real_streaming_processor():
    """Load the real processor under a synthetic package namespace."""
    original = {name: module for name, module in sys.modules.items() if name.startswith("vibevoice_contract_pkg")}
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "vibevoice"))
    package = types.ModuleType("vibevoice_contract_pkg")
    package.__path__ = [root]
    processor_package = types.ModuleType("vibevoice_contract_pkg.processor")
    processor_package.__path__ = [os.path.join(root, "processor")]
    token_module = types.ModuleType("vibevoice_contract_pkg.processor.vibevoice_tokenizer_processor")
    token_module.AudioNormalizer = _ContractAudioNormalizer
    sys.modules[package.__name__] = package
    sys.modules[processor_package.__name__] = processor_package
    sys.modules[token_module.__name__] = token_module
    module_name = "vibevoice_contract_pkg.processor.vibevoice_streaming_processor"
    spec = importlib.util.spec_from_file_location(
        module_name,
        os.path.join(root, "processor", "vibevoice_streaming_processor.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        for name in list(sys.modules):
            if name.startswith("vibevoice_contract_pkg"):
                del sys.modules[name]
        sys.modules.update(original)


def _branch(length: int) -> BaseModelOutputWithPast:
    return BaseModelOutputWithPast(
        last_hidden_state=torch.ones(1, length, 8),
        # A real DynamicCache instance without lazily-created layer state; this
        # exercises the official safe global without widening the allowlist.
        past_key_values=DynamicCache.__new__(DynamicCache),
    )


class _Token:
    pad_id = 0

    def encode(self, text, add_special_tokens=False):
        return [10, 11, 12, 13]


def test_real_processor_requires_cached_prompt_and_builds_prompt_sized_inputs(real_streaming_processor):
    processor = real_streaming_processor.VibeVoiceStreamingProcessor(tokenizer=_Token())
    preset = {key: _branch(index + 2) for index, key in enumerate(PRESET_CACHE_KEYS)}

    with pytest.raises(NotImplementedError):
        processor()
    inputs = processor.process_input_with_cached_prompt(
        text="Hello realtime world",
        cached_prompt=preset,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    assert inputs["input_ids"].shape == (1, preset["lm"].last_hidden_state.shape[1])
    assert inputs["tts_lm_input_ids"].shape == (1, preset["tts_lm"].last_hidden_state.shape[1])
    assert inputs["tts_text_ids"].shape == (1, 4)
    assert inputs["attention_mask"].dtype == torch.long
    assert inputs["speech_input_mask"].dtype == torch.bool


def test_real_processor_rejects_plain_cached_branch(real_streaming_processor):
    processor = real_streaming_processor.VibeVoiceStreamingProcessor(tokenizer=_Token())
    preset = {key: _branch(3) for key in PRESET_CACHE_KEYS}
    preset["lm"] = {"last_hidden_state": torch.ones(1, 3, 8)}
    with pytest.raises(ValueError, match="lm"):
        validate_voice_preset(preset, "synthetic.pt")


def test_official_shaped_serialization_contract(tmp_path):
    preset = {key: _branch(index + 2) for index, key in enumerate(PRESET_CACHE_KEYS)}
    path = tmp_path / "voice.pt"
    torch.save(preset, path)
    loaded = load_voice_preset(str(path), torch.device("cpu"))
    assert set(loaded) == set(PRESET_CACHE_KEYS)
    for key in PRESET_CACHE_KEYS:
        assert isinstance(loaded[key], BaseModelOutputWithPast)
        assert torch.equal(
            loaded[key].last_hidden_state,
            preset[key].last_hidden_state,
        )
        assert isinstance(loaded[key].past_key_values, DynamicCache)


# The following loop contract reuses the already isolated real inference module
# loader from the existing progress-callback contract test, but performs no
# model checkpoint or network loading. The module is loaded under its real
# source path only after the synthetic package is restored by that test's own
# fixture in the standard suite.
import tests.test_streaming_progress_callback as _stream_contract  # noqa: E402
from tests.test_streaming_progress_callback import (  # noqa: E402
    MAX_LENGTH,
    VibeVoiceGenerationOutput,
    _make_streaming_model,
    _run_generate,
)


def test_real_windowed_loop_is_finite_repeatable_and_progresses():
    model = _make_streaming_model()
    first_calls = []
    first = _run_generate(model, progress_callback=lambda current, total: first_calls.append((current, total)))
    second = _run_generate(model)

    assert isinstance(first, VibeVoiceGenerationOutput)
    assert first.speech_outputs[0].isfinite().all()
    assert second.speech_outputs[0].isfinite().all()
    assert first_calls and {total for _current, total in first_calls} == {MAX_LENGTH}
    assert first_calls == sorted(first_calls)


def test_real_windowed_loop_boundary_and_eos_flags(monkeypatch):
    monkeypatch.setattr(_stream_contract, "MAX_LENGTH", 7)
    model = _make_streaming_model()
    capped = _run_generate(model)
    assert bool(capped.reach_max_step_sample[0]) is True

    monkeypatch.setattr(_stream_contract, "MAX_LENGTH", 100)
    eos_model = _make_streaming_model()
    eos_model.tts_eos_classifier = MagicMock(return_value=torch.tensor([[10.0]]))
    eos = _run_generate(eos_model)
    assert bool(eos.reach_max_step_sample[0]) is False


def test_generation_config_compat_matches_installed_transformers():
    """The vendored loop calls ``_prepare_generation_config`` compatibly.

    transformers 4.x accepts a positional ``is_init`` flag; 5.x removed it and
    narrowed the signature to ``(generation_config, **kwargs)``. The detection
    helper must agree with the installed API, otherwise real generation raises
    ``TypeError: takes 2 positional arguments but 3 were given``.
    """
    import inspect

    from transformers.generation.configuration_utils import GenerationConfig
    from transformers.generation.utils import GenerationMixin

    module = sys.modules[
        "src.vibevoice.modular.modeling_vibevoice_streaming_inference"
    ]
    accepts_flag = module._generation_config_accepts_positional_flag()

    parameters = list(
        inspect.signature(GenerationMixin._prepare_generation_config).parameters.values()
    )
    has_positional_flag = any(
        parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        and parameter.name not in ("self", "generation_config")
        for parameter in parameters
    )
    assert accepts_flag is has_positional_flag

    # The call path itself must work with the installed API.
    model = _make_streaming_model()
    del model._build_generate_config_model_kwargs
    model.config.is_encoder_decoder = False
    model.generation_config = GenerationConfig()
    model._prepare_model_inputs = MagicMock(
        side_effect=lambda inputs, bos_token_id, model_kwargs: (
            inputs,
            "input_ids",
            model_kwargs,
        )
    )
    tokenizer = _stream_contract._FakeTokenizer()
    config, model_kwargs, _inputs_tensor = model._build_generate_config_model_kwargs(
        None,
        torch.zeros(1, 2, dtype=torch.long),
        tokenizer,
        return_processors=False,
        max_new_tokens=8,
    )
    assert config.speech_start_id == tokenizer.speech_start_id
    assert config.speech_end_id == tokenizer.speech_end_id
    assert config.speech_diffusion_id == tokenizer.speech_diffusion_id
    assert isinstance(model_kwargs, dict)
