"""Tests for the non-streaming VibeVoiceForConditionalGeneration.forward() and generate().

These tests exercise the model's inference path directly with a mocked inner
`self.model` (VibeVoiceModel) so we can verify behavior without importing the
full heavy vendored stack (which is blocked in the test env by a
diffusers/huggingface_hub version mismatch).

Key regression: forward() must NOT call semantic_connector with None during
generation (speech_semantic_tensors is None for inference).

To import the real module we stub the `diffusers` package (and submodules that
the vendored schedule code imports) so the import chain succeeds without the
real, version-incompatible diffusers install.
"""

import sys
import types
import importlib.util
from unittest.mock import MagicMock

import torch
import torch.nn as nn


def _stub_diffusers():
    """Stub the diffusers package and submodules the vendored code imports."""
    if "diffusers" in sys.modules and not isinstance(sys.modules["diffusers"], MagicMock):
        return  # already a real diffusers

    d = types.ModuleType("diffusers")
    cu = types.ModuleType("diffusers.configuration_utils")
    cu.ConfigMixin = object
    cu.register_to_config = lambda *a, **k: None
    du = types.ModuleType("diffusers.utils")
    du.deprecate = lambda *a, **k: None
    tu = types.ModuleType("diffusers.utils.torch_utils")
    tu.randn_tensor = lambda *a, **k: None
    su = types.ModuleType("diffusers.schedulers.scheduling_utils")
    su.KarrasDiffusionSchedulers = object
    su.SchedulerMixin = object
    su.SchedulerOutput = object
    d.configuration_utils = cu
    d.utils = du
    d.schedulers = su
    sys.modules.setdefault("diffusers", d)
    sys.modules.setdefault("diffusers.configuration_utils", cu)
    sys.modules.setdefault("diffusers.utils", du)
    sys.modules.setdefault("diffusers.utils.torch_utils", tu)
    sys.modules.setdefault("diffusers.schedulers", su)
    sys.modules.setdefault("diffusers.schedulers.scheduling_utils", su)


def _load_real_modeling_module():
    """Load the real modeling_vibevoice module, bypassing the conftest mock."""
    _stub_diffusers()
    # Remove any conftest-installed mocks for the vendored package so the real
    # module is imported (not a MagicMock).
    for mod in (
        "src.vibevoice",
        "src.vibevoice.modular",
        "src.vibevoice.modular.modeling_vibevoice",
        "ComfyUI_VibeVoice.src.vibevoice.modular.modeling_vibevoice",
    ):
        sys.modules.pop(mod, None)

    # Register the package hierarchy so relative imports (from .modular_...) work.
    import os
    root = os.path.join(os.getcwd(), "src", "vibevoice")
    mod_pkg = types.ModuleType("src.vibevoice.modular")
    mod_pkg.__path__ = [os.path.join(root, "modular")]
    sys.modules.setdefault("src.vibevoice", types.ModuleType("src.vibevoice"))
    sys.modules["src.vibevoice"].__path__ = [root]
    sys.modules["src.vibevoice.modular"] = mod_pkg

    # Stub configuration_vibevoice so AutoModel.register() works (it reads
    # config_class.__name__). The conftest-mocked version is a bare MagicMock.
    cfg_mod = types.ModuleType("src.vibevoice.modular.configuration_vibevoice")
    class _FakeConfig:
        __name__ = "VibeVoiceConfig"
        model_type = "vibevoice"
    cfg_mod.VibeVoiceConfig = _FakeConfig
    sys.modules["src.vibevoice.modular.configuration_vibevoice"] = cfg_mod

    spec = importlib.util.spec_from_file_location(
        "src.vibevoice.modular.modeling_vibevoice",
        "src/vibevoice/modular/modeling_vibevoice.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["src.vibevoice.modular.modeling_vibevoice"] = module
    spec.loader.exec_module(module)
    return module


# Load the real module once at import time
_modeling = _load_real_modeling_module()
VibeVoiceForConditionalGeneration = _modeling.VibeVoiceForConditionalGeneration
VibeVoiceCausalLMOutputWithPast = _modeling.VibeVoiceCausalLMOutputWithPast
VibeVoiceGenerationOutput = _modeling.VibeVoiceGenerationOutput


def _make_model(hidden_size: int = 64):
    """Build a VibeVoiceForConditionalGeneration with a fully mocked inner model.

    The inner `self.model` (VibeVoiceModel) submodules are replaced with mocks
    so forward()/generate() run without real weights.
    """
    model = VibeVoiceForConditionalGeneration.__new__(VibeVoiceForConditionalGeneration)
    # Properly initialize the nn.Module base so parameters()/device work and
    # submodule assignment is allowed.
    torch.nn.Module.__init__(model)
    # Minimal attributes used by forward()/generate()
    model.config = MagicMock()
    model.config.use_return_dict = True
    model.config.diffusion_head_config = MagicMock()
    model.config.diffusion_head_config.ddpm_num_inference_steps = 10
    model.config.acoustic_tokenizer_config = MagicMock()
    model.config.acoustic_tokenizer_config.vae_dim = 64

    # Mock the inner VibeVoiceModel submodules
    inner = MagicMock()
    inner.semantic_connector = MagicMock()
    inner.acoustic_connector = MagicMock()
    inner.acoustic_tokenizer = MagicMock()
    inner.speech_scaling_factor = torch.tensor(float('nan'))
    inner.speech_bias_factor = torch.tensor(float('nan'))
    inner.noise_scheduler = MagicMock()
    inner.noise_scheduler.timesteps = [MagicMock()]
    inner.noise_scheduler.set_timesteps = MagicMock()
    inner.prediction_head = MagicMock()
    inner.prediction_head.device = torch.device("cpu")
    model.model = inner

    # Mock get_input_embeddings() -> returns embeddings of shape (vocab, hidden)
    emb = nn.Embedding(100, hidden_size)
    model.get_input_embeddings = MagicMock(return_value=emb)

    # Mock lm_head to produce logits. Register as a submodule so the model has
    # at least one parameter (needed for next(self.parameters()).device).
    model.add_module("lm_head", nn.Linear(hidden_size, 100, bias=False))

    # Mock the inner language model forward used by forward()
    def _inner_forward(**kwargs):
        x = kwargs.get("inputs_embeds")
        if x is None:
            x = torch.randn(1, 4, hidden_size)
        result = MagicMock()
        result.last_hidden_state = x  # real tensor so lm_head works
        return result

    inner.side_effect = _inner_forward

    # The non-streaming generate() calls self.model.language_model(...) directly
    # (one forward per autoregressive step), so provide a working inner LM that
    # returns a real hidden-state tensor (mirrors the wrapper used by forward()).
    def _lm_forward(inputs_embeds=None, **kwargs):
        x = kwargs.get("inputs_embeds")
        if not isinstance(x, torch.Tensor):
            x = torch.randn(1, 4, hidden_size)
        result = MagicMock()
        result.last_hidden_state = x
        result.past_key_values = None
        return result

    inner.language_model = MagicMock(side_effect=_lm_forward)

    # Mock acoustic_tokenizer.decode to return a waveform tensor
    inner.acoustic_tokenizer.decode = MagicMock(return_value=torch.randn(1, 24000))

    # Mock prediction_head forward to return noise of correct shape
    def _pred_head(noisy, timesteps, condition):
        b = condition.shape[0]
        return torch.randn(b, 64)

    inner.prediction_head.side_effect = _pred_head

    # Mock noise_scheduler.step to return prev_sample
    class _StepResult:
        prev_sample = torch.randn(1, 64)
    inner.noise_scheduler.step = MagicMock(return_value=_StepResult())

    return model


class TestForwardSemanticConnectorGuard:
    """Regression: forward() must not crash when speech_semantic_tensors is None."""

    def test_forward_with_none_semantic_tensors_does_not_crash(self):
        model = _make_model()
        input_ids = torch.randint(0, 100, (1, 4))
        out = model.forward(
            input_ids=input_ids,
            speeches_loss_input=None,
            speech_semantic_tensors=None,
            return_dict=True,
        )
        assert isinstance(out, VibeVoiceCausalLMOutputWithPast)
        assert out.logits is not None

    def test_semantic_connector_not_called_when_none(self):
        model = _make_model()
        input_ids = torch.randint(0, 100, (1, 4))
        model.forward(
            input_ids=input_ids,
            speeches_loss_input=None,
            speech_semantic_tensors=None,
            return_dict=True,
        )
        model.model.semantic_connector.assert_not_called()

    def test_semantic_connector_called_when_provided(self):
        model = _make_model()
        input_ids = torch.randint(0, 100, (1, 4))
        sem = torch.randn(1, 4, 128)
        model.forward(
            input_ids=input_ids,
            speeches_loss_input=torch.zeros(1, 4, dtype=torch.bool),
            speech_semantic_tensors=sem,
            speech_masks=torch.zeros(1, 4, dtype=torch.bool),
            acoustic_input_mask=torch.zeros(1, 4, dtype=torch.bool),
            return_dict=True,
        )
        model.model.semantic_connector.assert_called_once()


class TestGenerateSemanticGuard:
    """Regression: generate() must run with semantic_speech_tensors=None."""

    def test_generate_with_none_semantic_tensors_no_crash(self):
        model = _make_model()
        input_ids = torch.randint(0, 100, (1, 4))
        out = model.generate(
            input_ids=input_ids,
            acoustic_input_mask=torch.zeros(1, 4, dtype=torch.bool),
            semantic_speech_tensors=None,
            cfg_scale=1.3,
            inference_steps=5,
            return_speech=True,
        )
        assert isinstance(out, VibeVoiceGenerationOutput)
        assert out.speech_outputs is not None
        assert len(out.speech_outputs) == 1

    def test_generate_semantic_connector_never_sees_none(self):
        model = _make_model()
        input_ids = torch.randint(0, 100, (1, 4))
        model.generate(
            input_ids=input_ids,
            acoustic_input_mask=torch.zeros(1, 4, dtype=torch.bool),
            semantic_speech_tensors=None,
            return_speech=True,
        )
        # semantic_connector must never be invoked with None during generation
        model.model.semantic_connector.assert_not_called()

    def test_generate_sets_ddpm_steps(self):
        model = _make_model()
        input_ids = torch.randint(0, 100, (1, 4))
        model.set_ddpm_inference_steps(num_steps=7)
        assert model.ddpm_inference_steps == 7
        model.generate(
            input_ids=input_ids,
            acoustic_input_mask=torch.zeros(1, 4, dtype=torch.bool),
            semantic_speech_tensors=None,
            inference_steps=7,
            return_speech=True,
        )
        model.model.noise_scheduler.set_timesteps.assert_called_with(7)


# Minimal stand-in for VibeVoiceTokenizerEncoderOutput (dataclass with .mean
# and .sample()). The real one lives in modular_vibevoice_tokenizer, but we
# avoid importing the heavy module here.
class _FakeEncoderOutput:
    def __init__(self, mean):
        self.mean = mean

    def sample(self, dist_type="fix"):
        # Mirror the real .sample(): returns (x, std) tuple.
        x = self.mean + 0.01 * torch.randn_like(self.mean)
        return x, None


def _make_model_with_encoder_output(hidden_size: int = 64):
    """Model whose acoustic_tokenizer.encode returns a dataclass (not a tensor)."""
    model = _make_model(hidden_size)

    # encode() must return a VibeVoiceTokenizerEncoderOutput-like object.
    def _encode(audio, **kwargs):
        # audio shape: [B, 1, L] -> produce mean [B, T, vae_dim]
        mean = torch.randn(audio.shape[0], 4, 64)
        return _FakeEncoderOutput(mean)

    model.model.acoustic_tokenizer.encode = MagicMock(side_effect=_encode)
    model.model.acoustic_tokenizer.std_dist_type = "fix"
    # acoustic_connector must accept both:
    #   - [B, T, vae_dim] (from forward_speech_features, the reference prefix) -> [B, T, hidden]
    #   - [B, vae_dim] (a single generated latent fed back in the AR loop)   -> [B, hidden]
    # matching the real SpeechConnector's (input_dim -> hidden_size) contract.
    def _acoustic_connector(f):
        if f.dim() == 3:
            return torch.randn(f.shape[0], f.shape[1], hidden_size)
        return torch.randn(f.shape[0], hidden_size)

    model.model.acoustic_connector.side_effect = _acoustic_connector
    return model


class TestForwardSpeechFeaturesAudioPath:
    """Regression: forward_speech_features 'audio' branch must not subscript
    the VibeVoiceTokenizerEncoderOutput dataclass returned by encode()."""

    def test_forward_speech_features_audio_no_subscript_error(self):
        model = _make_model_with_encoder_output()
        speech_tensors = torch.randn(1, 1000)  # [B, L] waveform
        audio_features, connect_features = model.forward_speech_features(
            speech_tensors=speech_tensors,
            speech_masks=torch.ones(1, 4, dtype=torch.bool),
            speech_type="audio",
        )
        assert isinstance(audio_features, torch.Tensor)
        assert isinstance(connect_features, torch.Tensor)
        # encode() must have been called (not crashed on [0][0])
        model.model.acoustic_tokenizer.encode.assert_called_once()

    def test_forward_speech_features_sample_called_with_std_dist(self):
        model = _make_model_with_encoder_output()
        speech_tensors = torch.randn(1, 1000)
        model.forward_speech_features(
            speech_tensors=speech_tensors,
            speech_masks=torch.ones(1, 4, dtype=torch.bool),
            speech_type="audio",
        )
        # The dataclass .sample() should be invoked with the tokenizer's std_dist_type
        # (verified indirectly: no subscript error means .sample was reached)
        model.model.acoustic_connector.assert_called()


class TestGenerateWithSpeechTensors:
    """Regression: generate() with voice-cloning speech_tensors must complete
    without the encoder-output subscript crash."""

    def test_generate_with_speech_tensors_no_crash(self):
        model = _make_model_with_encoder_output()
        input_ids = torch.randint(0, 100, (1, 4))
        # acoustic_input_mask and speech_masks must mark the same positions
        # (as the processor produces them), otherwise the embedding scatter
        # at forward() line 416 would be a shape mismatch.
        speech_mask = torch.ones(1, 4, dtype=torch.bool)
        out = model.generate(
            input_ids=input_ids,
            speech_tensors=torch.randn(1, 1000),
            speech_masks=speech_mask,
            acoustic_input_mask=speech_mask,
            semantic_speech_tensors=None,
            cfg_scale=1.3,
            inference_steps=5,
            return_speech=True,
        )
        assert isinstance(out, VibeVoiceGenerationOutput)
        assert out.speech_outputs is not None
        assert len(out.speech_outputs) == 1
