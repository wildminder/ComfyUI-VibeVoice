"""Tests for the ``progress_callback`` hook in the non-streaming
``VibeVoiceForConditionalGeneration.generate()`` (Phase 1 of the
2026-08-15 inference-progress-reporting plan).

The real vendored module is loaded with the same stub-diffusers /
real-module pattern used by ``tests/test_model_forward.py``. A fully
scripted mock inner model drives the AR loop deterministically:

* ``lm_head`` emits a fixed control-token script
  ``[diffusion, diffusion, speech_end, eos]`` (greedy decoding),
* the inner ``language_model`` returns fixed hidden states,
* diffusion / tokenizer submodules are mocked with fixed-shape outputs.

Expected loop trace (seq_len=4, max_length_times=2 -> max_steps=8):

    callback(0, 8)                  # initial budget announcement
    step 1: diffusion  -> callback(1, 8)
    step 2: diffusion  -> callback(2, 8)
    step 3: speech_end -> callback(3, 8)
    step 4: eos        -> finished -> break (no step increment)
"""

import os
import sys
import types
import importlib.util
from unittest.mock import MagicMock

import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Real-module loading (mirrors tests/test_model_forward.py)
# ----------------------------------------------------------------------
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
    for mod in (
        "src.vibevoice",
        "src.vibevoice.modular",
        "src.vibevoice.modular.modeling_vibevoice",
        "ComfyUI_VibeVoice.src.vibevoice.modular.modeling_vibevoice",
    ):
        sys.modules.pop(mod, None)

    root = os.path.join(os.getcwd(), "src", "vibevoice")
    mod_pkg = types.ModuleType("src.vibevoice.modular")
    mod_pkg.__path__ = [os.path.join(root, "modular")]
    sys.modules.setdefault("src.vibevoice", types.ModuleType("src.vibevoice"))
    sys.modules["src.vibevoice"].__path__ = [root]
    sys.modules["src.vibevoice.modular"] = mod_pkg

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


_modeling = _load_real_modeling_module()
VibeVoiceForConditionalGeneration = _modeling.VibeVoiceForConditionalGeneration
VibeVoiceGenerationOutput = _modeling.VibeVoiceGenerationOutput


# ----------------------------------------------------------------------
# Scripted mock model
# ----------------------------------------------------------------------
VOCAB = 100
HIDDEN = 64
SEQ_LEN = 4

# Control-token ids used by the scripted lm_head.
START_ID = 1
END_ID = 2
DIFFUSION_ID = 3
EOS_ID = 4
BOS_ID = 5

# The control-token script emitted by the scripted lm_head, one per AR step.
TOKEN_SCRIPT = [DIFFUSION_ID, DIFFUSION_ID, END_ID, EOS_ID]
EXPECTED_COMPLETED_STEPS = 3  # eos step breaks before `step += 1`
EXPECTED_MAX_STEPS = 8  # min(max_new_tokens, max_length_times * seq_len) = min(1020, 2*4)


class _ScriptedLMHead(nn.Module):
    """``lm_head`` stand-in: real ``.weight`` (for ``.weight.device`` reads)
    but scripted logits so greedy decoding follows ``TOKEN_SCRIPT``."""

    def __init__(self, script):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(HIDDEN, VOCAB))
        self._script = list(script)
        self._call = 0

    def forward(self, x):
        idx = self._script[min(self._call, len(self._script) - 1)]
        self._call += 1
        logits = torch.full((x.shape[0], VOCAB), -10.0)
        logits[:, idx] = 10.0
        return logits


class _FakeTokenizer:
    speech_start_id = START_ID
    speech_end_id = END_ID
    speech_diffusion_id = DIFFUSION_ID
    eos_id = EOS_ID
    eos_token_id = EOS_ID
    bos_token_id = BOS_ID


class _StepResult:
    def __init__(self, prev_sample):
        self.prev_sample = prev_sample


def _make_scripted_model():
    """Build a VibeVoiceForConditionalGeneration whose AR loop is fully driven
    by mocks and follows ``TOKEN_SCRIPT`` deterministically."""
    model = VibeVoiceForConditionalGeneration.__new__(VibeVoiceForConditionalGeneration)
    torch.nn.Module.__init__(model)

    model.config = MagicMock()
    model.config.use_return_dict = True
    model.config.diffusion_head_config = MagicMock()
    model.config.diffusion_head_config.ddpm_num_inference_steps = 5
    model.config.acoustic_tokenizer_config = MagicMock()
    model.config.acoustic_tokenizer_config.vae_dim = 64
    model.config.acoustic_vae_dim = 64
    model.config.decoder_config = MagicMock()
    model.config.decoder_config.max_position_embeddings = 1024
    model.vocab_size = VOCAB

    inner = MagicMock()
    inner.speech_scaling_factor = torch.tensor(float("nan"))
    inner.speech_bias_factor = torch.tensor(float("nan"))

    # Diffusion scheduler: one timestep, fixed prev_sample.
    inner.noise_scheduler = MagicMock()
    inner.noise_scheduler.timesteps = [torch.tensor(999)]
    inner.noise_scheduler.set_timesteps = MagicMock()
    inner.noise_scheduler.step = MagicMock(
        return_value=_StepResult(torch.zeros(2, 64))
    )

    # Prediction head: fixed zero noise (CFG combination stays zero).
    def _pred_head(noisy, timesteps, condition):
        return torch.zeros(condition.shape[0], 64)

    inner.prediction_head = MagicMock(side_effect=_pred_head)
    inner.prediction_head.device = torch.device("cpu")

    # Inner language model: fixed hidden states of the right length.
    def _lm_forward(inputs_embeds=None, **kwargs):
        x = inputs_embeds
        if not isinstance(x, torch.Tensor):
            x = torch.zeros(1, SEQ_LEN, HIDDEN)
        result = MagicMock()
        result.last_hidden_state = torch.zeros(x.shape[0], x.shape[1], HIDDEN)
        result.past_key_values = None
        return result

    inner.language_model = MagicMock(side_effect=_lm_forward)

    # Acoustic tokenizer: deterministic waveform chunk (Nd, 1, T).
    inner.acoustic_tokenizer = MagicMock()
    inner.acoustic_tokenizer.device = torch.device("cpu")
    inner.acoustic_tokenizer.decode = MagicMock(
        return_value=torch.full((1, 1, 1600), 0.25)
    )

    # Semantic tokenizer: encode -> object with .mean (Nd, sem_dim, T_sem).
    def _sem_encode(audio, **kwargs):
        nd = audio.shape[0]
        out = MagicMock()
        out.mean = torch.zeros(nd, 16, 4)
        return out

    inner.semantic_tokenizer = MagicMock()
    inner.semantic_tokenizer.encode = MagicMock(side_effect=_sem_encode)

    # Connectors: (Nd, H) outputs.
    def _acoustic_connector(f):
        if f.dim() == 3:
            return torch.zeros(f.shape[0], f.shape[1], HIDDEN)
        return torch.zeros(f.shape[0], HIDDEN)

    def _semantic_connector(f):
        return torch.zeros(f.shape[0], HIDDEN)

    inner.acoustic_connector = MagicMock(side_effect=_acoustic_connector)
    inner.semantic_connector = MagicMock(side_effect=_semantic_connector)
    model.model = inner

    model.get_input_embeddings = MagicMock(return_value=nn.Embedding(VOCAB, HIDDEN))
    model.add_module("lm_head", _ScriptedLMHead(TOKEN_SCRIPT))
    return model


def _generate(model, progress_callback=None, **overrides):
    """Run generate() with the standard scripted inputs."""
    kwargs = dict(
        input_ids=torch.randint(0, VOCAB, (1, SEQ_LEN)),
        attention_mask=torch.ones(1, SEQ_LEN, dtype=torch.long),
        acoustic_input_mask=torch.zeros(1, SEQ_LEN, dtype=torch.bool),
        cfg_scale=1.3,
        inference_steps=5,
        return_speech=True,
        do_sample=False,
        tokenizer=_FakeTokenizer(),
    )
    kwargs.update(overrides)
    if progress_callback is not None:
        kwargs["progress_callback"] = progress_callback
    return model.generate(**kwargs)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
class TestProgressCallbackContract:
    def test_callback_none_backward_compatible(self):
        """T1.1: generate() without progress_callback runs unchanged."""
        model = _make_scripted_model()
        out = _generate(model)
        assert isinstance(out, VibeVoiceGenerationOutput)
        assert out.speech_outputs is not None
        assert len(out.speech_outputs) == 1
        assert isinstance(out.speech_outputs[0], torch.Tensor)

    def test_callback_sequence_monotonic_and_bounded(self):
        """T1.2: first call is (0, max_steps); currents monotonic, <= total."""
        model = _make_scripted_model()
        calls = []
        _generate(model, progress_callback=lambda c, t: calls.append((c, t)))

        assert calls, "progress_callback was never invoked"
        assert calls[0] == (0, EXPECTED_MAX_STEPS)
        currents = [c for c, _ in calls]
        totals = [t for _, t in calls]
        assert currents == sorted(currents), "current must be monotonic non-decreasing"
        assert all(0 <= c <= EXPECTED_MAX_STEPS for c in currents)
        assert set(totals) == {EXPECTED_MAX_STEPS}, "total must stay constant"

    def test_callback_count_matches_completed_steps(self):
        """T1.3: exactly one initial call + one call per completed AR step."""
        model = _make_scripted_model()
        calls = []
        _generate(model, progress_callback=lambda c, t: calls.append((c, t)))

        # 1 initial + EXPECTED_COMPLETED_STEPS per-step calls.
        assert len(calls) == 1 + EXPECTED_COMPLETED_STEPS
        assert [c for c, _ in calls] == [0, 1, 2, 3]

    def test_callback_exception_propagates(self):
        """T1.4: a raising callback interrupts generation (exception propagates)."""
        model = _make_scripted_model()

        def _raiser(current, total):
            if current >= 2:
                raise RuntimeError("interrupt requested")

        try:
            _generate(model, progress_callback=_raiser)
            raised = False
        except RuntimeError as e:
            raised = "interrupt requested" in str(e)
        assert raised, "callback exception must propagate out of generate()"

    def test_output_identical_with_and_without_callback(self):
        """T1.5: progress plumbing must not change generated audio."""
        model_a = _make_scripted_model()
        out_a = _generate(model_a)

        model_b = _make_scripted_model()
        out_b = _generate(model_b, progress_callback=lambda c, t: None)

        assert torch.equal(out_a.speech_outputs[0], out_b.speech_outputs[0])
