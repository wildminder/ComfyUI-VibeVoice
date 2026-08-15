"""Tests for the ``progress_callback`` hook in the streaming
``VibeVoiceStreamingForConditionalGenerationInference.generate()`` (Phase 3 of
the 2026-08-15 inference-progress-reporting plan).

The real vendored module is loaded with heavy sibling modules stubbed. The
GenerationMixin plumbing (``_build_generate_config_model_kwargs``,
``prepare_inputs_for_generation``, ``forward_lm`` / ``forward_tts_lm``,
``sample_speech_tokens``) is mocked so the REAL windowed AR loop runs
deterministically:

* one text window of ``TTS_TEXT_WINDOW_SIZE`` (5) tokens -> 1 callback,
* ``TTS_SPEECH_WINDOW_SIZE`` (6) speech tokens -> 6 callbacks
  (the inner speech loop does not break on EOS; EOS only stops the outer
  while-loop on the next iteration),
* ``tts_eos_classifier`` returns a high logit so the outer loop stops after
  the first text window.

Expected callback trace (initial step = tts_lm_input_ids len = 3,
max_length = 100):

    (8, 100)                       # text window prefill (3 + 5)
    (9, 100) ... (14, 100)         # six speech tokens
"""

import os
import sys
import types
import importlib.util
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from transformers.modeling_utils import PreTrainedModel


HIDDEN = 64
MAX_LENGTH = 100
TEXT_LEN = 5          # one full text window
TTS_LM_INIT_LEN = 3   # initial tts_lm_input_ids length -> initial `step`
EXPECTED_TEXT_CALLBACK_STEP = TTS_LM_INIT_LEN + TEXT_LEN  # 8
EXPECTED_SPEECH_CALLBACKS = 6  # TTS_SPEECH_WINDOW_SIZE


# ----------------------------------------------------------------------
# Stubs + real-module loading
# ----------------------------------------------------------------------
class _FakeStreamingConfig:
    __name__ = "VibeVoiceStreamingConfig"
    model_type = "vibevoice_streaming"


class _FakeStreamingPreTrainedModel(PreTrainedModel):
    """Real PreTrainedModel subclass so the inference class can inherit from it."""

    config_class = _FakeStreamingConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        pass


def _install_stub_modules():
    """Replace the conftest MagicMocks for the streaming module's imports with
    stubs that provide real classes where inheritance/instantiation requires
    them."""
    streaming_pkg_root = os.path.join(os.getcwd(), "src", "vibevoice")

    # Package hierarchy (needed for relative imports during exec).
    if "src.vibevoice" not in sys.modules or not hasattr(sys.modules["src.vibevoice"], "__path__"):
        pkg = types.ModuleType("src.vibevoice")
        pkg.__path__ = [streaming_pkg_root]
        sys.modules["src.vibevoice"] = pkg
    else:
        sys.modules["src.vibevoice"].__path__ = [streaming_pkg_root]

    mod_pkg_name = "src.vibevoice.modular"
    if mod_pkg_name not in sys.modules or not hasattr(sys.modules[mod_pkg_name], "__path__"):
        mod_pkg = types.ModuleType(mod_pkg_name)
        mod_pkg.__path__ = [os.path.join(streaming_pkg_root, "modular")]
        sys.modules[mod_pkg_name] = mod_pkg
    else:
        sys.modules[mod_pkg_name].__path__ = [os.path.join(streaming_pkg_root, "modular")]

    # modeling_vibevoice_streaming: must expose a REAL PreTrainedModel subclass.
    ms = types.ModuleType("src.vibevoice.modular.modeling_vibevoice_streaming")
    ms.VibeVoiceStreamingPreTrainedModel = _FakeStreamingPreTrainedModel
    ms.VibeVoiceStreamingModel = MagicMock()
    ms.BinaryClassifier = MagicMock()
    sys.modules["src.vibevoice.modular.modeling_vibevoice_streaming"] = ms

    # configuration_vibevoice_streaming: plain class with model_type (register()).
    cfg = types.ModuleType("src.vibevoice.modular.configuration_vibevoice_streaming")
    cfg.VibeVoiceStreamingConfig = _FakeStreamingConfig
    sys.modules["src.vibevoice.modular.configuration_vibevoice_streaming"] = cfg

    # The remaining imports survive as conftest MagicMocks (names only), but
    # ensure they exist so `from .x import Y` resolves.
    for name in (
        "src.vibevoice.modular.modular_vibevoice_tokenizer",
        "src.vibevoice.modular.modular_vibevoice_diffusion_head",
        "src.vibevoice.modular.modular_vibevoice_text_tokenizer",
        "src.vibevoice.modular.streamer",
        "src.vibevoice.schedule.dpm_solver",
    ):
        sys.modules.setdefault(name, MagicMock())


def _load_real_streaming_module():
    _install_stub_modules()
    sys.modules.pop("src.vibevoice.modular.modeling_vibevoice_streaming_inference", None)

    spec = importlib.util.spec_from_file_location(
        "src.vibevoice.modular.modeling_vibevoice_streaming_inference",
        "src/vibevoice/modular/modeling_vibevoice_streaming_inference.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["src.vibevoice.modular.modeling_vibevoice_streaming_inference"] = module
    spec.loader.exec_module(module)
    return module


_streaming = _load_real_streaming_module()
VibeVoiceStreamingForConditionalGenerationInference = (
    _streaming.VibeVoiceStreamingForConditionalGenerationInference
)
VibeVoiceGenerationOutput = _streaming.VibeVoiceGenerationOutput


# ----------------------------------------------------------------------
# Scripted mock model
# ----------------------------------------------------------------------
class _FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    speech_start_id = 3
    speech_end_id = 4
    speech_diffusion_id = 5

    def convert_tokens_to_ids(self, token):
        return 7


class _FakeForwardOutput:
    def __init__(self, seq_len: int = 4):
        self.last_hidden_state = torch.zeros(1, seq_len, HIDDEN)
        self.past_key_values = None


def _make_streaming_model():
    cls = VibeVoiceStreamingForConditionalGenerationInference
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    # A real parameter so `self.device` resolves (PreTrainedModel.device).
    model.register_parameter("_dummy", nn.Parameter(torch.zeros(1)))

    model.config = MagicMock()
    model.config.acoustic_vae_dim = HIDDEN
    model.config.decoder_config = MagicMock()
    model.config.decoder_config.max_position_embeddings = 1024
    model.config.decoder_config.hidden_size = HIDDEN
    model.config.diffusion_head_config = MagicMock()
    model.config.diffusion_head_config.ddpm_num_inference_steps = 5
    model.ddpm_inference_steps = 5

    inner = MagicMock()
    inner.speech_scaling_factor = torch.tensor(1.0)
    inner.speech_bias_factor = torch.tensor(0.0)
    inner.acoustic_tokenizer = MagicMock()
    inner.acoustic_tokenizer.device = torch.device("cpu")
    inner.acoustic_tokenizer.decode = MagicMock(return_value=torch.full((1, 1600), 0.25))
    inner.acoustic_connector = MagicMock(return_value=torch.zeros(1, 1, HIDDEN))
    model.model = inner

    # EOS classifier: high logit -> sigmoid > 0.5 -> finished after window 1.
    model.tts_eos_classifier = MagicMock(return_value=torch.tensor([[10.0]]))

    # --- GenerationMixin plumbing mocks -------------------------------
    def _build_cfg(generation_config, inputs, tokenizer, return_processors=False, **kw):
        cfg = SimpleNamespace(max_length=MAX_LENGTH, min_length=0)
        ids = kw.get("input_ids")
        if ids is None:
            ids = torch.zeros(1, 1, dtype=torch.long)
        mk = {
            "input_ids": ids,
            "attention_mask": torch.ones(1, ids.shape[1], dtype=torch.long),
            "cache_position": torch.arange(ids.shape[1], dtype=torch.long),
            "past_key_values": None,
            "use_cache": True,
        }
        if return_processors:
            return cfg, mk, ids, [], []
        return cfg, mk, ids

    model._build_generate_config_model_kwargs = MagicMock(side_effect=_build_cfg)
    model.prepare_inputs_for_generation = MagicMock(return_value={})
    model.forward_lm = MagicMock(return_value=_FakeForwardOutput())
    model.forward_tts_lm = MagicMock(return_value=_FakeForwardOutput())
    model.sample_speech_tokens = MagicMock(return_value=torch.zeros(1, HIDDEN))
    # Bypass the GenerationMixin cache bookkeeping (module-level
    # _update_model_kwargs_for_generation still runs with real tensors).
    model._update_model_kwargs_for_generation = (
        lambda outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1: model_kwargs
    )
    return model


def _run_generate(model, progress_callback=None):
    kwargs = dict(
        input_ids=torch.randint(0, 100, (1, 4)),
        tts_text_ids=torch.randint(0, 100, (1, TEXT_LEN)),
        tts_lm_input_ids=torch.randint(0, 100, (1, TTS_LM_INIT_LEN)),
        tokenizer=_FakeTokenizer(),
        all_prefilled_outputs={
            "lm": _FakeForwardOutput(),
            "tts_lm": _FakeForwardOutput(),
            "neg_lm": _FakeForwardOutput(),
            "neg_tts_lm": _FakeForwardOutput(),
        },
        max_new_tokens=50,
        cfg_scale=1.0,
        return_speech=True,
        show_progress_bar=False,  # silence the console tqdm bar
    )
    if progress_callback is not None:
        kwargs["progress_callback"] = progress_callback
    return model.generate(**kwargs)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
class TestStreamingProgressCallbackContract:
    def test_callback_none_backward_compatible(self):
        """T3.1: generate() without progress_callback runs unchanged."""
        model = _make_streaming_model()
        out = _run_generate(model)
        assert isinstance(out, VibeVoiceGenerationOutput)
        assert out.speech_outputs is not None
        assert len(out.speech_outputs) == 1
        assert isinstance(out.speech_outputs[0], torch.Tensor)

    def test_callback_monotonic_with_constant_total(self):
        """T3.2: currents monotonic non-decreasing; total == max_length always."""
        model = _make_streaming_model()
        calls = []
        _run_generate(model, progress_callback=lambda c, t: calls.append((c, t)))

        assert calls, "progress_callback was never invoked"
        currents = [c for c, _ in calls]
        totals = [t for _, t in calls]
        assert currents == sorted(currents)
        assert set(totals) == {MAX_LENGTH}
        assert all(0 <= c <= MAX_LENGTH for c in currents)

    def test_callback_fires_for_text_window_and_speech_tokens(self):
        """T3.3: one callback after the text-window prefill and one per
        generated speech token (TTS_SPEECH_WINDOW_SIZE = 6)."""
        model = _make_streaming_model()
        calls = []
        _run_generate(model, progress_callback=lambda c, t: calls.append((c, t)))

        assert len(calls) == 1 + EXPECTED_SPEECH_CALLBACKS, f"got {calls}"
        # First call: text window prefill advanced step by TEXT_LEN.
        assert calls[0] == (EXPECTED_TEXT_CALLBACK_STEP, MAX_LENGTH)
        # Speech-token calls increment by exactly 1 each.
        speech_currents = [c for c, _ in calls[1:]]
        assert speech_currents == [
            EXPECTED_TEXT_CALLBACK_STEP + i + 1 for i in range(EXPECTED_SPEECH_CALLBACKS)
        ]

    def test_callback_exception_propagates(self):
        """T3.4: a raising callback interrupts generation."""
        model = _make_streaming_model()

        def _raiser(current, total):
            raise RuntimeError("interrupt requested")

        try:
            _run_generate(model, progress_callback=_raiser)
            raised = False
        except RuntimeError as e:
            raised = "interrupt requested" in str(e)
        assert raised, "callback exception must propagate out of generate()"
