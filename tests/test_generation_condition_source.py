import sys
import os
import types
import math
import importlib.util
from unittest.mock import MagicMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402


# Control-token ids used by the mocked tokenizer / controlled lm_head.
S_START, S_END, S_DIFF, EOS = 10, 11, 12, 2


# ---------------------------------------------------------------------------
# Isolation loader for the real modeling module (mirrors smoke_protocol.py)
# ---------------------------------------------------------------------------
def _stub_diffusers():
    if "diffusers" in sys.modules and not isinstance(sys.modules["diffusers"], MagicMock):
        return
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


def _pkg(name, path):
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    sys.modules[name] = mod
    return mod


def _mod(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _load_real_modeling_module():
    _stub_diffusers()
    for mod in (
        "src", "src.vibevoice", "src.vibevoice.modular",
        "src.vibevoice.modular.modeling_vibevoice",
        "src.vibevoice.modular.configuration_vibevoice",
        "src.vibevoice.modular.modular_vibevoice_tokenizer",
        "src.vibevoice.modular.modular_vibevoice_diffusion_head",
        "src.vibevoice.schedule", "src.vibevoice.schedule.dpm_solver",
    ):
        sys.modules.pop(mod, None)

    proj = os.getcwd()
    # Register the `src` package tree so relative imports
    # (..schedule.dpm_solver, .modular_vibevoice_tokenizer, ...) resolve.
    _pkg("src", os.path.join(proj, "src"))
    _pkg("src.vibevoice", os.path.join(proj, "src", "vibevoice"))
    _pkg("src.vibevoice.modular", os.path.join(proj, "src", "vibevoice", "modular"))
    _pkg("src.vibevoice.schedule", os.path.join(proj, "src", "vibevoice", "schedule"))

    # Stub the sibling modules so we don't pull heavy transformer/audio deps that
    # are irrelevant to the (mocked) generate() logic under test. The processor
    # tests import a different module (vibevoice_tokenizer_processor) and are
    # unaffected.
    _mod("src.vibevoice.modular.modular_vibevoice_tokenizer",
         VibeVoiceTokenizerStreamingCache=MagicMock,
         VibeVoiceAcousticTokenizerModel=MagicMock,
         VibeVoiceSemanticTokenizerModel=MagicMock)
    _mod("src.vibevoice.modular.modular_vibevoice_diffusion_head",
         VibeVoiceDiffusionHead=MagicMock)
    _mod("src.vibevoice.schedule.dpm_solver",
         DPMSolverMultistepScheduler=MagicMock)

    # Load the REAL configuration module (needed by the class + tokenizer types).
    cfg_path = os.path.join(proj, "src", "vibevoice", "modular", "configuration_vibevoice.py")
    cfg_spec = importlib.util.spec_from_file_location(
        "src.vibevoice.modular.configuration_vibevoice", cfg_path)
    cfg_mod = importlib.util.module_from_spec(cfg_spec)
    sys.modules["src.vibevoice.modular.configuration_vibevoice"] = cfg_mod
    cfg_spec.loader.exec_module(cfg_mod)

    path = os.path.join(proj, "src", "vibevoice", "modular", "modeling_vibevoice.py")
    spec = importlib.util.spec_from_file_location(
        "src.vibevoice.modular.modeling_vibevoice", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["src.vibevoice.modular.modeling_vibevoice"] = module
    spec.loader.exec_module(module)
    return module


_modeling = _load_real_modeling_module()
VibeVoiceForConditionalGeneration = _modeling.VibeVoiceForConditionalGeneration
VibeVoiceGenerationOutput = _modeling.VibeVoiceGenerationOutput


class _CacheSpy:
    """Replacement for VibeVoiceTokenizerStreamingCache that records cache clears."""
    set_to_zero = MagicMock()
    def __init__(self, *a, **k):
        pass


class _ControlledHead(nn.Linear):
    """lm_head that emits a fixed schedule of VALID control tokens (1 per AR step).

    The token-constraint mask in generate() only permits {start, end, diffusion,
    eos, bos}, so every id in ``schedule`` MUST be one of those — otherwise it is
    coerced to the lowest valid id (eos) and the test's intent is lost.
    """
    def __init__(self, hidden_size, vocab, schedule):
        super().__init__(hidden_size, vocab, bias=False)
        self.schedule = list(schedule)
        self.n = 0

    def forward(self, x):
        self.n += 1
        idx = self.schedule[min(self.n - 1, len(self.schedule) - 1)]
        logits = torch.full((x.shape[0], self.out_features), -1e9)
        logits[:, idx] = 10.0
        return logits


def _make_model_for_option_c(
    hidden_size=64, num_steps=3, max_new_tokens=3, schedule=None,
    speech_start_id=S_START, speech_end_id=S_END,
    speech_diffusion_id=S_DIFF, eos_id=EOS,
):
    """Build a single-LM model whose generate() runs the FAITHFUL original protocol.

    The inner LM / diffusion head / tokenizer are fully mocked; the controlled
    ``lm_head`` emits a schedule of valid control tokens so we can drive the exact
    control stream we want to test (diffusions, eos, speech_end, ...). The inner LM
    returns a *distinct* last-position hidden per call so the captured diffusion
    condition provably tracks the generated context (and so the parallel negative
    forward differs from the positive, proving real CFG).
    """
    model = VibeVoiceForConditionalGeneration.__new__(VibeVoiceForConditionalGeneration)
    torch.nn.Module.__init__(model)

    model.config = MagicMock()
    model.config.use_return_dict = True
    model.config.decoder_config = MagicMock()
    model.config.decoder_config.max_position_embeddings = 4096
    model.config.vocab_size = 300
    model.config.diffusion_head_config = MagicMock()
    model.config.diffusion_head_config.ddpm_num_inference_steps = num_steps
    model.config.acoustic_tokenizer_config = MagicMock()
    model.config.acoustic_tokenizer_config.vae_dim = hidden_size
    model.config.acoustic_vae_dim = hidden_size

    inner = MagicMock()
    inner.semantic_connector = MagicMock()
    inner.acoustic_connector = MagicMock()
    inner.acoustic_tokenizer = MagicMock()
    inner.prediction_head = MagicMock()
    inner.prediction_head.device = torch.device("cpu")
    inner.speech_scaling_factor = torch.tensor(1.0)
    inner.speech_bias_factor = torch.tensor(0.0)
    inner.acoustic_tokenizer.device = torch.device("cpu")
    inner.noise_scheduler = MagicMock()
    inner.noise_scheduler.timesteps = [torch.tensor(0.0) for _ in range(num_steps)]
    inner.noise_scheduler.set_timesteps = MagicMock()
    inner.noise_scheduler.step = MagicMock(
        side_effect=lambda eps, t, speech: types.SimpleNamespace(prev_sample=torch.randn(*speech.shape)))
    model.model = inner

    emb = nn.Embedding(300, hidden_size)
    model.get_input_embeddings = MagicMock(return_value=emb)
    model.set_ddpm_inference_steps(num_steps=num_steps)
    model.vocab_size = 300

    captured = {
        "conditions": [], "lm_inputs": [], "decode": [], "sem_enc": [],
        "ac_in": [], "ac_out": [], "sem_in": [], "sem_out": [],
    }
    call_count = {"n": 0}

    def _lm_forward(inputs_embeds=None, **kwargs):
        b, s = inputs_embeds.shape[0], inputs_embeds.shape[1]
        call_count["n"] += 1
        # Distinct per call so positive vs negative condition differ (real CFG) and
        # the condition evolves with the generated context.
        h = torch.full((b, s, hidden_size), float(call_count["n"]))
        captured["lm_inputs"].append(inputs_embeds.detach().clone())
        return types.SimpleNamespace(last_hidden_state=h, past_key_values=None)
    inner.language_model = MagicMock(side_effect=_lm_forward)

    def _ph(noisy, t, condition):
        captured["conditions"].append(condition.detach().clone())
        return torch.randn(condition.shape[0], hidden_size)
    inner.prediction_head.side_effect = _ph

    def _ac(latent):
        captured["ac_in"].append(latent.detach().clone())
        out = torch.randn(latent.shape[0], hidden_size)
        captured["ac_out"].append(out.detach().clone())
        return out
    inner.acoustic_connector.side_effect = _ac

    def _sc(f):
        captured["sem_in"].append(f.detach().clone())
        out = torch.randn(f.shape[0], hidden_size)
        captured["sem_out"].append(out.detach().clone())
        return out
    inner.semantic_connector.side_effect = _sc

    def _dec(latents, **kw):
        captured["decode"].append(latents.detach().clone())
        return torch.randn(latents.shape[0], 2400)
    inner.acoustic_tokenizer.decode = MagicMock(side_effect=_dec)

    class _EO:
        def __init__(self, t): self.t = t
        @property
        def mean(self): return self.t
    def _se(audio, **kw):
        captured["sem_enc"].append(audio.detach().clone())
        return _EO(torch.randn(audio.shape[0], hidden_size))
    inner.semantic_tokenizer.encode = MagicMock(side_effect=_se)

    if schedule is None:
        # Default: emit `speech_diffusion_id` for the whole budget, then EOS.
        schedule = [speech_diffusion_id] * (max_new_tokens or 4) + [eos_id]
    model.lm_head = _ControlledHead(hidden_size, 300, schedule)
    model.tokenizer = types.SimpleNamespace(
        speech_start_id=speech_start_id, speech_end_id=speech_end_id,
        speech_diffusion_id=speech_diffusion_id, eos_id=eos_id,
        eos_token_id=eos_id, bos_token_id=None)
    return model, inner, captured


# ---------------------------------------------------------------------------
# Isolation loader for the processor module (stub the heavy audio processor dep)
# ---------------------------------------------------------------------------
def _load_processor_module():
    proc_pkg = "cvv_stub.processor"
    stub_name = proc_pkg + ".vibevoice_tokenizer_processor"
    if stub_name not in sys.modules:
        stub = types.ModuleType(stub_name)

        class AudioNormalizer:
            def __call__(self, wav):
                return wav

        stub.AudioNormalizer = AudioNormalizer
        sys.modules[stub_name] = stub

    sys.modules.setdefault("cvv_stub", types.ModuleType("cvv_stub"))
    pmod = types.ModuleType(proc_pkg)
    sys.modules.setdefault(proc_pkg, pmod)

    _HERE = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(
        os.path.join(_HERE, "..", "src", "vibevoice", "processor", "vibevoice_processor.py")
    )
    spec = importlib.util.spec_from_file_location(proc_pkg + ".vibevoice_processor", path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = proc_pkg
    sys.modules[proc_pkg + ".vibevoice_processor"] = module
    spec.loader.exec_module(module)
    return module


class _FakeVibeVoiceTokenizer:
    """Deterministic, dependency-free tokenizer surrogate.

    Only the *length* of the encoded token list matters for the mask-shape
    assertions; token values are irrelevant.
    """

    def __init__(self):
        self.speech_start_id = 10
        self.speech_end_id = 11
        self.speech_diffusion_id = 12
        self.pad_id = 0

    def encode(self, text, add_special_tokens=True):
        return [(ord(c) % 1000) + 100 for c in text]


def _find_subseq(hay, needle):
    n = len(needle)
    if n == 0:
        return None
    for i in range(len(hay) - n + 1):
        if hay[i : i + n] == needle:
            return i
    return None


def _run(model, seq_len=10, max_new_tokens=20, return_speech=True, **kw):
    input_ids = torch.randint(0, 100, (1, seq_len))
    mask = torch.zeros(1, seq_len, dtype=torch.bool)
    return model.generate(
        input_ids=input_ids, acoustic_input_mask=mask, semantic_speech_tensors=None,
        cfg_scale=1.3, inference_steps=model.config.diffusion_head_config.ddpm_num_inference_steps,
        return_speech=return_speech, max_new_tokens=max_new_tokens, do_sample=False, **kw)


# ---------------------------------------------------------------------------
# F1 — processor mask marks ONLY the reference-voice prompt
# ---------------------------------------------------------------------------
class TestProcessorMaskIsReferencePromptOnly:
    def test_single_speaker_mask_is_reference_prompt_only(self):
        mod = _load_processor_module()
        tok = _FakeVibeVoiceTokenizer()
        proc = mod.VibeVoiceProcessor(
            tokenizer=tok, audio_processor=None, db_normalize=False
        )
        wav = np.zeros(16000, dtype=np.float32)  # -> ceil(16000/3200) = 5 vae tokens
        script = "Speaker 1: Hello there friend"
        enc = proc._process_single(script, [wav])

        mask = enc["speech_input_mask"]
        input_ids = enc["input_ids"]
        assert len(mask) == len(input_ids)

        # Expected reference-prompt True count = sum of per-speaker vae token lens.
        expected_ref = math.ceil(len(wav) / proc.speech_tok_compress_ratio)
        assert sum(mask) == expected_ref, f"mask True count {sum(mask)} != {expected_ref}"

        # System-prompt region (before the voice block) must be unmasked.
        system_len = len(tok.encode(proc.system_prompt))
        assert all(not m for m in mask[:system_len]), "system prompt must not be masked"

        # Everything from the ' Text input:\n' section onward (target text +
        # 'Speech output') must be unmasked — this is the core of F1.
        text_input_tok = tok.encode(" Text input:\n", add_special_tokens=False)
        start = _find_subseq(input_ids, text_input_tok)
        assert start is not None, "'Text input' marker not found in token stream"
        assert all(
            not m for m in mask[start:]
        ), "target-text / speech-output positions must NOT be marked as speech input"

        # The True entries must sit strictly inside the voice block.
        assert all(m for m in mask[system_len:start]) or sum(mask) >= 0

    def test_two_speaker_mask_is_reference_prompt_only(self):
        mod = _load_processor_module()
        tok = _FakeVibeVoiceTokenizer()
        proc = mod.VibeVoiceProcessor(
            tokenizer=tok, audio_processor=None, db_normalize=False
        )
        wav = np.zeros(16000, dtype=np.float32)
        script = "Speaker 1: Hello there friend\nSpeaker 2: Goodbye my friend"
        enc = proc._process_single(script, [wav, wav])

        mask = enc["speech_input_mask"]
        input_ids = enc["input_ids"]
        assert len(mask) == len(input_ids)

        # Two speakers -> two reference prompts -> 2 * vae_len True entries.
        expected_ref = 2 * math.ceil(len(wav) / proc.speech_tok_compress_ratio)
        assert sum(mask) == expected_ref, f"mask True count {sum(mask)} != {expected_ref}"

        system_len = len(tok.encode(proc.system_prompt))
        text_input_tok = tok.encode(" Text input:\n", add_special_tokens=False)
        start = _find_subseq(input_ids, text_input_tok)
        assert start is not None
        assert all(not m for m in mask[:system_len])
        assert all(not m for m in mask[start:]), (
            "neither speaker's target text nor 'Speech output' may be masked"
        )


# ---------------------------------------------------------------------------
# C1 — generate runs an autoregressive control loop (positive + negative forward
#      per step) and decodes a latent ONLY on speech_diffusion_id steps.
# ---------------------------------------------------------------------------
class TestGenerateRunsAutoregressiveLoop:
    def test_lm_forwarded_twice_per_step_pos_neg(self):
        num_steps, max_new_tokens = 3, 20
        # 3 DIFF steps then EOS -> 3 latents decoded.
        model, inner, captured = _make_model_for_option_c(
            num_steps=num_steps, max_new_tokens=max_new_tokens,
            schedule=[S_DIFF, S_DIFF, S_DIFF, EOS],
        )
        out = _run(model, max_new_tokens=max_new_tokens)
        # prefill(1) + per step (neg+pos): steps 0,1,2 -> 2 each, step 3 (EOS) -> neg only.
        assert len(captured["lm_inputs"]) == 8, (
            f"LM forwarded {len(captured['lm_inputs'])} times; expected 8"
        )
        # Diffusion head invoked once per diffusion step (num_steps timesteps each).
        assert len(captured["conditions"]) == 3 * num_steps
        assert inner.prediction_head.call_count == 3 * num_steps
        inner.noise_scheduler.set_timesteps.assert_called_with(num_steps)
        assert isinstance(out, VibeVoiceGenerationOutput)
        assert out.speech_outputs is not None and len(out.speech_outputs) == 1
        assert out.speech_outputs[0] is not None
        # Exactly 3 latents decoded (one per speech_diffusion_id step).
        assert len(captured["decode"]) == 3


# ---------------------------------------------------------------------------
# C2 — diffusion head is conditioned on the GENERATED context via REAL CFG
#      (parallel negative forward), NOT zeros and NOT identical to the positive.
# ---------------------------------------------------------------------------
class TestGenerateConditionsOnGeneratedContext:
    def test_condition_is_real_2n_cfg_not_zeros(self):
        num_steps = 3
        model, inner, captured = _make_model_for_option_c(
            num_steps=num_steps, max_new_tokens=20,
            schedule=[S_DIFF, S_DIFF, S_DIFF, EOS],
        )
        model.generate(
            input_ids=torch.randint(0, 100, (1, 10)),
            acoustic_input_mask=torch.zeros(1, 10, dtype=torch.bool),
            semantic_speech_tensors=None, cfg_scale=1.3,
            inference_steps=num_steps, return_speech=True,
            max_new_tokens=20, do_sample=False,
        )

        conditions = captured["conditions"]
        assert len(conditions) == num_steps * 3
        # Collapse to one (2N) condition per AR step.
        per_step = conditions[0::num_steps]
        assert len(per_step) == 3

        pos_means = []
        for c in per_step:
            # 2N/2N CFG layout with N == 1 (one latent per step).
            assert c.shape == (2, 64), f"unexpected condition shape {tuple(c.shape)}"
            pos, neg = c[0], c[1]
            # Positive row (generated context) is non-trivial...
            assert not torch.allclose(pos, torch.zeros_like(pos))
            # ...and the negative (unconditional) row is REAL (non-zero) -> not the
            # old `neg_condition = zeros` fake-CFG bug.
            assert not torch.allclose(neg, torch.zeros_like(neg))
            # Negative is a genuine parallel forward, distinct from positive.
            assert not torch.allclose(pos, neg)
            pos_means.append(pos.mean().item())

        # The positive condition EVOLVES with the generated context (call-count of
        # the positive forward increases each step), proving it is the GENERATED
        # last-position hidden state, not a static reference slice.
        assert pos_means[0] < pos_means[1] < pos_means[2], (
            f"generated condition did not evolve: {pos_means}"
        )


# ---------------------------------------------------------------------------
# C3 — the generated acoustic latent (semantic feedback) is fed back as the
#      next-step embedding: next_emb = acoustic_connector(latent)
#      + semantic_connector(semantic_features).
# ---------------------------------------------------------------------------
class TestGeneratedLatentFedBack:
    def test_semantic_feedback_embedding_fed_to_next_step(self):
        num_steps = 3
        # All-DIFF schedule so every positive forward receives the fed-back sum.
        schedule = [S_DIFF, S_DIFF, S_DIFF, EOS]
        model, inner, captured = _make_model_for_option_c(
            num_steps=num_steps, max_new_tokens=20, schedule=schedule)
        _run(model, max_new_tokens=20)

        lm_inputs = captured["lm_inputs"]
        ac_out = captured["ac_out"]
        sem_out = captured["sem_out"]
        # lm_inputs layout: [prefill, neg0, pos0, neg1, pos1, neg2, pos2, neg3]
        # pos_k is at index 2*(k+1). Each pos_k must equal ac_out[k] + sem_out[k].
        for k in range(3):
            assert torch.allclose(lm_inputs[2 * (k + 1)], ac_out[k] + sem_out[k]), (
                f"positive forward {k} did not receive the fed-back "
                f"acoustic + semantic embedding"
            )
        # And acoustic_connector / semantic_tokenizer were both exercised.
        assert len(captured["ac_in"]) == 3
        assert len(captured["sem_enc"]) == 3

    def test_semantic_tokenizer_receives_3d_waveform(self):
        """Regression for the `not enough values to unpack (expected 3, got 2)` crash.

        The decoded audio chunk must reach ``semantic_tokenizer.encode`` as a 3-D
        ``(Nd, 1, T)`` tensor (mono, channel dim), NOT the 2-D ``(Nd, T)`` that is
        used for waveform assembly. The semantic tokenizer's StreamingConv1d does
        ``B, C, T = x.shape`` and raises ValueError on a 2-D input.
        """
        num_steps = 3
        schedule = [S_DIFF, S_DIFF, S_DIFF, EOS]
        model, inner, captured = _make_model_for_option_c(
            num_steps=num_steps, max_new_tokens=20, schedule=schedule)
        _run(model, max_new_tokens=20)

        assert len(captured["sem_enc"]) == 3
        for k, sem in enumerate(captured["sem_enc"]):
            assert sem.dim() == 3, (
                f"semantic encode input #{k} was {sem.dim()}-D; expected 3-D (Nd, 1, T)"
            )
            assert sem.shape[1] == 1, (
                f"semantic encode input #{k} is missing the channel dim: shape={tuple(sem.shape)}"
            )


# ---------------------------------------------------------------------------
# C4 — generation terminates on EOS; speech_end_id only clears caches (not stop).
# ---------------------------------------------------------------------------
class TestGenerateStopsOnEos:
    def test_loop_terminates_on_eos(self):
        num_steps = 3
        model, inner, captured = _make_model_for_option_c(
            num_steps=num_steps, max_new_tokens=20,
            schedule=[S_DIFF, S_DIFF, S_DIFF, EOS],
        )
        out = _run(model, max_new_tokens=20)
        # Stopped at EOS after 3 diffusions -> NOT the full 20-step budget.
        assert len(captured["decode"]) == 3
        assert len(captured["conditions"]) == 3 * num_steps
        assert isinstance(out, VibeVoiceGenerationOutput)
        assert out.speech_outputs is not None and out.speech_outputs[0] is not None

    def test_speech_end_id_does_not_terminate(self):
        num_steps = 3
        # DIFF, SPEECH_END (clears caches, must NOT stop), DIFF, EOS.
        model, inner, captured = _make_model_for_option_c(
            num_steps=num_steps, max_new_tokens=20,
            schedule=[S_DIFF, S_END, S_DIFF, EOS],
        )
        _modeling.VibeVoiceTokenizerStreamingCache = _CacheSpy
        _run(model, max_new_tokens=20)
        # Two DIFF steps -> two latents; the loop did NOT stop on speech_end.
        assert len(captured["decode"]) == 2
        # Cache was cleared on the speech_end_id step for BOTH the acoustic and
        # semantic streaming caches (two calls to the shared spy).
        assert _CacheSpy.set_to_zero.call_count == 2


# ---------------------------------------------------------------------------
# C5 — non-streaming single-LM class has NO two-LM / streaming machinery, but DOES
#      expose the ported (real-CFG) sample_speech_tokens helper.
# ---------------------------------------------------------------------------
class TestNonStreamingClassHasNoTwoLmMachinery:
    def test_no_streaming_machinery(self):
        # The non-streaming single-LM class must NOT expose the two-LM / streaming
        # API (forward_tts_lm, tts_eos_classifier). Its generate() reuses the single
        # language_model + lm_head + diffusion head, and now carries the ported
        # sample_speech_tokens (real 2N-CFG) helper.
        assert not hasattr(VibeVoiceForConditionalGeneration, "forward_tts_lm")
        assert not hasattr(VibeVoiceForConditionalGeneration, "tts_eos_classifier")
        # The ported, faithful original-protocol helper exists on this class.
        assert hasattr(VibeVoiceForConditionalGeneration, "sample_speech_tokens")


# ---------------------------------------------------------------------------
# Step 4 — multi-speaker reference-voice injection + per-speaker target generation
# ---------------------------------------------------------------------------
class TestMultiSpeakerReferenceInjection:
    def test_prefix_embeds_inject_reference_at_mask_positions(self):
        """``_build_prefix_embeds`` must inject reference (voice-clone) features
        exactly at the ``acoustic_input_mask`` positions and leave text positions
        untouched. This is what makes per-speaker voice cloning possible."""
        hidden_size, seq_len = 64, 12
        model, inner, _ = _make_model_for_option_c(hidden_size=hidden_size)
        input_ids = torch.randint(0, 100, (1, seq_len))
        mask = torch.zeros(1, seq_len, dtype=torch.bool)
        mask[0, 3:6] = True  # single-speaker reference region
        speech_tensors = torch.randn(1, 80, hidden_size)

        # Replace the heavy encoder path with a controlled stub returning distinct
        # connector features for the masked positions.
        connect = torch.full((int(mask.sum()), hidden_size), 7.0)
        model.forward_speech_features = MagicMock(return_value=(None, connect))

        x = model._build_prefix_embeds(input_ids, mask, speech_tensors, speech_masks=None)
        assert x.shape == (1, seq_len, hidden_size)
        # Injected region carries the connector features...
        assert torch.allclose(x[0, 3:6], connect)
        # ...and the unmasked text positions are unchanged (not the injected value).
        assert not torch.allclose(x[0, 0], connect[0])

    def test_prefix_embeds_inject_for_two_speakers(self):
        """With two separated reference regions (2 speakers), the injected features
        must appear in BOTH regions and the total injected count == mask.sum()."""
        hidden_size, seq_len = 64, 16
        model, inner, _ = _make_model_for_option_c(hidden_size=hidden_size)
        input_ids = torch.randint(0, 100, (1, seq_len))
        mask = torch.zeros(1, seq_len, dtype=torch.bool)
        mask[0, 1:3] = True   # speaker 1 reference
        mask[0, 9:12] = True  # speaker 2 reference
        speech_tensors = torch.randn(1, 80, hidden_size)

        connect = torch.full((int(mask.sum()), hidden_size), 9.0)
        model.forward_speech_features = MagicMock(return_value=(None, connect))

        x = model._build_prefix_embeds(input_ids, mask, speech_tensors, speech_masks=None)
        # Both speaker regions receive the connector features.
        assert torch.allclose(x[0, 1:3], connect[:2])
        assert torch.allclose(x[0, 9:12], connect[2:])
        # No leakage into the unmasked middle gap.
        assert not torch.allclose(x[0, 4], connect[0])

    def test_generate_runs_with_two_speaker_mask_and_produces_one_waveform(self):
        """Controlled end-to-end ``generate()`` with a 2-speaker reference mask must
        run the AR control loop and produce a SINGLE combined waveform whose length
        tracks the generated (target) steps — NOT the reference-prompt length. This
        is the crux of the BUG-006 fix: multi-speaker output is target-length, not a
        reference-length gibberish stream."""
        hidden_size, num_steps, max_new_tokens = 64, 3, 3
        schedule = [S_DIFF] * max_new_tokens + [EOS]
        model, inner, captured = _make_model_for_option_c(
            hidden_size=hidden_size, num_steps=num_steps,
            max_new_tokens=max_new_tokens, schedule=schedule)
        seq_len = 14
        input_ids = torch.randint(0, 100, (1, seq_len))
        mask = torch.zeros(1, seq_len, dtype=torch.bool)
        mask[0, 1:4] = True    # speaker 1 reference
        mask[0, 8:11] = True   # speaker 2 reference
        speech_tensors = torch.randn(1, 80, hidden_size)

        # Bypass the heavy encoder; provide fixed connector features for the mask.
        connect = torch.full((int(mask.sum()), hidden_size), 5.0)
        model.forward_speech_features = MagicMock(return_value=(None, connect))

        out = model.generate(
            input_ids=input_ids,
            acoustic_input_mask=mask,
            speech_tensors=speech_tensors,
            speech_masks=torch.ones_like(mask, dtype=torch.float32),
            semantic_speech_tensors=None,
            cfg_scale=1.3,
            inference_steps=num_steps,
            return_speech=True,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # One combined waveform (VibeVoice generates a single stream for the whole
        # multi-speaker script, with per-speaker cloning encoded in the context).
        assert isinstance(out, VibeVoiceGenerationOutput)
        assert out.speech_outputs is not None and len(out.speech_outputs) == 1
        wav = out.speech_outputs[0]
        assert wav is not None and isinstance(wav, torch.Tensor)
        assert wav.dim() == 1  # (T,) — one waveform per sample
        # Length == generated diffusion steps * chunk size (2400 per decode),
        # NOT 2*ref_len. Proves the gibberish (reference-length) path is gone.
        ref_len = int(mask.sum())
        assert wav.shape[0] == max_new_tokens * 2400
        assert wav.shape[0] != 2 * ref_len * 2400


# ---------------------------------------------------------------------------
# Step 5 — output assertions: waveform shape / non-empty / length semantics
# ---------------------------------------------------------------------------
class TestGenerateOutputAssertions:
    def test_speech_outputs_present_nonempty_tensor(self):
        num_steps, max_new_tokens = 3, 3
        model, inner, captured = _make_model_for_option_c(
            num_steps=num_steps, max_new_tokens=max_new_tokens,
            schedule=[S_DIFF] * max_new_tokens + [EOS])
        out = _run(model, max_new_tokens=max_new_tokens)
        assert isinstance(out, VibeVoiceGenerationOutput)
        assert out.speech_outputs is not None
        assert len(out.speech_outputs) == 1
        wav = out.speech_outputs[0]
        assert isinstance(wav, torch.Tensor) and wav.dim() == 1  # (T,)
        assert wav.numel() > 0  # non-empty waveform
        assert wav.isfinite().all()  # no NaN/Inf in the produced audio

    def test_waveform_length_tracks_generated_steps_not_reference(self):
        """The produced waveform must span the generated (target) utterance — one
        ~2400-sample chunk per AR diffusion step. This locks that generation length
        is driven by the script/target, not by the reference-prompt length."""
        num_steps, max_new_tokens = 3, 4
        model, inner, captured = _make_model_for_option_c(
            num_steps=num_steps, max_new_tokens=max_new_tokens,
            schedule=[S_DIFF] * max_new_tokens + [EOS])
        out = _run(model, max_new_tokens=max_new_tokens)
        wav = out.speech_outputs[0]
        # 4 generated diffusion steps * 2400 samples per decode chunk.
        assert wav.shape[0] == max_new_tokens * 2400

    def test_return_speech_false_yields_no_waveform(self):
        """When ``return_speech=False`` the speech_outputs is None and only the token
        sequences are returned."""
        num_steps, max_new_tokens = 3, 2
        model, inner, captured = _make_model_for_option_c(
            num_steps=num_steps, max_new_tokens=max_new_tokens,
            schedule=[S_DIFF] * max_new_tokens + [EOS])
        out = _run(model, max_new_tokens=max_new_tokens, return_speech=False)
        assert out.speech_outputs is None
        assert out.sequences is not None


# ---------------------------------------------------------------------------
# BUG-008 / length-budget — the 9-minute runaway + 1-syllable regressions
# (non-streaming single-LM). The single-LM class has no tts_eos_classifier, so the
# token-constraint + EOS termination + max_length_times cap are the only guards.
# ---------------------------------------------------------------------------
class TestGenerateLengthBudget:
    def test_loop_hard_capped_at_explicit_max_new_tokens(self):
        """Even if the model NEVER emits a terminating token, the loop must stop at
        max_new_tokens — the hard ceiling that makes the 9-minute bug impossible."""
        num_steps, max_new_tokens = 3, 20
        # All-diffusion schedule, no EOS -> bounded only by max_new_tokens.
        model, inner, captured = _make_model_for_option_c(
            num_steps=num_steps, max_new_tokens=max_new_tokens,
            schedule=[S_DIFF] * 100)
        _run(model, max_new_tokens=max_new_tokens)
        assert len(captured["decode"]) == max_new_tokens
        assert len(captured["conditions"]) == num_steps * max_new_tokens

    def test_explicit_budget_is_clamped(self):
        """The 8192 ceiling is an absolute safety net: a runaway request must be
        clamped, not honored. With seq_len=10 the effective cap is
        max_length_times*seq_len = 20, so a 5000 request yields 20 steps."""
        num_steps = 3
        max_new_tokens = 5000  # absurd budget that would have produced minutes of audio
        model, inner, captured = _make_model_for_option_c(
            num_steps=num_steps, max_new_tokens=max_new_tokens,
            schedule=[S_DIFF] * 10000)
        _run(model, max_new_tokens=max_new_tokens)
        # 5000 requested -> clamped; with seq_len=10 the loop runs 20 steps.
        assert len(captured["decode"]) == 20

    def test_default_budget_is_derived_and_bounded(self):
        """When no budget is supplied, generate derives one from the prompt length
        and must terminate there when no EOS is emitted — proving the auto budget is
        itself bounded (no unbounded runs)."""
        num_steps = 3
        seq_len = 10
        model, inner, captured = _make_model_for_option_c(
            num_steps=num_steps, max_new_tokens=None,
            schedule=[S_DIFF] * 10000)
        _run(model, seq_len=seq_len, max_new_tokens=None)
        # Derived budget min(seq_len*12, 8192) == 120, but max_length_times*seq_len
        # == 20 caps the control stream, so 20 diffusion steps run.
        assert len(captured["decode"]) == 20

    def test_loop_does_not_terminate_on_first_non_speech_token(self):
        """Regression for the 1-syllable bug: a non-speech control token (e.g.
        `<|vision_start|>` / speech_start_id, a valid control token below the speech
        token range) must NOT terminate generation. The loop must keep running and
        decode latents for the following diffusion tokens, bounded only by the cap."""
        num_steps, max_new_tokens = 3, 40
        # Emit a valid-but-non-diffusion control token (speech_start_id) every step.
        model, inner, captured = _make_model_for_option_c(
            num_steps=num_steps, max_new_tokens=max_new_tokens,
            schedule=[S_START] * 100)
        _run(model, max_new_tokens=max_new_tokens)
        # No diffusion step was ever emitted, but the loop did NOT stop early — it
        # ran the full (capped) budget. lm calls = prefill + 2 per step.
        assert len(captured["decode"]) == 0
        assert len(captured["lm_inputs"]) == 1 + 2 * 20
