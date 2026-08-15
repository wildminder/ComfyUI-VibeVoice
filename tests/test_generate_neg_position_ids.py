"""Regression test for BUG-011: negative-branch RoPE position_ids desync.

During non-streaming ``generate()``, the negative (unconditional) CFG forward
feeds a SINGLE-token ``inputs_embeds`` ``(B,1,H)`` per autoregressive step. In
transformers 5.x the explicit ``position_ids`` drive RoPE directly, so they MUST
be the CURRENT position only (``(B,1)``). The previous code passed the
full-length ``neg_position_ids`` ``(B, step+1)``, which made q/k broadcast
against full-length cos/sin and silently expand to seq-len ``step+1`` while v
(never rotated) stayed length 1. The KV cache then accumulated ``step+1`` keys
but only 1 value per step, and SDPA's ``attn @ value`` crashed with::

    RuntimeError: Expected size for first two dimensions of batch2 tensor to be:
    [B*num_attention_heads, key_len] but got: [B*num_attention_heads, value_len]

(e.g. ``[12, 3] but got: [12, 2]`` at AR step 1 for VibeVoice-1.5B with
``num_attention_heads=12``).

This test loads the REAL ``modeling_vibevoice`` module (bypassing the conftest
mock), drives ``generate()`` through several AR steps with a recording inner
language model, and locks the invariant:

    For EVERY ``self.model.language_model(...)`` call,
    ``position_ids.shape[-1] == inputs_embeds.shape[-2]``.

Before the fix, the step>=1 negative-branch call violates this (position_ids
length 2 vs embeds length 1). After the fix, all calls satisfy it.
"""

import os
import sys
import types
import importlib.util
from unittest.mock import MagicMock

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Real-module loading (mirrors tests/test_model_forward.py)
# ---------------------------------------------------------------------------
def _stub_diffusers():
    """Stub diffusers so the vendored schedule import chain succeeds."""
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


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
HIDDEN = 32
VOCAB = 100
# Distinct control-token ids (all inside the valid constraint set).
SPEECH_START = 10
SPEECH_END = 11
SPEECH_DIFFUSION = 12
EOS = 13
BOS = 14


class _FixedLogitHead(nn.Module):
    """lm_head that always emits a fixed token (deterministic argmax).

    Exposes a real ``.weight`` parameter so ``generate()`` can read
    ``self.lm_head.weight.device``.
    """

    def __init__(self, vocab: int, token_id: int, hidden: int):
        super().__init__()
        self.vocab = vocab
        self.token_id = token_id
        self.weight = nn.Parameter(torch.zeros(hidden, 1))

    def forward(self, x):
        logits = torch.full((x.shape[0], self.vocab), -1e9)
        logits[:, self.token_id] = 1e9
        return logits


def _make_tokenizer():
    tok = MagicMock()
    tok.speech_start_id = SPEECH_START
    tok.speech_end_id = SPEECH_END
    tok.speech_diffusion_id = SPEECH_DIFFUSION
    tok.eos_id = EOS
    tok.bos_token_id = BOS
    return tok


def _make_recording_model():
    """Build a VibeVoiceForConditionalGeneration with a recording inner LM.

    Returns ``(model, calls)`` where ``calls`` is a list of dicts capturing the
    ``inputs_embeds`` / ``position_ids`` / ``attention_mask`` shapes of every
    ``self.model.language_model(...)`` invocation.
    """
    model = VibeVoiceForConditionalGeneration.__new__(VibeVoiceForConditionalGeneration)
    torch.nn.Module.__init__(model)

    model.config = MagicMock()
    model.config.use_return_dict = True
    model.config.diffusion_head_config = MagicMock()
    model.config.diffusion_head_config.ddpm_num_inference_steps = 10
    model.config.acoustic_tokenizer_config = MagicMock()
    model.config.acoustic_tokenizer_config.vae_dim = 64
    model.config.acoustic_vae_dim = 64
    model.config.decoder_config = MagicMock()
    model.config.decoder_config.max_position_embeddings = 4096

    model.vocab_size = VOCAB

    # Recording inner language model.
    calls = []

    def _lm_forward(inputs_embeds=None, attention_mask=None, position_ids=None,
                    cache_position=None, past_key_values=None, **kwargs):
        calls.append({
            "embeds_seq": None if inputs_embeds is None else int(inputs_embeds.shape[-2]),
            "pos_shape": None if position_ids is None else tuple(position_ids.shape),
            "attn_shape": None if attention_mask is None else tuple(attention_mask.shape),
            "cache_position": None if cache_position is None else cache_position.tolist(),
            "has_past": past_key_values is not None,
        })
        seq = int(inputs_embeds.shape[-2])
        bsz = int(inputs_embeds.shape[0])
        out = MagicMock()
        out.last_hidden_state = torch.zeros(bsz, seq, HIDDEN)
        out.past_key_values = MagicMock()  # truthy -> neg_past / past set for next step
        return out

    inner = MagicMock()
    inner.language_model = MagicMock(side_effect=_lm_forward)
    inner.noise_scheduler = MagicMock()
    inner.noise_scheduler.timesteps = []
    inner.noise_scheduler.set_timesteps = MagicMock()
    inner.prediction_head = MagicMock()
    inner.prediction_head.device = torch.device("cpu")
    inner.acoustic_tokenizer = MagicMock()
    inner.acoustic_tokenizer.device = torch.device("cpu")
    inner.speech_scaling_factor = torch.tensor(float("nan"))
    inner.speech_bias_factor = torch.tensor(float("nan"))
    model.model = inner

    # Input embeddings + deterministic lm_head (always emits SPEECH_START so the
    # loop runs several AR steps without hitting EOS / the diffusion branch).
    emb = nn.Embedding(VOCAB, HIDDEN)
    model.get_input_embeddings = MagicMock(return_value=emb)
    model.add_module("lm_head", _FixedLogitHead(VOCAB, SPEECH_START, HIDDEN))

    # Neutralize the streaming-cache construction (diffusion branch is skipped).
    _modeling.VibeVoiceTokenizerStreamingCache = lambda: MagicMock()

    return model, calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestNegativeBranchPositionIdsInvariant:
    """BUG-011: position_ids length must match inputs_embeds seq length."""

    def _run_generate(self, seq_len=4, max_new_tokens=3):
        model, calls = _make_recording_model()
        input_ids = torch.randint(0, VOCAB, (1, seq_len))
        attention_mask = torch.ones((1, seq_len), dtype=torch.long)
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            speech_tensors=None,
            speech_masks=None,
            acoustic_input_mask=None,
            cfg_scale=1.3,
            inference_steps=2,
            return_speech=True,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            tokenizer=_make_tokenizer(),
        )
        return model, calls, out

    def test_generate_completes_without_shape_error(self):
        """generate() must run to completion (no SDPA shape crash)."""
        model, calls, out = self._run_generate()
        assert isinstance(out, VibeVoiceGenerationOutput)
        # Several AR steps were executed.
        assert len(calls) >= 5

    def test_position_ids_match_embeds_length_on_every_call(self):
        """THE core invariant: position_ids[-1] == inputs_embeds seq len, always."""
        model, calls, out = self._run_generate()
        assert calls, "no language_model calls recorded"
        for i, c in enumerate(calls):
            embeds_seq = c["embeds_seq"]
            pos_shape = c["pos_shape"]
            assert embeds_seq is not None, f"call {i}: inputs_embeds missing"
            assert pos_shape is not None, f"call {i}: position_ids missing"
            pos_seq = pos_shape[-1]
            assert pos_seq == embeds_seq, (
                f"call {i}: position_ids length {pos_seq} != "
                f"inputs_embeds seq length {embeds_seq} "
                f"(has_past={c['has_past']}, cache_position={c['cache_position']})"
            )

    def test_negative_ar_branch_exercised_with_past(self):
        """Ensure we actually drove the negative branch with an existing KV cache.

        The bug only manifests when ``neg_past`` is not None (AR step >= 1), so
        confirm at least one single-token call ran with ``has_past=True``.
        """
        model, calls, out = self._run_generate()
        ar_with_past = [
            c for c in calls if c["embeds_seq"] == 1 and c["has_past"]
        ]
        assert ar_with_past, (
            "no single-token AR call with an existing KV cache was recorded; "
            "the test did not exercise the buggy path"
        )
        # And each such call must have a single (current-only) position id.
        for c in ar_with_past:
            assert c["pos_shape"][-1] == 1, (
                f"single-token AR call has {c['pos_shape'][-1]} position ids, "
                f"expected 1 (current-only)"
            )

    def test_attention_mask_stays_full_length_on_ar_steps(self):
        """The attention mask must still address the whole cache on AR steps.

        Only ``position_ids`` shrinks to current-only; the mask remains
        full-length. The negative and positive branches each grow their own
        mask independently (calls interleave), so we assert the branch-agnostic
        invariant: on every AR call with an existing KV cache, the attention
        mask length is STRICTLY greater than the (current-only) position_ids
        length — i.e. the mask addresses the whole cache while position_ids
        addresses only the new token.
        """
        model, calls, out = self._run_generate()
        ar_calls = [
            c for c in calls
            if c["has_past"] and c["attn_shape"] and c["pos_shape"]
        ]
        assert ar_calls, "no AR-step calls with an existing KV cache recorded"
        for c in ar_calls:
            mask_len = c["attn_shape"][-1]
            pos_len = c["pos_shape"][-1]
            assert mask_len > pos_len, (
                f"attention mask length {mask_len} must exceed current-only "
                f"position_ids length {pos_len} on AR steps "
                f"(cache_position={c['cache_position']})"
            )
        # The mask must have grown past a single token (cache accumulated).
        assert max(c["attn_shape"][-1] for c in ar_calls) > 1, (
            "attention mask never grew past a single token"
        )
