"""Phase 2 tests: assign-based weight loading (plan 2026-08-18, D2/D3).

Covers ``VibeVoiceLoader._apply_state_dict``:
- assign semantics replace parameter objects (no copy, data_ptr identity)
- tied weights are re-tied after assign (un-defers DF-005)
- meta stragglers are zero-materialized
- missing/unexpected keys are reported
- tie is skipped when the config is not tied
- end-to-end: meta-instantiated tiny model + assign load yields checkpoint values
"""

import torch
import pytest
from unittest.mock import MagicMock

from ComfyUI_VibeVoice.modules.loader import VibeVoiceLoader


# ---------------------------------------------------------------------------
# Tiny test doubles
# ---------------------------------------------------------------------------
class _TiedTinyModel(torch.nn.Module):
    """Minimal model with a tied lm_head (mirrors VibeVoice-1.5B)."""

    def __init__(self, vocab=4, dim=4):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        self.lm_head = torch.nn.Linear(dim, vocab, bias=False)
        self.lm_head.weight = self.embed.weight  # tie
        # Config gate used by _apply_state_dict
        self.config = MagicMock()
        self.config.decoder_config.tie_word_embeddings = True
        self.config.tie_word_embeddings = False

    def tie_weights(self):
        self.lm_head.weight = self.embed.weight


class _UntiedTinyModel(_TiedTinyModel):
    """Same model but with tying disabled in the config."""

    def __init__(self, vocab=4, dim=4):
        super().__init__(vocab, dim)
        self.config.decoder_config.tie_word_embeddings = False
        self.tie_called = False

    def tie_weights(self):
        self.tie_called = True


class _MetaStragglerModel(torch.nn.Module):
    """Model with one parameter absent from the checkpoint (stays meta)."""

    def __init__(self):
        super().__init__()
        self.present = torch.nn.Linear(2, 2, bias=False)
        self.absent = torch.nn.Linear(2, 2, bias=False)
        # Put 'absent' on meta to simulate a checkpoint that omits the key.
        with torch.device("meta"):
            self.absent = torch.nn.Linear(2, 2, bias=False)
        self.config = MagicMock()
        self.config.decoder_config.tie_word_embeddings = False
        self.config.tie_word_embeddings = False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestApplyStateDict:
    def test_assign_replaces_parameters_no_copy(self):
        model = _UntiedTinyModel()
        w = torch.ones(4, 4)
        sd = {"embed.weight": w}
        VibeVoiceLoader._apply_state_dict(model, sd)
        # assign=True: the parameter object IS the checkpoint tensor.
        assert model.embed.weight.data_ptr() == w.data_ptr()
        assert torch.equal(model.embed.weight, torch.ones(4, 4))

    def test_tie_restored_after_assign(self):
        model = _TiedTinyModel()
        sd = {"embed.weight": torch.ones(4, 4)}  # no lm_head.weight in sd
        VibeVoiceLoader._apply_state_dict(model, sd)
        # After re-tie, lm_head shares the (new) embed parameter.
        assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()
        assert torch.equal(model.lm_head.weight, torch.ones(4, 4))

    def test_missing_meta_params_materialized_zero(self):
        model = _MetaStragglerModel()
        assert model.absent.weight.is_meta
        sd = {"present.weight": torch.full((2, 2), 3.0)}
        VibeVoiceLoader._apply_state_dict(model, sd)
        # 'present' got the checkpoint value; 'absent' was zero-materialized.
        assert torch.equal(model.present.weight, torch.full((2, 2), 3.0))
        assert not model.absent.weight.is_meta
        assert model.absent.weight.device.type == "cpu"
        assert torch.equal(model.absent.weight, torch.zeros(2, 2))
        assert model.absent.weight.shape == (2, 2)

    def test_missing_unexpected_keys_reported(self):
        model = _UntiedTinyModel()
        sd = {"embed.weight": torch.ones(4, 4), "bogus.key": torch.zeros(1)}
        missing, unexpected = VibeVoiceLoader._apply_state_dict(model, sd)
        assert "bogus.key" in unexpected
        # lm_head.weight is tied; with assign it is reported missing before re-tie.
        assert isinstance(missing, list)

    def test_tie_skipped_when_not_tied(self):
        model = _UntiedTinyModel()
        sd = {"embed.weight": torch.ones(4, 4)}
        VibeVoiceLoader._apply_state_dict(model, sd)
        assert model.tie_called is False

    def test_meta_instantiate_then_assign_yields_checkpoint_values(self):
        """End-to-end fast path: meta init + assign load (Phase 3 preview)."""
        with torch.device("meta"):
            model = _UntiedTinyModel()
        assert all(p.is_meta for p in model.parameters())
        sd = {"embed.weight": torch.full((4, 4), 7.0),
              "lm_head.weight": torch.full((4, 4), 9.0)}
        VibeVoiceLoader._apply_state_dict(model, sd)
        assert not any(p.is_meta for p in model.parameters())
        assert torch.equal(model.embed.weight, torch.full((4, 4), 7.0))
        assert torch.equal(model.lm_head.weight, torch.full((4, 4), 9.0))

    def test_meta_buffers_materialized_after_assign(self):
        """Meta buffers (not in checkpoint) must be materialized to CPU.

        Without this, ComfyUI's unpatch_model → model.to(device) raises
        NotImplementedError on meta buffers.
        """
        class _ModelWithBuffer(_UntiedTinyModel):
            def __init__(self, vocab=4, dim=4):
                super().__init__(vocab, dim)
                self.register_buffer("position_ids", torch.arange(vocab))

        with torch.device("meta"):
            model = _ModelWithBuffer()
        # Buffer is on meta.
        assert model.position_ids.is_meta
        # Checkpoint has no buffer key.
        sd = {"embed.weight": torch.ones(4, 4)}
        VibeVoiceLoader._apply_state_dict(model, sd)
        # Buffer is now on CPU (materialized).
        assert not model.position_ids.is_meta
        assert model.position_ids.device.type == "cpu"

    def test_no_meta_tensors_remain_after_assign(self):
        """After _apply_state_dict, NO tensor (param or buffer) is on meta."""
        class _ModelWithBuffer(_UntiedTinyModel):
            def __init__(self, vocab=4, dim=4):
                super().__init__(vocab, dim)
                self.register_buffer("mask", torch.ones(vocab))

        with torch.device("meta"):
            model = _ModelWithBuffer()
        sd = {"embed.weight": torch.ones(4, 4), "lm_head.weight": torch.ones(4, 4)}
        VibeVoiceLoader._apply_state_dict(model, sd)
        # All parameters AND buffers must be off meta.
        assert not any(p.is_meta for p in model.parameters())
        assert not any(b.is_meta for b in model.buffers())


class TestSentinelBufferRegression:
    """v2.3.1: meta-init must not zero sentinel buffers (silent-output bug).

    The VibeVoice model registers ``speech_scaling_factor`` /
    ``speech_bias_factor`` as ``float('nan')`` and computes them at inference
    time. The diffusion-inversion gate (modeling_vibevoice.py:623) is
    ``if not torch.isnan(sf) and not torch.isnan(bf)``. Zeroing these buffers
    makes the gate TRUE and applies ``speech / 0 - 0`` → silent output. The
    hotfix in 6b99be9 zero-materialized ALL meta buffers, which regressed this.
    """

    def test_sentinel_buffers_restored_to_nan(self):
        """speech_scaling_factor / speech_bias_factor stay nan after meta load."""
        class _SentinelModel(_UntiedTinyModel):
            def __init__(self, vocab=4, dim=4):
                super().__init__(vocab, dim)
                self.register_buffer("speech_scaling_factor", torch.tensor(float("nan")))
                self.register_buffer("speech_bias_factor", torch.tensor(float("nan")))

        with torch.device("meta"):
            model = _SentinelModel()
        # Buffers are on meta after meta-init.
        assert model.speech_scaling_factor.is_meta
        assert model.speech_bias_factor.is_meta
        sd = {"embed.weight": torch.ones(4, 4), "lm_head.weight": torch.ones(4, 4)}
        VibeVoiceLoader._apply_state_dict(model, sd)
        # They must be restored to nan (NOT zero) so the isnan gate works.
        assert not model.speech_scaling_factor.is_meta
        assert not model.speech_bias_factor.is_meta
        assert torch.isnan(model.speech_scaling_factor).item()
        assert torch.isnan(model.speech_bias_factor).item()

    def test_fix_std_sentinel_restored_to_config_value(self):
        """fix_std (persistent=False) is restored to its config value (0.5)."""
        class _FixStdModel(_UntiedTinyModel):
            def __init__(self, vocab=4, dim=4):
                super().__init__(vocab, dim)
                self.register_buffer("fix_std", torch.tensor(0.5), persistent=False)

        with torch.device("meta"):
            model = _FixStdModel()
        assert model.fix_std.is_meta
        sd = {"embed.weight": torch.ones(4, 4), "lm_head.weight": torch.ones(4, 4)}
        VibeVoiceLoader._apply_state_dict(model, sd)
        assert not model.fix_std.is_meta
        assert model.fix_std.item() == 0.5

    def test_non_sentinel_buffer_still_zeroed(self):
        """Ordinary meta buffers (e.g. position_ids) are still zeroed."""
        class _PlainBufferModel(_UntiedTinyModel):
            def __init__(self, vocab=4, dim=4):
                super().__init__(vocab, dim)
                self.register_buffer("position_ids", torch.arange(vocab))

        with torch.device("meta"):
            model = _PlainBufferModel()
        sd = {"embed.weight": torch.ones(4, 4), "lm_head.weight": torch.ones(4, 4)}
        VibeVoiceLoader._apply_state_dict(model, sd)
        assert not model.position_ids.is_meta
        assert torch.equal(model.position_ids, torch.zeros(4))
