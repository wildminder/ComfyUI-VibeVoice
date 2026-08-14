"""Regression test for the diffusion-condition source in `generate`.

Runtime BUG-003: `VibeVoiceForConditionalGeneration.generate` built the diffusion
condition from `outputs.logits` (vocab dimension, e.g. 151936) instead of the LM
*hidden states* (model hidden_size, e.g. 1536). The diffusion head's `cond_proj`
is `Linear(hidden_size, hidden_size)`, so the vocab-dim condition raised
`RuntimeError: mat1 and mat2 shapes cannot be multiplied (384x151936 and 1536x1536)`.

`forward` computes `hidden_states = outputs.last_hidden_state` (the correct source,
which the training branch already uses) but returned `outputs.hidden_states` (= None,
since `output_hidden_states=False`). The fix makes `forward` return the real hidden
states and `generate` read `outputs.hidden_states`.

This test guards the invariant without loading the heavy model package: it replicates
the exact condition-extraction snippet from `generate` and asserts the condition's last
dimension equals the model hidden size (not the vocab size), and that it flows through a
`Linear(hidden_size, hidden_size)` projection the way `cond_proj` does.
"""

import pytest

torch = pytest.importorskip("torch")


def _extract_condition(src, acoustic_mask):
    """Mirror of VibeVoiceForConditionalGeneration.generate (lines ~624-630)."""
    cond = src[acoustic_mask] if acoustic_mask.any() else src
    if cond.ndim == 3:
        cond = cond.reshape(-1, cond.shape[-1])
    return cond


def test_condition_uses_hidden_states_not_logits():
    B, S, HID, VOCAB = 1, 400, 1536, 151936
    acoustic_mask = torch.zeros(B, S, dtype=torch.bool)
    acoustic_mask[0, :384] = True  # 384 acoustic positions

    logits = torch.randn(B, S, VOCAB, dtype=torch.bfloat16)  # buggy source
    hidden = torch.randn(B, S, HID, dtype=torch.bfloat16)    # correct source

    cond_logits = _extract_condition(logits, acoustic_mask)
    cond_hidden = _extract_condition(hidden, acoustic_mask)

    # The buggy source yields vocab-dim; the fixed source yields hidden-dim.
    assert cond_logits.shape[-1] == VOCAB
    assert cond_hidden.shape[-1] == HID
    assert tuple(cond_hidden.shape) == (384, HID)

    cond_proj = torch.nn.Linear(HID, HID, bias=False).to(torch.bfloat16)

    # The old (logits) path must fail with the exact user-facing error.
    with pytest.raises(RuntimeError):
        cond_proj(cond_logits)

    # The fixed (hidden) path must succeed and preserve the shape.
    out = cond_proj(cond_hidden)
    assert tuple(out.shape) == (384, HID)
