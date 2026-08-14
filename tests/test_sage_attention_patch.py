"""Regression test for the SageAttention RoPE call.

Runtime BUG-001: `sage_attention_forward` called
`apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids=None)`.
transformers >= 5 removed the `position_ids` kwarg from that function, so the
call raised `TypeError: apply_rotary_pos_emb() got an unexpected keyword
argument 'position_ids'` at generation time when the sage attention mode was
selected. The fix drops the kwarg and calls positionally.

This test loads the REAL module by file path (bypassing conftest's
`src.vibevoice` mock), stubs the CUDA sage kernel so it runs without a GPU,
and asserts the forward executes the RoPE call without TypeError and produces
the correct output shape. If the `position_ids` kwarg is ever reintroduced the
real `apply_rotary_pos_emb` raises and this test fails.
"""

import importlib.util
import os
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATCH_PATH = os.path.abspath(
    os.path.join(_HERE, "..", "src", "vibevoice", "modular", "sage_attention_patch.py")
)

spec = importlib.util.spec_from_file_location("sage_attention_patch_rt", _PATCH_PATH)
sage_attention_patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sage_attention_patch)

from torch import nn  # noqa: E402


def _make_fake_attention():
    head_dim = 16
    num_heads = 4
    num_kv = 4
    hidden = num_heads * head_dim  # 64

    cfg = types.SimpleNamespace(
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv,
        head_dim=head_dim,
        hidden_size=hidden,
    )

    attn = types.SimpleNamespace(
        config=cfg,
        layer_idx=0,
        head_dim=head_dim,
        q_proj=nn.Linear(hidden, num_heads * head_dim),
        k_proj=nn.Linear(hidden, num_kv * head_dim),
        v_proj=nn.Linear(hidden, num_kv * head_dim),
        o_proj=nn.Linear(hidden, hidden),
    )

    return attn, head_dim, num_heads, hidden


def test_sage_attention_forward_no_position_ids_kwarg():
    fake, head_dim, num_heads, hidden = _make_fake_attention()

    called = {}

    def stub_sage(q, k, v, tensor_layout="HND", is_causal=False,
                  qk_quant_gran="per_warp", pv_accum_dtype="fp32"):
        called["ok"] = True
        # Mirror the expected output shape: (bsz, heads, q_len, head_dim)
        return q

    prev = sage_attention_patch.SAGE_ATTENTION_FUNCTION
    sage_attention_patch.SAGE_ATTENTION_FUNCTION = stub_sage
    try:
        bsz, q_len = 1, 8
        hidden_states = torch.randn(bsz, q_len, hidden)
        cos = torch.randn(bsz, q_len, head_dim)
        sin = torch.randn(bsz, q_len, head_dim)

        out, attn_weights = sage_attention_patch.sage_attention_forward(
            fake,
            hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=None,
            cache_position=None,
        )
    finally:
        sage_attention_patch.SAGE_ATTENTION_FUNCTION = prev

    assert called.get("ok") is True
    assert tuple(out.shape) == (bsz, q_len, hidden)
    assert attn_weights is None
