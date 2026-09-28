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


# ---------------------------------------------------------------------------
# Arch dispatch: exact match, not a threshold.
# ---------------------------------------------------------------------------
# The previous chain tested ``arch_code >= 120``, then ``>= 90``, then
# ``== 89``, then ``>= 80``. sm100 / sm103 (Blackwell datacenter, CC 10.x)
# therefore fell into the SM90 branch. This module calls the kernels directly
# and never routes through sage's ``sageattn()``, so the pre-fix failure was
# not sage's ``ValueError: Unsupported CUDA architecture`` but SILENT
# wrong-kernel selection: the sm90 kernel asserts only that it COMPILED, not
# that it is running on the right silicon, and nothing downstream caught it.

_SAGE_KERNELS = (
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda_sm90",
)


def _make_kernel_stub(name):
    # The dispatcher SELECTS a kernel, it never calls it, so identity has to be
    # carried by the function object itself. One stable object per name, built
    # once, so `is` identity can be asserted in the tests below.
    def _fn(*a, **kw):
        raise AssertionError("the dispatcher must not call the kernel")
    _fn.kernel_name = name
    return _fn


_KERNEL_STUBS = {name: _make_kernel_stub(name) for name in _SAGE_KERNELS}


def _dispatch(major, minor):
    """Run the real dispatcher against a faked device.

    Returns (selected_kernel, pv_accum_dtype, arch_code). The selected kernel is
    the stub object itself, so callers can assert identity.
    """
    prev_funcs = {
        name: getattr(sage_attention_patch, name, None) for name in _SAGE_KERNELS
    }
    prev_available = sage_attention_patch.SAGE_ATTENTION_AVAILABLE
    prev_cuda = torch.cuda.is_available
    prev_cap = torch.cuda.get_device_capability
    for name in _SAGE_KERNELS:
        setattr(sage_attention_patch, name, _KERNEL_STUBS[name])
    sage_attention_patch.SAGE_ATTENTION_AVAILABLE = True
    torch.cuda.is_available = lambda: True
    torch.cuda.get_device_capability = lambda *a, **kw: (major, minor)
    try:
        attn_func, _, pv_accum_dtype = (
            sage_attention_patch.get_sage_attention_function_and_params()
        )
    finally:
        for name, fn in prev_funcs.items():
            setattr(sage_attention_patch, name, fn)
        sage_attention_patch.SAGE_ATTENTION_AVAILABLE = prev_available
        torch.cuda.is_available = prev_cuda
        torch.cuda.get_device_capability = prev_cap
    return attn_func, pv_accum_dtype, major * 10 + minor


@pytest.mark.parametrize(
    "capability,expected,pv_accum",
    [
        # int8 QK / fp16 PV on Ampere.
        ((8, 0), "sageattn_qk_int8_pv_fp16_cuda", "fp32"),
        ((8, 6), "sageattn_qk_int8_pv_fp16_cuda", "fp32"),
        # int8 QK / fp8 PV everywhere else, accumulating in fp32+fp32.
        ((8, 9), "sageattn_qk_int8_pv_fp8_cuda", "fp32+fp32"),
        ((9, 0), "sageattn_qk_int8_pv_fp8_cuda_sm90", "fp32+fp32"),
        ((12, 0), "sageattn_qk_int8_pv_fp8_cuda", "fp32+fp32"),
    ],
)
def test_supported_arch_selects_the_right_kernel(capability, expected, pv_accum):
    kernel, pv_accum_dtype, arch_code = _dispatch(*capability)
    # Object identity, not a name: a threshold rule would hand back the SM90
    # kernel for sm100/sm103 and a name-based assertion could be fooled by a
    # renamed or aliased symbol.
    assert kernel is _KERNEL_STUBS[expected], (
        f"sm{arch_code} should select {expected}"
    )
    assert pv_accum_dtype == pv_accum


@pytest.mark.parametrize("capability", [(7, 5), (8, 7), (8, 8), (10, 0), (10, 3), (11, 0)])
def test_unsupported_arch_selects_no_kernel(capability):
    """No kernel at all — explicitly not the SM90 one."""
    kernel, pv_accum_dtype, arch_code = _dispatch(*capability)
    assert kernel is None, (
        f"sm{arch_code} is not in sage's dispatch table; selecting a kernel for "
        f"it is how a Hopper kernel ends up on silicon it was never built for"
    )
    assert pv_accum_dtype is None


def test_dispatch_agrees_with_the_project_capability_check():
    """The dispatcher and check_sage_attention_compatible must accept exactly
    the same set, or the node offers sage on a machine the loader refuses.

    Both directions are asserted: the check must not reject an arch the
    dispatcher can serve, and must not accept one the dispatcher refuses. That
    keeps the two literals — the dispatcher's exact-code chain and
    SAGE_SUPPORTED_ARCHS — from drifting apart.
    """
    from ComfyUI_VibeVoice.modules.attention_utils import (
        SAGE_SUPPORTED_ARCHS,
        check_sage_attention_compatible,
    )
    from unittest.mock import patch

    # Every real arch the check knows about, accepted or not. Codes 75-120 step
    # by 1; sm75 is sage's triton-only arch and the project excludes it.
    all_codes = [c for c in range(75, 121)]

    for code in all_codes:
        major, minor = code // 10, code % 10
        kernel, _, _ = _dispatch(major, minor)
        arch = f"sm{code}"
        with patch("ComfyUI_VibeVoice.modules.attention_utils.SAGE_ATTENTION_AVAILABLE", True), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_capability", return_value=(major, minor)):
            check_says_yes = check_sage_attention_compatible()

        assert (kernel is not None) == check_says_yes, (
            f"{arch}: dispatcher selects {getattr(kernel, 'kernel_name', None)!r} "
            f"but check_sage_attention_compatible() says {check_says_yes}. "
            f"Project-supported archs are {sorted(SAGE_SUPPORTED_ARCHS)}; sage's "
            f"own table is sm75/sm80/sm86/sm89/sm90/sm120."
        )
        assert check_says_yes == (arch in SAGE_SUPPORTED_ARCHS), (
            f"{arch}: check_sage_attention_compatible() disagrees with the "
            f"SAGE_SUPPORTED_ARCHS literal {sorted(SAGE_SUPPORTED_ARCHS)}"
        )



def test_non_cuda_returns_no_kernel():
    prev = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        attn_func, gran, accum =             sage_attention_patch.get_sage_attention_function_and_params()
    finally:
        torch.cuda.is_available = prev
    assert (attn_func, gran, accum) == (None, None, None)


# ---------------------------------------------------------------------------
# Mask refusal
# ---------------------------------------------------------------------------
# sageattn 2.2.0 has NO attn_mask parameter. The old code used the mask only
# to pick ``is_causal`` and then dropped it, so a padded prefill (the ASR
# processor left-pads every batch) ran non-causally over the pad columns —
# silently wrong output. It now refuses.

def test_sage_attention_forward_refuses_a_non_none_mask():
    fake, head_dim, num_heads, hidden = _make_fake_attention()
    called = {"n": 0}

    def stub_sage(q, k, v, tensor_layout="HND", is_causal=False,
                  qk_quant_gran="per_warp", pv_accum_dtype="fp32"):
        called["n"] += 1
        return q

    prev = sage_attention_patch.SAGE_ATTENTION_FUNCTION
    sage_attention_patch.SAGE_ATTENTION_FUNCTION = stub_sage
    try:
        bsz, q_len = 1, 8
        hidden_states = torch.randn(bsz, q_len, hidden)
        cos = torch.randn(bsz, q_len, head_dim)
        sin = torch.randn(bsz, q_len, head_dim)
        additive_mask = torch.zeros(bsz, 1, q_len, q_len)
        with pytest.raises(ValueError, match="attention_mask"):
            sage_attention_patch.sage_attention_forward(
                fake,
                hidden_states,
                position_embeddings=(cos, sin),
                attention_mask=additive_mask,
                past_key_values=None,
                cache_position=None,
            )
    finally:
        sage_attention_patch.SAGE_ATTENTION_FUNCTION = prev

    assert called["n"] == 0, (
        "the kernel must never run with a mask it cannot honour — the mask is "
        "the whole reason for the refusal"
    )


def test_sage_attention_forward_still_runs_unmasked():
    """The guard must not disturb the working TTS path."""
    fake, head_dim, num_heads, hidden = _make_fake_attention()
    seen = {}

    def stub_sage(q, k, v, tensor_layout="HND", is_causal=False,
                  qk_quant_gran="per_warp", pv_accum_dtype="fp32"):
        seen["is_causal"] = is_causal
        return q

    prev = sage_attention_patch.SAGE_ATTENTION_FUNCTION
    sage_attention_patch.SAGE_ATTENTION_FUNCTION = stub_sage
    try:
        bsz, q_len = 1, 8
        hidden_states = torch.randn(bsz, q_len, hidden)
        cos = torch.randn(bsz, q_len, head_dim)
        sin = torch.randn(bsz, q_len, head_dim)
        out, _ = sage_attention_patch.sage_attention_forward(
            fake,
            hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=None,
            cache_position=None,
        )
    finally:
        sage_attention_patch.SAGE_ATTENTION_FUNCTION = prev

    assert tuple(out.shape) == (bsz, q_len, hidden)
    # Causality is now decided purely by the query length, since the mask is
    # guaranteed None by the guard above.
    assert seen["is_causal"] is True
