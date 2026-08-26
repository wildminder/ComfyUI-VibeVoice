"""Phase C tests: GGUFLinear forward parity, replacement helper, meta-safety."""

import pytest
import torch
from torch import nn

import numpy as np
from gguf.constants import GGMLQuantizationType as T
from gguf.quants import dequantize as oracle_dequantize, quantize as oracle_quantize

from ComfyUI_VibeVoice.modules.gguf_quant import GGUFLinear, gguf_linear_factory
from ComfyUI_VibeVoice.modules.quant_common import (
    QuantTargetMismatch,
    replace_linears_for_quant,
    validate_weight_plan,
)
from conftest import craft_kquant_blocks


def _q8_linear_pair(out_f=16, in_f=32, seed=3):
    """(float reference weight, installed raw bytes) for a Q8_0 weight."""
    x = (np.random.default_rng(seed).standard_normal((out_f, in_f)) * 0.05).astype(np.float32)
    raw = oracle_quantize(x, T.Q8_0)  # (n_blocks, 34) uint8
    ref = oracle_dequantize(raw.view(np.uint8), T.Q8_0).reshape(out_f, in_f)
    return (
        torch.from_numpy(ref.copy()),
        torch.from_numpy(np.ascontiguousarray(raw)).view(torch.uint8).reshape(-1),
    )


class TestGGUFLinearForwardParity:
    @pytest.mark.parametrize("bias", [False, True])
    def test_q8_0_parity_vs_float_reference(self, bias):
        w_ref, w_raw = _q8_linear_pair()
        lin = GGUFLinear(w_ref.shape[1], w_ref.shape[0], bias=bias, ggml_type=T.Q8_0)
        lin.set_raw_weight(w_raw)
        if bias:
            lin.bias.data = torch.randn(lin.out_features) * 0.01

        x = torch.randn(5, w_ref.shape[1])
        y = lin(x)
        y_ref = torch.nn.functional.linear(
            x, w_ref, lin.bias.detach().clone() if bias else None
        )
        # Our kernel is bitwise vs the oracle; F.linear on the same weights is
        # deterministic on CPU -> near-bitwise.
        assert torch.allclose(y, y_ref, atol=1e-6)

    def test_kquant_crafted_parity(self):
        blocks = craft_kquant_blocks("Q4_K", 256 * 8, seed=9)  # 8 blocks flat
        ref = oracle_dequantize(blocks.view(np.uint8), T.Q4_K).reshape(8, 256)
        w_raw = torch.from_numpy(blocks.copy()).reshape(-1)

        lin = GGUFLinear(256, 8, bias=False, ggml_type=T.Q4_K)
        lin.set_raw_weight(w_raw)
        x = torch.randn(3, 256)
        y = lin(x)
        y_ref = torch.nn.functional.linear(x, torch.from_numpy(ref.copy()))
        assert torch.allclose(y, y_ref, atol=1e-4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_out_dtype_follows_activation(self, dtype):
        w_ref, w_raw = _q8_linear_pair()
        lin = GGUFLinear(32, 16, bias=False, ggml_type=T.Q8_0)
        lin.set_raw_weight(w_raw)
        x = torch.randn(2, 32, dtype=dtype)
        y = lin(x)
        assert y.dtype == dtype

    def test_set_raw_weight_byte_count_mismatch(self):
        lin = GGUFLinear(32, 16, bias=False, ggml_type=T.Q8_0)
        with pytest.raises(ValueError, match="byte count"):
            lin.set_raw_weight(torch.zeros(10, dtype=torch.uint8))

    def test_extra_repr_shows_type_and_bytes(self):
        _, w_raw = _q8_linear_pair()
        lin = GGUFLinear(32, 16, bias=False, ggml_type=T.Q8_0)
        lin.set_raw_weight(w_raw)
        r = repr(lin)
        assert "Q8_0" in r and "raw_bytes" in r

    def test_state_dict_exposes_raw_params_only(self):
        _, w_raw = _q8_linear_pair()
        lin = GGUFLinear(32, 16, bias=True, ggml_type=T.Q8_0)
        lin.set_raw_weight(w_raw)
        sd = lin.state_dict()
        assert set(sd.keys()) == {"weight", "bias"}
        assert sd["weight"].dtype == torch.uint8


class TestReplaceLinearsForQuant:
    def _tree(self):
        class _Tree(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(8, 8, bias=False)
                self.b = nn.Sequential(nn.Linear(8, 4, bias=False))
                self.c = nn.Conv2d(1, 1, 1)

        return _Tree()

    def test_replaces_exactly_the_planned_set(self):
        m = self._tree()
        plan = {
            "a": gguf_linear_factory(T.Q8_0),
            "b.0": gguf_linear_factory(T.Q8_0),
        }
        replaced = replace_linears_for_quant(m, plan)
        assert sorted(replaced) == ["a", "b.0"]
        assert isinstance(m.a, GGUFLinear)
        assert isinstance(m.b[0], GGUFLinear)
        assert m.a.in_features == 8 and m.a.out_features == 8

    def test_missing_target_raises(self):
        m = self._tree()
        with pytest.raises(QuantTargetMismatch, match="not found"):
            replace_linears_for_quant(m, {"nope": gguf_linear_factory(T.Q8_0)})

    def test_non_linear_target_raises(self):
        m = self._tree()
        with pytest.raises(QuantTargetMismatch, match="expected nn.Linear"):
            replace_linears_for_quant(m, {"c": gguf_linear_factory(T.Q8_0)})

    def test_meta_safe_no_kernel_calls(self, monkeypatch):
        """Construction + replacement under a meta context must not invoke any
        dequantization or kitchen kernel."""
        import ComfyUI_VibeVoice.modules.gguf_quant as G

        def _boom(*a, **k):
            raise AssertionError("dequantize called during meta construction")

        monkeypatch.setattr(G, "dequantize_blocks", _boom)

        class _Tree(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(8, 8, bias=False)

        m = _Tree()
        with torch.device("meta"):
            replaced = replace_linears_for_quant(m, {"lin": gguf_linear_factory(T.Q8_0)})
        assert replaced == ["lin"]
        assert m.lin.weight.is_meta


class TestValidateWeightPlan:
    def test_gguf_and_convrot_conflict(self):
        with pytest.raises(ValueError, match="mutually"):
            validate_weight_plan(is_gguf_file=True,
                                 convrot_quant_map={"x": object()},
                                 use_llm_4bit=False, attention_mode="sdpa")

    def test_convrot_and_bnb_conflict(self):
        with pytest.raises(ValueError, match="quantize_llm_4bit"):
            validate_weight_plan(is_gguf_file=False,
                                 convrot_quant_map={"x": object()},
                                 use_llm_4bit=True, attention_mode="sdpa")

    def test_sage_with_kquants_allowed_but_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="ComfyUI_VibeVoice.modules.quant_common"):
            validate_weight_plan(is_gguf_file=True, convrot_quant_map={},
                                 use_llm_4bit=False, attention_mode="sage",
                                 gguf_kquant_present=True)
        assert any("K-quant" in r.message for r in caplog.records)

    def test_plain_dense_passes(self):
        validate_weight_plan(is_gguf_file=False, convrot_quant_map={},
                             use_llm_4bit=True, attention_mode="sage")


# ====================================================================
# Sage wrapper interaction (regression: hidden states were cast to the
# raw uint8 weight dtype, corrupting resident dequantization)
# ====================================================================

def _load_sage_module():
    """Import the REAL vendored sage patch file under a private name.

    conftest mocks src.vibevoice.modular.sage_attention_patch; loading by
    path bypasses that mock so the production logic is what's tested.
    """
    import importlib.util
    import os

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "vibevoice", "modular", "sage_attention_patch.py",
    )
    spec = importlib.util.spec_from_file_location(
        "_sage_patch_under_test", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestSageTargetDtypeResolution:
    def test_quant_resident_passes_hidden_dtype_through(self):
        sage = _load_sage_module()
        from gguf.constants import GGMLQuantizationType as T

        res = GGUFLinear(32, 16, bias=False, ggml_type=T.Q8_0)
        res.set_raw_weight(torch.zeros(16 * 34, dtype=torch.uint8))
        hidden = torch.randn(1, 4, 32, dtype=torch.bfloat16)
        assert sage.resolve_sage_target_dtype(res, hidden) is torch.bfloat16

    def test_bnb_style_weight_selects_bf16(self):
        sage = _load_sage_module()

        class _Fake4Bit(torch.nn.Module):
            pass

        m = _Fake4Bit()
        m.weight = torch.nn.Parameter(torch.zeros(4, dtype=torch.uint8),
                                      requires_grad=False)
        m.quant_state = object()
        hidden = torch.randn(1, 4, 8, dtype=torch.float32)
        assert sage.resolve_sage_target_dtype(m, hidden) is torch.bfloat16

    def test_plain_float_linear_matches_weight(self):
        sage = _load_sage_module()
        lin = torch.nn.Linear(8, 8)
        lin.weight.data = lin.weight.data.to(torch.float16)
        hidden = torch.randn(1, 4, 8, dtype=torch.float32)
        assert sage.resolve_sage_target_dtype(lin, hidden) is torch.float16


class TestResidentRejectsNonFloatInput:
    def test_gguf_linear_loudly_rejects_byte_input(self):
        from gguf.constants import GGMLQuantizationType as T

        res = GGUFLinear(32, 16, bias=False, ggml_type=T.Q8_0)
        res.set_raw_weight(torch.zeros(16 * 34, dtype=torch.uint8))
        with pytest.raises(TypeError, match="floating-point"):
            res(torch.zeros(2, 32, dtype=torch.uint8))

    def test_convrot_loudly_rejects_byte_input(self):
        pytest.importorskip("comfy_kitchen")
        from ComfyUI_VibeVoice.modules.convrot_quant import ConvRotInt8Linear

        layer = ConvRotInt8Linear(64, 32, bias=False, group_size=16)
        with pytest.raises(TypeError, match="floating-point"):
            layer(torch.zeros(2, 64, dtype=torch.int8))
