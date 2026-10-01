"""Unit tests for the FP8-resident linear runtime (plan 2026-08-27, Phase 1).

Parity references are the MANUAL dequant formula the kitchen eager backend
implements — ``(w.float() * scale).to(out_dtype)`` — verified bit-exact
against ``comfy_kitchen.dequantize_per_tensor_fp8`` on this box. All tests
run on CPU (eager backend); no GPU required.
"""

import importlib.util
import json
import sys
import types
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ComfyUI_VibeVoice.modules.fp8_quant import (
    FP8Linear,
    make_fp8_linear,
    probe_fp8_backend,
)
from ComfyUI_VibeVoice.modules.convrot_quant import QuantLayerInfo
from ComfyUI_VibeVoice.modules.dtype_utils import (
    cast_model_to_dtype,
    cast_model_to_dtype_if_needed,
)


# comfy-kitchen provides the fp8 dequant kernels FP8Linear.forward calls. It is
# in neither requirements.txt nor pyproject.toml, so a machine without it must
# SKIP these rather than error: the fp8 path is simply unavailable there.
#
# find_spec is wrapped because it can raise on a broken or partially-installed
# package, and this runs at COLLECTION time -- an exception here would error the
# whole file instead of skipping the fp8 tests.
def _has_comfy_kitchen() -> bool:
    try:
        return importlib.util.find_spec("comfy_kitchen") is not None
    except (ImportError, ValueError):
        return False


requires_kitchen = pytest.mark.skipif(
    not _has_comfy_kitchen(),
    reason="comfy_kitchen (optional fp8 dequant backend) is not installed",
)


def _fill(m, seed=0, scale=2.0, bias_val=None):
    """Install deterministic fp8 storage + scalar scale into a fresh module."""
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(m.out_features, m.in_features, generator=g) * 0.3).to(m.fp8_dtype)
    m.weight.data.copy_(w)
    m.weight_scale.data.copy_(torch.tensor(scale, dtype=torch.float32))
    if m.bias is not None:
        b = torch.randn(m.out_features, generator=g) * 0.1
        if bias_val is not None:
            b = torch.full((m.out_features,), bias_val)
        m.bias.data.copy_(b)
    return m


def _reference(m, x):
    ref_w = (m.weight.float() * m.weight_scale).to(x.dtype)
    bias = m.bias.to(x.dtype) if m.bias is not None else None
    return F.linear(x, ref_w, bias)


@requires_kitchen
class TestProbeFP8Backend:
    def test_returns_backend_on_this_box(self):
        assert probe_fp8_backend() in ("triton", "cuda", "eager")

    def test_none_when_kitchen_missing(self):
        with patch.dict(sys.modules, {"comfy_kitchen": None}):
            assert probe_fp8_backend() is None

    def test_none_when_callable_missing_despite_capability(self):
        # D-3 trap guard: capability strings have shipped without callables.
        fake = types.ModuleType("comfy_kitchen")
        fake.list_backends = lambda: {
            "eager": {"available": True,
                      "capabilities": ["dequantize_per_tensor_fp8"]}
        }
        with patch.dict(sys.modules, {"comfy_kitchen": fake}):
            assert probe_fp8_backend() is None

    def test_none_when_no_backend_available(self):
        # comfy_kitchen is in neither requirements.txt nor pyproject.toml, so a
        # bare import here ERRORS on a machine without it instead of skipping.
        comfy_kitchen = pytest.importorskip("comfy_kitchen")

        fake = types.ModuleType("comfy_kitchen")
        fake.dequantize_per_tensor_fp8 = comfy_kitchen.dequantize_per_tensor_fp8
        fake.list_backends = lambda: {
            "eager": {"available": False,
                      "capabilities": ["dequantize_per_tensor_fp8"]}
        }
        with patch.dict(sys.modules, {"comfy_kitchen": fake}):
            assert probe_fp8_backend() is None


class TestFP8LinearConstruction:
    def test_rejects_non_fp8_dtype(self):
        with pytest.raises(ValueError):
            FP8Linear(4, 4, False, torch.float16)

    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_storage_dtypes_and_markers(self, fp8_dtype):
        m = FP8Linear(6, 8, True, fp8_dtype)
        assert m.weight.dtype == fp8_dtype
        assert m.weight_scale.dtype == torch.float32
        assert m.weight_scale.shape == ()
        assert m.bias is not None and m.bias.dtype == torch.float32
        assert m._quant_resident is True
        assert m.comfy_cast_weights is True
        assert m.weight_comfy_model_dtype == fp8_dtype

    def test_no_bias(self):
        m = FP8Linear(6, 8, False, torch.float8_e4m3fn)
        assert m.bias is None

    def test_meta_init_compatible(self):
        with torch.device("meta"):
            m = FP8Linear(6, 8, False, torch.float8_e4m3fn)
        assert m.weight.is_meta
        assert m.weight_scale.is_meta
        assert tuple(m.weight.shape) == (8, 6)


@requires_kitchen
class TestFP8LinearForward:
    @pytest.mark.parametrize("act_dtype", [torch.bfloat16, torch.float32])
    def test_parity_bit_exact(self, act_dtype):
        m = _fill(FP8Linear(6, 8, False, torch.float8_e4m3fn), seed=1)
        x = torch.randn(3, 6, dtype=act_dtype)
        out = m(x)
        assert out.dtype == act_dtype
        assert torch.equal(out, _reference(m, x))

    def test_parity_e5m2(self):
        m = _fill(FP8Linear(6, 8, False, torch.float8_e5m2), seed=2)
        x = torch.randn(3, 6, dtype=torch.bfloat16)
        assert torch.equal(m(x), _reference(m, x))

    def test_parity_with_bias(self):
        m = _fill(FP8Linear(6, 8, True, torch.float8_e4m3fn), seed=3)
        x = torch.randn(2, 5, 6, dtype=torch.bfloat16)  # batched 3-D input
        assert torch.equal(m(x), _reference(m, x))

    def test_non_float_activation_raises(self):
        m = _fill(FP8Linear(6, 8, False, torch.float8_e4m3fn))
        with pytest.raises(TypeError):
            m(torch.randint(0, 4, (2, 6)))

    def test_streamed_path_parity(self):
        """A non-empty weight_function forces _forward_streamed; the pulled
        weight must stay fp8 (dtype-preserving) and the result must match."""
        m = _fill(FP8Linear(6, 8, True, torch.float8_e4m3fn), seed=4)
        seen = []

        def _identity(w):
            seen.append(w.dtype)
            return w

        m.weight_function = [_identity]
        x = torch.randn(3, 6, dtype=torch.bfloat16)
        assert torch.equal(m(x), _reference(m, x))
        assert seen == [torch.float8_e4m3fn], "streamed pull must preserve fp8 dtype"
        m.weight_function = []


class TestResidentConstructionIsMetaOnly:
    """2026-09-30: swapping in quant residents must not allocate host RAM.

    The model tree is meta (VibeVoiceLoader._instantiate_model use_meta=True),
    but the replacement factories used to run OUTSIDE that context, so every
    FP8Linear allocated a real torch.empty of its full weight — 8.08 GB for
    the 7B fp8 checkpoint, all of it replaced by the checkpoint assign. The
    live signature was peak_ws 5.6 GB next to peak_private 27.94 GB:
    committed, never touched.
    """

    def test_swapped_residents_start_on_meta(self):
        from ComfyUI_VibeVoice.modules.convrot_quant import QuantLayerInfo
        from ComfyUI_VibeVoice.modules.fp8_quant import make_fp8_linear
        from ComfyUI_VibeVoice.modules.quant_common import replace_linears_for_quant

        class Tree(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(32, 16)

        info = QuantLayerInfo(
            prefix="proj",
            group_size=32,
            in_features=32,
            out_features=16,
            has_bias=True,
            rowwise_dtype=torch.float8_e4m3fn,
            resident_fp8=True,
        )
        with torch.device("meta"):
            tree = Tree()
        replaced = replace_linears_for_quant(tree, {"proj": make_fp8_linear(info)})
        assert replaced == ["proj"]
        weight = dict(tree.named_parameters())["proj.weight"]
        assert weight.is_meta, (
            "the replacement resident allocated real host memory — this is "
            "the 8GB-per-model waste the meta-only construction removes"
        )


class TestFP8LinearStateDict:
    def test_load_consumes_comfy_quant_meta(self):
        m = FP8Linear(6, 8, False, torch.float8_e4m3fn)
        g = torch.Generator().manual_seed(5)
        w = (torch.randn(8, 6, generator=g) * 0.3).to(torch.float8_e4m3fn)
        meta = json.dumps(
            {"format": "float8_e4m3fn", "orig_dtype": "torch.bfloat16"}
        ).encode()
        sd = {
            "weight": w,
            "weight_scale": torch.tensor(1.5),
            "comfy_quant": torch.frombuffer(bytearray(meta), dtype=torch.uint8),
        }
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert list(missing) == []
        assert list(unexpected) == []
        assert torch.equal(m.weight, w)
        assert m.weight_scale.item() == pytest.approx(1.5)

    def test_state_dict_roundtrip_keeps_storage_dtypes(self):
        m = _fill(FP8Linear(6, 8, True, torch.float8_e4m3fn), seed=6)
        sd = m.state_dict()
        assert set(sd.keys()) == {"weight", "weight_scale", "bias"}
        assert sd["weight"].dtype == torch.float8_e4m3fn
        assert sd["weight_scale"].dtype == torch.float32


class TestMakeFP8Linear:
    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_factory_builds_matching_module(self, fp8_dtype):
        info = QuantLayerInfo(
            prefix="model.language_model.layers.0.self_attn.q_proj",
            group_size=0,
            in_features=6,
            out_features=8,
            convrot=False,
            rowwise_dtype=fp8_dtype,
        )
        mod = make_fp8_linear(info)(6, 8, True)
        assert isinstance(mod, FP8Linear)
        assert mod.fp8_dtype == fp8_dtype
        assert mod.bias is not None


# ====================================================================
# Ranked hypothesis (task fp8-peak, 2026-09-29): "_quant_protected_names
# fails to protect FP8Linear, so cast_model_to_dtype_if_needed dequantises
# the resident fp8 weights to bf16 — 9.47GB of fp8 becoming ~17GB, the
# measured 1.8x on the 7B file."
#
# MEASURED, 2026-09-29: REFUTED. The ranked hypothesis was the whole defect
# story and it does not hold. ``_quant_protected_names`` protects the fp8
# weight and the fp32 scale of every ``_quant_resident`` module
# (modules/dtype_utils.py:138-152), and FP8Linear sets that marker
# (modules/fp8_quant.py:93), so the cast skips the residents. These tests
# exist to keep it refuted: the protection is load-bearing for a 1.8x host-RAM
# claim and a future edit that drops the marker (or the weight entry) would
# silently reintroduce it, with no other test noticing.
# ====================================================================

class _WitnessModel(nn.Module):
    """fp8 resident + one mismatched fp32 linear, so the cast is NOT a no-op.

    The plain fp32 linear is the control: it proves the cast actually ran.
    Without a mismatched castable parameter, ``cast_model_to_dtype_if_needed``
    takes its fast path and returns untouched — which would make every
    "storage survived" assertion below pass vacuously.
    """

    def __init__(self, in_features=8, out_features=4):
        super().__init__()
        self.proj = FP8Linear(in_features, out_features, True,
                              torch.float8_e4m3fn, torch.bfloat16)
        self.head = nn.Linear(out_features, out_features, bias=False)
        self.head.weight.data = torch.zeros(out_features, out_features,
                                            dtype=torch.float32)


def _fp8_resident_tree():
    model = _WitnessModel()
    _fill(model.proj, seed=3, scale=0.75)
    return model


class TestFp8ResidentSurvivesDtypeCast:
    """KB-scale: a resident fp8 param keeps float8 STORAGE after the cast."""

    def test_cast_runs_and_leaves_fp8_storage_untouched(self):
        model = _fp8_resident_tree()
        before_bytes = model.proj.weight.untyped_storage().nbytes()
        assert before_bytes == 4 * 8  # 1 byte per fp8 element, no rounding

        cast_model_to_dtype_if_needed(model, torch.bfloat16)

        # Control: the cast really happened (else the rest proves nothing).
        assert model.head.weight.dtype == torch.bfloat16
        # The claim under test.
        assert model.proj.weight.dtype == torch.float8_e4m3fn
        assert model.proj.weight_scale.dtype == torch.float32
        assert model.proj.weight.untyped_storage().nbytes() == before_bytes
        assert model.proj.weight_scale.item() == pytest.approx(0.75)

    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_both_fp8_storage_dtypes_survive(self, fp8_dtype):
        model = _WitnessModel()
        model.proj = FP8Linear(8, 4, True, fp8_dtype, torch.bfloat16)
        _fill(model.proj, seed=4)

        cast_model_to_dtype_if_needed(model, torch.bfloat16)

        assert model.proj.weight.dtype == fp8_dtype
        assert model.proj.weight_scale.dtype == torch.float32

    def test_unconditional_cast_model_to_dtype_also_protects(self):
        """``cast_model_to_dtype`` shares the filtered walk, so it is safe too."""
        model = _fp8_resident_tree()
        cast_model_to_dtype(model, torch.bfloat16)
        assert model.proj.weight.dtype == torch.float8_e4m3fn

    def test_fp8_bytes_stay_half_the_bf16_footprint(self):
        """The 1.8x hypothesis, in bytes: fp8 storage is HALF a bf16 recast.

        9.47GB of fp8 is ~17GB if the residents are cast — the exact shape of
        the reported 25->42GB spike. With the protection in place the model
        keeps the fp8 footprint, so the ceiling of the quant route is ~1x
        file (owned copies), not ~2x.
        """
        model = _fp8_resident_tree()
        fp8_bytes = model.proj.weight.numel()
        cast_model_to_dtype_if_needed(model, torch.bfloat16)
        assert model.proj.weight.untyped_storage().nbytes() == fp8_bytes
        assert fp8_bytes * 2 == model.proj.weight.numel() * 2  # bf16 would be this

    def test_census_reads_the_resident_as_private_fp8(self):
        """The census half of the same claim, at the same scale.

        A quant-assigned parameter is a PRIVATE host allocation BY DESIGN
        (the stream assigns owned clones — under the aimdo arm they are
        cloned from zero-copy file views, see modules/base_loader.py) —
        so the census must count it as private
        fp8 bytes at 1 byte/element, never as an aimdo file view. That 1x is
        the ceiling of the quant route: with views the family would read as
        page cache, and with a bf16 recast it would read as twice these bytes.
        """
        from ComfyUI_VibeVoice.modules.memory_census import census

        model = _fp8_resident_tree()
        cast_model_to_dtype_if_needed(model, torch.bfloat16)
        report = census(model)

        # No view, no mmap: the whole tree is private host memory.
        assert report["param_view_bytes"] == 0
        assert report["param_mmap_bytes"] == 0
        assert report["param_private_bytes"] == sum(
            p.untyped_storage().nbytes() for p in model.parameters())
        # The resident weight is counted at fp8 width (1 byte/element) plus
        # its fp32 scalar scale and its (castable, so cast) bf16 bias.
        assert report["families"]["FP8Linear/params"] == (
            model.proj.weight.numel()
            + model.proj.weight_scale.untyped_storage().nbytes()
            + model.proj.bias.untyped_storage().nbytes()
        )
        assert report["families"]["Linear/params"] == model.head.weight.numel() * 2
