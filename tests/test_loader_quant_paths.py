"""Phase D tests: quant-resident paths through load_external_vibevoice_model.

Exercises the REAL loader with a synthetic GGUF file and a REAL stub module
tree (meta-initialized, like production), asserting:
- residents are GGUFLinear modules holding raw uint8 bytes,
- dense floats keep native dtypes (BF16 embed stays BF16),
- NO gguf.dequantize call happens on the load path (the RAM-spike proof),
- bundle records weight_family/quant_stats,
- ConvRot safetensors branch swaps linears before assignment.
"""

import json
import pytest
import torch
from unittest.mock import MagicMock, patch

import gguf
from safetensors.torch import save_file

from ComfyUI_VibeVoice.modules import external_loader as EL
from ComfyUI_VibeVoice.modules.external_loader import (
    _install_gguf_weights,
    load_external_vibevoice_model,
)
from ComfyUI_VibeVoice.modules.gguf_quant import GGUFLinear
from ComfyUI_VibeVoice.modules.convrot_quant import ConvRotInt8Linear
from ComfyUI_VibeVoice.modules.fp8_quant import FP8Linear
from conftest import build_stub_vv, stub_vv_gguf_spec


class _FakeStreamingCfg:
    pass


@pytest.fixture
def stubbed_load_env():
    """Common mocks for load_external_vibevoice_model around a REAL model tree.

    Yields (run_fn, state) where run_fn(weight_path, **kw) executes the loader.
    Pass dims=(n_layers, hidden, ffn) to match the GGUF spec under test
    (K-quants require in_features % 256 == 0).
    """
    holder = {"dims": (1, 256, 512)}

    def _instantiate(config, is_streaming, attn_implementation, final_load_dtype,
                     use_meta=True):
        n_layers, hidden, ffn = holder["dims"]
        return build_stub_vv(n_layers=n_layers, hidden=hidden, ffn=ffn)

    def _run(weight_path, dims=None, **kwargs):
        if dims is not None:
            holder["dims"] = dims
        kwargs.setdefault("config_name", "VibeVoice-1.5B")
        kwargs.setdefault("attention_mode", "sdpa")
        kwargs.setdefault("use_llm_4bit", False)
        kwargs.setdefault("dtype_str", "auto")
        with patch.object(EL.VibeVoiceLoader, "_load_config",
                          return_value=holder.get("config", MagicMock())), \
             patch.object(EL.VibeVoiceLoader, "_load_tokenizer", return_value=MagicMock()), \
             patch.object(EL.VibeVoiceLoader, "_load_processor", return_value=MagicMock()), \
             patch.object(EL.VibeVoiceLoader, "_instantiate_model",
                          side_effect=_instantiate), \
             patch.object(EL, "resolve_sidecar_config", return_value="/fake/config.json"), \
             patch.object(EL, "resolve_sidecar_preprocessor", return_value=""), \
             patch.object(EL, "resolve_sidecar_tokenizer_dir", return_value="/fake/dir"), \
             patch.object(EL, "resolve_dtype", return_value=torch.bfloat16), \
             patch.object(EL, "resolve_attention_mode", side_effect=lambda m, q: m), \
             patch.object(EL, "get_attn_implementation_for_load", return_value="eager"), \
             patch.object(EL, "VibeVoiceStreamingConfig", _FakeStreamingCfg), \
             patch.object(EL.model_management, "get_torch_device",
                          return_value=torch.device("cpu")):
            bundle = load_external_vibevoice_model(
                weight_path=str(weight_path), **kwargs
            )
        holder["bundle"] = bundle
        holder["model"] = bundle["model"]
        return bundle

    return _run, holder


class TestGGUFLazyInstall:
    def test_residents_installed_and_dense_kept_native(
            self, make_gguf_file, stubbed_load_env):
        run, _ = stubbed_load_env
        path = make_gguf_file(stub_vv_gguf_spec(n_layers=1, qtype="Q8_0"),
                              tag="lazy")

        with patch.object(gguf, "dequantize",
                          side_effect=AssertionError(
                              "legacy full dequant ran on the lazy path")):
            bundle = run(path, dims=(1, 64, 128))

        assert bundle["weight_family"] == "gguf_block"
        assert bundle["quant_stats"]["n_resident_layers"] == 7

        model = bundle["model"]
        q = model.model.language_model.layers[0].self_attn.q_proj
        assert isinstance(q, GGUFLinear)
        assert q.weight.dtype == torch.uint8
        assert not q.weight.is_meta
        assert q._quant_resident

        # Dense passthrough keeps NATIVE dtype through install+cast:
        # BF16 embed stays BF16 under a bf16 target dtype.
        embed = model.model.language_model.embed_tokens.weight
        assert embed.dtype == torch.bfloat16
        assert not embed.is_meta

        # Resident byte accounting equals raw Q8_0 sizes.
        total = sum(
            m.weight.numel() for m in model.modules() if isinstance(m, GGUFLinear)
        )
        assert total == bundle["quant_stats"]["raw_bytes"]
        assert total > 0

    def test_no_full_dequant_on_real_reader(self, make_gguf_file, stubbed_load_env):
        """Second, stricter spike-proof variant: gguf.dequantize must be called
        ZERO times; even indirect numpy float expansion of quant blocks is
        avoided because we only touch .data bytes. Uses Q4_K, which requires
        in_features % 256 == 0 -> hidden=256, ffn=512."""
        run, _ = stubbed_load_env
        path = make_gguf_file(
            stub_vv_gguf_spec(n_layers=1, hidden=256, ffn=512, qtype="Q4_K"),
            tag="kq",
        )
        calls = []
        real_deq = gguf.dequantize

        def _spy(data, qtype):
            calls.append(qtype)
            return real_deq(data, qtype)

        with patch.object(gguf, "dequantize", side_effect=_spy):
            bundle = run(path, dims=(1, 256, 512))
        assert calls == [], "gguf.dequantize must not run on the lazy path"
        assert bundle["weight_family"] == "gguf_block"
        down = bundle["model"].model.language_model.layers[0].mlp.down_proj
        assert isinstance(down, GGUFLinear)
        assert down.ggml_type == gguf.constants.GGMLQuantizationType.Q4_K

    def test_forward_works_after_install(self, make_gguf_file, stubbed_load_env):
        """Smoke: a resident linear produces finite outputs post-install."""
        run, _ = stubbed_load_env
        path = make_gguf_file(stub_vv_gguf_spec(n_layers=1, qtype="Q8_0"),
                              tag="fwd")
        bundle = run(path, dims=(1, 64, 128))
        q = bundle["model"].model.language_model.layers[0].self_attn.q_proj
        x = torch.randn(2, q.in_features)
        y = q(x)
        assert y.shape == (2, q.out_features)
        assert torch.isfinite(y).all()


class TestConvRotLoaderBranch:
    def test_convrot_checkpoint_replaces_linears(self, tmp_path, stubbed_load_env):
        run, _ = stubbed_load_env
        prefix = "model.language_model.layers.0.self_attn.q_proj"
        tensors = {
            f"{prefix}.weight": torch.randint(-100, 100, (64, 64), dtype=torch.int8),
            f"{prefix}.weight_scale": torch.full((64, 1), 0.01),
        }
        meta = json.dumps({
            "format": "int8_tensorwise", "convrot": True,
            "convrot_groupsize": 16,
        }).encode("utf-8")
        tensors[f"{prefix}.comfy_quant"] = torch.frombuffer(
            bytearray(meta), dtype=torch.uint8
        )
        p = tmp_path / "convrot.safetensors"
        save_file(tensors, str(p))

        bundle = run(p, dims=(1, 64, 128))

        assert bundle["weight_family"] == "convrot_int8"
        assert bundle["quant_stats"]["n_resident_layers"] == 1
        q = bundle["model"].model.language_model.layers[0].self_attn.q_proj
        assert isinstance(q, ConvRotInt8Linear)
        assert q.weight.dtype == torch.int8
        assert not q.weight.is_meta
        assert q.convrot_groupsize == 16

    def test_convrot_with_bnb_rejected(self, tmp_path, stubbed_load_env):
        run, _ = stubbed_load_env
        prefix = "model.prediction_head.cond_proj"  # exists in the stub tree
        tensors = {
            f"{prefix}.weight": torch.randint(-100, 100, (64, 64), dtype=torch.int8),
            f"{prefix}.weight_scale": torch.full((64, 1), 0.01),
        }
        meta = json.dumps({"format": "int8_tensorwise", "convrot": True,
                           "convrot_groupsize": 64}).encode("utf-8")
        tensors[f"{prefix}.comfy_quant"] = torch.frombuffer(
            bytearray(meta), dtype=torch.uint8
        )
        p = tmp_path / "convrot2.safetensors"
        save_file(tensors, str(p))

        from ComfyUI_VibeVoice.modules.quant_common import QuantTargetMismatch
        with pytest.raises((ValueError, QuantTargetMismatch)):
            run(p, dims=(1, 64, 128), use_llm_4bit=True)


class TestGGUFInstallUnitLevel:
    def test_unmapped_module_raises_actionable(self, make_gguf_file):
        """A quantized tensor targeting a non-existent module fails loudly."""
        spec = [("does.not.exist.weight", "Q8_0", (32, 32))]
        path = make_gguf_file(spec, tag="badmap")
        reader = gguf.GGUFReader(str(path))
        model = build_stub_vv()
        from ComfyUI_VibeVoice.modules.quant_common import QuantTargetMismatch

        # Either layer of the pipeline reports the offending key: the
        # key-mapper (UnmappedKeyError, a ValueError) or the module
        #-resolution check (QuantTargetMismatch).
        with pytest.raises((QuantTargetMismatch, ValueError), match="does.not.exist"):
            _install_gguf_weights(model, reader)

    def test_shape_mismatch_raises(self, make_gguf_file):
        spec = [("model.language_model.norm.weight", "Q8_0", (128, 32))]
        # norm is a plain parameter (not Linear); shape (32,) vs file (32,128)
        path = make_gguf_file(spec, tag="badshape")
        reader = gguf.GGUFReader(str(path))
        model = build_stub_vv()
        from ComfyUI_VibeVoice.modules.quant_common import QuantTargetMismatch

        with pytest.raises(QuantTargetMismatch):
            _install_gguf_weights(model, reader)


class TestASRGGUFPath:
    def test_asr_gguf_lazy_branch(self, make_gguf_file):
        """The ASR branch uses the same lazy installer for GGUF files."""
        from ComfyUI_VibeVoice.modules.external_loader import (
            load_external_vibevoice_asr_model,
        )

        class _StubASR(torch.nn.Module):
            """Exposes the stub tree at top level so checkpoint keys
            (model.language_model.*, lm_head.*) resolve directly."""

            def __init__(self, config):
                super().__init__()
                vv = build_stub_vv(n_layers=1)
                self.model = vv.model
                self.lm_head = vv.lm_head

        path = make_gguf_file(stub_vv_gguf_spec(n_layers=1, qtype="Q8_0"),
                              tag="asr")
        with patch.object(EL, "VibeVoiceASRForConditionalGeneration", _StubASR), \
             patch.object(EL, "_load_asr_config", return_value=MagicMock()), \
             patch.object(EL, "_load_asr_tokenizer", return_value=MagicMock()), \
             patch.object(EL, "_load_asr_processor", return_value=MagicMock()), \
             patch.object(EL, "resolve_sidecar_config", return_value="/fake/c.json"), \
             patch.object(EL, "resolve_sidecar_preprocessor", return_value=""), \
             patch.object(EL, "resolve_sidecar_tokenizer_dir", return_value="/fake/d"), \
             patch.object(EL, "resolve_dtype", return_value=torch.float32), \
             patch.object(EL, "resolve_attention_mode",
                          side_effect=lambda m, quantize_4bit=False: m), \
             patch.object(EL, "get_attn_implementation_for_load", return_value="eager"), \
             patch.object(EL.model_management, "get_torch_device",
                          return_value=torch.device("cpu")), \
             patch.object(gguf, "dequantize",
                          side_effect=AssertionError("full dequant on ASR path")):
            bundle = load_external_vibevoice_asr_model(
                weight_path=str(path), config_name="VibeVoice-ASR",
                attention_mode="sdpa", dtype_str="auto",
            )
        assert bundle["is_asr"] is True
        assert bundle["weight_family"] == "gguf_block"


# ====================================================================
# Rowwise int8 / fp8 dequant-at-load + hard-fail propagation
# ====================================================================

import json as _json

from ComfyUI_VibeVoice.modules.convrot_quant import QuantLayerInfo
from ComfyUI_VibeVoice.modules.external_loader import (
    _assert_dense_loadable,
    _prepare_quantized_safetensors_load,
)


def _save_qcheckpoint(path, entries):
    """entries: list of (prefix, kind, shape) where kind in
    {'convrot', 'rowwise', 'rowwise_fp8'}."""
    from safetensors.torch import save_file

    tensors = {}
    for prefix, kind, shape in entries:
        if kind == "rowwise_fp8":
            tensors[f"{prefix}.weight"] = torch.randint(
                -100, 100, shape).to(torch.float8_e4m3fn)
            # Real fp8 checkpoints carry PER-TENSOR scalar scales.
            tensors[f"{prefix}.weight_scale"] = torch.tensor(0.5)
        else:
            tensors[f"{prefix}.weight"] = torch.randint(
                -100, 100, shape, dtype=torch.int8)
            if kind == "block":
                gs = 16
                tensors[f"{prefix}.weight_scale"] = torch.rand(
                    (shape[0] // gs, shape[1] // gs),
                    dtype=torch.float32) * 0.1
            else:
                tensors[f"{prefix}.weight_scale"] = torch.full(
                    (shape[0], 1), 0.5)
        if kind == "convrot":
            meta = {"convrot": True, "convrot_groupsize": 16,
                    "format": "int8_tensorwise",
                    "orig_dtype": "torch.bfloat16"}
        elif kind == "rowwise":
            meta = {"format": "int8_tensorwise", "per_row": True,
                    "orig_dtype": "torch.bfloat16"}
        elif kind == "block":
            meta = {"format": "int8_blockwise", "group_size": 16,
                    "orig_dtype": "torch.bfloat16"}
        else:
            meta = {"format": "float8_e4m3fn",
                    "orig_dtype": "torch.bfloat16"}
        tensors[f"{prefix}.comfy_quant"] = torch.frombuffer(
            bytearray(_json.dumps(meta).encode("utf-8")), dtype=torch.uint8
        )
    save_file(tensors, str(path))
    return path


class TestPrepareQuantizedSafetensorsLoad:
    def test_rowwise_dequant_math_and_key_stripping(self):
        sd = {
            "l.weight": torch.tensor([[100, -101], [50, -25]],
                                     dtype=torch.int8),
            "l.weight_scale": torch.tensor([[0.5], [2.0]]),
            "l.comfy_quant": torch.zeros(4, dtype=torch.uint8),
            "l.bias": torch.zeros(2),
        }
        qmap = {"l": QuantLayerInfo(
            prefix="l", group_size=0, in_features=2, out_features=2,
            convrot=False, orig_dtype="torch.bfloat16",
        )}
        plan, n_rowwise, n_fp8 = _prepare_quantized_safetensors_load(sd, qmap)

        assert n_rowwise == 1 and plan == {}
        expected = (
            torch.tensor([[100., -101.], [50., -25.]])
            * torch.tensor([[0.5], [2.0]])
        ).to(torch.bfloat16)
        assert sd["l.weight"].dtype == torch.bfloat16
        assert torch.equal(sd["l.weight"], expected)
        assert "l.weight_scale" not in sd
        assert "l.comfy_quant" not in sd
        assert "l.bias" in sd  # untouched

    def test_convrot_entries_become_plan_and_keep_scale(self):
        sd = {
            "c.weight": torch.zeros(64, 64, dtype=torch.int8),
            "c.weight_scale": torch.full((64, 1), 0.01),
            "c.comfy_quant": torch.zeros(3, dtype=torch.uint8),
        }
        qmap = {"c": QuantLayerInfo(
            prefix="c", group_size=16, in_features=64, out_features=64,
            convrot=True,
        )}
        plan, n_rowwise, n_fp8 = _prepare_quantized_safetensors_load(sd, qmap)

        assert n_rowwise == 0 and set(plan) == {"c"}
        assert "c.weight_scale" in sd      # consumed by resident assign
        assert "c.comfy_quant" not in sd   # metadata stripped up front

    def test_missing_scale_raises(self):
        sd = {"l.weight": torch.zeros(2, 2, dtype=torch.int8)}
        qmap = {"l": QuantLayerInfo(
            prefix="l", group_size=0, convrot=False,
            orig_dtype="torch.bfloat16",
        )}
        with pytest.raises(Exception, match="missing its"):
            _prepare_quantized_safetensors_load(sd, qmap)

    def test_scalar_scale_accepted(self):
        sd = {
            "l.weight": torch.tensor([[100, -101]], dtype=torch.int8),
            "l.weight_scale": torch.tensor(0.5),
        }
        qmap = {"l": QuantLayerInfo(
            prefix="l", group_size=0, in_features=2, out_features=1,
            convrot=False, orig_dtype="torch.bfloat16",
        )}
        plan, n_rowwise, n_fp8 = _prepare_quantized_safetensors_load(sd, qmap)
        assert n_rowwise == 1
        assert torch.equal(
            sd["l.weight"],
            torch.tensor([[50.0, -50.5]]).to(torch.bfloat16),
        )

    def test_bad_scale_shape_raises(self):
        sd = {
            "l.weight": torch.zeros(2, 2, dtype=torch.int8),
            "l.weight_scale": torch.zeros(2),  # neither scalar nor [out,1]
        }
        qmap = {"l": QuantLayerInfo(
            prefix="l", group_size=0, convrot=False,
            orig_dtype="torch.bfloat16", out_features=2, in_features=2,
        )}
        with pytest.raises(Exception, match="neither a scalar"):
            _prepare_quantized_safetensors_load(sd, qmap)

    def test_weight_metadata_mismatch_raises(self):
        sd = {
            "l.weight": torch.zeros(2, 2, dtype=torch.int8),
            "l.weight_scale": torch.tensor(0.5),
        }
        qmap = {"l": QuantLayerInfo(
            prefix="l", group_size=0, convrot=False,
            orig_dtype="torch.bfloat16", out_features=4, in_features=4,
        )}
        with pytest.raises(Exception, match="disagrees with metadata"):
            _prepare_quantized_safetensors_load(sd, qmap)


class TestRowwiseLoaderIntegration:
    def test_mixed_convrot_rowwise_end_to_end(self, tmp_path, stubbed_load_env):
        """Real-world layout: some layers rotated residents, the rest plain
        rowwise dequantized at load (the user's int8_convrot file)."""
        run, holder = stubbed_load_env
        p = _save_qcheckpoint(tmp_path / "mixed.safetensors", [
            ("model.language_model.layers.0.self_attn.q_proj",
             "convrot", (64, 64)),
            ("model.prediction_head.cond_proj", "rowwise", (64, 64)),
        ])

        bundle = run(p, dims=(1, 64, 128))

        assert bundle["weight_family"] == "convrot_int8"
        assert bundle["quant_stats"] == {
            "n_resident_layers": 1,
            "n_rowwise_layers": 1,
            "n_fp8_resident_layers": 0,
        }
        q = bundle["model"].model.language_model.layers[0].self_attn.q_proj
        assert isinstance(q, ConvRotInt8Linear)

        cond = bundle["model"].model.prediction_head.cond_proj
        assert isinstance(cond, torch.nn.Linear)
        assert not isinstance(cond, ConvRotInt8Linear)
        # Dequanted value matches q * per-row scale exactly (bf16 cast).
        from safetensors.torch import load_file
        raw = load_file(str(p))
        ref_w = (raw["model.prediction_head.cond_proj.weight"].to(torch.float32)
                 * raw["model.prediction_head.cond_proj.weight_scale"]).to(torch.bfloat16)
        assert cond.weight.dtype == torch.bfloat16
        assert torch.equal(cond.weight.data, ref_w)

    def test_fp8_per_row_scale_dequants_at_load(self, tmp_path, stubbed_load_env):
        """[out, 1] per-row fp8 scales cannot run through the per-tensor
        kitchen kernel -> legacy dequant-at-load (plan 2026-08-27, D2)."""
        run, holder = stubbed_load_env
        from safetensors.torch import save_file as _sf

        prefix = "model.prediction_head.cond_proj"
        tensors = {
            f"{prefix}.weight": torch.randint(-100, 100, (64, 64)).to(
                torch.float8_e4m3fn),
            f"{prefix}.weight_scale": torch.full((64, 1), 0.5),
        }
        tensors[f"{prefix}.comfy_quant"] = torch.frombuffer(
            bytearray(_json.dumps({"format": "float8_e4m3fn",
                                   "orig_dtype": "torch.bfloat16"})
                      .encode("utf-8")), dtype=torch.uint8
        )
        p = tmp_path / "fp8_perrow.safetensors"
        _sf(tensors, str(p))

        bundle = run(p, dims=(1, 64, 128))

        assert bundle["quant_stats"]["n_rowwise_layers"] == 1
        assert bundle["quant_stats"]["n_fp8_resident_layers"] == 0
        cond = bundle["model"].model.prediction_head.cond_proj
        assert not isinstance(cond, FP8Linear)  # stays a plain Linear
        assert cond.weight.dtype == torch.bfloat16
        ref_w = (tensors[f"{prefix}.weight"].float()
                 * tensors[f"{prefix}.weight_scale"]).to(torch.bfloat16)
        assert torch.equal(cond.weight.data, ref_w)


class TestFP8ResidentLoading:
    """Scalar-scale fp8 checkpoints execute resident (plan 2026-08-27, D1)."""

    def test_fp8_scalar_scale_resident_end_to_end(self, tmp_path, stubbed_load_env):
        run, holder = stubbed_load_env
        p = _save_qcheckpoint(tmp_path / "fp8.safetensors", [
            ("model.prediction_head.cond_proj", "rowwise_fp8", (64, 64)),
        ])

        bundle = run(p, dims=(1, 64, 128))

        assert bundle["weight_family"] == "fp8_resident"
        assert bundle["quant_stats"] == {
            "n_resident_layers": 1,
            "n_rowwise_layers": 0,
            "n_fp8_resident_layers": 1,
        }
        cond = bundle["model"].model.prediction_head.cond_proj
        assert isinstance(cond, FP8Linear)
        assert cond.weight.dtype == torch.float8_e4m3fn
        assert not cond.weight.is_meta
        assert cond.weight_scale.dtype == torch.float32
        assert not cond.weight_scale.is_meta

        # Forward parity vs the manual dequant reference (bit-exact eager).
        from safetensors.torch import load_file
        raw = load_file(str(p))
        w = raw["model.prediction_head.cond_proj.weight"]
        s = raw["model.prediction_head.cond_proj.weight_scale"]
        x = torch.randn(2, 64, dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(
            x, (w.float() * s).to(torch.bfloat16)
        )
        assert torch.equal(cond(x), ref)

    def test_fp8_resident_survives_final_dtype_cast(self, tmp_path,
                                                    stubbed_load_env):
        """cast_model_to_dtype_if_needed(bf16) after load must not touch the
        fp8 storage or the fp32 scale (_quant_protected_names)."""
        run, holder = stubbed_load_env
        p = _save_qcheckpoint(tmp_path / "fp8cast.safetensors", [
            ("model.prediction_head.cond_proj", "rowwise_fp8", (64, 64)),
        ])
        bundle = run(p, dims=(1, 64, 128))
        cond = bundle["model"].model.prediction_head.cond_proj
        assert cond.weight.dtype == torch.float8_e4m3fn
        assert cond.weight_scale.dtype == torch.float32

    def test_fp8_no_backend_falls_back_to_dequant(self, tmp_path,
                                                  stubbed_load_env):
        """Without a kitchen fp8 backend the scalar-scale file dequantizes at
        load (correct, just heavier) instead of failing (plan D3)."""
        run, holder = stubbed_load_env
        from ComfyUI_VibeVoice.modules import fp8_quant as FQ

        p = _save_qcheckpoint(tmp_path / "fp8fb.safetensors", [
            ("model.prediction_head.cond_proj", "rowwise_fp8", (64, 64)),
        ])
        with patch.object(FQ, "probe_fp8_backend", return_value=None):
            bundle = run(p, dims=(1, 64, 128))

        assert bundle["quant_stats"]["n_rowwise_layers"] == 1
        assert bundle["quant_stats"]["n_fp8_resident_layers"] == 0
        cond = bundle["model"].model.prediction_head.cond_proj
        assert not isinstance(cond, FP8Linear)
        assert cond.weight.dtype == torch.bfloat16
        assert torch.isfinite(cond.weight.data.float()).all()

    def test_mixed_convrot_and_fp8_resident(self, tmp_path, stubbed_load_env):
        """ConvRot int8 + scalar fp8 in one file: both resident families
        install; the family label stays convrot_int8 (it dominates)."""
        run, holder = stubbed_load_env
        p = _save_qcheckpoint(tmp_path / "mixfp8.safetensors", [
            ("model.language_model.layers.0.self_attn.q_proj",
             "convrot", (64, 64)),
            ("model.prediction_head.cond_proj", "rowwise_fp8", (64, 64)),
        ])

        bundle = run(p, dims=(1, 64, 128))

        assert bundle["weight_family"] == "convrot_int8"
        assert bundle["quant_stats"] == {
            "n_resident_layers": 2,
            "n_rowwise_layers": 0,
            "n_fp8_resident_layers": 1,
        }
        q = bundle["model"].model.language_model.layers[0].self_attn.q_proj
        assert isinstance(q, ConvRotInt8Linear)
        cond = bundle["model"].model.prediction_head.cond_proj
        assert isinstance(cond, FP8Linear)

    def test_prepare_routes_resident_fp8_and_keeps_tensors(self):
        sd = {
            "l.weight": torch.randint(-100, 100, (4, 4)).to(
                torch.float8_e4m3fn),
            "l.weight_scale": torch.tensor(0.25),
            "l.comfy_quant": torch.zeros(3, dtype=torch.uint8),
        }
        qmap = {"l": QuantLayerInfo(
            prefix="l", group_size=0, in_features=4, out_features=4,
            convrot=False, orig_dtype="torch.bfloat16",
            rowwise_dtype=torch.float8_e4m3fn, resident_fp8=True,
        )}
        plan, n_rowwise, n_fp8 = _prepare_quantized_safetensors_load(sd, qmap)

        assert n_rowwise == 0 and n_fp8 == 1 and set(plan) == {"l"}
        # Storage stays fp8 + scale stays in the dict for assign.
        assert sd["l.weight"].dtype == torch.float8_e4m3fn
        assert "l.weight_scale" in sd
        assert "l.comfy_quant" not in sd

    def test_prepare_resident_fp8_rejects_wrong_storage_dtype(self):
        sd = {
            "l.weight": torch.zeros(4, 4, dtype=torch.int8),
            "l.weight_scale": torch.tensor(0.25),
        }
        qmap = {"l": QuantLayerInfo(
            prefix="l", group_size=0, in_features=4, out_features=4,
            convrot=False, orig_dtype="torch.bfloat16",
            rowwise_dtype=torch.float8_e4m3fn, resident_fp8=True,
        )}
        with pytest.raises(Exception, match="expected"):
            _prepare_quantized_safetensors_load(sd, qmap)

    def test_prepare_resident_fp8_rejects_per_row_scale(self):
        sd = {
            "l.weight": torch.randint(-100, 100, (4, 4)).to(
                torch.float8_e4m3fn),
            "l.weight_scale": torch.full((4, 1), 0.25),
        }
        qmap = {"l": QuantLayerInfo(
            prefix="l", group_size=0, in_features=4, out_features=4,
            convrot=False, orig_dtype="torch.bfloat16",
            rowwise_dtype=torch.float8_e4m3fn, resident_fp8=True,
        )}
        with pytest.raises(Exception, match="not a per-tensor scalar"):
            _prepare_quantized_safetensors_load(sd, qmap)


class TestTiedLmHeadGuard:
    """A quantized lm_head contradicts tie_word_embeddings (plan 2026-08-27)."""

    def _cfg(self, tied):
        class _Cfg:
            pass

        c = _Cfg()
        c.tie_word_embeddings = tied
        return c

    def test_tied_config_with_quantized_lm_head_raises(self):
        from ComfyUI_VibeVoice.modules.external_loader import (
            _assert_lm_head_not_tied,
        )
        from ComfyUI_VibeVoice.modules.quant_common import QuantTargetMismatch

        qmap = {"lm_head": QuantLayerInfo(prefix="lm_head", group_size=0)}
        with pytest.raises(QuantTargetMismatch, match="tie_word_embeddings"):
            _assert_lm_head_not_tied(self._cfg(True), qmap)

    def test_untied_config_passes(self):
        from ComfyUI_VibeVoice.modules.external_loader import (
            _assert_lm_head_not_tied,
        )

        qmap = {"lm_head": QuantLayerInfo(prefix="lm_head", group_size=0)}
        _assert_lm_head_not_tied(self._cfg(False), qmap)  # no raise

    def test_tied_config_without_lm_head_quant_passes(self):
        from ComfyUI_VibeVoice.modules.external_loader import (
            _assert_lm_head_not_tied,
        )

        qmap = {"model.layers.0.mlp": QuantLayerInfo(
            prefix="model.layers.0.mlp", group_size=0)}
        _assert_lm_head_not_tied(self._cfg(True), qmap)  # no raise

    def test_magicmock_config_not_treated_as_tied(self):
        """Test doubles must not read as tied (non-bool flags ignored)."""
        from ComfyUI_VibeVoice.modules.external_loader import (
            _assert_lm_head_not_tied,
        )

        qmap = {"lm_head": QuantLayerInfo(prefix="lm_head", group_size=0)}
        _assert_lm_head_not_tied(MagicMock(), qmap)  # no raise

    def test_loader_rejects_tied_fp8_lm_head_end_to_end(self, tmp_path,
                                                       stubbed_load_env):
        run, holder = stubbed_load_env
        p = _save_qcheckpoint(tmp_path / "tiedlm.safetensors", [
            ("lm_head", "rowwise_fp8", (96, 64)),
        ])
        cfg = MagicMock()
        cfg.tie_word_embeddings = True
        cfg.decoder_config = None
        holder["config"] = cfg

        with pytest.raises(RuntimeError, match="tie_word_embeddings"):
            run(p, dims=(1, 64, 128))

    def test_unknown_format_propagates_not_swallowed(self, tmp_path,
                                                     stubbed_load_env):
        """Unsupported formats must HARD-FAIL the node, never fall back to
        the dense loader (which would silently misload fp8/int8 weights)."""
        run, holder = stubbed_load_env
        p = _save_qcheckpoint(tmp_path / "mystery.safetensors", [])
        from safetensors.torch import save_file

        tensors = {
            "model.prediction_head.cond_proj.weight": torch.zeros(64, 64),
            "model.prediction_head.cond_proj.weight_scale": torch.ones(64, 1),
        }
        tensors["model.prediction_head.cond_proj.comfy_quant"] = (
            torch.frombuffer(
                bytearray(_json.dumps({"format": "brand_new_scheme"})
                          .encode("utf-8")), dtype=torch.uint8)
        )
        save_file(tensors, str(p))

        with pytest.raises(RuntimeError, match="brand_new_scheme"):
            run(p, dims=(1, 64, 128))

    def test_dense_net_rejects_unplanned_int_weights(self, tmp_path,
                                                     stubbed_load_env):
        """A 'quantized' file WITHOUT usable metadata must fail loudly at the
        dense gate instead of crashing inside torch or corrupting output."""
        run, holder = stubbed_load_env
        from safetensors.torch import save_file

        p = tmp_path / "naive.safetensors"
        save_file({
            "model.prediction_head.cond_proj.weight": torch.zeros(
                64, 64, dtype=torch.int8),
            "model.prediction_head.cond_proj.bias": torch.zeros(64),
        }, str(p))

        # Loader wraps load errors as RuntimeError; the actionable message
        # from the dense gate must survive the wrapping.
        with pytest.raises(RuntimeError, match="quantized-weight tensors"):
            run(p, dims=(1, 64, 128))


class TestBlockwiseLoading:
    """int8_blockwise checkpoints (unrotated, [out, in/gs] scales)."""

    def test_scanner_parses_blockwise(self, tmp_path):
        from ComfyUI_VibeVoice.modules.convrot_quant import (
            scan_checkpoint_quantization,
        )

        p = _save_qcheckpoint(tmp_path / "blk.safetensors", [
            ("model.prediction_head.cond_proj", "block", (64, 64)),
        ])
        qmap = scan_checkpoint_quantization(p)
        info = qmap["model.prediction_head.cond_proj"]
        assert info.convrot is False
        assert info.group_size == 16
        assert (info.out_features, info.in_features) == (64, 64)

    def test_blockwise_dequant_math(self):
        # gs=2 -> scale grid (out/2, in/2), one scale per 2x2 block.
        sd = {
            "l.weight": (torch.arange(16, dtype=torch.int8)
                         .reshape(4, 4) - 8),
            "l.weight_scale": torch.tensor([[0.5, 0.25], [2.0, 1.5]]),
        }
        qmap = {"l": QuantLayerInfo(
            prefix="l", group_size=2, in_features=4, out_features=4,
            convrot=False, orig_dtype="torch.float32",
        )}
        plan, n_rowwise, n_fp8 = _prepare_quantized_safetensors_load(sd, qmap)

        assert n_rowwise == 1 and plan == {}
        # Block layout: rows 0-1 scaled by s[:,0] column-group 0, etc.
        expected = torch.tensor([
            [-8 * 0.5, -7 * 0.5, -6 * 0.25, -5 * 0.25],
            [-4 * 0.5, -3 * 0.5, -2 * 0.25, -1 * 0.25],
            [-0 * 2.0, 1 * 2.0, 2 * 1.5, 3 * 1.5],
            [4 * 2.0, 5 * 2.0, 6 * 1.5, 7 * 1.5],
        ])
        assert torch.equal(sd["l.weight"].float(), expected)
        assert sd["l.weight"].dtype == torch.float32
        assert "l.weight_scale" not in sd

    def test_blockwise_end_to_end_through_loader(self, tmp_path,
                                                 stubbed_load_env):
        """File-C layout: whole checkpoint int8_blockwise incl. an
        Embedding-shaped target — dequant-at-load handles any module kind."""
        run, holder = stubbed_load_env
        # cond_proj is Linear; use a big 'embedding-like' target too by
        # pointing at the same Linear tree (module type irrelevant for the
        # dequant path).
        p = _save_qcheckpoint(tmp_path / "blockmodel.safetensors", [
            ("model.prediction_head.cond_proj", "block", (64, 64)),
            ("lm_head.weight-placeholder", "block", (64, 64)),
        ][0:1])

        bundle = run(p, dims=(1, 64, 128))
        assert bundle["quant_stats"] == {
            "n_resident_layers": 0,
            "n_rowwise_layers": 1,
            "n_fp8_resident_layers": 0,
        }
        cond = bundle["model"].model.prediction_head.cond_proj
        assert isinstance(cond, torch.nn.Linear)
        assert not isinstance(cond, ConvRotInt8Linear)
        assert cond.weight.dtype == torch.bfloat16
        assert torch.isfinite(cond.weight.data.float()).all()

    def test_rotated_embedding_fails_with_reexport_guidance(
            self, tmp_path, stubbed_load_env):
        """File-A/B layout: convrot-flagged embed_tokens must fail with an
        actionable message, never a silent misload."""
        run, holder = stubbed_load_env
        from safetensors.torch import save_file

        tensors = {
            "model.language_model.embed_tokens.weight": torch.randint(
                -100, 100, (96, 64), dtype=torch.int8),
            "model.language_model.embed_tokens.weight_scale":
                torch.rand((96, 1)) * 0.01,
            "model.language_model.embed_tokens.comfy_quant":
                torch.frombuffer(bytearray(_json.dumps({
                    "convrot": True, "convrot_groupsize": 16,
                    "format": "int8_tensorwise",
                    "orig_dtype": "torch.bfloat16", "per_row": True,
                }).encode("utf-8")), dtype=torch.uint8),
        }
        p = tmp_path / "rotembed.safetensors"
        save_file(tensors, str(p))

        with pytest.raises(RuntimeError, match="[Rr]e-export"):
            run(p, dims=(1, 64, 128))

    def test_plain_rowwise_embedding_dequants_fine(self, tmp_path,
                                                   stubbed_load_env):
        """Non-Linear targets are fine when NOT rotated (plain rowwise)."""
        run, holder = stubbed_load_env

        class _EmbHolder(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                vv = build_stub_vv(n_layers=1)
                self.model = vv.model
                self.lm_head = vv.lm_head
                self.embed_extra = torch.nn.Embedding(96, 64)

        # quick sanity of the prep path on an Embedding module directly:
        sd = {
            "emb.weight": torch.randint(-50, 50, (96, 64), dtype=torch.int8),
            "emb.weight_scale": torch.full((96, 1), 0.25),
        }
        qmap = {"emb": QuantLayerInfo(
            prefix="emb", group_size=0, in_features=64, out_features=96,
            convrot=False, orig_dtype="torch.bfloat16",
        )}
        plan, n_rowwise, n_fp8 = _prepare_quantized_safetensors_load(sd, qmap)
        assert n_rowwise == 1 and plan == {}
        assert sd["emb.weight"].dtype == torch.bfloat16
        assert torch.isfinite(sd["emb.weight"].float()).all()


class TestOptionalAbsentPrefixes:
    """A released checkpoint may omit a whole subtree; that is not a failure.

    VibeVoice-Realtime-0.5B ships an acoustic-tokenizer decoder only, and
    vanilla ``from_pretrained`` reports the same 276 encoder keys as MISSING.
    Our loader must not print an alarming warning for that, while still
    reporting genuinely missing keys.
    """

    def test_prefix_is_declared(self):
        from ComfyUI_VibeVoice.modules.loader import OPTIONAL_ABSENT_PREFIXES

        assert "acoustic_tokenizer.encoder." in OPTIONAL_ABSENT_PREFIXES

    def test_wholly_omitted_prefix_is_suppressed(self):
        from ComfyUI_VibeVoice.modules.loader import mark_optional_absent

        missing = [
            "acoustic_tokenizer.encoder.stages.0.0.weight",
            "acoustic_tokenizer.encoder.head.conv.bias",
        ]
        assigned = {"acoustic_tokenizer.decoder.head.weight", "model.language_model.x"}

        known = mark_optional_absent(missing, assigned)

        assert known == set(missing)

    def test_partially_supplied_prefix_is_still_reported(self):
        from ComfyUI_VibeVoice.modules.loader import mark_optional_absent

        missing = ["acoustic_tokenizer.encoder.stages.9.9.weight"]
        assigned = {"acoustic_tokenizer.encoder.stages.0.0.weight"}

        known = mark_optional_absent(missing, assigned)

        assert known == set()

    def test_unrelated_missing_keys_are_untouched(self):
        from ComfyUI_VibeVoice.modules.loader import mark_optional_absent

        missing = ["model.language_model.layers.0.weight", "tts_eos_classifier.fc1.bias"]
        assigned = set()

        known = mark_optional_absent(missing, assigned)

        assert known == set()

    def test_existing_known_missing_is_preserved(self):
        from ComfyUI_VibeVoice.modules.loader import mark_optional_absent

        missing = ["acoustic_tokenizer.encoder.head.conv.bias"]
        known = mark_optional_absent(missing, set(), {"lm_head.weight"})

        assert "lm_head.weight" in known
        assert "acoustic_tokenizer.encoder.head.conv.bias" in known

    def test_caller_set_is_not_mutated(self):
        from ComfyUI_VibeVoice.modules.loader import mark_optional_absent

        original = {"lm_head.weight"}
        mark_optional_absent(["acoustic_tokenizer.encoder.head.bias"], set(), original)

        assert original == {"lm_head.weight"}


# ====================================================================
# The realtime encoder namespace, measured rather than assumed
# ====================================================================
#
# Instantiating the real trees on ``torch.device("meta")`` from the packaged
# configs — a probe run OUTSIDE the suite, because conftest mocks the vendored
# model code wholesale — shows the SAME shape for both released architectures
# (streaming 881 state_dict keys, diffusion 1205):
#
#   * every key hangs off ``model.`` (plus ``tts_eos_classifier.`` on the
#     streaming model and ``lm_head.`` on the diffusion one). ZERO keys in
#     either tree begin with the unrooted ``acoustic_tokenizer.``;
#   * 276 acoustic-tokenizer encoder keys, 276 decoder keys.
#
# So the encoder namespace is exactly ``model.acoustic_tokenizer.encoder.``,
# and a prefix written without the ``model.`` root matches nothing at all.
# The tables below rebuild those 276 keys from the measured block layout.

_REALTIME_ENCODER_PREFIX = "model.acoustic_tokenizer.encoder."

# (downsample index, sub-stage index), as measured on the released tree.
_ENCODER_DOWNSAMPLE_BLOCKS = tuple((i, 0) for i in range(7))
# (stage index, residual-layer index): six 3-layer stages, then an 8-layer one.
_ENCODER_STAGE_BLOCKS = tuple(
    [(i, j) for i in range(6) for j in range(3)]
    + [(6, j) for j in range(8)]
)
_ENCODER_DOWNSAMPLE_SUFFIXES = ("conv.conv.weight", "conv.conv.bias")
_ENCODER_BLOCK_SUFFIXES = (
    "gamma", "ffn_gamma", "norm.weight", "ffn_norm.weight",
    "mixer.conv.conv.conv.weight", "mixer.conv.conv.conv.bias",
    "ffn.linear1.weight", "ffn.linear1.bias",
    "ffn.linear2.weight", "ffn.linear2.bias",
)
_ENCODER_HEAD_SUFFIXES = ("conv.conv.weight", "conv.conv.bias")

# Real key names from the same probe, one per remaining subtree. None of them
# may ever be silenced by the optional-absent rule.
_MEASURED_NON_ENCODER_KEYS = (
    "model.acoustic_tokenizer.decoder.upsample_layers.0.0.conv.conv.weight",
    "model.acoustic_tokenizer.decoder.head.conv.conv.weight",
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "tts_eos_classifier.fc1.weight",
)


def _measured_realtime_encoder_keys() -> set[str]:
    """The 276 real encoder keys, rebuilt from the measured block layout."""
    keys = {f"{_REALTIME_ENCODER_PREFIX}head.{s}" for s in _ENCODER_HEAD_SUFFIXES}
    for block, stage in _ENCODER_DOWNSAMPLE_BLOCKS:
        keys.update(
            f"{_REALTIME_ENCODER_PREFIX}downsample_layers.{block}.{stage}.{s}"
            for s in _ENCODER_DOWNSAMPLE_SUFFIXES
        )
    for stage, layer in _ENCODER_STAGE_BLOCKS:
        keys.update(
            f"{_REALTIME_ENCODER_PREFIX}stages.{stage}.{layer}.{s}"
            for s in _ENCODER_BLOCK_SUFFIXES
        )
    return keys


class TestRealtimeOptionalAbsentNamespace:
    """The prefix must name the namespace the checkpoint actually uses.

    ``OPTIONAL_ABSENT_PREFIXES`` used to carry only the unrooted spelling,
    which matches zero keys in either released tree — so a realtime load still
    warned about all 276 encoder keys while the rule looked armed.
    """

    def test_rebuilt_key_set_matches_the_published_count(self):
        # 14 downsample + 260 stage + 2 head, as measured on the real tree.
        assert len(_measured_realtime_encoder_keys()) == 276

    def test_realtime_namespace_prefix_is_declared(self):
        from ComfyUI_VibeVoice.modules.loader import OPTIONAL_ABSENT_PREFIXES

        assert _REALTIME_ENCODER_PREFIX in OPTIONAL_ABSENT_PREFIXES

    def test_every_measured_encoder_key_is_covered(self):
        from ComfyUI_VibeVoice.modules.loader import OPTIONAL_ABSENT_PREFIXES

        uncovered = [
            k for k in _measured_realtime_encoder_keys()
            if not any(k.startswith(p) for p in OPTIONAL_ABSENT_PREFIXES)
        ]
        assert uncovered == []

    def test_only_the_rooted_prefix_matches_the_released_tree(self):
        from ComfyUI_VibeVoice.modules.loader import OPTIONAL_ABSENT_PREFIXES

        matching = [
            p for p in OPTIONAL_ABSENT_PREFIXES
            if any(k.startswith(p) for k in _measured_realtime_encoder_keys())
        ]
        # Drift guard: the unrooted spelling covers nothing here, so if it ever
        # starts matching, the key layout changed and the comment above does.
        assert matching == [_REALTIME_ENCODER_PREFIX]

    def test_omitted_realtime_encoder_is_suppressed(self):
        from ComfyUI_VibeVoice.modules.loader import mark_optional_absent

        missing = sorted(_measured_realtime_encoder_keys())
        assigned = set(_MEASURED_NON_ENCODER_KEYS)

        known = mark_optional_absent(missing, assigned)

        assert known == set(missing)

    def test_partially_supplied_realtime_encoder_is_still_reported(self):
        from ComfyUI_VibeVoice.modules.loader import mark_optional_absent

        missing = sorted(_measured_realtime_encoder_keys())
        # One real encoder key arrived — the rest are a genuine finding.
        assigned = {
            "model.acoustic_tokenizer.encoder.stages.0.0.gamma",
            *_MEASURED_NON_ENCODER_KEYS,
        }

        assert mark_optional_absent(missing, assigned) == set()

    def test_non_encoder_subtrees_are_never_optional(self):
        from ComfyUI_VibeVoice.modules.loader import mark_optional_absent

        # No encoder key was assigned at all, so the rule fires — yet none of
        # these keys may be caught by it.
        known = mark_optional_absent(list(_MEASURED_NON_ENCODER_KEYS), set())

        assert known == set()

    def test_1p5b_style_tree_marks_nothing_optional(self):
        """The 1.5B tree carries a full encoder, and none of its keys match."""
        from ComfyUI_VibeVoice.modules.loader import (
            OPTIONAL_ABSENT_PREFIXES,
            mark_optional_absent,
        )

        stub_keys = set(build_stub_vv().state_dict())
        assert not [
            k for k in stub_keys
            if any(k.startswith(p) for p in OPTIONAL_ABSENT_PREFIXES)
        ]

        # A complete export: nothing missing, nothing marked.
        assert mark_optional_absent([], stub_keys | _measured_realtime_encoder_keys()) == set()

        # A truncated one still reports its own gap. This is the catcher for
        # an over-broad prefix: no encoder key arrived, so the rule fires, and
        # it must not swallow a missing key from the tree that never had one.
        dropped = "model.language_model.layers.0.self_attn.q_proj.weight"
        assert mark_optional_absent([dropped], stub_keys) == set()


class _QuantGuardModel(torch.nn.Module):
    """Tiny real module for the streaming quant-storage guard tests."""

    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(2, 2, bias=False)
        self.layer2 = torch.nn.Linear(2, 2, bias=False)
        self.config = MagicMock()
        self.config.decoder_config.tie_word_embeddings = False
        self.config.tie_word_embeddings = False


class TestStreamApplyDenseQuantGuard:
    """The streaming dense route must reject quantized storages too.

    ``external_loader._assert_dense_loadable`` guards the batch route; without
    a matching check the per-tensor route would happily assign an int8 storage
    into a float parameter. One shared dtype set keeps both routes speaking
    the same message.
    """

    @staticmethod
    def _apply(pairs):
        from ComfyUI_VibeVoice.modules.loader import VibeVoiceLoader

        return VibeVoiceLoader._stream_apply_dense(_QuantGuardModel(), iter(pairs))

    def test_int8_storage_rejected(self):
        with pytest.raises(ValueError) as exc:
            self._apply([("layer1.weight", torch.ones(2, 2, dtype=torch.int8))])

        assert "carries no executable quantization metadata" in str(exc.value)
        assert "layer1.weight" in str(exc.value)
        assert "torch.int8" in str(exc.value)

    def test_uint8_storage_rejected(self):
        with pytest.raises(ValueError) as exc:
            self._apply([("layer1.weight", torch.ones(2, 2, dtype=torch.uint8))])

        assert "carries no executable quantization metadata" in str(exc.value)
        assert "torch.uint8" in str(exc.value)

    def test_float8_e4m3fn_storage_rejected(self):
        with pytest.raises(ValueError) as exc:
            self._apply([("layer1.weight", torch.ones(2, 2).to(torch.float8_e4m3fn))])

        assert "carries no executable quantization metadata" in str(exc.value)
        assert "torch.float8_e4m3fn" in str(exc.value)

    def test_quant_storage_rejected_under_an_unexpected_key(self):
        """Guard sits at the top of the loop, before the params lookup.

        ``_assert_dense_loadable`` checks every key of the state dict, not
        just the ones the model has a target for; an int8 tensor under an
        unknown key must not slip through silently.
        """
        with pytest.raises(ValueError) as exc:
            self._apply([("no.such.key", torch.ones(2, 2, dtype=torch.int8))])

        assert "no.such.key" in str(exc.value)

    def test_bf16_and_fp32_dense_still_assign(self):
        from ComfyUI_VibeVoice.modules.loader import VibeVoiceLoader

        w1 = torch.full((2, 2), 3.0, dtype=torch.bfloat16)
        w2 = torch.full((2, 2), 4.0, dtype=torch.float32)

        model = _QuantGuardModel()
        missing, unexpected = VibeVoiceLoader._stream_apply_dense(
            model, iter([("layer1.weight", w1), ("layer2.weight", w2)])
        )

        assert missing == []
        assert unexpected == []
        assert model.layer1.weight.dtype == torch.bfloat16
        assert model.layer2.weight.dtype == torch.float32
        assert torch.equal(model.layer1.weight.data, w1)
        assert torch.equal(model.layer2.weight.data, w2)

    def test_dtype_set_is_shared_with_external_loader(self):
        # Drift guard: the dtype set has exactly ONE definition object,
        # module-level in ``modules.loader`` (external_loader imports it from
        # there, so the dependency runs one way only).
        from ComfyUI_VibeVoice.modules import loader as L

        assert L.QUANT_STORAGE_DTYPES is EL._QUANT_STORAGE_DTYPES
        assert L.QUANT_STORAGE_DTYPES == frozenset({
            torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2,
        })
