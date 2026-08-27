"""Phase C tests: ConvRot INT8 module, checkpoint scanning, group-size rules."""

import json

import pytest
import torch
from safetensors.torch import save_file

from ComfyUI_VibeVoice.modules.convrot_quant import (
    ConvRotInt8Linear,
    QuantLayerInfo,
    UnsupportedQuantFormat,
    scan_checkpoint_quantization,
    validate_group_size,
)


def _write_convrot_checkpoint(path, entries):
    """entries: list of (prefix, out_f, in_f, group_size)."""
    tensors = {}
    for prefix, out_f, in_f, g in entries:
        tensors[f"{prefix}.weight"] = torch.randint(
            -127, 127, (out_f, in_f), dtype=torch.int8
        )
        tensors[f"{prefix}.weight_scale"] = (
            torch.rand(out_f, 1, dtype=torch.float32) * 0.01 + 0.001
        )
        meta = json.dumps({
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": g,
            "in_features": in_f,
            "out_features": out_f,
        }).encode("utf-8")
        tensors[f"{prefix}.comfy_quant"] = torch.frombuffer(
            bytearray(meta), dtype=torch.uint8
        )
    save_file(tensors, str(path))
    return path


class TestValidateGroupSize:
    @pytest.mark.parametrize("g", [4, 16, 64, 256])
    def test_valid_powers_of_four(self, g):
        validate_group_size(g, g * 4)

    @pytest.mark.parametrize("g", [2, 3, 8, 12, 0])
    def test_non_power_of_four_rejected(self, g):
        with pytest.raises(ValueError, match="power of four"):
            validate_group_size(g, 64)

    def test_indivisible_in_features_rejected(self):
        with pytest.raises(ValueError, match="divisible"):
            validate_group_size(16, 100)


class TestScanCheckpoint:
    def test_plain_checkpoint_returns_empty(self, tmp_path):
        p = tmp_path / "plain.safetensors"
        save_file({"w": torch.zeros(4, 4)}, str(p))
        assert scan_checkpoint_quantization(p) == {}

    def test_missing_file_returns_empty(self, tmp_path):
        assert scan_checkpoint_quantization(tmp_path / "nope.safetensors") == {}

    def test_convrot_metadata_parsed(self, tmp_path):
        p = _write_convrot_checkpoint(tmp_path / "q.safetensors",
                                      [("model.layers.0.self_attn.q_proj", 64, 64, 16)])
        qmap = scan_checkpoint_quantization(p)
        assert list(qmap.keys()) == ["model.layers.0.self_attn.q_proj"]
        info = qmap["model.layers.0.self_attn.q_proj"]
        assert isinstance(info, QuantLayerInfo)
        assert info.group_size == 16
        assert info.in_features == 64 and info.out_features == 64
        assert info.has_bias is False

    def test_unsupported_format_hard_fail(self, tmp_path):
        from safetensors.torch import save_file as _sf

        p = tmp_path / "bad.safetensors"
        meta = json.dumps({"format": "mystery_pack", "convrot": True,
                           "convrot_groupsize": 16}).encode("utf-8")
        _sf({
            "layer.weight": torch.zeros(4, 4, dtype=torch.int8),
            "layer.comfy_quant": torch.frombuffer(bytearray(meta), dtype=torch.uint8),
        }, str(p))
        with pytest.raises(RuntimeError, match="mystery_pack"):
            scan_checkpoint_quantization(p)


class TestConvRotForwardContract:
    """ConvRot forward runs on the deterministic `eager` kitchen backend.

    A CPU numerical reference is NOT constructible: the checkpoint stores the
    offline-ROTATED int8 weight (W_rot = W @ H^T) and the rotation matrix H
    is an implementation detail of the quantizer, while `int8_linear` rotates
    activations online. The float-parity gate therefore runs on real hardware
    (plan Gate D manual verification); here we pin the execution CONTRACT.
    """

    def test_eager_backend_executes_and_returns_finite(self):
        pytest.importorskip("comfy_kitchen")

        torch.manual_seed(0)
        out_f, in_f, g = 32, 64, 16
        layer = ConvRotInt8Linear(in_f, out_f, bias=True, group_size=g)
        layer.weight.data = torch.randint(-100, 100, (out_f, in_f), dtype=torch.int8)
        layer.weight_scale.data = torch.rand(out_f, 1, dtype=torch.float32) * 0.02 + 0.005
        layer.bias.data = torch.randn(out_f) * 0.01

        x = torch.randn(4, in_f)
        y = layer(x)
        assert y.shape == (4, out_f)
        assert y.dtype == x.dtype
        assert torch.isfinite(y).all()

    def test_no_backend_raises(self, monkeypatch):
        """With every kitchen backend disabled the module must fail loudly,
        proving kernels genuinely come from comfy-kitchen."""
        pytest.importorskip("comfy_kitchen")
        import comfy_kitchen

        for name in ("triton", "cuda", "eager"):
            monkeypatch.setattr(
                comfy_kitchen, f"_test_disabled_{name}", None, raising=False
            )
        # Disable via the public API instead of internals.
        for name in ("triton", "cuda", "eager"):
            try:
                comfy_kitchen.disable_backend(name)
            except Exception:
                pass
        try:
            layer = ConvRotInt8Linear(64, 32, bias=False, group_size=16)
            layer.weight.data = torch.zeros(32, 64, dtype=torch.int8)
            layer.weight_scale.data = torch.ones(32, 1)
            with pytest.raises(Exception) as e:
                layer(torch.randn(2, 64))
            assert "backend" in str(e.value).lower()
        finally:
            for name in ("triton", "cuda", "eager"):
                try:
                    comfy_kitchen.enable_backend(name)
                except Exception:
                    pass

    def test_meta_construction_no_kernel_calls(self, monkeypatch):
        pytest.importorskip("comfy_kitchen")

        def _boom(*a, **k):
            raise AssertionError("kitchen kernel called during construction")

        monkeypatch.setattr(torch.nn.functional, "linear", _boom, raising=False)
        with torch.device("meta"):
            layer = ConvRotInt8Linear(64, 32, bias=False, group_size=16)
        assert layer.weight.is_meta
        assert layer.weight.dtype == torch.int8
        assert layer.weight_scale.dtype == torch.float32


# ====================================================================
# Rowwise / fp8 format support + hard-fail propagation
# ====================================================================

class TestRowwiseFormatScanning:
    def _meta(self, path, prefix, meta):
        from safetensors.torch import save_file as _sf

        tensors = {
            f"{prefix}.weight": torch.randint(-100, 100, (64, 64),
                                              dtype=torch.int8),
            f"{prefix}.weight_scale": torch.full((64, 1), 0.01),
        }
        import json as _json
        tensors[f"{prefix}.comfy_quant"] = torch.frombuffer(
            bytearray(_json.dumps(meta).encode("utf-8")), dtype=torch.uint8
        )
        _sf(tensors, str(path))
        return path

    def test_plain_rowwise_int8_recognized(self, tmp_path):
        p = self._meta(tmp_path / "row.safetensors",
                       "model.layers.0.self_attn.q_proj",
                       {"format": "int8_tensorwise",
                        "orig_dtype": "torch.bfloat16", "per_row": True})
        qmap = scan_checkpoint_quantization(p)
        info = qmap["model.layers.0.self_attn.q_proj"]
        assert info.convrot is False
        assert info.rowwise_dtype is None  # int8 storage
        assert info.orig_dtype == "torch.bfloat16"
        assert (info.out_features, info.in_features) == (64, 64)

    def test_fp8_rowwise_recognized(self, tmp_path):
        from safetensors.torch import save_file as _sf

        prefix = "model.prediction_head.cond_proj"
        tensors = {
            f"{prefix}.weight": torch.randint(-100, 100, (64, 64)).to(
                torch.float8_e4m3fn),
            f"{prefix}.weight_scale": torch.full((64, 1), 0.01),
        }
        import json as _json
        tensors[f"{prefix}.comfy_quant"] = torch.frombuffer(
            bytearray(_json.dumps({
                "format": "float8_e4m3fn",
                "orig_dtype": "torch.bfloat16",
            }).encode("utf-8")), dtype=torch.uint8)
        p = tmp_path / "fp8.safetensors"
        _sf(tensors, str(p))

        qmap = scan_checkpoint_quantization(p)
        info = qmap[prefix]
        assert info.convrot is False
        assert info.rowwise_dtype == torch.float8_e4m3fn
        # PER-ROW [out, 1] scales cannot use the per-tensor kitchen kernel.
        assert info.resident_fp8 is False

    def _write_fp8(self, path, prefix, scale):
        from safetensors.torch import save_file as _sf
        import json as _json

        tensors = {
            f"{prefix}.weight": torch.randint(-100, 100, (64, 64)).to(
                torch.float8_e4m3fn),
            f"{prefix}.weight_scale": scale,
            f"{prefix}.comfy_quant": torch.frombuffer(
                bytearray(_json.dumps({
                    "format": "float8_e4m3fn",
                    "orig_dtype": "torch.bfloat16",
                }).encode("utf-8")), dtype=torch.uint8),
        }
        _sf(tensors, str(path))
        return path

    def test_fp8_scalar_scale_marked_resident(self, tmp_path):
        """Scalar per-tensor scale + available kitchen backend -> resident."""
        p = self._write_fp8(tmp_path / "fp8s.safetensors",
                            "model.prediction_head.cond_proj",
                            torch.tensor(0.5))
        info = scan_checkpoint_quantization(p)["model.prediction_head.cond_proj"]
        assert info.resident_fp8 is True
        assert info.rowwise_dtype == torch.float8_e4m3fn
        assert (info.out_features, info.in_features) == (64, 64)

    def test_fp8_single_element_1d_scale_marked_resident(self, tmp_path):
        p = self._write_fp8(tmp_path / "fp8s1.safetensors",
                            "model.prediction_head.cond_proj",
                            torch.tensor([0.5]))
        info = scan_checkpoint_quantization(p)["model.prediction_head.cond_proj"]
        assert info.resident_fp8 is True

    def test_fp8_not_resident_without_backend(self, tmp_path):
        """No kitchen fp8 backend -> dequant-at-load fallback (plan D3)."""
        from unittest.mock import patch
        from ComfyUI_VibeVoice.modules import fp8_quant as FQ

        p = self._write_fp8(tmp_path / "fp8nb.safetensors",
                            "model.prediction_head.cond_proj",
                            torch.tensor(0.5))
        with patch.object(FQ, "probe_fp8_backend", return_value=None):
            info = scan_checkpoint_quantization(p)[
                "model.prediction_head.cond_proj"]
        assert info.resident_fp8 is False
        assert info.rowwise_dtype == torch.float8_e4m3fn

    def test_unknown_format_raises_specific_class(self, tmp_path):
        self._meta(tmp_path / "bad.safetensors", "layer",
                   {"format": "mystery_pack"})
        with pytest.raises(UnsupportedQuantFormat, match="mystery_pack"):
            scan_checkpoint_quantization(tmp_path / "bad.safetensors")

    def test_unknown_orig_dtype_raises(self, tmp_path):
        self._meta(tmp_path / "od.safetensors", "layer",
                   {"format": "int8_tensorwise", "per_row": True,
                    "orig_dtype": "torch.float4"})
        with pytest.raises(UnsupportedQuantFormat, match="float4"):
            scan_checkpoint_quantization(tmp_path / "od.safetensors")

    def test_mixed_real_world_shapes_parse(self, tmp_path):
        """340 convrot + 39 rowwise style mix parses into both kinds."""
        from safetensors.torch import save_file as _sf
        import json as _json

        tensors = {}
        specs = [
            ("a.conv", {"convrot": True, "convrot_groupsize": 256,
                        "format": "int8_tensorwise",
                        "orig_dtype": "torch.bfloat16"}, (256, 256)),
            ("b.row", {"format": "int8_tensorwise", "per_row": True,
                       "orig_dtype": "torch.bfloat16"}, (64, 64)),
        ]
        for prefix, meta, shape in specs:
            tensors[f"{prefix}.weight"] = torch.randint(
                -100, 100, shape, dtype=torch.int8)
            tensors[f"{prefix}.weight_scale"] = torch.full(
                (shape[0], 1), 0.01)
            tensors[f"{prefix}.comfy_quant"] = torch.frombuffer(
                bytearray(_json.dumps(meta).encode("utf-8")),
                dtype=torch.uint8)
        p = tmp_path / "mixed.safetensors"
        _sf(tensors, str(p))

        qmap = scan_checkpoint_quantization(p)
        assert qmap["a.conv"].convrot is True
        assert qmap["a.conv"].group_size == 256
        assert qmap["b.row"].convrot is False
