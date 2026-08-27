"""Phase 2 tests: streaming conversion wired into the load pipeline.

Every weight-loading path funnels through ``VibeVoiceLoader._apply_state_dict``,
so one integration assertion covers dropdown/sharded, external dense, external
GGUF, and external ConvRot flows.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

import gguf
from gguf.constants import GGMLQuantizationType as T

from ComfyUI_VibeVoice.modules import external_loader as EL
from ComfyUI_VibeVoice.modules.external_loader import (
    _install_gguf_weights,
    load_external_vibevoice_model,
)
from ComfyUI_VibeVoice.modules.gguf_quant import GGUFLinear
from ComfyUI_VibeVoice.modules.loader import VibeVoiceLoader
from conftest import build_stub_vv, stub_vv_gguf_spec


class _FakeStreamingCfg:
    pass


class TestApplyStateDictConverts:
    def test_dense_apply_converts_tree(self):
        model = build_stub_vv(n_layers=1)
        with torch.device("meta"):
            meta_model = build_stub_vv(n_layers=1)

        dense = {k: v for k, v in model.state_dict().items()}
        VibeVoiceLoader._apply_state_dict(meta_model, dense)

        q = meta_model.model.language_model.layers[0].self_attn.q_proj
        assert getattr(q, "comfy_cast_weights", False) is True
        assert isinstance(q, torch.nn.Linear)  # base preserved for sage etc.

    def test_residents_skipped_by_conversion(self, make_gguf_file):
        """GGUF residents stream natively; conversion must not rewrap them."""
        path = make_gguf_file(stub_vv_gguf_spec(n_layers=1, qtype="Q8_0"),
                              tag="conv-skip")
        reader = gguf.GGUFReader(str(path))
        model = build_stub_vv(n_layers=1)
        _install_gguf_weights(model, reader)

        q = model.model.language_model.layers[0].self_attn.q_proj
        assert isinstance(q, GGUFLinear)
        assert type(q).__name__ == "GGUFLinear"  # not a subclass rewrap


class TestExternalLoaderEndToEnd:
    def test_external_fp8_style_load_converts_tree(self, tmp_path):
        """Full loader flow: dequant-at-load + conversion in one pass."""
        from safetensors.torch import save_file
        import json as _json

        prefix = "model.prediction_head.cond_proj"
        tensors = {
            f"{prefix}.weight": torch.randint(-100, 100, (64, 64),
                                              dtype=torch.int8),
            f"{prefix}.weight_scale": torch.full((64, 1), 0.5),
        }
        tensors[f"{prefix}.comfy_quant"] = torch.frombuffer(
            bytearray(_json.dumps({
                "format": "int8_tensorwise", "per_row": True,
                "orig_dtype": "torch.bfloat16",
            }).encode("utf-8")), dtype=torch.uint8)
        p = tmp_path / "row.safetensors"
        save_file(tensors, str(p))

        def _instantiate(config, is_streaming, attn_implementation,
                         final_load_dtype, use_meta=True):
            return build_stub_vv(n_layers=1)

        with patch.object(EL.VibeVoiceLoader, "_load_config",
                          return_value=MagicMock()), \
             patch.object(EL.VibeVoiceLoader, "_load_tokenizer",
                          return_value=MagicMock()), \
             patch.object(EL.VibeVoiceLoader, "_load_processor",
                          return_value=MagicMock()), \
             patch.object(EL.VibeVoiceLoader, "_instantiate_model",
                          side_effect=_instantiate), \
             patch.object(EL, "resolve_sidecar_config",
                          return_value="/fake/config.json"), \
             patch.object(EL, "resolve_sidecar_preprocessor",
                          return_value=""), \
             patch.object(EL, "resolve_sidecar_tokenizer_dir",
                          return_value="/fake/dir"), \
             patch.object(EL, "resolve_dtype", return_value=torch.bfloat16), \
             patch.object(EL, "resolve_attention_mode",
                          side_effect=lambda m, q: m), \
             patch.object(EL, "get_attn_implementation_for_load",
                          return_value="eager"), \
             patch.object(EL, "VibeVoiceStreamingConfig", _FakeStreamingCfg),              patch.object(EL.model_management, "get_torch_device",
                          return_value=torch.device("cpu")):
            bundle = load_external_vibevoice_model(
                weight_path=str(p), config_name="VibeVoice-1.5B",
                attention_mode="sdpa", use_llm_4bit=False, dtype_str="auto",
            )

        cond = bundle["model"].model.prediction_head.cond_proj
        assert isinstance(cond, torch.nn.Linear)
        assert getattr(cond, "comfy_cast_weights", False) is True
        assert cond.weight.dtype == torch.bfloat16  # dequanted + intact

    def test_conversion_failure_does_not_break_loading_external(self, tmp_path,
                                                                monkeypatch):
        """External-path mirror of the defensive contract."""
        from ComfyUI_VibeVoice.modules import comfy_stream
        from safetensors.torch import save_file

        def boom(root, skip=()):
            raise RuntimeError("conversion blew up")

        monkeypatch.setattr(comfy_stream, "convert_tree_for_streaming", boom)

        p = tmp_path / "plain.safetensors"
        save_file({"model.prediction_head.cond_proj.weight":
                   torch.zeros(64, 64)}, str(p))

        def _instantiate(config, is_streaming, attn_implementation,
                         final_load_dtype, use_meta=True):
            return build_stub_vv(n_layers=1)

        with patch.object(EL.VibeVoiceLoader, "_load_config",
                          return_value=MagicMock()),              patch.object(EL.VibeVoiceLoader, "_load_tokenizer",
                          return_value=MagicMock()),              patch.object(EL.VibeVoiceLoader, "_load_processor",
                          return_value=MagicMock()),              patch.object(EL.VibeVoiceLoader, "_instantiate_model",
                          side_effect=_instantiate),              patch.object(EL, "resolve_sidecar_config",
                          return_value="/fake/config.json"),              patch.object(EL, "resolve_sidecar_preprocessor",
                          return_value=""),              patch.object(EL, "resolve_sidecar_tokenizer_dir",
                          return_value="/fake/dir"),              patch.object(EL, "resolve_dtype", return_value=torch.bfloat16),              patch.object(EL, "resolve_attention_mode",
                          side_effect=lambda m, q: m),              patch.object(EL, "get_attn_implementation_for_load",
                          return_value="eager"),              patch.object(EL, "VibeVoiceStreamingConfig", _FakeStreamingCfg),              patch.object(EL.model_management, "get_torch_device",
                          return_value=torch.device("cpu")):
            bundle = load_external_vibevoice_model(
                weight_path=str(p), config_name="VibeVoice-1.5B",
                attention_mode="sdpa", use_llm_4bit=False, dtype_str="auto",
            )
        cond = bundle["model"].model.prediction_head.cond_proj
        assert not getattr(cond, "comfy_cast_weights", False)
        assert not cond.weight.is_meta

    def test_conversion_failure_does_not_break_loading(self, tmp_path,
                                                       monkeypatch):
        """Defensive: if the conversion pass explodes, loading must still
        succeed (worst case: today's status quo for that model)."""
        from ComfyUI_VibeVoice.modules import comfy_stream

        def boom(root, skip=()):
            raise RuntimeError("conversion blew up")

        monkeypatch.setattr(comfy_stream, "convert_tree_for_streaming", boom)
        model = build_stub_vv(n_layers=1)
        meta_model = build_stub_vv(n_layers=1)
        VibeVoiceLoader._apply_state_dict(meta_model, model.state_dict())
        q = meta_model.model.language_model.layers[0].self_attn.q_proj
        assert getattr(q, "comfy_cast_weights", False) is False
        assert not q.weight.is_meta
