"""Phase B tests: GGUF key-scheme detection and mapping onto the module tree."""

import pytest
import torch
from torch import nn

from ComfyUI_VibeVoice.modules.gguf_quant import (
    UnmappedKeyError,
    detect_key_scheme,
    map_keys,
)
from conftest import build_stub_vv


class TestSchemeDetection:
    def test_hf_passthrough(self):
        keys = [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
        ]
        assert detect_key_scheme(keys) == "hf"

    def test_llamacpp(self):
        keys = [
            "blk.0.attn_q.weight",
            "tok_embeddings.weight",
            "output.weight",
        ]
        assert detect_key_scheme(keys) == "llamacpp"

    def test_mixed_rejected_by_map_keys(self):
        keys = ["blk.0.attn_q.weight", "model.language_model.norm.weight"]
        with pytest.raises(UnmappedKeyError, match="mixed"):
            map_keys(keys)


class TestHFIdentityMapping:
    def test_identity_for_stub_tree(self):
        model = build_stub_vv(n_layers=2)
        param_paths = {name for name, _ in model.named_parameters()}
        keys = sorted(param_paths)[:50]
        mapping = map_keys(keys, scheme="hf")
        assert all(mapping[k] == k for k in keys)


class TestLlamacppMapping:
    def test_golden_keys_map_onto_stub_tree(self):
        """Golden llamacpp key list (mirrors VibeVoice-1.5B topology) maps 100%
        onto synthesized stub named_parameters."""
        model = build_stub_vv(n_layers=1)
        params = {name for name, _ in model.named_parameters()}

        golden = {
            "tok_embeddings.weight": "model.language_model.embed_tokens.weight",
            "output.weight": "lm_head.weight",
            "output_norm.weight": "model.language_model.norm.weight",
            "blk.0.attn_norm.weight": "model.language_model.layers.0.input_layernorm.weight",
            "blk.0.ffn_norm.weight": "model.language_model.layers.0.post_attention_layernorm.weight",
            "blk.0.attn_q.weight": "model.language_model.layers.0.self_attn.q_proj.weight",
            "blk.0.attn_k.weight": "model.language_model.layers.0.self_attn.k_proj.weight",
            "blk.0.attn_v.weight": "model.language_model.layers.0.self_attn.v_proj.weight",
            "blk.0.attn_output.weight": "model.language_model.layers.0.self_attn.o_proj.weight",
            "blk.0.ffn_gate.weight": "model.language_model.layers.0.mlp.gate_proj.weight",
            "blk.0.ffn_up.weight": "model.language_model.layers.0.mlp.up_proj.weight",
            "blk.0.ffn_down.weight": "model.language_model.layers.0.mlp.down_proj.weight",
        }
        mapping = map_keys(list(golden.keys()), scheme="llamacpp")
        for src, expected in golden.items():
            assert mapping[src] == expected, src
            # Every mapped weight must exist on the real module tree.
            if mapping[src].endswith(".weight"):
                assert mapping[src] in params, f"{mapping[src]} not in stub tree"

    def test_unknown_key_lists_offenders_and_hints(self):
        with pytest.raises(UnmappedKeyError) as e:
            map_keys(["blk.0.weird_tensor.weight"], scheme="llamacpp")
        msg = str(e.value)
        assert "weird_tensor" in msg
        assert "sidecar config" in msg

    def test_layer_number_preserved(self):
        mapping = map_keys(["blk.17.attn_q.weight"], scheme="llamacpp")
        assert "layers.17.self_attn.q_proj.weight" in mapping["blk.17.attn_q.weight"]
