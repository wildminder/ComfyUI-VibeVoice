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

    def test_llamacpp_modern_names(self):
        """Modern llama.cpp names (token_embd) count as llamacpp."""
        keys = ["blk.0.attn_q.weight", "token_embd.weight", "output_norm.weight"]
        assert detect_key_scheme(keys) == "llamacpp"

    def test_mixed_resolved_by_majority(self):
        """A mostly-HF file with llamacpp aliases maps fully (quantui-rs 7B)."""
        keys = [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.norm.weight",
            "model.language_model.embed_tokens.weight",
            "output.weight",  # llamacpp alias for lm_head
        ]
        mapping = map_keys(keys)
        assert mapping["output.weight"] == "lm_head.weight"
        assert all(mapping[k] == k for k in keys if k != "output.weight")

    def test_mixed_reversed_majority(self):
        """A mostly-llamacpp file with an HF straggler maps via llamacpp."""
        keys = [
            "blk.0.attn_q.weight",
            "tok_embeddings.weight",
            "output_norm.weight",
            "model.language_model.layers.1.mlp.up_proj.weight",
        ]
        mapping = map_keys(keys)
        assert mapping["tok_embeddings.weight"] == (
            "model.language_model.embed_tokens.weight"
        )
        assert mapping["model.language_model.layers.1.mlp.up_proj.weight"] == (
            "model.language_model.layers.1.mlp.up_proj.weight"
        )

    def test_exact_tie_still_rejected(self):
        keys = ["blk.0.attn_q.weight", "model.language_model.norm.weight"]
        with pytest.raises(UnmappedKeyError, match="mixed"):
            map_keys(keys)

    def test_hf_majority_unknown_alias_raises(self):
        """Majority resolution must not swallow unknown minority keys."""
        keys = [
            "model.language_model.norm.weight",
            "model.language_model.embed_tokens.weight",
            "blk.0.weird_tensor.weight",  # llamacpp-ish but unmapped
        ]
        with pytest.raises(UnmappedKeyError, match="weird_tensor"):
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

    def test_token_embd_maps_to_embed_tokens(self):
        """Modern llama.cpp 'token_embd' aliases the input embedding."""
        mapping = map_keys(
            ["token_embd.weight", "blk.0.attn_q.weight", "output_norm.weight"],
            scheme="llamacpp",
        )
        assert mapping["token_embd.weight"] == (
            "model.language_model.embed_tokens.weight"
        )

    def test_mixed_llamacpp_lm_hf_rest_maps(self):
        """quantui-rs 'new' layout: llamacpp-named language model + HF names
        everywhere else resolves by majority vote (HF wins here)."""
        keys = [
            "model.acoustic_connector.fc1.weight",           # HF
            "model.language_model.norm.weight",               # HF (7B case)
            "model.prediction_head.cond_proj.weight",        # HF
            "blk.0.attn_q.weight",                           # llamacpp
            "blk.0.ffn_norm.weight",                         # llamacpp
            "token_embd.weight",                              # llamacpp
            "output_norm.weight",                             # llamacpp
        ]
        mapping = map_keys(keys)
        assert mapping["model.acoustic_connector.fc1.weight"] == (
            "model.acoustic_connector.fc1.weight"
        )
        assert mapping["blk.0.attn_q.weight"] == (
            "model.language_model.layers.0.self_attn.q_proj.weight"
        )
        assert mapping["token_embd.weight"] == (
            "model.language_model.embed_tokens.weight"
        )
        assert mapping["output_norm.weight"] == (
            "model.language_model.norm.weight"
        )

    def test_layer_number_preserved(self):
        mapping = map_keys(["blk.17.attn_q.weight"], scheme="llamacpp")
        assert "layers.17.self_attn.q_proj.weight" in mapping["blk.17.attn_q.weight"]
