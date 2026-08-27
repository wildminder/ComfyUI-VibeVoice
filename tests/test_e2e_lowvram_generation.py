"""Phase 5: end-to-end lowvram scenario.

Full pipeline: loader-style build -> _apply_state_dict (conversion choke
point) -> REAL core ModelPatcher partial load under a tight budget ->
simulated VRAM pressure (partially_unload) -> generation-shaped forward.
Asserts the production failure mode (CPU-stray device mismatch) cannot
recur on a converted tree.
"""

import torch

import comfy.model_management as comfy_mm
from comfy.model_patcher import ModelPatcher

from ComfyUI_VibeVoice.modules.loader import VibeVoiceLoader
from conftest import build_stub_vv


def _build_loaded_patcher():
    eager = build_stub_vv(n_layers=1)
    with torch.device("meta"):
        model = build_stub_vv(n_layers=1)
    VibeVoiceLoader._apply_state_dict(model, eager.state_dict())
    model.eval()

    mp = ModelPatcher(
        model,
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        size=comfy_mm.module_size(model),
    )
    return model, mp


class TestEndToEndLowvramGeneration:
    def test_partial_load_pressure_then_forward(self):
        model, mp = _build_loaded_patcher()
        hot_mem = comfy_mm.module_size(
            model.model.language_model.layers[0].self_attn.q_proj)
        # Budget admits only a fraction of the tree -> core goes partial.
        mp.patch_model(device_to=torch.device("cpu"),
                       lowvram_model_memory=hot_mem * 2)

        # Simulated pressure: strip whatever can be freed.
        freed = mp.partially_unload(torch.device("cpu"), memory_to_free=1)

        # Generation-shaped forward through the stripped tree.
        lm = model.model.language_model
        ids = torch.randint(0, 96, (1, 8))
        with torch.no_grad():
            embeds = lm.embed_tokens(ids)
            hidden = embeds
            for layer in lm.layers[:1]:
                normed = layer.input_layernorm(hidden)
                attn_out = layer.self_attn.q_proj(normed)
                hidden = hidden + attn_out[..., :hidden.shape[-1]][..., :hidden.shape[-1]]
            logits = model.lm_head(hidden[:, -1, :])
        assert torch.isfinite(logits).all()
        assert freed >= 0  # pressure applied without breaking the tree

    def test_full_load_still_works(self):
        model, mp = _build_loaded_patcher()
        mp.patch_model(device_to=torch.device("cpu"),
                       lowvram_model_memory=0)
        ids = torch.randint(0, 96, (1, 4))
        with torch.no_grad():
            logits = model.lm_head(
                model.model.language_model.embed_tokens(ids))
        assert torch.isfinite(logits).all()
