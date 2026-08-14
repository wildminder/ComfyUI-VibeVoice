"""Regression test for the CFG diffusion sampling loop in `generate`.

Runtime BUG-004: `VibeVoiceForConditionalGeneration.generate` built the diffusion
condition with 2N rows (conditioned + unconditioned concatenated for classifier-free
guidance) but fed the diffusion head a noisy-input batch of only N rows (a leftover
`half = speech[:len//2]; combined = cat([half, half])` trick). Inside the head:

    condition = self.cond_proj(condition)   # (2N, hidden_size)
    t        = self.t_embedder(timesteps)   # (N,  hidden_size)
    c = condition + t                        # 2N vs N -> RuntimeError

raising `RuntimeError: The size of tensor a (384) must match the size of tensor b
(192)` (the user's `384 vs 192` is exactly 2N vs N with N=192).

The fix aligns the batch to 2N (`combined = cat([speech, speech])`, `t` repeated 2N)
and collapses the two branches after the head returns (`eps = uncond_eps +
cfg_scale * (cond_eps - uncond_eps)`) back to N.

This test loads the REAL diffusion head by file path (bypassing the heavy `vibevoice`
package __init__ which would pull in `diffusers`) and asserts:
  * the OLD 2N-condition / N-batch call still raises the mismatch, and
  * the NEW 2N / 2N call runs and the CFG collapse yields a correct (N, vae_dim) tensor.
"""

import importlib.util
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

_HERE = os.path.dirname(os.path.abspath(__file__))
_HEAD_PATH = os.path.abspath(
    os.path.join(
        _HERE, "..", "src", "vibevoice", "modular",
        "modular_vibevoice_diffusion_head.py",
    )
)


def _load_module():
    # Stub the relative-import target so we don't pull in the heavy vibevoice package.
    config_name = "src.vibevoice.modular.configuration_vibevoice"
    stub = types.ModuleType(config_name)
    stub.VibeVoiceDiffusionHeadConfig = object
    sys.modules[config_name] = stub

    # Register the package chain so PreTrainedModel.__init__ can resolve cls.__module__.
    sys.modules.setdefault("src", types.ModuleType("src"))
    sys.modules.setdefault("src.vibevoice", types.ModuleType("src.vibevoice"))
    sys.modules.setdefault("src.vibevoice.modular", types.ModuleType("src.vibevoice.modular"))

    spec = importlib.util.spec_from_file_location(
        "src.vibevoice.modular.modular_vibevoice_diffusion_head", _HEAD_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "src.vibevoice.modular"
    sys.modules["src.vibevoice.modular.modular_vibevoice_diffusion_head"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_head():
    from transformers import PretrainedConfig

    class _Cfg(PretrainedConfig):
        def __init__(self, **kw):
            super().__init__()
            for k, v in kw.items():
                setattr(self, k, v)

    mod = _load_module()
    cfg = _Cfg(hidden_size=64, latent_size=32, head_ffn_ratio=2.0, head_layers=2, rms_norm_eps=1e-6)
    head = mod.VibeVoiceDiffusionHead(cfg).to(torch.bfloat16)
    return head


def _timesteps(bsz):
    # Mimic DDPM: an int64 scalar repeated across the batch (see BUG-002).
    return torch.tensor([500], dtype=torch.int64).repeat(bsz)


def test_old_cfg_layout_raises_size_mismatch():
    """The pre-fix layout (2N condition, N batch) must raise the BUG-004 error."""
    head = _make_head()
    N, hidden, vae_dim = 16, 64, 32

    condition = torch.randn(N, hidden, dtype=torch.bfloat16)
    neg_condition = torch.zeros_like(condition)
    cond_in = torch.cat([condition, neg_condition], dim=0)          # 2N rows
    speech = torch.randn(N, vae_dim, dtype=torch.bfloat16)
    combined = torch.cat([speech[: N // 2], speech[: N // 2]], dim=0)  # N rows (old trick)

    with pytest.raises(RuntimeError):
        head(combined, _timesteps(combined.shape[0]), condition=cond_in)


def test_new_cfg_layout_runs_and_collapses_to_n():
    """The fixed layout (2N condition, 2N batch) runs and collapses to (N, vae_dim)."""
    head = _make_head()
    N, hidden, vae_dim = 16, 64, 32
    cfg_scale = 1.3

    condition = torch.randn(N, hidden, dtype=torch.bfloat16)
    neg_condition = torch.zeros_like(condition)
    cond_in = torch.cat([condition, neg_condition], dim=0)          # 2N rows
    speech = torch.randn(N, vae_dim, dtype=torch.bfloat16)
    combined = torch.cat([speech, speech], dim=0)                  # 2N rows (fixed)

    eps = head(combined, _timesteps(combined.shape[0]), condition=cond_in)
    assert tuple(eps.shape) == (2 * N, vae_dim), f"head output shape was {tuple(eps.shape)}"

    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

    assert tuple(eps.shape) == (N, vae_dim), f"collapsed shape was {tuple(eps.shape)}"
    assert torch.isfinite(eps).all(), "CFG collapse produced non-finite values"
