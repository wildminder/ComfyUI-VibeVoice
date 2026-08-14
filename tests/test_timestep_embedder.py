"""Regression test for the TimestepEmbedder dtype handling.

Runtime BUG-002: `TimestepEmbedder.timestep_embedding` returned
`embedding.to(t.dtype)`. When the timesteps tensor is integer (`int64` — which is
what `noise_scheduler.timesteps` yields), that (a) truncated the cos/sin embedding
to int64, destroying it, and (b) fed an int64 tensor into `self.mlp`, whose weights
are `c10::BFloat16` when the model is loaded in bf16 (e.g. under SageAttention) —
raising `RuntimeError: expected mat1 and mat2 to have the same dtype`.

The fix keeps the embedding floating and casts it to the MLP's compute dtype in
`TimestepEmbedder.forward`. This test loads the REAL module by file path (bypassing
the heavy `vibevoice` package __init__ chain, which would otherwise pull in
`diffusers`) and asserts the forward runs for int64/float32/bf16 timesteps and
produces a correct bf16 output that is not destroyed.
"""

import importlib.util
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATCH_PATH = os.path.abspath(
    os.path.join(
        _HERE, "..", "src", "vibevoice", "modular",
        "modular_vibevoice_diffusion_head.py",
    )
)


def _load_module():
    # Stub the relative-import target so we don't pull in the heavy vibevoice
    # package (which transitively imports diffusers and trips a huggingface_hub
    # version clash in some venvs). We only need TimestepEmbedder here.
    config_name = "src.vibevoice.modular.configuration_vibevoice"
    stub = types.ModuleType(config_name)
    stub.VibeVoiceDiffusionHeadConfig = type("VibeVoiceDiffusionHeadConfig", (), {})
    sys.modules[config_name] = stub

    spec = importlib.util.spec_from_file_location(
        "src.vibevoice.modular.modular_vibevoice_diffusion_head", _PATCH_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "src.vibevoice.modular"
    spec.loader.exec_module(mod)
    return mod


def test_timestep_embedder_handles_int64_timesteps_in_bf16():
    mod = _load_module()
    TimestepEmbedder = mod.TimestepEmbedder

    embedder = TimestepEmbedder(hidden_size=64, frequency_embedding_size=256).to(
        torch.bfloat16
    )

    # The exact failing case: integer scheduler timesteps + bf16 model weights.
    t_int = torch.arange(0, 50, dtype=torch.int64)
    out = embedder(t_int)

    assert out.dtype == torch.bfloat16, f"output dtype was {out.dtype}"
    assert out.shape == (50, 64)
    # The embedding must not have been destroyed by an int downcast.
    assert out.abs().max().item() > 0, "embedding was destroyed (all zeros)"

    # Sanity: float inputs also work and stay bf16.
    for t in (
        torch.linspace(0, 1000, 50, dtype=torch.float32),
        torch.linspace(0, 1000, 50, dtype=torch.bfloat16),
    ):
        o = embedder(t)
        assert o.dtype == torch.bfloat16
        assert o.abs().max().item() > 0
