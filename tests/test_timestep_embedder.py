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


# ====================================================================
# N-13 regression (2026-08-28): fp8 checkpoints replace the diffusion
# head's mlp linears with FP8Linear. `TimestepEmbedder.forward` used to
# cast `t_freq.to(self.mlp[0].weight.dtype)` — with fp8-resident weights
# that cast the activation to torch.float8_e4m3fn, and FP8Linear passed
# it straight into comfy_kitchen.dequantize_per_tensor_fp8 as the output
# dtype -> NoCapableBackendError. The forward must derive the compute
# dtype via _mlp_compute_dtype (declared compute_dtype first), never the
# storage dtype.
# ====================================================================

def test_mlp_compute_dtype_selection():
    mod = _load_module()

    # Plain float mlp: first layer's weight dtype (legacy behavior).
    emb = mod.TimestepEmbedder(hidden_size=32).to(torch.bfloat16)
    assert mod._mlp_compute_dtype(emb.mlp) == torch.bfloat16

    # fp8 storage without a declared compute dtype: falls back to bf16,
    # never the fp8 storage dtype.
    class _FakeFp8Lin(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.empty(8, 8, dtype=torch.float8_e4m3fn)
            )
            self.compute_dtype = None

        def forward(self, x):
            return x

    mlp = torch.nn.Sequential(_FakeFp8Lin(), torch.nn.SiLU())
    assert mod._mlp_compute_dtype(mlp) == torch.bfloat16

    # A declared compute dtype wins.
    mlp[0].compute_dtype = torch.float16
    assert mod._mlp_compute_dtype(mlp) == torch.float16


def test_timestep_embedder_with_fp8_mlp_stays_bf16():
    from ComfyUI_VibeVoice.modules.convrot_quant import QuantLayerInfo
    from ComfyUI_VibeVoice.modules.fp8_quant import (
        make_fp8_linear,
        probe_fp8_backend,
    )

    if probe_fp8_backend() is None:
        pytest.skip("no comfy_kitchen fp8 backend on this box")

    mod = _load_module()
    embedder = mod.TimestepEmbedder(hidden_size=64, frequency_embedding_size=256)

    def _fp8_linear(in_f, out_f):
        info = QuantLayerInfo(
            prefix="t_embedder.mlp",
            group_size=0,
            in_features=in_f,
            out_features=out_f,
            has_bias=False,
            convrot=False,
            orig_dtype="torch.bfloat16",
            rowwise_dtype=torch.float8_e4m3fn,
            resident_fp8=True,
        )
        lin = make_fp8_linear(info)(in_f, out_f, False)
        g = torch.Generator().manual_seed(0)
        w = torch.randint(
            -100, 100, (out_f, in_f), generator=g
        ).to(torch.float8_e4m3fn)
        lin.weight.data.copy_(w)
        lin.weight_scale.data.fill_(0.01)
        return lin

    # Mirror the real fp8 checkpoint: both mlp linears replaced
    # (bias=False, so the tree has NO bf16 parameter to fall back on).
    embedder.mlp[0] = _fp8_linear(256, 64)
    embedder.mlp[2] = _fp8_linear(64, 64)

    assert mod._mlp_compute_dtype(embedder.mlp) == torch.bfloat16

    # The exact failing case from the GPU gate: integer scheduler
    # timesteps into the fp8-resident mlp.
    t_int = torch.arange(0, 50, dtype=torch.int64)
    out = embedder(t_int)

    assert out.dtype == torch.bfloat16, f"output dtype was {out.dtype}"
    assert out.shape == (50, 64)
    assert torch.isfinite(out).all()
    assert out.abs().max().item() > 0, "embedding was destroyed (all zeros)"
