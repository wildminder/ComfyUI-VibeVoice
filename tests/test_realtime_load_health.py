"""S3.1 + S3.3 — the refuted causes, as regression tests, on the real checkpoint.

Three hypotheses were investigated and refuted by measurement (plan §0):

* **F4** — the ``speech_scaling_factor`` / ``speech_bias_factor`` buffers are
  loaded correctly (``0.2333984375`` / ``-0.0703125``, bf16).
* **F5** — rotary ``inv_freq`` survives the load (it is a non-persistent buffer
  that the meta-instantiation path zero-materialises and the loader recomputes).
* **F6** — no parameter or buffer of the loaded model is non-finite.

They are pinned here so nobody spends a day on them again. Two more
measurements live in the same file because they are the same question about
the same load:

* the **t2** contract, on the real checkpoint rather than on a stub: after
  ``from_pretrained``, ``acoustic_connector`` and ``tts_eos_classifier`` must
  still hold the checkpoint's values;
* **S3.3**, the first-latent EOS logit in bf16 and fp32 on the same seed.

The reference is always the **checkpoint's own tensors, read off disk**
(``realtime_e2e_support.checkpoint_tensors``) — never a second
``from_pretrained``, which on transformers 5.3 re-runs ``_init_weights`` over
exactly the tensors the t2 contract is about.

Tier G: opt-in via ``RUN_VIBEVOICE_E2E=1`` plus the model/preset environment
variables documented in ``realtime_e2e_support``.
"""

from __future__ import annotations

import pytest
import torch

from tests.realtime_e2e_support import (
    checkpoint_tensors,
    diag,
    named_nonfinite,
    node_model,
    realtime_env,
    real_vibevoice,
    rotary_modules,
)

# The node path's default; the audit is about the shipped configuration.
ATTENTION_MODE = "sdpa"

# F4's measured values, pinned so a silent change of checkpoint or of the
# buffer's dtype is visible in the failure message rather than in a diff.
F4_SPEECH_SCALING_FACTOR = 0.2333984375
F4_SPEECH_BIAS_FACTOR = -0.0703125
F4_ROPE_THETA = 1000000.0

# t2: the tensors transformers 5.3's _init_weights used to overwrite.
INIT_WEIGHTS_SENSITIVE_TENSORS = (
    "model.acoustic_connector.fc1.weight",
    "model.acoustic_connector.fc1.bias",
    "model.acoustic_connector.fc2.weight",
    "model.acoustic_connector.fc2.bias",
    "tts_eos_classifier.fc1.weight",
    "tts_eos_classifier.fc1.bias",
    "tts_eos_classifier.fc2.weight",
    "tts_eos_classifier.fc2.bias",
)

# S3.3 (measured 2026-09-26, node load path, sdpa, cfg 1.5, seed 42):
#   bf16  eos=0.000017  fp32  eos=0.000016  -> 1e-6 apart, same verdict.
# The gate is stated as a tolerance rather than a transcription of those
# numbers: the two dtypes must agree on the EOS verdict, and their logits must
# land within this absolute distance of each other.
EOS_DTYPE_TOLERANCE = 5e-3
EOS_VERDICT_THRESHOLD = 0.5


# ------------------------------------------------------------------- S3.1 --
def test_speech_scaling_factors_match_the_checkpoint(node_model, realtime_env):
    """F4: the two scaling buffers are finite, non-zero, and the checkpoint's."""
    model, _processor, _preset = node_model(attention_mode=ATTENTION_MODE, dtype_str="bf16")
    reference = checkpoint_tensors(
        realtime_env["model_dir"],
        ("model.speech_scaling_factor", "model.speech_bias_factor"),
    )

    for name, expected_literal in (
        ("speech_scaling_factor", F4_SPEECH_SCALING_FACTOR),
        ("speech_bias_factor", F4_SPEECH_BIAS_FACTOR),
    ):
        loaded = getattr(model, name)
        assert torch.isfinite(loaded).all(), f"{name} is not finite after load"
        assert float(loaded.abs().min()) > 0.0, (
            f"{name} is zero: the diffusion-inversion gate "
            "'if not isnan(sf) and not isnan(bf)' would then apply speech/0 - 0"
        )
        expected = reference[f"model.{name}"].to(loaded.dtype)
        assert torch.equal(loaded.detach().cpu(), expected), (
            f"{name} drifted from the checkpoint value {expected.tolist()}"
        )
        assert float(loaded.float()) == expected_literal


def _rope_theta(config) -> float:
    """The config's RoPE base, wherever this transformers version keeps it.

    5.x moved it into ``config.rope_parameters``; 4.x exposed ``config.rope_theta``.
    """
    parameters = getattr(config, "rope_parameters", None)
    if isinstance(parameters, dict) and "rope_theta" in parameters:
        return float(parameters["rope_theta"])
    return float(config.rope_theta)


def test_rotary_inv_freq_is_healthy(node_model, realtime_env):
    """F5: RoPE frequencies are rebuilt from the config, not left at zero.

    ``inv_freq[i] = rope_theta ** (-2i/dim)``, so the table always starts at
    exactly 1.0 whatever the base is — the base shows up in the *decay*. The
    check that therefore has teeth is the smallest frequency: with
    ``rope_theta=1e6`` and ``head_dim=128`` it is ~1.2e-6, while a default
    10000.0 base gives ~1.2e-4. A zero-materialised buffer (the meta-init
    regression this pins) fails both ends.
    """
    model, _processor, _preset = node_model(attention_mode=ATTENTION_MODE, dtype_str="bf16")
    modules = rotary_modules(model)
    assert modules, "no rotary inv_freq buffer found — the audit is vacuous"

    rope_theta = _rope_theta(model.config.decoder_config)
    assert rope_theta == F4_ROPE_THETA

    for name, module in modules.items():
        flat = module.inv_freq.detach().float().flatten()
        assert torch.isfinite(flat).all(), f"{name}.inv_freq holds a non-finite frequency"
        assert float(flat.min()) > 0.0, (
            f"{name}.inv_freq contains a zero frequency: RoPE collapses to "
            "cos(0)=1/sin(0)=0, i.e. no positional encoding at all"
        )
        assert bool((flat[1:] <= flat[:-1]).all()), (
            f"{name}.inv_freq is not monotonically decreasing; it is not a RoPE table"
        )
        head_dim = getattr(module, "dim", None) or getattr(
            getattr(module, "config", None), "head_dim", None
        )
        if head_dim:
            expected_min = rope_theta ** (-(head_dim - 2) / head_dim)
            torch.testing.assert_close(
                flat[-1:].to(device="cpu", dtype=module.inv_freq.dtype),
                torch.tensor([expected_min], dtype=module.inv_freq.dtype),
                rtol=5e-2,
                atol=0.0,
                msg=lambda m, name=name: (
                    f"{name}.inv_freq decays to {float(flat[-1]):.3g}, but "
                    f"rope_theta={rope_theta:g} with head_dim={head_dim} requires "
                    f"{expected_min:.3g} — the base was not applied"
                ),
            )
    sample = next(iter(modules.values())).inv_freq
    print(
        f"[rope] {len(modules)} rotary buffers, dtype={sample.dtype}, "
        f"span=[{min(float(m.inv_freq.float().min()) for m in modules.values()):.3g}, "
        f"{max(float(m.inv_freq.float().max()) for m in modules.values()):.3g}], "
        f"rope_theta={rope_theta:g}"
    )


def test_no_non_finite_parameters_or_buffers(node_model):
    """F6: nothing in the loaded model is NaN or infinite."""
    model, _processor, _preset = node_model(attention_mode=ATTENTION_MODE, dtype_str="bf16")
    offenders = named_nonfinite(model.named_parameters(), "param")
    offenders += named_nonfinite(model.named_buffers(), "buffer")
    assert offenders == [], f"non-finite tensors after load: {offenders[:10]}"


def test_from_pretrained_keeps_the_connector_and_eos_head(
    real_vibevoice, realtime_env
):
    """t2 on the real checkpoint: ``_init_weights`` must not clobber them.

    On transformers 5.x ``from_pretrained`` builds on meta, loads the
    checkpoint, then initialises "missing" keys over the whole graph. Those
    eight tensors are checkpoint-present, so the only thing the pass can do to
    them is overwrite them with ``normal_(0, initializer_range)`` — silently,
    while reporting ``missing=276 unexpected=0 mismatched=0``. The node loader
    is unaffected (it instantiates the class directly); this is the contract
    that keeps ``from_pretrained`` usable as a reference at all.
    """
    model_class = real_vibevoice["streaming_inference"].VibeVoiceStreamingForConditionalGenerationInference
    model = model_class.from_pretrained(
        str(realtime_env["model_dir"]),
        dtype=torch.bfloat16,
        attn_implementation=ATTENTION_MODE,
        device_map=None,
    )
    reference = checkpoint_tensors(
        realtime_env["model_dir"], INIT_WEIGHTS_SENSITIVE_TENSORS
    )
    state = model.state_dict()
    for name, expected in reference.items():
        loaded = state[name]
        assert loaded.dtype == expected.dtype, f"{name} changed dtype on load"
        assert torch.equal(loaded.cpu(), expected.cpu()), (
            f"{name} no longer holds the checkpoint's value after from_pretrained: "
            "transformers 5.x re-initialised a loaded tensor"
        )


# ------------------------------------------------------------------- S3.3 --
def test_first_latent_eos_agrees_across_dtypes(node_model, diag):
    """S3.3: bf16 and fp32 reach the same EOS verdict on the same seed.

    Both runs go through the node load path and the loop's own first-latent
    step, so this compares the shipping configuration against a wider precision
    rather than two ways of loading the same weights.
    """
    results = {}
    conditions = {}
    for dtype_str in ("bf16", "fp32"):
        model, processor, preset = node_model(attention_mode=ATTENTION_MODE, dtype_str=dtype_str)
        inputs = diag.processor_inputs(processor, preset)
        results[dtype_str] = diag.first_latent_eos(model, preset, inputs, seed=42)
        conditions[dtype_str] = diag.first_window_conditioning(
            model, preset, inputs
        )["condition"]
        print(
            f"[dtype] {dtype_str:<5} weights={results[dtype_str]['dtype']:<16} "
            f"eos={results[dtype_str]['eos'][0]:.6f} "
            f"stop={results[dtype_str]['stopped']} "
            f"latent_norm={results[dtype_str]['latent_norm']:.3f}"
        )
        diag.release_node_model(model)

    cosine = torch.nn.functional.cosine_similarity(
        conditions["fp32"].unsqueeze(0), conditions["bf16"].unsqueeze(0)
    ).item()
    print(f"[dtype] cos(condition[fp32], condition[bf16]) = {cosine:.6f}")
    assert cosine >= 0.999, f"the dtypes condition differently: cos={cosine:.6f}"

    verdicts = {name: result["stopped"] for name, result in results.items()}
    assert len(set(verdicts.values())) == 1, (
        f"precision flips the EOS verdict across {EOS_VERDICT_THRESHOLD}: {verdicts}. "
        "That is a dtype recommendation for the node tooltip and README (S3.3)."
    )

    logits = {name: result["eos"][0] for name, result in results.items()}
    drift = abs(logits["bf16"] - logits["fp32"])
    assert drift <= EOS_DTYPE_TOLERANCE, (
        f"first-latent EOS logit moved {drift:.6f} between bf16 and fp32 "
        f"({logits}), tolerance {EOS_DTYPE_TOLERANCE}"
    )
