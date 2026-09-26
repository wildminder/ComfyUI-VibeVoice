"""S3.2 — attention-backend parity for the realtime (streaming) path.

The realtime model does something no other VibeVoice path does: it feeds a
5-token text window *on top of* a 316-token cached voice prefill, and the
conditioning vector that comes out is what the diffusion head consumes. That
makes the attention backend numerically load-bearing here in a way it is not
elsewhere, so every advertised backend is measured rather than assumed
non-crashing.

Measured on VibeVoice-Realtime-0.5B, RTX SM89, bf16, one fixed voice prompt,
through the node load path, as cosine of the first-window conditioning against
eager (gate: 0.999):

    sdpa               0.999962  (rel_l2 0.0088)
    flash_attention_2  0.999940  (rel_l2 0.0110)
    sage               0.994651  (rel_l2 0.1033)  -> excluded

Tier C (default suite, no weights): the mask-interface contract, the exclusion
policy, the sage kernel's mask handling, and an eager-vs-sdpa numeric check on
a small real Qwen2 driven exactly like the realtime loop drives its LM.
Tier G (opt-in, real checkpoint): every available backend through the node load
path — see ``realtime_e2e_support`` for the opt-in environment variables.
"""

from __future__ import annotations

import copy

import pytest
import torch

from ComfyUI_VibeVoice.modules.attention_utils import (
    REALTIME_ATTENTION_FALLBACK,
    REALTIME_EXCLUDED_ATTENTION_MODES,
    get_attn_implementation_for_load,
    resolve_realtime_attention_mode,
)

# Imported for their side effect: pytest only collects fixtures it can see in
# the test module's namespace, so every fixture in this module's closure
# (including the ones only ``node_model`` depends on) must be named here.
from tests.realtime_e2e_support import (  # noqa: F401
    diag,
    node_model,
    realtime_env,
    real_vibevoice,
)

# Plan S3.2 acceptance: conditioning vectors agree to this cosine.
PARITY_COSINE = 0.999

# The prefill/window shape the realtime loop actually runs.
PREFILL_TOKENS = 16
WINDOW_TOKENS = 5

NEVER_EXCLUDE = ("eager", "sdpa", "flash_attention_2")


# --------------------------------------------------------------- tier C ----
def test_sage_is_not_a_mask_interface_implementation():
    """Record *why* sage may never be handed to the model as an implementation.

    transformers 5.3 builds the causal mask through
    ``masking_utils.AttentionMaskInterface._global_mapping``; a name that is
    absent from that mapping makes ``create_causal_mask`` take its early-exit
    path and return no mask at all. ``sage`` is absent, so the loaders map it to
    "sdpa" for loading and patch ``Qwen2Attention.forward`` afterwards
    (``get_attn_implementation_for_load``). If a future transformers release
    adds sage to the mapping this test tells us the two mechanisms now overlap
    and the workaround can be revisited.
    """
    from transformers.masking_utils import AttentionMaskInterface

    mapping = AttentionMaskInterface._global_mapping
    assert "sdpa" in mapping
    assert "eager" in mapping
    assert "sage" not in mapping, (
        "transformers now builds masks for sage; the post-load monkey-patch and "
        "the sdpa load-time mapping are no longer the only route."
    )


def test_every_attention_mode_still_loads_through_the_masked_interface():
    """Whatever we hand ``from_pretrained``/``_instantiate_model`` must be maskable."""
    from transformers.masking_utils import AttentionMaskInterface

    mapping = AttentionMaskInterface._global_mapping
    for mode in ("eager", "sdpa", "flash_attention_2", "sage"):
        impl = get_attn_implementation_for_load(mode)
        assert impl in mapping, f"{mode} loads as {impl!r}, which builds no mask"


def test_sage_forward_ignores_the_attention_mask(real_vibevoice, monkeypatch):
    """The mechanism behind sage's exclusion, asserted rather than assumed.

    ``sage_attention_forward`` derives causality from whether a mask was passed,
    not from the mask itself, so any masked (i.e. every windowed) call runs
    non-causal over the whole cache. Both branches are recorded here.
    """
    import torch.nn as nn

    sage_patch = real_vibevoice["sage_attention_patch"]
    recorded: list[dict] = []

    def _fake_kernel(q, k, v, tensor_layout, is_causal, qk_quant_gran, pv_accum_dtype):
        # (batch, heads, sequence, head_dim) — the sequence axis is -2.
        recorded.append(
            {"is_causal": is_causal, "q_len": q.shape[-2], "kv_len": k.shape[-2]}
        )
        return torch.zeros_like(q)

    monkeypatch.setattr(sage_patch, "SAGE_ATTENTION_FUNCTION", _fake_kernel)

    class _Cfg:
        num_attention_heads = 2
        num_key_value_heads = 1
        hidden_size = 16
        head_dim = 8

    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Cfg()
            self.layer_idx = 0
            self.head_dim = self.config.head_dim
            self.q_proj = nn.Linear(16, 16, bias=False)
            self.k_proj = nn.Linear(16, 8, bias=False)
            self.v_proj = nn.Linear(16, 8, bias=False)
            self.o_proj = nn.Linear(16, 16, bias=False)

    attn = _Attn()
    hidden = torch.randn(1, WINDOW_TOKENS, 16)
    position_embeddings = (torch.randn(1, WINDOW_TOKENS, 8), torch.randn(1, WINDOW_TOKENS, 8))
    additive_mask = torch.zeros(1, 1, WINDOW_TOKENS, PREFILL_TOKENS + WINDOW_TOKENS)

    class _PrefilledCache:
        """Just enough cache to put the prefill in front of the window."""

        def update(self, key_states, value_states, layer_idx, cache_kwargs):
            prefill = torch.zeros(
                1, attn.config.num_key_value_heads, PREFILL_TOKENS, attn.head_dim
            )
            return (
                torch.cat([prefill, key_states], dim=-2),
                torch.cat([prefill, value_states], dim=-2),
            )

    cache = _PrefilledCache()
    sage_patch.sage_attention_forward(
        attn,
        hidden,
        position_embeddings=position_embeddings,
        attention_mask=None,
        past_key_values=cache,
    )
    sage_patch.sage_attention_forward(
        attn,
        hidden,
        position_embeddings=position_embeddings,
        attention_mask=additive_mask,
        past_key_values=cache,
    )

    # Same keys either way — the mask changes nothing about what is attended to,
    # only about the is_causal flag, which is derived from the mask's presence.
    assert [entry["kv_len"] for entry in recorded] == [
        PREFILL_TOKENS + WINDOW_TOKENS
    ] * 2
    assert [entry["q_len"] for entry in recorded] == [WINDOW_TOKENS] * 2
    assert recorded[0]["is_causal"] is True
    assert recorded[1]["is_causal"] is False, (
        "a masked call must not fall back to is_causal; the mask itself is never "
        "read, so a windowed step attends to keys ahead of its own positions"
    )


def test_realtime_excludes_a_diverging_backend_loudly(caplog):
    """Excluded backends are downgraded, and the reason reaches the log."""
    assert REALTIME_EXCLUDED_ATTENTION_MODES, "the exclusion registry is empty"

    with caplog.at_level("WARNING", logger="ComfyUI_VibeVoice.modules.attention_utils"):
        resolved = resolve_realtime_attention_mode("sage")
    assert resolved == REALTIME_ATTENTION_FALLBACK
    assert "sage" in caplog.text
    # The log must carry the measured reason, not just the mode name.
    assert "diverg" in caplog.text or "cos=" in caplog.text

    for mode in NEVER_EXCLUDE:
        assert mode not in REALTIME_EXCLUDED_ATTENTION_MODES
        assert resolve_realtime_attention_mode(mode) == mode


def _tiny_qwen2(attn_implementation: str) -> torch.nn.Module:
    """A small real Qwen2 body with an explicit attention implementation."""
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

    config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    config._attn_implementation = attn_implementation
    torch.manual_seed(1234)
    return Qwen2Model(config).eval()


def _window_condition(model, cache, input_ids) -> torch.Tensor:
    """One windowed step against a prefilled cache, exactly as the loop issues it."""
    from transformers.cache_utils import DynamicCache

    assert isinstance(cache, DynamicCache)
    cache_len = cache.get_seq_length()
    cache_position = torch.arange(cache_len, cache_len + input_ids.shape[1])
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            attention_mask=torch.ones(1, cache_len + input_ids.shape[1], dtype=torch.long),
            position_ids=cache_position.unsqueeze(0),
            cache_position=cache_position,
        )
    return outputs.last_hidden_state[0, -1].float()


def test_eager_and_sdpa_agree_on_a_windowed_step(real_vibevoice):
    """Numeric parity on a real Qwen2, no checkpoint and no GPU required.

    The step mirrors the realtime loop: a prefill, then a 5-token window on top
    of it with the mask and ``cache_position`` advanced by the window size.
    """
    from transformers.cache_utils import DynamicCache

    eager = _tiny_qwen2("eager")
    sdpa = _tiny_qwen2("sdpa")
    sdpa.load_state_dict(copy.deepcopy(eager.state_dict()))

    prefill_ids = torch.randint(0, 128, (1, PREFILL_TOKENS))
    window_ids = torch.randint(0, 128, (1, WINDOW_TOKENS))
    conditions = {}
    for name, model in (("eager", eager), ("sdpa", sdpa)):
        cache = DynamicCache()
        with torch.no_grad():
            model(
                input_ids=prefill_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
                attention_mask=torch.ones(1, PREFILL_TOKENS, dtype=torch.long),
            )
        conditions[name] = _window_condition(model, cache, window_ids)

    cosine = torch.nn.functional.cosine_similarity(
        conditions["sdpa"].unsqueeze(0), conditions["eager"].unsqueeze(0)
    ).item()
    assert cosine >= PARITY_COSINE, f"sdpa diverges from eager: cos={cosine:.6f}"


# --------------------------------------------------------------- tier G ----
def _backends_to_measure() -> list[str]:
    """eager and sdpa always; the rest only where the hardware offers them."""
    from ComfyUI_VibeVoice.modules.attention_utils import get_available_attention_modes

    available = set(get_available_attention_modes())
    modes = ["eager", "sdpa"]
    modes += [m for m in ("sage", "flash_attention_2") if m in available]
    return modes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real-checkpoint parity needs CUDA.")
def test_conditioning_parity_across_attention_backends(node_model, diag):
    """Every backend's conditioning must match, or be explicitly excluded.

    This is the enforcement mechanism for ``REALTIME_EXCLUDED_ATTENTION_MODES``:
    a backend that drops below the cosine gate only passes by being named in
    the exclusion registry, which is a user-visible decision rather than a
    silent downgrade.
    """
    model, processor, preset = node_model(attention_mode="eager", dtype_str="bf16")
    inputs = diag.processor_inputs(processor, preset)
    reference = diag.first_window_conditioning(model, preset, inputs)
    diag.release_node_model(model)

    measured = {"eager": reference}
    for mode in _backends_to_measure():
        if mode == "eager":
            continue
        model, processor, _preset = node_model(attention_mode=mode, dtype_str="bf16")
        inputs = diag.processor_inputs(processor, _preset)
        measured[mode] = diag.first_window_conditioning(model, _preset, inputs)
        diag.release_node_model(model)

    for mode, probe in measured.items():
        condition = probe["condition"]
        cosine = float(
            torch.nn.functional.cosine_similarity(
                condition.unsqueeze(0), reference["condition"].unsqueeze(0)
            )
        )
        rel_l2 = float(
            (condition - reference["condition"]).norm()
            / reference["condition"].norm()
        )
        print(
            f"[parity] {mode:<18} cos_vs_eager={cosine:.6f} rel_l2={rel_l2:.6f} "
            f"config_attn={probe['config_attn']!r} eos={probe['eos']}"
        )
        assert (
            cosine >= PARITY_COSINE or mode in REALTIME_EXCLUDED_ATTENTION_MODES
        ), (
            f"{mode} diverges from eager (cos={cosine:.6f}, gate {PARITY_COSINE}). "
            "Either fix the backend or exclude it from the realtime path in "
            "modules/attention_utils.py with a log line and a README note."
        )

    # sage stays offered in the widget (it is still valid for the standard
    # family) but must not reach the realtime model.
    if "sage" in measured:
        assert "sage" in REALTIME_EXCLUDED_ATTENTION_MODES
        assert measured["sage"]["config_attn"] == "sdpa", (
            "a sage request reached the realtime model unexcluded: the loaded "
            "config still says sdpa, so the post-load patch was applied"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real-checkpoint parity needs CUDA.")
def test_sage_diverges_from_every_other_backend(node_model, diag, real_vibevoice, monkeypatch):
    """Keep the exclusion's justification measurable, not folklore.

    The exclusion happens in ``modules/generation.py``; neutralising it for one
    load measures the raw sage kernel on the realtime step. If this ever
    passes the cosine gate, sage has been fixed and
    ``REALTIME_EXCLUDED_ATTENTION_MODES`` should lose its entry.
    """
    monkeypatch.setattr(
        real_vibevoice["generation"], "resolve_realtime_attention_mode", lambda mode: mode
    )
    model, processor, preset = node_model(attention_mode="sage", dtype_str="bf16")
    assert model.config.decoder_config._attn_implementation == "sdpa"

    inputs = diag.processor_inputs(processor, preset)
    sage = diag.first_window_conditioning(model, preset, inputs)
    diag.release_node_model(model)

    model, processor, _preset = node_model(attention_mode="eager", dtype_str="bf16")
    inputs = diag.processor_inputs(processor, _preset)
    eager = diag.first_window_conditioning(model, _preset, inputs)
    diag.release_node_model(model)

    cosine = float(
        torch.nn.functional.cosine_similarity(
            sage["condition"].unsqueeze(0), eager["condition"].unsqueeze(0)
        )
    )
    rel_l2 = float(
        (sage["condition"] - eager["condition"]).norm() / eager["condition"].norm()
    )
    print(f"[parity] raw sage (policy neutralised) cos_vs_eager={cosine:.6f} rel_l2={rel_l2:.6f}")
    assert cosine < PARITY_COSINE, (
        f"sage now agrees with eager (cos={cosine:.6f}); the realtime exclusion "
        "is no longer justified and can be removed from "
        "REALTIME_EXCLUDED_ATTENTION_MODES"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real-checkpoint parity needs CUDA.")
def test_sage_request_loads_the_realtime_model_as_sdpa(node_model, real_vibevoice):
    """End-to-end: asking for sage on a realtime model never patches attention."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

    sage_forward = real_vibevoice["sage_attention_patch"].sage_attention_forward
    model, _processor, _preset = node_model(attention_mode="sage", dtype_str="bf16")

    assert model.config.decoder_config._attn_implementation == "sdpa"
    patched = [
        name
        for name, module in model.named_modules()
        if isinstance(module, Qwen2Attention)
        and getattr(getattr(module, "forward", None), "__func__", None) is sage_forward
    ]
    assert patched == [], (
        f"sage's Qwen2Attention.forward was monkey-patched onto {patched[:3]} "
        "despite being excluded from the realtime path"
    )
