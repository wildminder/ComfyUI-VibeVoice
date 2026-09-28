"""Queue-time rejection of dtype='fp32' + attention_mode='sage'.

The sage kernels hard-assert ``dtype in [torch.float16, torch.bfloat16]``
(``sageattn_qk_int8_pv_fp8_cuda_sm90``) and
``resolve_sage_target_dtype`` returns the *stored weight dtype* for a plain
float linear. So a user who picks "fp32" in the dtype widget (offered by
``dtype_utils.get_dtype_options``) and "sage" in the attention widget crashes
inside the CUDA kernel minutes into a model load, with an assert that names
neither widget. The two inputs are independent and nothing cross-checked them.

These tests drive the three node validators that expose the pair:
the unified TTS node, the ASR node, and the external loader node.
"""

from unittest.mock import patch

import pytest

from ComfyUI_VibeVoice.nodes.tts_node import VibeVoiceTTSNode
from ComfyUI_VibeVoice.nodes.asr_node import VibeVoiceASRNode
from ComfyUI_VibeVoice.nodes.external_loader_node import VibeVoiceExternalLoaderNode


_MODELS = {"VibeVoice-1.5B": {}, "VibeVoice-Realtime-0.5B": {}}


def _tts(**kwargs):
    with patch(
        "ComfyUI_VibeVoice.nodes.tts_node.AVAILABLE_VIBEVOICE_MODELS", _MODELS
    ), patch(
        "ComfyUI_VibeVoice.nodes.tts_node.get_available_attention_modes",
        return_value=["eager", "sdpa", "flash_attention_2", "sage"],
    ):
        return VibeVoiceTTSNode.validate_inputs(**kwargs)


def _asr(**kwargs):
    with patch(
        "ComfyUI_VibeVoice.nodes.asr_node.AVAILABLE_VIBEVOICE_MODELS",
        {"VibeVoice-ASR-HF": {"type": "local_dir", "path": "some/dir"}},
    ):
        return VibeVoiceASRNode.validate_inputs(**kwargs)


def _external(**kwargs):
    from ComfyUI_VibeVoice.nodes.external_loader_node import EXTERNAL_CONFIG_OPTIONS

    payload = {"config_name": EXTERNAL_CONFIG_OPTIONS[0]}
    payload.update(kwargs)
    return VibeVoiceExternalLoaderNode.validate_inputs(**payload)


_NODES = [
    pytest.param(_tts, id="tts"),
    pytest.param(_asr, id="asr"),
    pytest.param(_external, id="external_loader"),
]


@pytest.mark.parametrize("call", _NODES)
def test_fp32_with_sage_is_rejected(call):
    result = call(dtype="fp32", attention_mode="sage")
    assert isinstance(result, str), (
        "fp32 + sage must be refused at queue time; the kernel asserts fp16/bf16 "
        "and fails with a message that names neither widget"
    )
    assert "fp32" in result and "sage" in result
    assert "bf16" in result and "sdpa" in result


@pytest.mark.parametrize("call", _NODES)
def test_fp32_with_sdpa_is_accepted(call):
    assert call(dtype="fp32", attention_mode="sdpa") is True


@pytest.mark.parametrize("call", _NODES)
@pytest.mark.parametrize("dtype_str", ["auto", "bf16", "fp16"])
def test_sage_with_half_dtypes_is_accepted(call, dtype_str):
    assert call(dtype=dtype_str, attention_mode="sage") is True


@pytest.mark.parametrize("call", _NODES)
def test_absent_attention_mode_is_accepted(call):
    """``attention_mode`` is optional in the prompt (a node that doesn't set
    it, or a workflow predating the widget). There is nothing to cross-check
    against, so the check must stay silent — the loader picks a default."""
    assert call(dtype="fp32") is True


def test_tts_4bit_exempts_fp32():
    """The loader forces an fp32 bnb compute dtype for 4-bit + sage, but the
    quantized linears carry a quant_state so sage still reads bf16."""
    assert _tts(
        model_name="VibeVoice-1.5B",
        dtype="fp32",
        attention_mode="sage",
        quantize_llm_4bit=True,
    ) is True


def test_tts_rejects_a_stale_attention_mode():
    """Declaring `attention_mode` in the validator signature opts it out of
    core's own combo check (execution.py range-checks only inputs the
    validator does not declare), so the node has to re-check it — a stale
    workflow must not be silently downgraded to eager by
    resolve_attention_mode, because a backend swap changes the audio."""
    with patch(
        "ComfyUI_VibeVoice.nodes.tts_node.get_available_attention_modes",
        return_value=["eager", "sdpa"],
    ):
        result = VibeVoiceTTSNode.validate_inputs(
            model_name="VibeVoice-1.5B", attention_mode="sage"
        )
    assert isinstance(result, str)
    assert "sage" in result and "not available" in result


def test_tts_dtype_check_is_skipped_for_a_linked_external_model():
    """A linked external_model is validated by the external loader node, which
    owns the effective dtype/attention for the bundle it already checked."""
    import ComfyUI_VibeVoice.nodes.tts_node as m

    result = VibeVoiceTTSNode.validate_inputs(
        model_name=None,
        external_model=None,
        dtype="fp32",
        attention_mode="sage",
    )
    assert result is True
    assert m._EXTERNAL_UNSET is not None


def test_asr_returns_early_for_a_linked_external_model():
    """A linked external_model is always present in kwargs (resolved to None by
    core), so the presence test short-circuits before the dtype check. The
    external loader node owns the pair for that bundle and already validated it.
    """
    result = VibeVoiceASRNode.validate_inputs(
        external_model=None,
        dtype="fp32",
        attention_mode="sage",
    )
    assert result is True
