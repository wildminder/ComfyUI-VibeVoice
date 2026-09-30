"""End-to-end load of a NATIVE transformers-5.3 VibeVoice-ASR checkpoint
through the EXTERNAL loader (plan t5).

Drives ``external_loader.load_external_vibevoice_asr_model`` with real files
and real transformers classes — no vendored-class stubbing, no MagicMock
processor. A tiny but structurally complete ASR model is built on CPU (113
state-dict keys, ~202 KB of parameters) and written to a temp directory
exactly the way a user's single-file checkpoint sits on disk: the weights
plus a ``<weight>.config.json`` sidecar, and deliberately NO tokenizer, so the
loader must source its own assets.

What this pins:

* the config / processor / model CLASSES chosen are the transformers-native
  ones — the vendored ASR tree nests ``model.language_model.*`` and cannot
  consume the native checkpoint's ``language_model.model.*`` keys at all, and
  the vendored processor returns keys the native forward does not accept;
* the processor is classified ``"native"`` by
  ``asr_generation._asr_processor_kind``, i.e. transcription takes the
  ``_transcribe_native`` branch;
* every parameter equals the tensor saved in the file (no silent
  random-init survivor from the meta instantiation);
* NO tensor is left on the ``meta`` device — a meta rotary buffer means
  uninitialised RoPE, i.e. gibberess output (the failure
  ``loader._recompute_rope_buffers`` exists to prevent).

The 16.6 GB real ``VibeVoice-ASR-HF-bf16.safetensors`` is NEVER loaded by
any test in this repo; only the tiny model below is.

Named ``..._native_load`` (not ``test_asr_external_native``) because that
path is owned by the packaged-asset tests (plan t2).
"""

import json
import os
from unittest.mock import patch

import pytest
import torch

from ComfyUI_VibeVoice.modules import external_loader as EL

transformers = pytest.importorskip("transformers")

# The acoustic / semantic tokenizer towers. ``depths`` MUST be
# len(downsampling_ratios) + 1 or the encoder raises IndexError.
_ENCODER_CONFIG = {
    "model_type": "vibevoice_acoustic_tokenizer_encoder",
    "hidden_size": 16,
    "num_filters": 8,
    "depths": [1, 1, 1],
    "downsampling_ratios": [2, 2],
    "ffn_expansion": 2,
    "kernel_size": 3,
    "channels": 1,
}

# A two-layer Qwen2 backbone — the smallest tree that still has the
# architecture the real 28-layer / 3584-hidden checkpoint reduces to.
_TEXT_CONFIG = {
    "model_type": "qwen2",
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 128,
    "tie_word_embeddings": False,
}

_SEED = 0


def _build_tiny_native_asr_model():
    """Build the tiny native ASR model on CPU with a fixed seed."""
    try:
        from transformers import VibeVoiceAsrForConditionalGeneration
    except ImportError as e:  # pragma: no cover - depends on the env
        pytest.skip(
            f"transformers {transformers.__version__} has no native "
            f"VibeVoice ASR: {e}"
        )

    config = transformers.AutoConfig.for_model(
        model_type="vibevoice_asr",
        text_config=dict(_TEXT_CONFIG),
        acoustic_tokenizer_encoder_config=dict(_ENCODER_CONFIG),
        semantic_tokenizer_encoder_config=dict(_ENCODER_CONFIG),
        acoustic_tokenizer_chunk_size=48000,
    )
    torch.manual_seed(_SEED)
    return VibeVoiceAsrForConditionalGeneration(config)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    """Write the tiny native ASR checkpoint + sidecar config to a temp dir.

    Returns ``(weight_path, state_dict)``.
    """
    from safetensors.torch import save_file

    model = _build_tiny_native_asr_model()
    state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

    directory = tmp_path_factory.mktemp("native_asr")
    weight_path = str(directory / "model.safetensors")
    save_file(state_dict, weight_path)

    # Sidecar config, exactly as a user's single-file checkpoint ships it.
    with open(weight_path + ".config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "architectures": ["VibeVoiceAsrForConditionalGeneration"],
                "model_type": "vibevoice_asr",
                "text_config": dict(_TEXT_CONFIG),
                "acoustic_tokenizer_encoder_config": dict(_ENCODER_CONFIG),
                "semantic_tokenizer_encoder_config": dict(_ENCODER_CONFIG),
                "acoustic_tokenizer_chunk_size": 48000,
            },
            f,
        )

    # No tokenizer.json / preprocessor_config.json on purpose: the loader must
    # source its own assets rather than depend on the user's folder.
    assert not os.path.exists(os.path.join(str(directory), "tokenizer.json"))
    return weight_path, state_dict


def _load_native_asr(weight_path):
    """Run the real external ASR loader, pinned to CPU."""
    with patch.object(
        EL.model_management, "get_torch_device", return_value=torch.device("cpu")
    ):
        return EL.load_external_vibevoice_asr_model(
            weight_path=weight_path,
            config_name="VibeVoice-ASR",
            attention_mode="sdpa",
            dtype_str="auto",
        )


class TestNativeAsrExternalLoad:
    """load_external_vibevoice_asr_model on a real native checkpoint."""

    def test_bundle_shape_and_identity(self, tiny_checkpoint):
        weight_path, _ = tiny_checkpoint
        bundle = _load_native_asr(weight_path)

        assert bundle["is_asr"] is True
        assert bundle["model_name"] == "VibeVoice-ASR"
        assert bundle["weight_family"] == "dense"
        assert bundle["is_streaming"] is False
        assert bundle["use_llm_4bit"] is False
        assert bundle["source_path"] == weight_path
        assert bundle["source_size"] == os.path.getsize(weight_path)

    def test_config_is_native_vibevoice_asr(self, tiny_checkpoint):
        """The real checkpoint's structure: a text backbone plus two encoder
        towers. The vendored config class reads decoder_config /
        acoustic_tokenizer_config and leaves these at their defaults."""
        weight_path, _ = tiny_checkpoint
        config = _load_native_asr(weight_path)["config"]

        assert getattr(config, "model_type", None) == "vibevoice_asr"
        assert config.text_config.hidden_size == _TEXT_CONFIG["hidden_size"]
        assert config.text_config.vocab_size == _TEXT_CONFIG["vocab_size"]
        assert config.text_config.num_hidden_layers == _TEXT_CONFIG["num_hidden_layers"]

    def test_model_consumes_the_native_checkpoint_keys(self, tiny_checkpoint):
        """The native tree nests ``language_model.model.*`` — the exact key
        layout of VibeVoice-ASR-HF. The vendored class nests
        ``model.language_model.*`` and would load nothing here."""
        weight_path, state_dict = tiny_checkpoint
        model = _load_native_asr(weight_path)["model"]

        assert set(model.state_dict()) == set(state_dict)

    def test_processor_is_classified_native(self, tiny_checkpoint):
        """The vendored processor returns keys the native forward does not
        accept and takes a different call signature — a native model paired
        with it cannot transcribe."""
        from ComfyUI_VibeVoice.modules.asr_generation import _asr_processor_kind

        weight_path, _ = tiny_checkpoint
        processor = _load_native_asr(weight_path)["processor"]
        kind = _asr_processor_kind(processor)
        assert kind == "native", (
            f"expected the transformers VibeVoiceAsrProcessor, got "
            f"{type(processor).__module__}.{type(processor).__name__} "
            f"(kind={kind!r})"
        )

    def test_parameters_equal_the_saved_tensors(self, tiny_checkpoint):
        """Every parameter equals what is in the file — no random-init
        survivor from the meta instantiation."""
        weight_path, state_dict = tiny_checkpoint
        loaded = _load_native_asr(weight_path)["model"].state_dict()

        assert set(loaded) == set(state_dict)
        for key, expected in state_dict.items():
            actual = loaded[key]
            assert actual.shape == expected.shape, key
            assert actual.dtype == expected.dtype, key
            assert torch.equal(actual, expected), key

    def test_no_meta_buffers_survive_the_load(self, tiny_checkpoint):
        """A tensor still on ``meta`` is uninitialised memory; for the rotary
        embeddings that means wrong RoPE and gibberess output — exactly the
        failure ``loader._recompute_rope_buffers`` exists to prevent."""
        weight_path, _ = tiny_checkpoint
        model = _load_native_asr(weight_path)["model"]

        assert [n for n, b in model.named_buffers() if b.is_meta] == []
        assert [n for n, p in model.named_parameters() if p.is_meta] == []

    def test_streaming_conversion_is_applied_to_the_loaded_tree(self, tiny_checkpoint):
        """``convert_tree_for_streaming`` must actually have run on this tree.

        The call sits in ``loader._apply_state_dict`` inside a bare
        ``except Exception`` that downgrades any failure to a warning
        (modules/loader.py:816-822), so "the load succeeded" says nothing
        about it: a refactor that dropped the import, renamed the helper or
        started passing a wrong root would leave the 16.6 GB ASR checkpoint
        entirely unstreamable while every other test stayed green. Core's
        partial-load machinery only streams modules carrying
        ``comfy_cast_weights``, so that flag on the leaves is the observable
        end state of the conversion.
        """
        weight_path, _ = tiny_checkpoint
        model = _load_native_asr(weight_path)["model"]

        leaves = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
        ]
        assert leaves, "fixture must expose Linear leaves to convert"

        unconverted = [name for name, m in leaves
                       if getattr(m, "comfy_cast_weights", False) is not True]
        assert not unconverted, (
            f"{len(unconverted)} of {len(leaves)} Linear leaves were never "
            f"converted for streaming (convert_tree_for_streaming did not "
            f"reach this tree). First few: {unconverted[:5]}"
        )

        # The bulk of the weight mass is the text backbone; name it
        # explicitly so a partial conversion of the towers alone cannot pass.
        lm_unconverted = [
            name for name, m in leaves
            if name.startswith("language_model.")
            and getattr(m, "comfy_cast_weights", False) is not True
        ]
        assert not lm_unconverted, (
            "language_model leaves not converted for streaming: "
            f"{lm_unconverted[:5]}"
        )

    def test_streaming_conversion_does_not_change_the_weights(self, tiny_checkpoint):
        """The conversion is a class swap only: values must survive it.

        Without this, a "fix" that made the assertions above pass by
        re-wrapping modules around fresh tensors would still look green.
        """
        weight_path, state_dict = tiny_checkpoint
        loaded = _load_native_asr(weight_path)["model"].state_dict()

        for key, expected in state_dict.items():
            assert torch.equal(loaded[key], expected), key

    def test_rope_inv_freq_is_materialized(self, tiny_checkpoint):
        """The concrete RoPE regression: ``inv_freq`` must be a real CPU
        tensor after an external (meta-instantiated) load."""
        weight_path, _ = tiny_checkpoint
        model = _load_native_asr(weight_path)["model"]
        rotary = model.language_model.model.rotary_emb

        assert rotary.inv_freq.device.type == "cpu"
        assert torch.isfinite(rotary.inv_freq).all()
        assert float(rotary.inv_freq.sum()) > 0.0
