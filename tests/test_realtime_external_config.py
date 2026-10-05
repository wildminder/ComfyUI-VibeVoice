"""Tests for the packaged VibeVoice-Realtime-0.5B architecture config.

The realtime checkpoint (``microsoft/VibeVoice-Realtime-0.5B``, revision
``6bce5f06``) ships its own ``config.json`` inside the model directory, but
ComfyUI users routinely point ``VibeVoiceExternalLoader`` at a bare weight file
with no sidecar — so the node has to carry a packaged default of its own. This
module owns that asset: it is a BYTE-VERBATIM copy of the published
``config.json``, and the tests here exist to keep it that way.

Why verbatim matters: ``VibeVoiceLoader._load_config`` (``loader.py:353-359``)
dispatches on ``model_type`` alone — anything not exactly ``"vibevoice_streaming"``
falls through to the non-streaming ``VibeVoiceConfig``. A paraphrased asset (a
"streamed" flag added here, a dropped ``tie_word_embeddings``, a renamed sub-key)
either dispatches to the wrong class or builds a wrong-sized tree, and the
failure surfaces much later as a missing-key or shape error at generation time.
A digest pin catches that at the asset boundary, in milliseconds, with no
checkpoint on disk.

The asset is addressed by PATH (through ``_packaged_configs_dir``), never
through ``import src.vibevoice.configs`` — the root ``conftest.py`` replaces
that whole package with ``MagicMock`` (``conftest.py:123-134``).
"""

import hashlib
import json
import os

from ComfyUI_VibeVoice.modules.config_detect import config_fingerprint
from ComfyUI_VibeVoice.modules.external_loader import _packaged_configs_dir

PACKAGED_REALTIME_CONFIG_FILE = "default_VibeVoice-Realtime-0.5B_config.json"

# sha256 + byte length of the published `config.json` at the revision named in
# the module docstring. Pinned rather than read from a checkpoint on disk: the
# whole point is that this test must be hermetic and run everywhere, while the
# digest still fails loudly the moment anyone hand-edits the asset.
UPSTREAM_SHA256 = "caee2691e790b04054bbe14a753b40149fa7c0c16fadb58d9adf5412343dcf57"
UPSTREAM_BYTE_LENGTH = 2117

# Top-level keys of the streaming architecture config. Note the absence of the
# non-streaming `semantic_tokenizer_config` / `semantic_vae_dim` pair — the
# realtime architecture has no semantic tokenizer at all, so their presence
# would mean the wrong asset was copied.
EXPECTED_TOP_LEVEL_KEYS = {
    "acoustic_tokenizer_config",
    "acoustic_vae_dim",
    "architectures",
    "decoder_config",
    "diffusion_head_config",
    "model_type",
    "torch_dtype",
    "transformers_version",
    "tts_backbone_num_hidden_layers",
}


def _asset_path() -> str:
    return os.path.normpath(os.path.join(_packaged_configs_dir(), PACKAGED_REALTIME_CONFIG_FILE))


def _asset_bytes() -> bytes:
    with open(_asset_path(), "rb") as f:
        return f.read()


def _asset_json() -> dict:
    with open(_asset_path(), encoding="utf-8") as f:
        return json.load(f)


class TestPackagedRealtimeAsset:
    """The packaged asset must stay a byte-exact copy of the upstream config."""

    def test_asset_ships_in_the_packaged_configs_dir(self):
        """The file lives where the loader looks for packaged defaults.

        ``_packaged_configs_dir`` is the single resolution point shared with
        ``_get_packaged_config_path`` (``external_loader.py:223-233``); the
        node ships ``src/vibevoice/configs/`` as-is with no packaging manifest,
        so "resolves through that helper" is the whole contract.
        """
        path = _asset_path()

        assert os.path.isfile(path)
        assert os.path.basename(path) == PACKAGED_REALTIME_CONFIG_FILE
        assert os.path.dirname(path) == os.path.normpath(_packaged_configs_dir())

    def test_asset_is_byte_identical_to_the_upstream_config(self):
        """Any hand edit — even whitespace — changes the digest and fails here."""
        raw = _asset_bytes()

        assert len(raw) == UPSTREAM_BYTE_LENGTH
        assert hashlib.sha256(raw).hexdigest() == UPSTREAM_SHA256

    def test_asset_is_json_with_the_streaming_key_set(self):
        """Parses, and carries exactly the streaming architecture's top-level keys."""
        config = _asset_json()

        assert isinstance(config, dict)
        assert set(config) == EXPECTED_TOP_LEVEL_KEYS

    def test_asset_declares_the_streaming_model_type(self):
        """``model_type`` is the sole dispatch key in ``_load_config``."""
        config = _asset_json()

        assert config["model_type"] == "vibevoice_streaming"
        assert config["architectures"] == ["VibeVoiceStreamingForConditionalGenerationInference"]

    def test_decoder_config_is_the_realtime_llm_shape(self):
        """The 0.5B decoder: 896 hidden, 151936 vocab, 24 layers, untied embeddings."""
        decoder = _asset_json()["decoder_config"]

        assert decoder["model_type"] == "qwen2"
        assert decoder["hidden_size"] == 896
        assert decoder["vocab_size"] == 151936
        assert decoder["num_hidden_layers"] == 24
        assert decoder["num_attention_heads"] == 14
        assert decoder["num_key_value_heads"] == 2
        assert decoder["intermediate_size"] == 4864
        # An untied embedding is what makes the checkpoint carry its own
        # `lm_head.weight`; a typo here reads as a tied head at load time.
        assert decoder["tie_word_embeddings"] is False

    def test_decoder_fingerprint_is_896_by_151936(self):
        """``config_fingerprint`` reads the asset, not the loader.

        This is the asset side of the detector contract: the signature the
        auto-detect path must learn to recognise is whatever this pair says.
        """
        assert config_fingerprint(_asset_json()) == (896, 151936)

    def test_sub_configs_and_backbone_split_are_streaming_typed(self):
        """Both towers are typed for streaming; the TTS split is 20 of 24 layers."""
        config = _asset_json()

        assert config["acoustic_tokenizer_config"]["model_type"] == "vibevoice_acoustic_tokenizer"
        assert config["diffusion_head_config"]["model_type"] == "vibevoice_diffusion_head"
        assert config["acoustic_vae_dim"] == config["acoustic_tokenizer_config"]["vae_dim"]

        # The lower layers encode text only; the upper 20 do TTS as well.
        assert config["tts_backbone_num_hidden_layers"] == 20
        assert config["tts_backbone_num_hidden_layers"] < config["decoder_config"]["num_hidden_layers"]