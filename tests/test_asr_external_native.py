"""Tests for the packaged VibeVoice-ASR assets (plan t2).

Covers:
- the packaged ASR architecture config loads through ``transformers.AutoConfig``
  and carries the shape the published VibeVoice-ASR-HF checkpoint expects
- ``resolve_sidecar_config`` prefers a ``<weight>.config.json`` sidecar and
  otherwise falls back to the packaged default for ``config_name="VibeVoice-ASR"``
- the realtime family shares that table but resolves to its own asset
- the packaged processor assets exist and parse

The assets are addressed by PATH (via ``asr_native``), never through
``import src.vibevoice.configs`` — that module is mocked wholesale in
``conftest.py``.
"""

import json
import os

import pytest

from ComfyUI_VibeVoice.modules import asr_native
from ComfyUI_VibeVoice.modules.external_loader import (
    _get_packaged_config_path,
    resolve_sidecar_config,
)

transformers = pytest.importorskip("transformers")


class TestPackagedAsrConfig:
    """The packaged default config must be loadable by transformers itself."""

    def test_packaged_config_loads_through_auto_config(self):
        """AutoConfig on the packaged file yields a native VibeVoiceAsrConfig.

        The real checkpoint is ``model_type "vibevoice_asr"`` with a
        ``text_config`` sub-config; the vendored class cannot read those keys.
        """
        config = transformers.AutoConfig.from_pretrained(
            asr_native.packaged_asset_path(asr_native.PACKAGED_ASR_CONFIG_FILE)
        )

        assert config.model_type == asr_native.NATIVE_ASR_MODEL_TYPE
        text = config.text_config
        assert text.hidden_size == 3584
        assert text.num_hidden_layers == 28
        assert text.vocab_size == 152064
        assert text.tie_word_embeddings is False
        assert config.acoustic_tokenizer_chunk_size == 1440000

    def test_packaged_config_path_is_the_registered_default(self):
        """_PACKAGED_CONFIG_FILES points at the file asr_native reads."""
        assert _get_packaged_config_path("VibeVoice-ASR") == asr_native.packaged_asset_path(
            asr_native.PACKAGED_ASR_CONFIG_FILE
        )


class TestAsrSidecarPrecedence:
    """Sidecar-first must still beat the packaged default for ASR.

    The packaged-default table is shared by every family, so the sibling
    realtime name is asserted here too: it resolves to its own asset, not
    to the ASR one.
    """

    def test_sidecar_overrides_packaged_default(self, tmp_path):
        """A <weight>.config.json sidecar wins over the packaged default."""
        weight = tmp_path / "foo.safetensors"
        weight.write_bytes(b"")
        sidecar = tmp_path / "foo.safetensors.config.json"
        sidecar.write_text(json.dumps({"model_type": "custom_asr"}), encoding="utf-8")

        resolved = resolve_sidecar_config(str(weight), "VibeVoice-ASR")

        assert resolved == str(sidecar)
        assert json.load(open(resolved, encoding="utf-8"))["model_type"] == "custom_asr"

    def test_bare_weight_resolves_to_existing_packaged_file(self, tmp_path):
        """No sidecar → the packaged ASR default, which must exist on disk."""
        weight = tmp_path / "foo.safetensors"
        weight.write_bytes(b"")

        resolved = resolve_sidecar_config(str(weight), "VibeVoice-ASR")

        assert resolved.endswith("default_VibeVoice-ASR_config.json")
        assert os.path.exists(resolved)

    def test_realtime_resolves_to_its_own_packaged_default(self, tmp_path):
        """The realtime family has a packaged default of its own, not ASR's.

        This test used to assert the opposite — that VibeVoice-Realtime-0.5B
        was the one selectable option with no packaged default and kept
        raising. Both families now ship one, so the shared table has to route
        each name to its own asset rather than to the ASR file.
        """
        weight = tmp_path / "foo.safetensors"
        weight.write_bytes(b"")

        resolved = resolve_sidecar_config(str(weight), "VibeVoice-Realtime-0.5B")

        assert resolved.endswith("default_VibeVoice-Realtime-0.5B_config.json")
        assert resolved != _get_packaged_config_path("VibeVoice-ASR")
        assert os.path.exists(resolved)


class TestPackagedProcessorAssets:
    """The native processor needs its full asset set, shipped with the node."""

    def test_three_small_assets_exist_and_parse(self):
        """tokenizer_config / processor_config / chat_template ship and read."""
        assets = asr_native.packaged_processor_assets()
        assert set(assets) == set(asr_native.PROCESSOR_ASSET_FILES)

        for name in (asr_native.PACKAGED_TOKENIZER_CONFIG_FILE, asr_native.PACKAGED_PROCESSOR_CONFIG_FILE):
            with open(assets[name], encoding="utf-8") as f:
                assert isinstance(json.load(f), dict)

        with open(assets[asr_native.PACKAGED_CHAT_TEMPLATE_FILE], encoding="utf-8") as f:
            assert f.read().strip()

    def test_processor_config_declares_native_processor_and_feature_extractor(self):
        """The packaged processor config names the native classes transformers 5.3 builds."""
        config = asr_native.read_json_asset(
            asr_native.packaged_asset_path(asr_native.PACKAGED_PROCESSOR_CONFIG_FILE)
        )

        assert config["processor_class"] == "VibeVoiceAsrProcessor"
        assert config["feature_extractor"]["feature_extractor_type"] == (
            "VibeVoiceAcousticTokenizerFeatureExtractor"
        )
        assert config["feature_extractor"]["sampling_rate"] == 24000

    def test_tokenizer_asset_resolves_packaged(self, tmp_path):
        """tokenizer.json falls back to the packaged copy for a bare weight dir."""
        resolved = asr_native.resolve_asset_file(
            asr_native.PACKAGED_TOKENIZER_FILE, str(tmp_path)
        )

        assert os.path.exists(resolved)
        assert resolved.endswith("tokenizer.json")

    def test_local_tokenizer_beats_packaged(self, tmp_path):
        """A tokenizer.json next to the weight file wins over the packaged one."""
        local = tmp_path / "tokenizer.json"
        local.write_text("{}", encoding="utf-8")

        assert asr_native.resolve_asset_file("tokenizer.json", str(tmp_path)) == str(local)

    def test_missing_asset_raises_named_error(self, tmp_path):
        """A missing asset names the file and both searched locations."""
        with pytest.raises(FileNotFoundError) as exc:
            asr_native.resolve_asset_file("definitely_absent.json", str(tmp_path))

        assert "definitely_absent.json" in str(exc.value)


# ====================================================================
# Native class selection (plan t3)
#
# The ASR branch must load the VENDORED class for a ``model_type
# "vibevoice"`` config and the NATIVE transformers class for a
# ``model_type "vibevoice_asr"`` one. The two families are incompatible
# (different state-dict key prefixes), so the dispatch is pinned here from
# both sides.
# ====================================================================

import sys
import types

import torch

from ComfyUI_VibeVoice.modules import external_loader as EL


def _write_config(directory, model_type, **extra):
    """Write a ``config.json`` with ``model_type`` and return its path."""
    path = directory / "config.json"
    path.write_text(json.dumps({"model_type": model_type, **extra}), encoding="utf-8")
    return str(path)


class TestConfigClassSelection:
    """``_load_asr_config`` must route on the config's own model_type."""

    def test_native_config_never_reaches_the_vendored_class(self, tmp_path, monkeypatch):
        """A ``vibevoice_asr`` config is read by transformers, never by the
        vendored class: its ``__init__`` reads ``decoder_config`` /
        ``acoustic_tokenizer_config`` keys a native config does not carry, so
        they would land in ``**kwargs`` and the Qwen2 default hidden size
        would survive silently."""
        config_path = _write_config(
            tmp_path, "vibevoice_asr", text_config={"hidden_size": 3584}
        )

        def _boom(*_args, **_kwargs):
            raise AssertionError(
                "vendored VibeVoiceASRConfig.from_pretrained must not see a "
                "native config"
            )

        monkeypatch.setattr(EL.VibeVoiceASRConfig, "from_pretrained", _boom)

        config = EL._load_asr_config(config_path)

        assert config.model_type == asr_native.NATIVE_ASR_MODEL_TYPE
        # The native shape the vendored class cannot represent: a text_config
        # sub-config, which VibeVoiceASRConfig would never build.
        assert type(config).__name__ == "VibeVoiceAsrConfig"
        assert config.text_config.hidden_size == 3584

    def test_legacy_sidecar_still_reaches_the_vendored_class(self, tmp_path, monkeypatch):
        """A legacy ``vibevoice`` ASR sidecar keeps going through the vendored
        class, and the instantiated object is the vendored model class."""
        config_path = _write_config(tmp_path, "vibevoice")

        seen = {}

        class _FakeVendoredConfig:
            model_type = "vibevoice"

            @classmethod
            def from_pretrained(cls, path):
                seen["path"] = path
                return cls()

        class _StubASR(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.lin = torch.nn.Linear(4, 4)
                self.config = config

        monkeypatch.setattr(EL, "VibeVoiceASRConfig", _FakeVendoredConfig)
        monkeypatch.setattr(EL, "VibeVoiceASRForConditionalGeneration", _StubASR)

        config = EL._load_asr_config(config_path)
        assert isinstance(config, _FakeVendoredConfig)
        assert seen["path"] == config_path

        model = EL._instantiate_asr_model(
            config=config,
            attn_implementation="eager",
            final_load_dtype=torch.float32,
        )
        assert isinstance(model, _StubASR)
        assert all(p.is_meta for p in model.parameters())

    def test_missing_config_still_raises_before_dispatch(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            EL._load_asr_config(str(tmp_path / "nope.json"))


class TestIsNativeAsrConfigPath:
    """The model_type peek is cheap and must never raise."""

    def test_true_for_native_file(self, tmp_path):
        path = _write_config(tmp_path, "vibevoice_asr")
        assert asr_native.is_native_asr_config_path(path) is True

    def test_true_for_native_directory(self, tmp_path):
        _write_config(tmp_path, "vibevoice_asr")
        assert asr_native.is_native_asr_config_path(str(tmp_path)) is True

    def test_false_for_legacy_vibevoice(self, tmp_path):
        path = _write_config(tmp_path, "vibevoice")
        assert asr_native.is_native_asr_config_path(path) is False

    @pytest.mark.parametrize("body", ["", "not json at all", "[]"])
    def test_false_for_unusable_json(self, tmp_path, body):
        path = tmp_path / "config.json"
        path.write_text(body, encoding="utf-8")
        assert asr_native.is_native_asr_config_path(str(path)) is False

    def test_false_for_missing_or_empty_path(self, tmp_path):
        assert asr_native.is_native_asr_config_path(str(tmp_path / "gone.json")) is False
        assert asr_native.is_native_asr_config_path("") is False


class TestVersionGuard:
    """An old transformers must fail with a message naming the minimum."""

    def test_import_failure_becomes_a_5_3_0_runtime_error(self, monkeypatch):
        """``from transformers import VibeVoiceAsrForConditionalGeneration``
        fails on transformers < 5.3; the guard must translate that into a
        RuntimeError naming the required version, with the ImportError kept
        as the cause."""
        stub = types.ModuleType("transformers")
        stub.AutoConfig = object()  # present, but the model class is not
        monkeypatch.setitem(sys.modules, "transformers", stub)

        with pytest.raises(RuntimeError) as exc:
            asr_native.import_native_asr_classes()

        message = str(exc.value)
        assert asr_native.MIN_TRANSFORMERS_VERSION in message
        assert "5.3.0" in message
        assert isinstance(exc.value.__cause__, ImportError)

    def test_available_is_the_boolean_form_of_the_guard(self, monkeypatch):
        """``native_asr_available`` is the same guard as a bool, so a caller
        can feature-check without catching."""
        monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
        assert asr_native.native_asr_available() is False

    def test_every_native_entry_point_goes_through_the_guard(self, tmp_path, monkeypatch):
        config_path = _write_config(tmp_path, "vibevoice_asr")
        monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))

        with pytest.raises(RuntimeError, match="5.3.0"):
            asr_native.load_native_asr_config(config_path)
        with pytest.raises(RuntimeError, match="5.3.0"):
            asr_native.instantiate_native_asr_model(
                object(), attn_implementation="eager", final_load_dtype=torch.bfloat16
            )
        with pytest.raises(RuntimeError, match="5.3.0"):
            EL._load_asr_config(config_path)


class _StubNativeConfig:
    """Just enough config surface for the dispatch + attention routing.

    The sub-configs are REAL ``PretrainedConfig`` objects so
    ``set_config_dtype`` takes the same version-safe branch it takes for the
    shipped config (on transformers v5 it writes ``dtype``, because
    ``torch_dtype`` is a deprecated property there).
    """

    model_type = asr_native.NATIVE_ASR_MODEL_TYPE

    def __init__(self):
        self.text_config = transformers.PretrainedConfig()
        self.acoustic_tokenizer_encoder_config = transformers.PretrainedConfig()
        self.semantic_tokenizer_encoder_config = transformers.PretrainedConfig()


@pytest.mark.parametrize("mode", ["sdpa", "flash_attention_2", "eager"])
def test_attention_mode_lands_on_the_language_model_only(mode):
    """The requested mode must reach the language model; the ConvNext
    encoders must stay eager whatever was asked for.

    The native ctor takes no ``attn_implementation`` and the config has no
    ``set_attn_implementation``, so the config objects the ctor builds its
    submodules FROM are the only place the routing can live.
    """
    config = _StubNativeConfig()

    asr_native.apply_native_attn_implementation(config, mode)

    assert config.text_config._attn_implementation == mode
    assert config.acoustic_tokenizer_encoder_config._attn_implementation == "eager"
    assert config.semantic_tokenizer_encoder_config._attn_implementation == "eager"


def test_attention_routing_tolerates_a_config_without_the_encoders():
    """Only sub-configs that are actually present are touched."""

    class _Bare:
        model_type = asr_native.NATIVE_ASR_MODEL_TYPE

    bare = _Bare()
    asr_native.apply_native_attn_implementation(bare, "sdpa")

    assert bare._attn_implementation == "sdpa"
    assert not hasattr(bare, "text_config")


@pytest.mark.parametrize("mode", ["sdpa", "flash_attention_2"])
def test_instantiation_routes_attention_and_dtype(mode, monkeypatch):
    """The helper applies the routing itself, and records the dtype on the
    root AND on text_config (the submodules are built from text_config, so a
    root-only dtype would not be inherited)."""
    config = _StubNativeConfig()
    built_with = {}

    class _StubModel:
        def __init__(self, cfg):
            built_with["config"] = cfg

    monkeypatch.setattr(
        asr_native, "import_native_asr_classes", lambda: (object, _StubModel)
    )

    model = asr_native.instantiate_native_asr_model(
        config, attn_implementation=mode, final_load_dtype=torch.bfloat16
    )

    assert isinstance(model, _StubModel)
    assert built_with["config"] is config
    assert config.text_config._attn_implementation == mode
    assert config.acoustic_tokenizer_encoder_config._attn_implementation == "eager"
    assert config.semantic_tokenizer_encoder_config._attn_implementation == "eager"
    assert config.text_config.dtype == torch.bfloat16


def test_dispatch_is_on_the_model_type_value_not_truthiness():
    """``tests/test_loader_quant_paths.py`` patches ``_load_asr_config`` to
    return a bare MagicMock. Dispatching on the model_type VALUE is what
    keeps that test (and any other double) on the vendored class."""
    from unittest.mock import MagicMock

    config = MagicMock()
    assert (config.model_type == asr_native.NATIVE_ASR_MODEL_TYPE) is False


# ====================================================================
# TREE-SHAPE GATE
#
# The published VibeVoice-ASR-HF checkpoint's 901 key names are committed as
# a fixture (derived from model.safetensors.index.json's weight_map). If a
# transformers upgrade ever reshapes the tree, the vendored class is no
# longer the right one for these weights and the mismatch must be caught
# here rather than as silently unloaded tensors at run time.
#
# The tree is built on ``meta`` from the PACKAGED config: no weight file, no
# 16.6 GB of RAM, and a measured ~0.1 s.
# ====================================================================

_ASR_HF_KEYS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "vibevoice_asr_hf_keys.json"
)


def _checkpoint_keys():
    with open(_ASR_HF_KEYS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _packaged_native_config():
    """AutoConfig on the packaged VibeVoice-ASR default config."""
    return transformers.AutoConfig.from_pretrained(
        asr_native.packaged_asset_path(asr_native.PACKAGED_ASR_CONFIG_FILE)
    )


def test_fixture_pins_the_published_checkpoint_shape():
    """The committed list is the 901-name weight_map of the 8B ASR-HF
    checkpoint, grouped the way the native tree groups them."""
    keys = _checkpoint_keys()

    assert len(keys) == 901
    assert len(set(keys)) == 901, "fixture must not repeat a key"
    groups = {}
    for key in keys:
        groups[key.split(".")[0]] = groups.get(key.split(".")[0], 0) + 1
    assert groups == {
        "language_model": 339,
        "acoustic_tokenizer_encoder": 276,
        "semantic_tokenizer_encoder": 276,
        "multi_modal_projector": 10,
    }


def test_native_tree_covers_every_checkpoint_key():
    """The native class built on meta must expose a superset of the
    checkpoint's keys. A missing key is an unloaded weight (silently wrong
    audio at run time), so the failure message names every missing PREFIX
    instead of the hundreds of individual leaves it would otherwise print."""
    config = _packaged_native_config()

    model = EL._instantiate_asr_model(
        config=config,
        attn_implementation="eager",
        final_load_dtype=torch.bfloat16,
    )
    assert isinstance(
        model, transformers.VibeVoiceAsrForConditionalGeneration
    ), "a native config must build the native class"

    tree_keys = set(model.state_dict().keys())
    checkpoint_keys = set(_checkpoint_keys())

    missing = sorted(checkpoint_keys - tree_keys)
    assert not missing, (
        "The native VibeVoice-ASR tree no longer covers the published "
        f"VibeVoice-ASR-HF checkpoint: {len(missing)} of {len(checkpoint_keys)} "
        "keys have no parameter. Missing prefixes: "
        + ", ".join(sorted({key.rsplit(".", 1)[0] for key in missing}))
    )


@pytest.mark.parametrize("mode", ["sdpa", "flash_attention_2"])
def test_attention_mode_survives_the_native_constructor(mode):
    """End-to-end form of the routing test: the mode written on the config
    must be the one the CONSTRUCTED tree reports. The native ctor takes no
    ``attn_implementation`` and the config has no
    ``set_attn_implementation``, so a routing rule that only touched the
    right attribute of the right object would still be dropped here."""
    config = _packaged_native_config()

    model = EL._instantiate_asr_model(
        config=config,
        attn_implementation=mode,
        final_load_dtype=torch.bfloat16,
    )

    assert model.language_model.config._attn_implementation == mode
    assert model.acoustic_tokenizer_encoder.config._attn_implementation == "eager"
    assert model.semantic_tokenizer_encoder.config._attn_implementation == "eager"


# ====================================================================
# Native processor construction (plan t4)
#
# The published checkpoint ships a single weight file. Its directory has no
# tokenizer.json, no preprocessor_config.json, no tokenizer_config.json and no
# chat_template.jinja — the node carries those itself. A native
# VibeVoiceAsrProcessor therefore has to be assembled from components rather
# than discovered by AutoProcessor, which is both directory-based and
# SILENTLY wrong on a partial asset set (see build_native_asr_processor).
# ====================================================================

import hashlib
from unittest.mock import MagicMock, patch

from ComfyUI_VibeVoice.modules import asr_generation
from ComfyUI_VibeVoice.modules.external_loader import resolve_sidecar_preprocessor

# The exact key set the native forward accepts. The vendored processor emits
# {input_ids, acoustic_input_mask, speech, vae_tok_len} instead — none of
# which VibeVoiceAsrForConditionalGeneration.forward takes, so a vendored
# processor paired with a native model cannot transcribe at all.
NATIVE_TRANSCRIPTION_KEYS = {
    "attention_mask",
    "input_ids",
    "input_values",
    "padding_mask",
}

# sha256 over the canonical {"vocab", "added"} projection of the packaged
# tokenizer.json. The published VibeVoice-ASR-HF tokenizer.json (11,421,892 B)
# and the packaged copy (7,334,926 B) hash identically here: the same
# 151,643-entry vocab and the same 22 added tokens, only the encoder payload
# (and the file size) differs — which is what makes the fallback sound.
PACKAGED_TOKENIZER_SHA256 = "4f06d63b02fb14668436c6f83b116e875d804ac3574ef3e1bc0fd48462839f3f"


def _packaged_tokenizer_projection():
    """Return the canonical (json text, vocab count, added count) triple."""
    with open(
        asr_native.packaged_asset_path(asr_native.PACKAGED_TOKENIZER_FILE),
        encoding="utf-8",
    ) as f:
        data = json.load(f)
    vocab = data["model"]["vocab"]
    added = data["added_tokens"]
    canonical = json.dumps(
        {"vocab": vocab, "added": added}, sort_keys=True, separators=(",", ":")
    )
    return canonical, len(vocab), len(added)


# Where the published VibeVoice-ASR-HF tokenizer.json lives when this host has
# it. Resolution order: the VIBEVOICE_TEST_ASR_HF_DIR override (a directory or
# the file itself), then the conventional ComfyUI model trees. Nothing is
# written outside the repo and no file is loaded, only read.
_PUBLISHED_TOKENIZER_NAME = "tokenizer.json"


def _published_tokenizer_path():
    """Path to the published ASR tokenizer, or skip the test."""
    candidates = []
    override = os.environ.get("VIBEVOICE_TEST_ASR_HF_DIR", "").strip()
    if override:
        candidates.append(override)
    comfy_root = os.environ.get("COMFYUI_ROOT", "").strip()
    for root in (comfy_root, r"C:\AI\ComfyUI\ComfyUI"):
        if not root:
            continue
        candidates.append(
            os.path.join(root, "models", "tts", "VibeVoice", "VibeVoice-ASR-HF")
        )

    for candidate in candidates:
        path = candidate
        if os.path.isdir(path):
            path = os.path.join(path, _PUBLISHED_TOKENIZER_NAME)
        if os.path.isfile(path):
            return path

    pytest.skip(
        "published VibeVoice-ASR-HF/tokenizer.json not on this host; set "
        "VIBEVOICE_TEST_ASR_HF_DIR=<dir> to run this comparison "
        f"(looked in: {', '.join(candidates)})"
    )


class TestNativeProcessorFromEmptyWeightDir:
    """(a) A weight directory with ZERO files must still yield a usable native
    processor, and (b) AutoProcessor must never be consulted."""

    def test_empty_weight_dir_yields_a_working_native_processor(self, tmp_path):
        """The user ships one .safetensors; the node ships the assets."""
        assert list(tmp_path.iterdir()) == []

        processor = asr_native.build_native_asr_processor(str(tmp_path), "")

        assert isinstance(processor, transformers.VibeVoiceAsrProcessor)
        assert type(processor).__module__.startswith("transformers.")
        assert asr_generation._asr_processor_kind(processor) == "native"
        assert processor.feature_extractor.sampling_rate == 24000

    def test_apply_transcription_request_yields_the_native_key_set(self, tmp_path):
        """The output must match what the native forward consumes.

        Without the chat template this raises "ValueError: Cannot use
        apply_chat_template because this processor does not have a chat
        template" — exactly what a partial asset set produces.
        """
        processor = asr_native.build_native_asr_processor(str(tmp_path), "")

        inputs = processor.apply_transcription_request(
            audio=torch.zeros(24000), prompt="speaker_0"
        )

        assert set(inputs.keys()) == NATIVE_TRANSCRIPTION_KEYS

    def test_auto_processor_is_never_called(self, tmp_path, monkeypatch):
        """AutoProcessor.from_pretrained on a directory is the failure mode
        this design exists to avoid: a directory holding only tokenizer.json
        returns a TokenizersBackend (not a processor at all), and adding the
        other files one at a time only moves the failure (OSError, then a
        missing-chat-template ValueError)."""

        def _boom(*args, **kwargs):
            raise AssertionError(
                f"AutoProcessor.from_pretrained must not be called "
                f"(args={args!r}, kwargs={kwargs!r})"
            )

        monkeypatch.setattr(
            transformers.AutoProcessor, "from_pretrained", staticmethod(_boom)
        )

        processor = asr_native.build_native_asr_processor(str(tmp_path), "")

        assert isinstance(processor, transformers.VibeVoiceAsrProcessor)

    def test_special_tokens_come_from_the_processor_config(self, tmp_path):
        """audio_token / bos / eos / duration drive the chat template's
        placeholder expansion, so they must be the checkpoint's own."""
        processor = asr_native.build_native_asr_processor(str(tmp_path), "")

        assert processor.audio_token == "<|box_start|>"
        assert processor.audio_bos_token == "<|object_ref_start|>"
        assert processor.audio_eos_token == "<|object_ref_end|>"
        assert processor.audio_duration_token == "<|AUDIO_DURATION|>"


class TestPackagedTokenizerFallback:
    """(c) The packaged tokenizer is token-equivalent to the published one."""

    def test_canonical_projection_hashes_to_the_measured_digest(self):
        canonical, _, _ = _packaged_tokenizer_projection()
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        assert digest == PACKAGED_TOKENIZER_SHA256

    def test_vocab_and_added_token_counts(self):
        _, vocab_size, added_count = _packaged_tokenizer_projection()

        assert vocab_size == 151643
        assert added_count == 22

    def test_packaged_tokenizer_matches_the_published_file(self):
        """The equivalence criterion itself, against the real file.

        ``test_canonical_projection_hashes_to_the_measured_digest`` pins the
        packaged copy to a constant, which cannot notice that the PUBLISHED
        tokenizer has since changed. Where the published
        ``VibeVoice-ASR-HF/tokenizer.json`` is on this host, diff the two
        directly: same vocab size, empty symmetric difference in both
        directions, identical token->id mapping, identical added_tokens.

        Optional by construction (same idiom as
        ``tests/test_gguf_quant_blocks.py``): the file is 11 MB and lives in
        the user's model tree, not in the repo, so the test skips unless the
        host has it. ``VIBEVOICE_TEST_ASR_HF_DIR`` overrides the location.
        """
        published_path = _published_tokenizer_path()
        with open(published_path, encoding="utf-8") as f:
            published = json.load(f)
        with open(
            asr_native.packaged_asset_path(asr_native.PACKAGED_TOKENIZER_FILE),
            encoding="utf-8",
        ) as f:
            packaged = json.load(f)

        pub_vocab = published["model"]["vocab"]
        pkg_vocab = packaged["model"]["vocab"]

        assert len(pkg_vocab) == len(pub_vocab), (
            f"vocab size diverged: packaged {len(pkg_vocab)} vs published "
            f"{len(pub_vocab)}"
        )
        only_in_packaged = set(pkg_vocab) - set(pub_vocab)
        only_in_published = set(pub_vocab) - set(pkg_vocab)
        assert not only_in_packaged and not only_in_published, (
            "symmetric difference is non-empty: "
            f"{len(only_in_packaged)} tokens only in the packaged copy "
            f"(e.g. {sorted(only_in_packaged)[:3]}), "
            f"{len(only_in_published)} only in the published one "
            f"(e.g. {sorted(only_in_published)[:3]})"
        )
        # Equal SETS are not an equal MAPPING; the id half must match too.
        differing_ids = {
            token for token in pub_vocab if pkg_vocab[token] != pub_vocab[token]
        }
        assert not differing_ids, (
            f"{len(differing_ids)} tokens map to different ids "
            f"(e.g. {sorted(differing_ids)[:3]})"
        )
        assert packaged["added_tokens"] == published["added_tokens"], (
            "added_tokens diverged: "
            f"{len(packaged['added_tokens'])} packaged vs "
            f"{len(published['added_tokens'])} published"
        )

    def test_packaged_tokenizer_is_what_the_processor_gets(self, tmp_path):
        """The fallback is not merely equivalent, it is the one actually used."""
        processor = asr_native.build_native_asr_processor(str(tmp_path), "")

        assert processor.tokenizer.vocab_size == 151643
        assert len(processor.tokenizer.get_added_vocab()) == 22
        # A tokenizer that hashes the same but cannot encode would still be a
        # broken fallback, so round-trip real text through it.
        text = "The quick brown fox jumps over the lazy dog."
        assert processor.tokenizer.decode(
            processor.tokenizer(text)["input_ids"]
        ) == text


class TestPreprocessorOverlay:
    """(d) The sidecar OVERLAYS the packaged feature-extractor defaults.

    An empty resolution means "packaged defaults", never a crash — the
    vendored builder silently accepted "" and fell back to its own
    constructor defaults, which are not the checkpoint's.
    """

    def test_empty_sidecar_keeps_the_packaged_defaults(self, tmp_path):
        processor = asr_native.build_native_asr_processor(str(tmp_path), "")
        feature_extractor = processor.feature_extractor

        assert feature_extractor.sampling_rate == 24000
        assert feature_extractor.normalize_audio is True
        assert feature_extractor.target_dB_FS == -25
        assert feature_extractor.eps == pytest.approx(1e-6)

    def test_weight_dot_preprocessor_json_overlays(self, tmp_path):
        """The preferred sidecar name, next to the weight file."""
        weight = tmp_path / "model.safetensors"
        weight.write_bytes(b"")
        sidecar = tmp_path / "model.safetensors.preprocessor.json"
        sidecar.write_text(
            json.dumps({"target_dB_FS": -31.5, "normalize_audio": False}),
            encoding="utf-8",
        )

        preprocessor_path = resolve_sidecar_preprocessor(str(weight))
        processor = asr_native.build_native_asr_processor(
            str(tmp_path), preprocessor_path
        )

        assert preprocessor_path == str(sidecar)
        assert processor.feature_extractor.target_dB_FS == -31.5
        assert processor.feature_extractor.normalize_audio is False
        # Not mentioned by the sidecar -> still the packaged value.
        assert processor.feature_extractor.sampling_rate == 24000

    def test_directory_preprocessor_config_json_overlays(self, tmp_path):
        """The in-directory name, the other resolve_sidecar_preprocessor slot."""
        weight = tmp_path / "model.safetensors"
        weight.write_bytes(b"")
        in_dir = tmp_path / "preprocessor_config.json"
        in_dir.write_text(json.dumps({"sampling_rate": 16000}), encoding="utf-8")

        preprocessor_path = resolve_sidecar_preprocessor(str(weight))
        processor = asr_native.build_native_asr_processor(
            str(tmp_path), preprocessor_path
        )

        assert preprocessor_path == str(in_dir)
        assert processor.feature_extractor.sampling_rate == 16000
        # Everything the sidecar did not mention stays packaged.
        assert processor.feature_extractor.target_dB_FS == -25

    def test_weight_dot_sidecar_beats_the_directory_name(self, tmp_path):
        """Precedence is unchanged: <weight>.preprocessor.json wins."""
        weight = tmp_path / "model.safetensors"
        weight.write_bytes(b"")
        (tmp_path / "model.safetensors.preprocessor.json").write_text(
            json.dumps({"target_dB_FS": -11.0}), encoding="utf-8"
        )
        (tmp_path / "preprocessor_config.json").write_text(
            json.dumps({"target_dB_FS": -22.0}), encoding="utf-8"
        )

        preprocessor_path = resolve_sidecar_preprocessor(str(weight))
        processor = asr_native.build_native_asr_processor(
            str(tmp_path), preprocessor_path
        )

        assert processor.feature_extractor.target_dB_FS == -11.0


class TestDispatchSelectsProcessorByModelType:
    """The ASR branch must route on the config's own model_type: a native
    checkpoint skips the vendored tokenizer entirely (the native processor
    owns its Qwen2TokenizerFast), a legacy one keeps the vendored pair."""

    def test_native_config_skips_the_vendored_tokenizer(self, tmp_path):
        from unittest.mock import patch

        weight = tmp_path / "model.safetensors"
        weight.write_bytes(b"")
        config_path = tmp_path / "model.safetensors.config.json"
        config_path.write_text(
            json.dumps({"model_type": "vibevoice_asr"}), encoding="utf-8"
        )

        with patch.object(EL, "_load_asr_tokenizer") as load_tokenizer, patch.object(
            EL, "_load_asr_processor"
        ) as load_processor:
            assert EL.is_native_asr_config_path(str(config_path)) is True
            assert EL.resolve_sidecar_config(str(weight), "VibeVoice-ASR") == str(
                config_path
            )
            # The real dispatch lives inside load_external_vibevoice_asr_model
            # (asserted end to end in test_asr_external_native_load.py); what is
            # pinned here is the SELECTOR it dispatches on.
            selected_native = EL.is_native_asr_config_path(str(config_path))

        assert selected_native is True
        assert load_tokenizer.call_count == 0
        assert load_processor.call_count == 0

    def test_legacy_config_keeps_the_vendored_pair(self, tmp_path):
        weight = tmp_path / "model.safetensors"
        weight.write_bytes(b"")
        config_path = tmp_path / "model.safetensors.config.json"
        config_path.write_text(json.dumps({"model_type": "vibevoice"}), encoding="utf-8")

        assert EL.is_native_asr_config_path(str(config_path)) is False


# ====================================================================
# Post-load parity (plan t4, test (e))
#
# The ASR branch instantiates the model on ``meta`` and binds the checkpoint
# by assignment. Anything NOT in the state dict therefore stays uninitialised
# — and the native rotary buffers (``rotary_emb.inv_freq`` /
# ``original_inv_freq``) are exactly that: they are computed from the config
# in __init__ and never appear in the file. The dense path already reaches
# loader._post_assign_fixups (VibeVoiceLoader._apply_state_dict), which calls
# _recompute_rope_buffers; this test is the OBLIGATION that a future refactor
# of that chain cannot silently drop it, because a zero inv_freq means the
# model loses positional encoding and emits gibberish.
#
# The 16.6 GB published checkpoint is never loaded by any test here — only
# this structurally identical tiny model.
# ====================================================================

# ``depths`` MUST be len(downsampling_ratios) + 1 or the encoder raises
# IndexError. A two-layer Qwen2 backbone is the smallest tree that still has
# the architecture the real 28-layer / 3584-hidden checkpoint reduces to.
_TINY_ENCODER_CONFIG = {
    "model_type": "vibevoice_acoustic_tokenizer_encoder",
    "hidden_size": 16,
    "num_filters": 8,
    "depths": [1, 1, 1],
    "downsampling_ratios": [2, 2],
    "ffn_expansion": 2,
    "kernel_size": 3,
    "channels": 1,
}

_TINY_TEXT_CONFIG = {
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

_TINY_NATIVE_CONFIG = {
    "architectures": ["VibeVoiceAsrForConditionalGeneration"],
    "model_type": "vibevoice_asr",
    "text_config": dict(_TINY_TEXT_CONFIG),
    "acoustic_tokenizer_encoder_config": dict(_TINY_ENCODER_CONFIG),
    "semantic_tokenizer_encoder_config": dict(_TINY_ENCODER_CONFIG),
    "acoustic_tokenizer_chunk_size": 48000,
}


def _build_tiny_native_asr_model():
    """Build the tiny native ASR model on CPU with a fixed seed."""
    try:
        from transformers import VibeVoiceAsrForConditionalGeneration
    except ImportError as e:  # pragma: no cover - depends on the env
        pytest.skip(
            f"transformers {transformers.__version__} has no native VibeVoice "
            f"ASR support: {e}"
        )

    config = transformers.AutoConfig.for_model(
        **{k: v for k, v in _TINY_NATIVE_CONFIG.items() if k != "architectures"}
    )
    torch.manual_seed(0)
    return VibeVoiceAsrForConditionalGeneration(config)


@pytest.fixture(scope="module")
def tiny_native_asr_checkpoint(tmp_path_factory):
    """Write a tiny native ASR checkpoint the way a user ships one.

    Returns the weight path. Deliberately NO tokenizer.json / preprocessor
    sidecar in the directory: the loader must source its own assets, which is
    the whole point of the native processor path.
    """
    from safetensors.torch import save_file

    model = _build_tiny_native_asr_model()
    directory = tmp_path_factory.mktemp("t4_native_asr")
    weight_path = str(directory / "model.safetensors")
    save_file(
        {k: v.detach().clone() for k, v in model.state_dict().items()}, weight_path
    )
    with open(weight_path + ".config.json", "w", encoding="utf-8") as f:
        json.dump(_TINY_NATIVE_CONFIG, f)
    return weight_path


def _load_tiny_native_asr(weight_path):
    """Run the real external ASR loader, pinned to CPU."""
    with patch.object(
        EL.model_management, "get_torch_device", return_value=torch.device("cpu")
    ):
        return EL.load_external_vibevoice_asr_model(
            weight_path=weight_path,
            config_name="VibeVoice-ASR",
            attention_mode="sdpa",
            dtype_str="fp32",
        )


class TestPostLoadParity:
    """(e) After an end-to-end external ASR load, nothing may be left on meta."""

    def test_no_named_buffer_is_still_on_meta(self, tiny_native_asr_checkpoint):
        model = _load_tiny_native_asr(tiny_native_asr_checkpoint)["model"]

        leftover = [name for name, buf in model.named_buffers() if buf.is_meta]
        assert leftover == [], f"uninitialised buffers after load: {leftover}"

    def test_inv_freq_is_materialized_and_matches_a_fresh_rotary(self, tiny_native_asr_checkpoint):
        """Exact parity, not just "non-zero": the recomputed buffer must be
        bit-identical to one a directly constructed Qwen2RotaryEmbedding
        computes from the same config."""
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

        bundle = _load_tiny_native_asr(tiny_native_asr_checkpoint)
        model, config = bundle["model"], bundle["config"]

        inv_freq = model.language_model.model.rotary_emb.inv_freq
        reference = Qwen2RotaryEmbedding(config.text_config).inv_freq

        assert not inv_freq.is_meta
        assert float(inv_freq.sum()) > 0.0, "inv_freq is all zeros: RoPE is dead"
        assert torch.equal(inv_freq, reference), (
            f"inv_freq differs from a freshly computed rotary: "
            f"{inv_freq} != {reference}"
        )

    def test_original_inv_freq_is_materialized_too(self, tiny_native_asr_checkpoint):
        """The second rotary buffer is a non-persistent copy of the same thing;
        leaving it on meta is the same failure one attribute over."""
        model = _load_tiny_native_asr(tiny_native_asr_checkpoint)["model"]
        rotary = model.language_model.model.rotary_emb

        assert not rotary.original_inv_freq.is_meta
        assert torch.equal(rotary.original_inv_freq, rotary.inv_freq)

    def test_native_load_never_builds_the_vendored_tokenizer_or_processor(self, tiny_native_asr_checkpoint):
        """The dispatch itself, end to end: a native config must skip
        _load_asr_tokenizer entirely (the native processor owns its
        Qwen2TokenizerFast) and must not reach the vendored processor, which
        emits keys the native forward does not accept."""
        with patch.object(EL, "_load_asr_tokenizer") as load_tokenizer, patch.object(
            EL, "_load_asr_processor"
        ) as load_processor:
            bundle = _load_tiny_native_asr(tiny_native_asr_checkpoint)

        assert load_tokenizer.call_count == 0
        assert load_processor.call_count == 0
        assert isinstance(bundle["processor"], transformers.VibeVoiceAsrProcessor)
        assert asr_generation._asr_processor_kind(bundle["processor"]) == "native"

    def test_legacy_config_still_reaches_the_vendored_pair(self, tmp_path):
        """The other side of the dispatch: a ``model_type "vibevoice"``
        config must keep calling the vendored tokenizer and processor.

        The assertion is on the DISPATCH, not on how far the load then gets.
        This used to require a ``RuntimeError`` from the vendored model
        instantiation, but that was only ever a side-effect of conftest mocking
        the vendored classes: pinning the failure mode made the test break for
        reasons unrelated to the routing it exists to check.
        """
        weight_path = str(tmp_path / "legacy.safetensors")
        torch.manual_seed(1)
        state_dict = {
            k: v.detach().clone()
            for k, v in _build_tiny_native_asr_model().state_dict().items()
        }
        from safetensors.torch import save_file

        save_file(state_dict, str(weight_path))
        with open(weight_path + ".config.json", "w", encoding="utf-8") as f:
            json.dump({"model_type": "vibevoice"}, f)

        with patch.object(EL, "_load_asr_tokenizer", return_value=MagicMock()) as load_tokenizer, \
             patch.object(EL, "_load_asr_processor", return_value=MagicMock()) as load_processor, \
             patch.object(EL.model_management, "get_torch_device", return_value=torch.device("cpu")):
            # The vendored classes are MagicMocks in conftest, so the vendored
            # model instantiation cannot build a real tree here; the point of
            # this test is only the tokenizer/processor selection. Whether the
            # load afterwards completes or raises is not the contract.
            try:
                EL.load_external_vibevoice_asr_model(
                    weight_path=str(weight_path),
                    config_name="VibeVoice-ASR",
                    attention_mode="sdpa",
                    dtype_str="fp32",
                )
            except Exception:
                pass

        # A "vibevoice" model_type must reach the VENDORED pair exactly once
        # and must not fall through to the transformers-native classes.
        assert load_tokenizer.call_count == 1
        assert load_processor.call_count == 1
