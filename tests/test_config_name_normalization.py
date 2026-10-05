"""Tests for config_name de-duplication + legacy alias normalization.

Plan 2026-08-27 (Phase 1): 'VibeVoice-Large' was removed from the dropdown
(it mapped to the exact same packaged config as 'VibeVoice-7B'), but saved
workflows carrying the removed value must keep loading via the alias map.
"""

import inspect
from unittest.mock import MagicMock, patch

import pytest

from ComfyUI_VibeVoice.modules.external_loader import (
    AUTO_CONFIG_NAME,
    ASR_CONFIG_NAMES,
    EXTERNAL_CONFIG_OPTIONS,
    _get_packaged_config_path,
    is_asr_config_name,
    normalize_config_name,
)
from ComfyUI_VibeVoice.nodes.external_loader_node import VibeVoiceExternalLoaderNode


class TestNormalizeConfigName:
    """normalize_config_name(): alias map + passthrough semantics."""

    def test_legacy_large_maps_to_7b(self):
        assert normalize_config_name("VibeVoice-Large") == "VibeVoice-7B"

    def test_alias_lookup_is_case_insensitive(self):
        assert normalize_config_name("vibevoice-large") == "VibeVoice-7B"
        assert normalize_config_name("VIBEVOICE-LARGE") == "VibeVoice-7B"

    def test_current_options_pass_through_unchanged(self):
        for name in EXTERNAL_CONFIG_OPTIONS:
            assert normalize_config_name(name) == name

    def test_unknown_value_returned_as_is(self):
        assert normalize_config_name("Not-A-Model") == "Not-A-Model"

    def test_empty_and_none_safe(self):
        assert normalize_config_name("") == ""
        assert normalize_config_name(None) is None


class TestRealtimeAliases:
    """The lowercase realtime spellings normalize to the one canonical option.

    Saved workflows carry the realtime family in the spelling their author
    typed — ``vibevoice-realtime`` with a hyphen, ``vibevoice_realtime`` with
    an underscore, or either in a different case. Lookup is ``.lower()``-keyed,
    so two alias entries cover all of them, and every one must reach the
    canonical option instead of falling through to the no-packaged-default
    error. The aliases never become visible options: the dropdown keeps one
    canonical entry.
    """

    REALTIME = "VibeVoice-Realtime-0.5B"
    ALIASES = ("vibevoice-realtime", "vibevoice_realtime")

    def test_canonical_realtime_passes_through(self):
        assert normalize_config_name(self.REALTIME) == self.REALTIME

    @pytest.mark.parametrize(
        "alias",
        [
            "vibevoice-realtime",
            "VibeVoice-Realtime",
            "VIBEVOICE-REALTIME",
            "vibevoice_realtime",
            "VibeVoice_Realtime",
            "VIBEVOICE_REALTIME",
        ],
    )
    def test_alias_maps_to_canonical_option(self, alias):
        assert normalize_config_name(alias) == self.REALTIME

    def test_alias_is_idempotent(self):
        once = normalize_config_name("vibevoice_realtime")
        assert normalize_config_name(once) == once

    def test_aliases_resolve_to_the_packaged_realtime_config(self):
        """The alias must land on the file the packaged-defaults task shipped."""
        for alias in self.ALIASES:
            path = _get_packaged_config_path(normalize_config_name(alias))
            assert path, f"{alias!r} resolved to no packaged config"
            assert path.replace("\\", "/").endswith(
                "src/vibevoice/configs/default_VibeVoice-Realtime-0.5B_config.json"
            )

    def test_aliases_are_not_visible_options(self):
        """Aliases normalize, but the dropdown keeps one canonical entry only."""
        for alias in self.ALIASES:
            assert alias not in EXTERNAL_CONFIG_OPTIONS
        assert EXTERNAL_CONFIG_OPTIONS.count(self.REALTIME) == 1

    @pytest.mark.parametrize("alias", list(ALIASES) + ["VibeVoice-Realtime"])
    def test_validate_inputs_accepts_alias(self, alias):
        assert VibeVoiceExternalLoaderNode.validate_inputs(config_name=alias) is True

    def test_unknown_name_is_not_silently_rewritten(self):
        """Aliases are a closed map, not a fuzzy matcher."""
        assert normalize_config_name("VibeVoice-99B") == "VibeVoice-99B"
        assert normalize_config_name("vibevoice-realtime-99b") == "vibevoice-realtime-99b"

    def test_existing_large_alias_unchanged(self):
        """Adding the realtime pair must not disturb the 7B alias."""
        assert normalize_config_name("vibevoice-large") == "VibeVoice-7B"

    def test_execute_normalizes_alias_before_identity(self):
        """The node normalizes BEFORE the cache identity, so an aliased saved
        workflow reuses the same bundle as the canonical spelling."""
        fake_bundle = {"model": MagicMock()}

        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.load_external_vibevoice_model",
            return_value=fake_bundle,
        ) as mock_load, patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_full_path_or_raise",
            return_value="/fake/path/model.safetensors",
        ), patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.identity_for_external",
            return_value="external:key",
        ) as mock_identity:
            VibeVoiceExternalLoaderNode.execute(
                model_file="model.safetensors",
                config_name="vibevoice_realtime",
                attention_mode="sdpa",
                quantize_llm_4bit=False,
                dtype="auto",
            )

        assert mock_identity.call_args[0][1] == self.REALTIME
        assert mock_load.call_args[1]["config_name"] == self.REALTIME


class TestDropdownDeDuplication:
    """The visible option list no longer contains the ambiguous alias."""

    def test_large_not_in_options(self):
        assert "VibeVoice-Large" not in EXTERNAL_CONFIG_OPTIONS

    def test_7b_still_in_options(self):
        assert "VibeVoice-7B" in EXTERNAL_CONFIG_OPTIONS
        assert "VibeVoice-1.5B" in EXTERNAL_CONFIG_OPTIONS

    def test_large_has_no_packaged_default_anymore(self):
        assert _get_packaged_config_path("VibeVoice-Large") == ""

    def test_7b_still_maps_to_large_config_file(self):
        # The packaged FILE keeps its historical name; only the option was
        # removed. Pin the mapping so a rename accident breaks the build.
        path = _get_packaged_config_path("VibeVoice-7B")
        assert path.replace("\\", "/").endswith(
            "src/vibevoice/configs/default_VibeVoice-Large_config.json"
        )


class TestNodeValidateInputs:
    """validate_inputs override: accepts legacy aliases, rejects junk.

    The **kwargs signature is load-bearing: ComfyUI core's execution.py skips
    its built-in combo-membership check when the node's validate function has
    var-keywords, which is what lets saved 'VibeVoice-Large' workflows queue.
    """

    def test_signature_has_varkw(self):
        spec = inspect.getfullargspec(VibeVoiceExternalLoaderNode.validate_inputs)
        assert spec.varkw is not None

    def test_accepts_legacy_alias(self):
        assert VibeVoiceExternalLoaderNode.validate_inputs(
            config_name="VibeVoice-Large"
        ) is True

    def test_accepts_current_options(self):
        for name in EXTERNAL_CONFIG_OPTIONS:
            assert VibeVoiceExternalLoaderNode.validate_inputs(
                config_name=name
            ) is True

    def test_rejects_unknown_with_message(self):
        result = VibeVoiceExternalLoaderNode.validate_inputs(config_name="bogus")
        assert isinstance(result, str)
        assert "bogus" in result
        assert "VibeVoice-7B" in result


class TestExecuteNormalization:
    """execute() normalizes BEFORE identity computation (plan F11/D1)."""

    def test_legacy_alias_reaches_loader_as_7b(self):
        fake_bundle = {"model": MagicMock()}

        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.load_external_vibevoice_model",
            return_value=fake_bundle,
        ) as mock_load, patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_full_path_or_raise",
            return_value="/fake/path/model.safetensors",
        ):
            VibeVoiceExternalLoaderNode.execute(
                model_file="model.safetensors",
                config_name="VibeVoice-Large",
                attention_mode="sdpa",
                quantize_llm_4bit=False,
                dtype="auto",
            )

        assert mock_load.call_args[1]["config_name"] == "VibeVoice-7B"

    def test_identity_computed_with_normalized_name(self):
        """Legacy and modern workflows must produce the same cache identity."""
        fake_bundle = {"model": MagicMock()}

        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.load_external_vibevoice_model",
            return_value=fake_bundle,
        ), patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_full_path_or_raise",
            return_value="/fake/path/model.safetensors",
        ), patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.identity_for_external",
            return_value="external:key",
        ) as mock_identity:
            VibeVoiceExternalLoaderNode.execute(
                model_file="model.safetensors",
                config_name="VibeVoice-Large",
                attention_mode="sdpa",
                quantize_llm_4bit=False,
                dtype="auto",
            )

        assert mock_identity.call_args[0][1] == "VibeVoice-7B"


class TestLoaderEntryNormalization:
    """load_external_vibevoice_model normalizes at its own entry (defense in
    depth: direct callers/tests bypass the node)."""

    def test_tts_entry_normalizes_before_dispatch(self, tmp_path):
        """The ASR dispatch gate sees the NORMALIZED name."""
        fake_weights = tmp_path / "model.safetensors"
        fake_weights.write_bytes(b"not a real checkpoint")

        _captured = []

        with patch(
            "ComfyUI_VibeVoice.modules.external_loader.is_asr_config_name",
            wraps=lambda name: (_captured.append(name), False)[1],
        ), patch(
            "ComfyUI_VibeVoice.modules.external_loader.load_external_vibevoice_asr_model",
        ):
            try:
                from ComfyUI_VibeVoice.modules.external_loader import (
                    load_external_vibevoice_model,
                )

                load_external_vibevoice_model(
                    weight_path=str(fake_weights),
                    config_name="VibeVoice-Large",
                )
            except Exception:
                # Downstream loading will fail on the fake file — the
                # assertion below only needs the dispatch-gate capture.
                pass

        assert _captured == ["VibeVoice-7B"]

    def test_asr_entry_normalizes_idempotently(self, tmp_path):
        """The ASR function's own normalization is a safe no-op on canonical
        names and maps aliases when called directly."""
        fake_weights = tmp_path / "asr.safetensors"
        fake_weights.write_bytes(b"not a real checkpoint")

        captured = []

        def _fake_resolve(weight_path, config_name):
            captured.append(config_name)
            raise RuntimeError("stop here")

        with patch(
            "ComfyUI_VibeVoice.modules.external_loader._load_weight_state_dict",
            return_value={},
        ), patch(
            "ComfyUI_VibeVoice.modules.external_loader.resolve_sidecar_config",
            side_effect=_fake_resolve,
        ):
            from ComfyUI_VibeVoice.modules.external_loader import (
                load_external_vibevoice_asr_model,
            )

            with pytest.raises(RuntimeError, match="stop here"):
                load_external_vibevoice_asr_model(
                    weight_path=str(fake_weights),
                    config_name="VibeVoice-ASR",
                )

        assert captured == ["VibeVoice-ASR"]


class TestAutoDetectOption:
    """Step 4.1: the Auto-detect sentinel option (plan 2026-08-27, D6)."""

    def test_options_content_and_order(self):
        assert EXTERNAL_CONFIG_OPTIONS == [
            "Auto-detect",
            "VibeVoice-1.5B",
            "VibeVoice-7B",
            "VibeVoice-Realtime-0.5B",
            "VibeVoice-ASR",
        ]

    def test_auto_constant_is_first_option(self):
        assert AUTO_CONFIG_NAME == "Auto-detect"
        assert EXTERNAL_CONFIG_OPTIONS[0] == AUTO_CONFIG_NAME

    def test_auto_is_not_asr(self):
        assert is_asr_config_name("Auto-detect") is False
        assert AUTO_CONFIG_NAME not in ASR_CONFIG_NAMES

    def test_normalize_passes_auto_through(self):
        assert normalize_config_name(AUTO_CONFIG_NAME) == AUTO_CONFIG_NAME

    def test_validate_inputs_accepts_auto(self):
        assert VibeVoiceExternalLoaderNode.validate_inputs(
            config_name=AUTO_CONFIG_NAME
        ) is True

    def test_schema_default_is_auto_detect(self):
        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node."
            "list_external_model_files",
            return_value=["fake.safetensors"],
        ):
            schema = VibeVoiceExternalLoaderNode.define_schema()
        cfg = next(i for i in schema.inputs if i.id == "config_name")
        assert cfg.default == "Auto-detect"
        assert cfg.options[0] == "Auto-detect"


class TestNodeAutoDetectResolution:
    """execute() resolves Auto-detect BEFORE the identity is computed.

    The consumer (generation.py) keys its patcher cache off the bundle's
    recorded model_name, so the node's request key must carry the same
    resolved name — otherwise the unload-before-load gate and patcher cache
    churn on every run (deviation from plan D7; see implementation notes).
    """

    def _execute(self, config_name, resolve_side_effect):
        fake_bundle = {"model": MagicMock()}
        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.load_external_vibevoice_model",
            return_value=fake_bundle,
        ) as mock_load, patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_full_path_or_raise",
            return_value="/fake/path/model.safetensors",
        ), patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.resolve_auto_config_name",
            side_effect=resolve_side_effect,
        ) as mock_resolve, patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.identity_for_external",
            return_value="external:key",
        ) as mock_identity:
            result = VibeVoiceExternalLoaderNode.execute(
                model_file="model.safetensors",
                config_name=config_name,
                attention_mode="sdpa",
                quantize_llm_4bit=False,
                dtype="auto",
            )
        return result, mock_load, mock_resolve, mock_identity

    def test_auto_resolved_before_identity(self):
        _, mock_load, mock_resolve, mock_identity = self._execute(
            "Auto-detect", resolve_side_effect=lambda wp: "VibeVoice-7B"
        )
        mock_resolve.assert_called_once_with("/fake/path/model.safetensors")
        # Identity and the loader both see the RESOLVED family name.
        assert mock_identity.call_args[0][1] == "VibeVoice-7B"
        assert mock_load.call_args[1]["config_name"] == "VibeVoice-7B"

    def test_explicit_selection_skips_detection(self):
        _, mock_load, mock_resolve, mock_identity = self._execute(
            "VibeVoice-1.5B", resolve_side_effect=lambda wp: "SHOULD-NOT-RUN"
        )
        mock_resolve.assert_not_called()
        assert mock_identity.call_args[0][1] == "VibeVoice-1.5B"
        assert mock_load.call_args[1]["config_name"] == "VibeVoice-1.5B"

    def test_auto_inconclusive_fails_fast_without_loading(self):
        fake_bundle = {"model": MagicMock()}
        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.load_external_vibevoice_model",
            return_value=fake_bundle,
        ) as mock_load, patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_full_path_or_raise",
            return_value="/fake/path/model.bin",
        ), patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.resolve_auto_config_name",
            side_effect=ValueError("could not determine architecture"),
        ):
            with pytest.raises(ValueError, match="could not determine"):
                VibeVoiceExternalLoaderNode.execute(
                    model_file="model.bin",
                    config_name="Auto-detect",
                    attention_mode="sdpa",
                    quantize_llm_4bit=False,
                    dtype="auto",
                )
        mock_load.assert_not_called()
