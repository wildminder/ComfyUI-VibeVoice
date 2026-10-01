"""CPU tests for realtime voice-prompt discovery, validation, and caching."""

from __future__ import annotations

import logging
import os
import pickle
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from ComfyUI_VibeVoice.modules import voice_presets
from ComfyUI_VibeVoice.modules.voice_presets import (
    PRESET_CACHE_KEYS,
    clear_voice_preset_cache,
    get_cached_voice_preset,
    list_voice_presets,
    load_voice_preset,
    resolve_voice_preset_path,
    validate_voice_preset,
    voice_preset_search_dirs,
)


class _UnapprovedPayload:
    """Serializable class intentionally absent from the production allowlist."""


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_voice_preset_cache()
    yield
    clear_voice_preset_cache()


def _branch(length: int = 3) -> BaseModelOutputWithPast:
    return BaseModelOutputWithPast(
        last_hidden_state=torch.arange(length * 4, dtype=torch.float32).reshape(1, length, 4),
        # A real DynamicCache instance without lazily-created DynamicLayer state.
        # This exercises the official safe global without widening the
        # production allowlist beyond the two prescribed classes.
        past_key_values=DynamicCache.__new__(DynamicCache),
    )


def _valid_preset() -> dict[str, BaseModelOutputWithPast]:
    return {
        key: _branch(2 + index)
        for index, key in enumerate(PRESET_CACHE_KEYS)
    }


def _write_preset(path, preset=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(preset if preset is not None else _valid_preset(), path)
    return path


class TestVoicePresetSearchDirs:
    def test_explicit_voice_roots_precede_default_tts_roots(self, tmp_path):
        explicit = str(tmp_path / "explicit")
        default = str(tmp_path / "tts" / "VibeVoice" / "voices")
        fake = SimpleNamespace(
            folder_names_and_paths={
                "vibevoice_voices": ([str(tmp_path / "explicit")], {".pt"})
            },
            get_folder_paths=lambda key: [str(tmp_path / "tts")],
        )
        with patch.object(voice_presets, "folder_paths", fake):
            assert voice_preset_search_dirs() == [explicit, default]

    def test_duplicate_roots_are_deduplicated_case_insensitively(self, tmp_path):
        root = tmp_path / "tts" / "VibeVoice" / "voices"
        fake = SimpleNamespace(
            folder_names_and_paths={"vibevoice_voices": ([str(root)], {".pt"})},
            get_folder_paths=lambda key: [str(tmp_path / "tts")],
        )
        with patch.object(voice_presets, "folder_paths", fake):
            assert voice_preset_search_dirs() == [str(root)]

    def test_schema_safe_when_folder_registry_fails(self):
        fake = SimpleNamespace(
            folder_names_and_paths=property(lambda _self: (_ for _ in ()).throw(RuntimeError())),
            get_folder_paths=lambda key: [],
        )
        with patch.object(voice_presets, "folder_paths", fake):
            assert voice_preset_search_dirs() == []


class TestVoicePresetDiscovery:
    def test_recursive_case_insensitive_discovery_and_ordering(self, tmp_path):
        root = tmp_path / "voices"
        _write_preset(root / "nested" / "zeta.PT")
        _write_preset(root / "alpha.pt")
        (root / "ignored.bin").write_bytes(b"")

        result = list_voice_presets([str(root)])

        assert list(result) == ["alpha", "zeta"]
        assert result["alpha"] == str(root / "alpha.pt")
        assert result["zeta"] == str(root / "nested" / "zeta.PT")

    def test_missing_directory_returns_empty_mapping(self, tmp_path):
        assert list_voice_presets([str(tmp_path / "missing")]) == {}

    def test_first_registration_wins_collision_and_warns_both_paths(self, tmp_path, caplog):
        first = _write_preset(tmp_path / "first" / "Voice.pt")
        second = _write_preset(tmp_path / "second" / "voice.pt")
        with caplog.at_level(logging.WARNING):
            result = list_voice_presets([str(first.parent), str(second.parent)])

        assert result == {"Voice": str(first)}
        warning = "\n".join(record.message for record in caplog.records)
        assert str(first) in warning
        assert str(second) in warning

    def test_resolve_is_case_insensitive(self, tmp_path):
        preset_path = _write_preset(tmp_path / "en-Carter_man.pt")
        with patch.object(voice_presets, "voice_preset_search_dirs", return_value=[str(tmp_path)]):
            assert resolve_voice_preset_path("EN-CARTER_MAN") == str(preset_path)

    def test_missing_name_lists_all_search_roots(self, tmp_path):
        roots = [str(tmp_path / "one"), str(tmp_path / "two")]
        with patch.object(voice_presets, "voice_preset_search_dirs", return_value=roots):
            with pytest.raises(FileNotFoundError) as excinfo:
                resolve_voice_preset_path("missing")
        message = str(excinfo.value)
        assert os.path.basename(roots[0]) in message
        assert os.path.basename(roots[1]) in message


class TestVoicePresetValidation:
    def test_valid_real_outputs(self):
        validate_voice_preset(_valid_preset(), "valid.pt")

    @pytest.mark.parametrize("missing_key", PRESET_CACHE_KEYS)
    def test_missing_key_is_named(self, missing_key):
        preset = _valid_preset()
        del preset[missing_key]
        with pytest.raises(ValueError, match=missing_key):
            validate_voice_preset(preset, "broken.pt")

    def test_plain_mapping_branch_is_rejected(self):
        preset = _valid_preset()
        preset["lm"] = {"last_hidden_state": torch.ones(1, 2, 3)}
        with pytest.raises(ValueError, match="key 'lm'.*plain mapping"):
            validate_voice_preset(preset, "broken.pt")

    def test_missing_last_hidden_state_is_named(self):
        preset = _valid_preset()
        preset["tts_lm"] = SimpleNamespace()
        with pytest.raises(ValueError, match="key 'tts_lm'.*last_hidden_state"):
            validate_voice_preset(preset, "broken.pt")

    @pytest.mark.parametrize("shape", [(0, 2, 3), (1, 0, 3), (3,)])
    def test_empty_or_invalid_sequence_dimensions_are_rejected(self, shape):
        preset = _valid_preset()
        preset["neg_lm"] = SimpleNamespace(last_hidden_state=torch.empty(shape))
        with pytest.raises(ValueError, match="key 'neg_lm'.*last_hidden_state"):
            validate_voice_preset(preset, "broken.pt")

    def test_non_mapping_payload_is_rejected(self):
        with pytest.raises(ValueError, match="expected a mapping"):
            validate_voice_preset([], "broken.pt")


class TestVoicePresetLoadingAndCache:
    def test_real_cpu_base_outputs_round_trip(self, tmp_path):
        source = _valid_preset()
        path = _write_preset(tmp_path / "official-shaped.pt", source)

        loaded = load_voice_preset(str(path), torch.device("cpu"))

        assert set(loaded) == set(PRESET_CACHE_KEYS)
        for key in PRESET_CACHE_KEYS:
            assert isinstance(loaded[key], BaseModelOutputWithPast)
            assert torch.equal(loaded[key].last_hidden_state, source[key].last_hidden_state)

    def test_torch_load_uses_public_weights_only_api_with_safe_globals(self, tmp_path):
        path = _write_preset(tmp_path / "voice.pt")
        loaded = _valid_preset()
        real_safe_globals = torch.serialization.safe_globals

        @contextmanager
        def recording_safe_globals(values):
            with real_safe_globals(values):
                yield

        with patch.object(
            voice_presets.torch,
            "load",
            return_value=loaded,
        ) as mock_load, patch.object(
            voice_presets.torch.serialization,
            "safe_globals",
            side_effect=recording_safe_globals,
        ) as mock_safe_globals:
            result = load_voice_preset(str(path), torch.device("cpu"))

        mock_safe_globals.assert_called_once_with(
            list(voice_presets.PRESET_SAFE_GLOBALS)
        )
        mock_load.assert_called_once_with(
            str(path),
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        assert result is loaded

    def test_safe_globals_allowlist_matches_official_contract(self):
        assert voice_presets.PRESET_SAFE_GLOBALS == (
            BaseModelOutputWithPast,
            DynamicCache,
        )

    def test_non_allowlisted_global_is_rejected_by_weights_only_loader(self, tmp_path):
        path = tmp_path / "unapproved.pt"
        torch.save(_UnapprovedPayload(), path)
        with pytest.raises(pickle.UnpicklingError, match="not an allowed global"):
            load_voice_preset(str(path), torch.device("cpu"))

    def test_structurally_invalid_branch_is_rejected_after_load(self, tmp_path):
        payload = _valid_preset()
        payload["lm"] = torch.arange(4)
        path = tmp_path / "broken-branch.pt"
        torch.save(payload, path)
        with pytest.raises(ValueError, match="last_hidden_state"):
            load_voice_preset(str(path), torch.device("cpu"))

    def test_cache_hit_returns_same_object_without_reload(self, tmp_path):
        _write_preset(tmp_path / "voice.pt")
        loaded = _valid_preset()
        with patch.object(voice_presets, "voice_preset_search_dirs", return_value=[str(tmp_path)]), patch.object(
            voice_presets, "load_voice_preset", return_value=loaded
        ) as mock_load:
            first = get_cached_voice_preset("voice", torch.device("cpu"))
            second = get_cached_voice_preset("VOICE", torch.device("cpu"))
        assert second is first
        mock_load.assert_called_once()

    def test_mtime_invalidation(self, tmp_path):
        path = _write_preset(tmp_path / "voice.pt")
        with patch.object(voice_presets, "voice_preset_search_dirs", return_value=[str(tmp_path)]), patch.object(
            voice_presets, "load_voice_preset", side_effect=lambda *_args, **_kwargs: _valid_preset()
        ) as mock_load:
            first = get_cached_voice_preset("voice", torch.device("cpu"))
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            second = get_cached_voice_preset("voice", torch.device("cpu"))
        assert second is not first
        assert mock_load.call_count == 2

    def test_size_invalidation(self, tmp_path):
        path = _write_preset(tmp_path / "voice.pt")
        with patch.object(voice_presets, "voice_preset_search_dirs", return_value=[str(tmp_path)]), patch.object(
            voice_presets, "load_voice_preset", side_effect=lambda *_args, **_kwargs: _valid_preset()
        ) as mock_load:
            first = get_cached_voice_preset("voice", torch.device("cpu"))
            path.write_bytes(path.read_bytes() + b"changed")
            second = get_cached_voice_preset("voice", torch.device("cpu"))
        assert second is not first
        assert mock_load.call_count == 2

    def test_device_invalidation(self, tmp_path):
        _write_preset(tmp_path / "voice.pt")
        with patch.object(voice_presets, "voice_preset_search_dirs", return_value=[str(tmp_path)]), patch.object(
            voice_presets, "load_voice_preset", side_effect=lambda *_args, **_kwargs: _valid_preset()
        ) as mock_load:
            first = get_cached_voice_preset("voice", torch.device("cpu"))
            second = get_cached_voice_preset("voice", torch.device("meta"))
        assert first is not second
        assert mock_load.call_count == 2

    def test_explicit_clear_forces_reload(self, tmp_path):
        _write_preset(tmp_path / "voice.pt")
        with patch.object(voice_presets, "voice_preset_search_dirs", return_value=[str(tmp_path)]), patch.object(
            voice_presets, "load_voice_preset", side_effect=lambda *_args, **_kwargs: _valid_preset()
        ) as mock_load:
            first = get_cached_voice_preset("voice", torch.device("cpu"))
            clear_voice_preset_cache()
            second = get_cached_voice_preset("voice", torch.device("cpu"))
        assert second is not first
        assert mock_load.call_count == 2

    def test_cached_object_is_returned_without_copy_or_mutation(self, tmp_path):
        _write_preset(tmp_path / "voice.pt")
        loaded = _valid_preset()
        with patch.object(voice_presets, "voice_preset_search_dirs", return_value=[str(tmp_path)]), patch.object(
            voice_presets, "load_voice_preset", return_value=loaded
        ):
            preset = get_cached_voice_preset("voice", torch.device("cpu"))
            before = {
                key: value.last_hidden_state.clone() for key, value in preset.items()
            }
            get_cached_voice_preset("voice", torch.device("cpu"))
        for key, hidden_state in before.items():
            assert torch.equal(preset[key].last_hidden_state, hidden_state)
