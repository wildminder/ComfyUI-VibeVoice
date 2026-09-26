"""Tests for importable, idempotent ComfyUI folder registration helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace

from ComfyUI_VibeVoice.modules.folder_registration import (
    register_vibevoice_folders,
    register_voice_preset_folder,
)


def _fake_folder_paths(models_dir: str) -> SimpleNamespace:
    return SimpleNamespace(
        models_dir=models_dir,
        supported_pt_extensions={".pt", ".bin", ".ckpt"},
        folder_names_and_paths={},
    )


class TestFolderRegistration:
    def test_registers_primary_tts_root_and_supported_extensions(self, tmp_path):
        folder_paths = _fake_folder_paths(str(tmp_path))

        paths = register_vibevoice_folders(folder_paths)

        expected = str(tmp_path / "tts")
        assert paths == [expected]
        assert folder_paths.folder_names_and_paths["tts"] == (
            [expected],
            {".pt", ".bin", ".ckpt", ".safetensors", ".json"},
        )

    def test_preserves_existing_tts_registrations(self, tmp_path):
        existing = str(tmp_path / "existing-tts")
        folder_paths = _fake_folder_paths(str(tmp_path))
        folder_paths.folder_names_and_paths["tts"] = (
            [existing],
            {".pt"},
        )

        paths = register_vibevoice_folders(folder_paths)

        assert paths == [existing, str(tmp_path / "tts")]

    def test_registration_is_idempotent(self, tmp_path):
        folder_paths = _fake_folder_paths(str(tmp_path))

        first = register_vibevoice_folders(folder_paths)
        second = register_vibevoice_folders(folder_paths)

        assert first == second
        assert len(second) == 1

    def test_registers_voice_preset_folder(self, tmp_path):
        folder_paths = _fake_folder_paths(str(tmp_path))
        tts_root = tmp_path / "tts"
        tts_root.mkdir()
        register_vibevoice_folders(folder_paths)

        paths = register_voice_preset_folder(folder_paths, str(tts_root))

        expected = str(tts_root / "VibeVoice" / "voices")
        assert paths == [expected]
        assert folder_paths.folder_names_and_paths["vibevoice_voices"] == (
            [expected],
            {".pt"},
        )

    def test_voice_registration_is_idempotent_and_preserves_roots(self, tmp_path):
        folder_paths = _fake_folder_paths(str(tmp_path))
        custom = str(tmp_path / "custom-voices")
        folder_paths.folder_names_and_paths["vibevoice_voices"] = (
            [custom],
            {".pt"},
        )

        first = register_voice_preset_folder(folder_paths, str(tmp_path / "tts"))
        second = register_voice_preset_folder(folder_paths, str(tmp_path / "tts"))

        assert first == second
        assert first[0] == custom
        assert first[1] == str(tmp_path / "tts" / "VibeVoice" / "voices")


class TestMultiRootVoiceRegistration:
    """Every TTS root needs a voices candidate, not just the first.

    ``extra_model_paths.yaml`` can add a TTS root that holds the prompts while
    the primary root has none — registering only ``roots[0]`` made the node
    report every preset as missing even though it was installed.
    """

    def test_all_tts_roots_get_a_voices_candidate(self, tmp_path):
        folder_paths = _fake_folder_paths(str(tmp_path / "models"))
        primary = str(tmp_path / "models" / "tts")
        extra = str(tmp_path / "C-drive" / "ComfyUI" / "models" / "tts")

        paths = register_voice_preset_folder(folder_paths, primary, [extra])

        assert paths == [
            str(tmp_path / "models" / "tts" / "VibeVoice" / "voices"),
            str(tmp_path / "C-drive" / "ComfyUI" / "models" / "tts" / "VibeVoice" / "voices"),
        ]

    def test_primary_comes_first_so_it_keeps_priority(self, tmp_path):
        folder_paths = _fake_folder_paths(str(tmp_path / "models"))
        primary = str(tmp_path / "models" / "tts")
        extra = str(tmp_path / "other" / "tts")

        paths = register_voice_preset_folder(folder_paths, primary, [extra])

        assert paths[0] == str(tmp_path / "models" / "tts" / "VibeVoice" / "voices")

    def test_nonexistent_roots_are_still_registered(self, tmp_path):
        # Discovery skips paths that do not exist, and a root may be populated
        # after startup, so registering them costs nothing.
        folder_paths = _fake_folder_paths(str(tmp_path / "models"))
        missing = str(tmp_path / "not-created-yet" / "tts")

        paths = register_voice_preset_folder(
            folder_paths, str(tmp_path / "models" / "tts"), [missing]
        )

        assert str(tmp_path / "not-created-yet" / "tts" / "VibeVoice" / "voices") in [
            os.path.normpath(p) for p in paths
        ]

    def test_duplicate_roots_are_collapsed(self, tmp_path):
        folder_paths = _fake_folder_paths(str(tmp_path / "models"))
        primary = str(tmp_path / "models" / "tts")

        paths = register_voice_preset_folder(
            folder_paths, primary, [primary, str(primary) + os.sep]
        )

        assert len(paths) == 1

    def test_no_additional_roots_still_registers_the_primary(self, tmp_path):
        folder_paths = _fake_folder_paths(str(tmp_path / "models"))
        paths = register_voice_preset_folder(
            folder_paths, str(tmp_path / "models" / "tts"), None
        )
        assert len(paths) == 1

    def test_existing_custom_entries_are_preserved(self, tmp_path):
        folder_paths = _fake_folder_paths(str(tmp_path / "models"))
        custom = str(tmp_path / "my-voices")
        folder_paths.folder_names_and_paths["vibevoice_voices"] = (
            [custom],
            {".pt"},
        )

        paths = register_voice_preset_folder(
            folder_paths, str(tmp_path / "models" / "tts"), []
        )

        assert custom in [os.path.normpath(p) for p in paths]
