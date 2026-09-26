"""Contract tests for the e2e smoke script's CLI and realtime branch wiring."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parent.parent / "e2e_smoke_test.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("e2e_smoke_contract", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def smoke_script():
    return _load_script()


def test_help_does_not_import_realtime_modules(smoke_script, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["e2e_smoke_test.py", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        smoke_script.main()
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--realtime-model" in out
    assert "--voice-preset" in out
    assert "generate_realtime_audio" not in out


def test_realtime_requires_both_arguments(smoke_script, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["e2e_smoke_test.py", "--realtime-model", "VibeVoice-Realtime-0.5B"],
    )
    with pytest.raises(SystemExit) as excinfo:
        smoke_script.main()
    assert excinfo.value.code == 2
    assert "--voice-preset" in capsys.readouterr().err

    monkeypatch.setattr(
        sys,
        "argv",
        ["e2e_smoke_test.py", "--voice-preset", "C:/voices/en-Carter_man.pt"],
    )
    with pytest.raises(SystemExit) as excinfo:
        smoke_script.main()
    assert excinfo.value.code == 2
    assert "--realtime-model" in capsys.readouterr().err


def test_realtime_branch_rejects_missing_preset_file(smoke_script, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "e2e_smoke_test.py",
            "--realtime-model",
            "VibeVoice-Realtime-0.5B",
            "--voice-preset",
            str(tmp_path / "missing.pt"),
        ],
    )
    assert smoke_script.main() == 2
    assert "voice preset not found" in capsys.readouterr().err


def test_realtime_branch_uses_cached_prompt_adapter(smoke_script, monkeypatch, capsys, tmp_path):
    import torch

    preset_path = tmp_path / "en-Carter_man.pt"
    preset_path.write_bytes(b"stub")
    same_stem_elsewhere = tmp_path / "other" / "en-Carter_man.pt"
    same_stem_elsewhere.parent.mkdir()
    same_stem_elsewhere.write_bytes(b"other stub")
    calls = {}

    def _fake_load(**kwargs):
        calls["load"] = kwargs
        return object(), object(), object()

    def _fake_realtime(**kwargs):
        calls["realtime"] = kwargs
        return torch.full((1, 1, 2400), 0.25), 24000

    def _fake_preset(path, device):
        calls["preset"] = (path, device)
        return {"stub": True}

    fake_mm = type("MM", (), {"get_torch_device": staticmethod(lambda: torch.device("cpu"))})
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake_mm)
    monkeypatch.setitem(
        sys.modules,
        "ComfyUI_VibeVoice.modules.generation",
        type("Gen", (), {"load_vibevoice_model": staticmethod(_fake_load)}),
    )
    monkeypatch.setitem(
        sys.modules,
        "ComfyUI_VibeVoice.modules.realtime_generation",
        type("RT", (), {"generate_realtime_audio": staticmethod(_fake_realtime)}),
    )
    monkeypatch.setitem(
        sys.modules,
        "ComfyUI_VibeVoice.modules.voice_presets",
        type("VP", (), {"load_voice_preset": staticmethod(_fake_preset)}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "e2e_smoke_test.py",
            "--realtime-model",
            "VibeVoice-Realtime-0.5B",
            "--voice-preset",
            str(preset_path),
            "--outdir",
            str(tmp_path / "out"),
            "--inference_steps",
            "7",
        ],
    )

    assert smoke_script.main() == 0
    assert calls["load"]["model_name"] == "VibeVoice-Realtime-0.5B"
    assert calls["preset"][0] == str(preset_path.resolve())
    assert calls["preset"][0] != str(same_stem_elsewhere.resolve())
    assert calls["realtime"]["diffusion_steps"] == 7
    assert calls["realtime"]["max_new_tokens"] is None
    assert (tmp_path / "out" / "e2e_realtime.wav").is_file()
    assert "Wrote wav" in capsys.readouterr().out


def test_realtime_imports_stay_inside_the_realtime_branch():
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "def _run_realtime(args)" in text
    branch = text.split("def _run_realtime(args)", 1)[1]
    branch = branch.split("\ndef ", 1)[0]
    for symbol in (
        "realtime_generation",
        "load_voice_preset",
        "generate_realtime_audio",
    ):
        assert symbol in branch
    # The standard branch must not import the realtime adapter at module scope.
    head = text.split("def _run_realtime(args)", 1)[0]
    assert "import generate_realtime_audio" not in head
    assert "from ComfyUI_VibeVoice.modules.realtime_generation" not in head


def test_script_never_advises_realtime_as_standard_fallback():
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "fallback is Option B" not in text
    assert "never as a standard fallback" in text
