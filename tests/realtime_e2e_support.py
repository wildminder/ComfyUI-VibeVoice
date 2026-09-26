"""Shared plumbing for the opt-in real-checkpoint numeric audit (plan S3.1-S3.3).

Three things both audit modules need and neither should own:

* the opt-in gate and the un-mocking of the ``src.vibevoice`` tree, so the
  genuine vendored classes are the ones under test (the root ``conftest.py``
  replaces that whole tree with ``MagicMock``);
* a reader for the checkpoint's own tensors, read straight off disk. This is
  the reference the audit compares against — deliberately NOT a second
  ``from_pretrained``: on transformers 5.3 that re-runs ``_init_weights`` over
  ``acoustic_connector`` and ``tts_eos_classifier`` (F4/t2), so it describes a
  random head, not this checkpoint;
* the two GPU measurements (first-window conditioning vector, first-latent EOS
  logit). They live in ``diag_realtime_quality.py`` — that is where the plan
  puts the S3.2/S3.3 experiments ("Change: none", a recorded measurement) — and
  are loaded here by path so the printed number and the asserted number cannot
  drift apart.

Opt-in contract, unchanged from ``tests/test_realtime_e2e_gpu.py``:

- ``RUN_VIBEVOICE_E2E=1``
- ``VIBEVOICE_REALTIME_MODEL_DIR`` -> local ``VibeVoice-Realtime-0.5B`` folder
- ``VIBEVOICE_REALTIME_VOICE_PRESET`` -> official ``.pt`` cached voice prompt

Reference command (Windows cmd.exe)::

    cmd.exe /d /c "set COMFYUI_ROOT=<ComfyUI root>&& set RUN_VIBEVOICE_E2E=1&& set VIBEVOICE_REALTIME_MODEL_DIR=<model dir>&& set VIBEVOICE_REALTIME_VOICE_PRESET=<preset.pt>&& <python> -m pytest tests\\test_realtime_load_health.py -s"
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DIAG_PATH = REPO_ROOT / "diag_realtime_quality.py"

OPT_IN_ENV = "RUN_VIBEVOICE_E2E"
MODEL_DIR_ENV = "VIBEVOICE_REALTIME_MODEL_DIR"
PRESET_ENV = "VIBEVOICE_REALTIME_VOICE_PRESET"

# The name the probes load under; registered test-locally so no development
# ComfyUI configuration has to be mutated (same approach as the e2e gpu test).
REALTIME_MODEL_NAME = "VibeVoice-Realtime-0.5B"

_MOCKED_PREFIXES = ("src.vibevoice", "ComfyUI_VibeVoice.src.vibevoice")
_REIMPORTED_MODULES = (
    "ComfyUI_VibeVoice.modules.loader",
    "ComfyUI_VibeVoice.modules.generation",
    "ComfyUI_VibeVoice.modules.utils",
    "ComfyUI_VibeVoice.nodes.tts_node",
)


# ------------------------------------------------------------------ gating --
def env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "y", "yes", "true"}


def require_opt_in() -> None:
    if not env_enabled(OPT_IN_ENV):
        pytest.skip(f"Set {OPT_IN_ENV}=1 to run the real-checkpoint numeric audit.")


def require_path_env(name: str) -> Path:
    raw = os.environ.get(name, "").strip()
    if not raw:
        pytest.fail(f"{name} is required when {OPT_IN_ENV}=1.")
    path = Path(raw).expanduser()
    if not path.exists():
        pytest.fail(f"{name} does not exist: {path}")
    return path


# ------------------------------------------------------- vendored un-mock --
@pytest.fixture(scope="module")
def real_vibevoice():
    """Import the genuine ``src.vibevoice`` tree and the loader that uses it.

    The root ``conftest.py`` replaces ``src.vibevoice`` with ``MagicMock``
    entries so the default suite stays light. The audit needs the real model
    class (to test ``from_pretrained``) and the real loader modules, so the
    mocked entries are dropped, the modules are re-imported from disk, and
    every original entry is restored on teardown.
    """
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name.startswith(_MOCKED_PREFIXES)
    }
    for name in saved:
        del sys.modules[name]
    saved_nodes = {
        name: sys.modules.pop(name, None) for name in _REIMPORTED_MODULES
    }
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        modules = {
            "loader": importlib.import_module("ComfyUI_VibeVoice.modules.loader"),
            "generation": importlib.import_module("ComfyUI_VibeVoice.modules.generation"),
            "utils": importlib.import_module("ComfyUI_VibeVoice.modules.utils"),
            "voice_presets": importlib.import_module(
                "ComfyUI_VibeVoice.modules.voice_presets"
            ),
            "model_info": importlib.import_module(
                "ComfyUI_VibeVoice.modules.model_info"
            ),
            "streaming": importlib.import_module(
                "ComfyUI_VibeVoice.src.vibevoice.modular.modeling_vibevoice_streaming"
            ),
            "sage_attention_patch": importlib.import_module(
                "ComfyUI_VibeVoice.src.vibevoice.modular.sage_attention_patch"
            ),
            "streaming_inference": importlib.import_module(
                "ComfyUI_VibeVoice.src.vibevoice.modular"
                ".modeling_vibevoice_streaming_inference"
            ),
        }
        yield modules
    finally:
        for name in [
            name
            for name in list(sys.modules)
            if name.startswith("src.vibevoice") or name.startswith("ComfyUI_VibeVoice.src")
        ]:
            del sys.modules[name]
        sys.modules.update(saved)
        for name, module in saved_nodes.items():
            if module is not None:
                sys.modules[name] = module


@pytest.fixture(scope="module")
def realtime_env():
    """Validate the opt-in environment and the checkpoint assets it names."""
    require_opt_in()
    model_dir = require_path_env(MODEL_DIR_ENV)
    preset_path = require_path_env(PRESET_ENV)

    missing = [
        name
        for name in ("config.json", "preprocessor_config.json", "tokenizer.json")
        if not (model_dir / name).is_file()
    ]
    if missing:
        pytest.skip(f"Realtime checkpoint is incomplete; missing {missing} in {model_dir}.")
    if not any(model_dir.glob("*.safetensors")) and not any(model_dir.glob("*.bin")):
        pytest.skip(f"No model weight files found in {model_dir}.")
    if preset_path.suffix.casefold() != ".pt":
        pytest.fail(f"{preset_path} must be an official cached .pt voice prompt.")
    return {"model_dir": model_dir, "preset_path": preset_path}


# ----------------------------------------------------- checkpoint as ground --
def checkpoint_tensors(model_dir: Path, names) -> dict[str, torch.Tensor]:
    """Read the named tensors straight out of the checkpoint files on disk.

    This is the ground truth for "did the load keep the checkpoint's values".
    Reading a second ``from_pretrained`` would not answer that question on
    transformers 5.3 (see the module docstring), so the tensors come from
    safetensors (or ``.bin``) and are compared in the loaded model's dtype.
    """
    model_dir = Path(model_dir)
    wanted = set(names)
    found: dict[str, torch.Tensor] = {}
    index = model_dir / "model.safetensors.index.json"
    single = model_dir / "model.safetensors"
    if index.is_file():
        import json

        shards = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        for name in wanted:
            shard = model_dir / shards[name]
            from safetensors import safe_open

            with safe_open(str(shard), framework="pt") as handle:
                found[name] = handle.get_tensor(name)
    elif single.is_file():
        from safetensors import safe_open

        with safe_open(str(single), framework="pt") as handle:
            available = set(handle.keys())
            missing = wanted - available
            if missing:
                pytest.fail(
                    f"checkpoint {single} does not contain {sorted(missing)}; "
                    f"the audit's reference list is stale."
                )
            for name in wanted:
                found[name] = handle.get_tensor(name)
    else:
        pytest.skip(f"No model.safetensors[/index.json] in {model_dir}.")
    return found


# --------------------------------------------------------- GPU measurements --
@pytest.fixture(scope="module")
def diag():
    """The diagnostic module, loaded from disk.

    S3.2/S3.3 measure through ``diag_realtime_quality``'s node-path probes so
    the test asserts the number the diagnostic prints, not a reimplementation
    of it. Loading by path keeps ``import diag_realtime_quality`` out of the
    default suite's collection.
    """
    spec = importlib.util.spec_from_file_location("diag_realtime_quality", DIAG_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def node_model(real_vibevoice, realtime_env, diag, monkeypatch):
    """Factory loading the realtime checkpoint through the NODE load path.

    Yields ``load(attention_mode, dtype_str) -> (model, processor, preset)``.
    Model and preset are registered test-locally, so a run never depends on the
    machine's ComfyUI folder configuration, and every load is released when the
    test ends (the loader and patcher both cache by name).
    """
    model_info = real_vibevoice["model_info"]
    voice_presets = real_vibevoice["voice_presets"]
    monkeypatch.setitem(
        model_info.AVAILABLE_VIBEVOICE_MODELS,
        REALTIME_MODEL_NAME,
        {
            "type": "local_dir",
            "path": str(realtime_env["model_dir"]),
            "tokenizer_repo": "Qwen/Qwen2.5-1.5B",
        },
    )
    monkeypatch.setattr(
        voice_presets,
        "list_voice_presets",
        lambda: {realtime_env["preset_path"].stem: str(realtime_env["preset_path"])},
    )
    monkeypatch.setattr(diag, "REALTIME", REALTIME_MODEL_NAME, raising=False)
    monkeypatch.setattr(
        diag, "VOICE", realtime_env["preset_path"].stem, raising=False
    )

    loaded: list = []

    def load(attention_mode: str = "sdpa", dtype_str: str = "auto"):
        model, processor = diag.load_node_model(
            attention_mode=attention_mode, dtype_str=dtype_str
        )
        preset = diag.load_voice_preset_for_probe()
        loaded.append(model)
        return model, processor, preset

    yield load

    # release_node_model drops every patcher-cache entry for this model name and
    # returns the VRAM, so a test that forgot to release still cleans up.
    for model in loaded:
        diag.release_node_model(model)


# ------------------------------------------------------------- assertions --
def named_nonfinite(named_tensors, kind: str) -> list[str]:
    """Names of the given tensors holding a NaN or an infinity."""
    offenders = []
    for name, tensor in named_tensors:
        if tensor is None or not torch.is_tensor(tensor):
            continue
        if not tensor.is_floating_point():
            continue
        if not bool(torch.isfinite(tensor).all()):
            offenders.append(f"{kind}:{name}")
    return offenders


def rotary_modules(model) -> dict[str, torch.nn.Module]:
    """Every module carrying a rotary ``inv_freq`` buffer, by module name."""
    return {
        name: module
        for name, module in model.named_modules()
        if torch.is_tensor(getattr(module, "inv_freq", None))
    }
