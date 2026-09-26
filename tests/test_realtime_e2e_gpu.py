"""Opt-in real-checkpoint acceptance tests for VibeVoice-Realtime-0.5B.

These tests are skipped by default. They require an explicit opt-in plus
absolute model and voice-prompt paths supplied through environment variables so
no machine-specific path is embedded in the repository:

- ``RUN_VIBEVOICE_E2E=1``
- ``VIBEVOICE_REALTIME_MODEL_DIR`` -> local ``VibeVoice-Realtime-0.5B`` folder
- ``VIBEVOICE_REALTIME_VOICE_PRESET`` -> official ``.pt`` cached voice prompt
- ``VIBEVOICE_REALTIME_VOICE_PRESET_ALT`` (optional) -> a second official ``.pt``
  prompt for the two-voices-condition-differently test. When unset, any other
  ``.pt`` next to the configured one is used; when neither exists the test is
  skipped with an explicit reason.
- ``VIBEVOICE_STANDARD_MODEL_DIR`` (optional) -> real standard checkpoint directory
  used by the standard forced-offload acceptance test; without it that test is
  skipped with an explicit reason

The realtime test additionally requires CUDA. Model and voice-prompt folders
are registered test-locally so the development ComfyUI configuration is never
mutated.

Reference command (Windows cmd.exe)::

    cmd.exe /d /c "set COMFYUI_ROOT=<ComfyUI root>&& set RUN_VIBEVOICE_E2E=1&& set VIBEVOICE_REALTIME_MODEL_DIR=<model dir>&& set VIBEVOICE_REALTIME_VOICE_PRESET=<preset.pt>&& <python> -m pytest tests\\test_realtime_e2e_gpu.py -s"
"""

from __future__ import annotations

import copy
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from ComfyUI_VibeVoice.modules.voice_presets import (
    clear_voice_preset_cache,
    load_voice_preset,
    validate_voice_preset,
)
from ComfyUI_VibeVoice.modules.realtime_generation import (
    REALTIME_MAX_AUTO_BUDGET_UNITS,
    generate_realtime_audio,
)

TEST_MODEL_NAME = "E2E-VibeVoice-Realtime-0.5B"
TEST_STANDARD_MODEL_NAME = "E2E-VibeVoice-Standard"
TEST_SCRIPT = "This is an acceptance test for the VibeVoice realtime model."
REPO_ROOT = Path(__file__).parent.parent


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "y", "yes", "true"}


def _require_opt_in() -> None:
    if not _env_enabled("RUN_VIBEVOICE_E2E"):
        pytest.skip("Set RUN_VIBEVOICE_E2E=1 to run the real-checkpoint acceptance tests.")


def _require_path_env(name: str) -> Path:
    raw = os.environ.get(name, "").strip()
    if not raw:
        pytest.fail(f"{name} is required when RUN_VIBEVOICE_E2E=1.")
    path = Path(raw).expanduser()
    if not path.exists():
        pytest.fail(f"{name} does not exist: {path}")
    return path


@pytest.fixture(scope="module")
def realtime_env():
    """Validate the opt-in environment and required checkpoint assets."""
    _require_opt_in()
    model_dir = _require_path_env("VIBEVOICE_REALTIME_MODEL_DIR")
    preset_path = _require_path_env("VIBEVOICE_REALTIME_VOICE_PRESET")

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


@pytest.fixture(scope="module")
def real_vendored_modules():
    """Load the genuine vendored VibeVoice modules for this module only.

    ``conftest.py`` replaces the whole ``src.vibevoice`` tree with MagicMocks so
    the default suite stays lightweight. The real-checkpoint tests need the real
    configuration, model, and processor classes, so the mocked entries are
    temporarily removed, the loader is re-imported against the on-disk sources,
    and every original entry is restored on teardown.
    """
    import importlib

    mocked_prefixes = ("src.vibevoice", "ComfyUI_VibeVoice.src.vibevoice")
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name.startswith(mocked_prefixes)
    }
    for name in saved:
        del sys.modules[name]

    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    saved_submodules = {
        dotted: sys.modules.pop(dotted, None)
        for dotted in (
            "ComfyUI_VibeVoice.modules.loader",
            "ComfyUI_VibeVoice.modules.generation",
            "ComfyUI_VibeVoice.nodes.tts_node",
        )
    }
    try:
        yield {
            "loader": importlib.import_module("ComfyUI_VibeVoice.modules.loader"),
            "generation": importlib.import_module("ComfyUI_VibeVoice.modules.generation"),
        }
    finally:
        for name in [
            name
            for name in list(sys.modules)
            if name.startswith("src.vibevoice") or name.startswith("ComfyUI_VibeVoice.src")
        ]:
            del sys.modules[name]
        sys.modules.update(saved)
        _restore_submodule_attributes(saved_submodules)


def _restore_submodule_attributes(saved: dict[str, object]) -> None:
    """Put every re-imported submodule back on its parent package as well.

    ``sys.modules`` is not the only place a submodule is registered: importing
    ``ComfyUI_VibeVoice.nodes.tts_node`` also *rebinds the attribute* ``tts_node``
    on the already-imported ``ComfyUI_VibeVoice.nodes`` package. ``import a.b as
    m`` resolves that attribute first, so restoring only ``sys.modules`` leaves
    the freshly-built module object reachable and the second instance leaks into
    every later test — e.g.
    ``tests/test_unified_tts_node.py::test_external_model_default_is_a_sentinel_not_none``
    compares the sentinel of one instance against the default of another, and the
    opt-in GPU run reports a failure that reproduces in no other order.
    """
    for dotted, module in saved.items():
        parent_name, _, leaf = dotted.rpartition(".")
        parent = sys.modules.get(parent_name)
        if module is None:
            sys.modules.pop(dotted, None)
            if parent is not None and getattr(parent, leaf, None) is not None:
                delattr(parent, leaf)
        else:
            sys.modules[dotted] = module
            if parent is not None:
                setattr(parent, leaf, module)


@pytest.fixture
def realtime_assets(realtime_env, real_vendored_modules, monkeypatch):
    """Register the model/preset test-locally and yield loaded GPU objects."""
    import folder_paths

    from ComfyUI_VibeVoice.modules.folder_registration import VOICE_PRESET_FOLDER_KEY
    from ComfyUI_VibeVoice.modules.model_info import AVAILABLE_VIBEVOICE_MODELS
    from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_PATCHER_CACHE

    generation = real_vendored_modules["generation"]
    model_dir = realtime_env["model_dir"]
    preset_path = realtime_env["preset_path"]

    monkeypatch.setitem(
        AVAILABLE_VIBEVOICE_MODELS,
        TEST_MODEL_NAME,
        {
            "type": "local_dir",
            "path": str(model_dir),
            "tokenizer_repo": "Qwen/Qwen2.5-1.5B",
        },
    )
    monkeypatch.setitem(
        folder_paths.folder_names_and_paths,
        VOICE_PRESET_FOLDER_KEY,
        ([str(preset_path.parent)], {".pt"}),
    )
    clear_voice_preset_cache()

    cache_key = f"{TEST_MODEL_NAME}_attn_sdpa_q4_0"
    patcher = model = processor = None
    try:
        patcher, model, processor = generation.load_vibevoice_model(
            model_name=TEST_MODEL_NAME,
            device="cuda",
            dtype="auto",
            attention_mode="sdpa",
            quantize_4bit=False,
        )
        device = torch.device("cuda")
        preset = load_voice_preset(str(preset_path), device)
        validate_voice_preset(preset, str(preset_path))
        yield {
            "patcher": patcher,
            "model": model,
            "processor": processor,
            "preset": preset,
            "device": device,
            "preset_path": preset_path,
        }
    finally:
        clear_voice_preset_cache()
        VIBEVOICE_PATCHER_CACHE.pop(cache_key, None)
        AVAILABLE_VIBEVOICE_MODELS.pop(TEST_MODEL_NAME, None)
        del patcher, model, processor


def cap_of(observed_generated: int) -> int:
    """Return a length cap safely below half the observed generated length."""
    return max(1, observed_generated // 2 - 1)


def _reference_tone() -> torch.Tensor:
    """A short, non-silent reference waveform for the standard-model test.

    ``audio_utils`` drops completely silent inputs, which would make the
    generated-audio RMS assertion vacuous. A 220 Hz tone at speech-like
    amplitude passes the silence check and is a real reference voice.
    """
    sample_rate = 24000
    t = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    return (0.2 * torch.sin(2 * torch.pi * 220.0 * t)).reshape(1, -1)


@contextmanager
def _comfyui_preview_stub():
    """Stub only ``ui.PreviewAudio`` for direct ``execute()`` calls.

    ``VibeVoiceTTSNode.execute`` builds ``ui.PreviewAudio(output, cls=cls)``,
    and the real helper reads ``cls.hidden.prompt``. Under pytest there is no
    ComfyUI hidden execution context, so ``cls.hidden`` is None and the real
    helper raises ``'NoneType' object has no attribute 'prompt'`` *after*
    generation has already succeeded. Only the preview is stubbed here —
    generation, patching and force-offload all stay real.
    """
    with patch(
        "ComfyUI_VibeVoice.nodes.tts_node.ui.PreviewAudio",
        return_value=MagicMock(),
    ):
        yield


def _assert_waveform(waveform: torch.Tensor, sample_rate: int) -> None:
    assert waveform.device.type == "cpu"
    assert waveform.dtype == torch.float32
    assert waveform.dim() == 3 and waveform.shape[:2] == (1, 1)
    assert waveform.shape[2] > 0
    assert sample_rate == 24000
    assert torch.isfinite(waveform).all()
    assert float(waveform.float().pow(2).mean().sqrt()) > 1e-5


def _install_generate_probe(model) -> dict:
    """Wrap the bound ``generate`` to capture the raw generation output."""
    captured: dict = {}
    original = model.generate

    def _probe(**kwargs):
        output = original(**kwargs)
        captured["output"] = output
        captured["max_new_tokens"] = kwargs.get("max_new_tokens")
        tts_lm_input_ids = kwargs.get("tts_lm_input_ids")
        captured["tts_lm_input_length"] = (
            int(tts_lm_input_ids.shape[1]) if tts_lm_input_ids is not None else None
        )
        return output

    model.generate = _probe
    captured["restore"] = lambda: setattr(model, "generate", original)
    return captured


def test_real_voice_prompt_loads_with_installed_transformers(realtime_env):
    """The official .pt prompt deserializes safely on the installed API."""
    preset = load_voice_preset(str(realtime_env["preset_path"]), torch.device("cpu"))
    validate_voice_preset(preset, str(realtime_env["preset_path"]))
    for key in ("lm", "tts_lm", "neg_lm", "neg_tts_lm"):
        assert preset[key].last_hidden_state.shape[0] == 1
        assert preset[key].last_hidden_state.shape[1] > 0


def _processor_inputs(processor, preset, text: str = TEST_SCRIPT) -> dict:
    """The real processor inputs for one script against a cached prompt."""
    inputs = processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=preset,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    return {
        key: (value.to("cuda") if torch.is_tensor(value) else value)
        for key, value in inputs.items()
    }


def _empty_cached_prompt(preset) -> dict:
    """The same four prefill slots, holding no voice at all.

    This is the control for the conditioning test: whatever the TTS-LM produces
    here is what it produces when the cached voice prompt contributes nothing.
    """
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    return {
        key: BaseModelOutputWithPast(
            last_hidden_state=torch.zeros(
                1,
                1,
                preset[key]["last_hidden_state"].shape[-1],
                dtype=preset[key]["last_hidden_state"].dtype,
                device="cuda",
            ),
            past_key_values=DynamicCache(),
        )
        for key in preset
    }


def _first_window_conditioning(model, inputs, source) -> dict:
    """One pass of the vendored loop, returning the conditioning and cache lens.

    Mirrors what ``generate()`` issues for the first text window: the new
    window through the base LM, then through the TTS-LM with the LM state
    spliced in. The returned ``condition`` is the tensor
    ``sample_speech_tokens`` consumes, so two caches are compared on exactly the
    quantity that decides the audio.
    """
    from ComfyUI_VibeVoice.src.vibevoice.modular.modeling_vibevoice_streaming_inference import (
        TTS_TEXT_WINDOW_SIZE,
        _ensure_cache_has_layers,
    )

    tts_cache = _ensure_cache_has_layers(
        copy.deepcopy(source["tts_lm"]).past_key_values
    )
    lm_cache = _ensure_cache_has_layers(copy.deepcopy(source["lm"]).past_key_values)

    def _cache_len(cache) -> int:
        layers = getattr(cache, "layers", None) or []
        return int(layers[0].get_seq_length()) if layers else 0

    tts_cache_len = _cache_len(tts_cache)
    lm_cache_len = _cache_len(lm_cache)
    new_ids = inputs["tts_text_ids"][:, :TTS_TEXT_WINDOW_SIZE]
    assert new_ids.shape[1] > 0, "first text window is empty; nothing to compare"

    def _step_kwargs(cache_len: int) -> dict:
        cache_position = torch.arange(
            cache_len, cache_len + new_ids.shape[1], device="cuda", dtype=torch.long
        )
        return {
            "attention_mask": torch.ones(
                1, cache_len + new_ids.shape[1], dtype=torch.long, device="cuda"
            ),
            "position_ids": cache_position.unsqueeze(0),
            "cache_position": cache_position,
        }

    with torch.no_grad():
        lm_result = model.forward_lm(
            input_ids=new_ids,
            past_key_values=lm_cache,
            use_cache=True,
            return_dict=True,
            **_step_kwargs(lm_cache_len),
        )
        result = model.forward_tts_lm(
            input_ids=new_ids,
            past_key_values=tts_cache,
            tts_text_masks=torch.ones_like(new_ids[:, -1:]),
            lm_last_hidden_state=lm_result.last_hidden_state,
            use_cache=True,
            return_dict=True,
            **_step_kwargs(tts_cache_len),
        )
    return {
        "condition": result.last_hidden_state[0, -1, :].detach().float().cpu(),
        "tts_cache_len": tts_cache_len,
        "lm_cache_len": lm_cache_len,
    }


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    )


def _second_voice_preset_path(preset_path: Path) -> Path | None:
    """Another official prompt from the same folder, if the env names one.

    ``VIBEVOICE_REALTIME_VOICE_PRESET_ALT`` wins; otherwise any other ``.pt``
    next to the configured one is used, so no machine-specific path is embedded
    in the repository.
    """
    explicit = os.environ.get("VIBEVOICE_REALTIME_VOICE_PRESET_ALT", "").strip()
    if explicit:
        candidate = Path(explicit).expanduser()
        return candidate if candidate.is_file() else None
    for candidate in sorted(preset_path.parent.glob("*.pt")):
        if candidate != preset_path:
            return candidate
    return None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Realtime E2E requires CUDA.")
def test_realtime_cache_conditions_on_the_voice_prefill(realtime_assets):
    """S2.4: the shimmed cache must really carry the voice into the TTS-LM.

    Three claims, all on real weights:

    1. the prefilled cache and an empty one condition the model differently
       (F3 measured cos = 0.667; anything at or above 0.99 would mean the
       prefill is being read but is irrelevant, i.e. effectively invisible);
    2. ``cache_len == tts_lm_input_ids.shape[1]`` exactly. The processor builds
       ``tts_lm_input_ids`` as one pad per cached position
       (vibevoice_streaming_processor.py:220-225), so the cache length and the
       pseudo input length are the same number by construction - a tolerance
       here would hide a partially-applied prefill;
    3. the cache survives the shim with its full length, before any forward.
    """
    model = realtime_assets["model"]
    processor = realtime_assets["processor"]
    preset = realtime_assets["preset"]

    inputs = _processor_inputs(processor, preset)
    prefilled = _first_window_conditioning(model, inputs, preset)
    empty = _first_window_conditioning(model, inputs, _empty_cached_prompt(preset))

    expected = int(inputs["tts_lm_input_ids"].shape[1])
    assert prefilled["tts_cache_len"] == expected
    assert prefilled["lm_cache_len"] == int(inputs["input_ids"].shape[1])

    cosine = _cosine(prefilled["condition"], empty["condition"])
    print(
        f"[e2e] conditioning cos(prefilled, empty)={cosine:.4f}  "
        f"cache_len={prefilled['tts_cache_len']}  "
        f"tts_lm_input_ids={tuple(inputs['tts_lm_input_ids'].shape)}"
    )
    assert cosine < 0.99, (
        "the voice prefill does not change the conditioning - the cache is "
        "invisible to the TTS-LM"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Realtime E2E requires CUDA.")
def test_two_voice_presets_condition_differently(realtime_assets, realtime_env):
    """S2.4: two different voices must produce two different conditionings.

    Without this, a cache that returned a constant would pass every other
    assertion in this module.
    """
    model = realtime_assets["model"]
    processor = realtime_assets["processor"]

    other_path = _second_voice_preset_path(realtime_env["preset_path"])
    if other_path is None:
        pytest.skip(
            "no second .pt voice prompt next to VIBEVOICE_REALTIME_VOICE_PRESET"
        )

    other_preset = load_voice_preset(str(other_path), torch.device("cuda"))
    validate_voice_preset(other_preset, str(other_path))

    first = _first_window_conditioning(
        model, _processor_inputs(processor, realtime_assets["preset"]),
        realtime_assets["preset"],
    )
    second = _first_window_conditioning(
        model, _processor_inputs(processor, other_preset), other_preset
    )

    cosine = _cosine(first["condition"], second["condition"])
    print(
        f"[e2e] cos({realtime_env['preset_path'].stem}, {other_path.stem})={cosine:.4f}"
    )
    assert cosine < 0.99, "two different voices produced the same conditioning"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Realtime E2E requires CUDA.")
def test_realtime_checkpoint_generation_and_controls(realtime_assets):
    """Real generation: audio validity, length cap, step control."""
    model = realtime_assets["model"]
    processor = realtime_assets["processor"]
    preset = realtime_assets["preset"]

    probe = _install_generate_probe(model)
    try:
        # 1) The acceptance run must exercise the model's approved auto-length
        #    contract. Zero is mapped to None by the production adapter.
        started = time.perf_counter()
        waveform, sample_rate = generate_realtime_audio(
            model=model,
            processor=processor,
            text=TEST_SCRIPT,
            voice_preset=preset,
            cfg_scale=1.3,
            diffusion_steps=10,
            max_new_tokens=0,
            seed=42,
        )
        latency = time.perf_counter() - started
        _assert_waveform(waveform, sample_rate)
        auto_length = waveform.shape[2]
        auto_rms = float(waveform.float().pow(2).mean().sqrt())
        print(
            f"[e2e] auto run (max_new_tokens=0): samples={auto_length} "
            f"rms={auto_rms:.6f} latency={latency:.2f}s (informational)"
        )

        auto_output = probe["output"]
        # 0 means "auto", and auto must resolve to a bounded budget planned
        # from the script. It must never reach the model unchanged (None), which
        # would let the loop run its full 8192-token context (~18 minutes of
        # audio) and exhaust VRAM in the acoustic-decoder cache.
        assert probe["max_new_tokens"] is not None
        assert probe["max_new_tokens"] > 0
        assert probe["max_new_tokens"] <= REALTIME_MAX_AUTO_BUDGET_UNITS
        auto_input_length = probe["tts_lm_input_length"]
        assert auto_input_length is not None
        observed_generated = int(auto_output.sequences.shape[1]) - auto_input_length
        assert observed_generated > 0
        assert bool(auto_output.reach_max_step_sample[0]) is True

        # 2) Second run from the same cached preset proves it is not corrupted.
        waveform_again, sample_rate_again = generate_realtime_audio(
            model=model,
            processor=processor,
            text=TEST_SCRIPT,
            voice_preset=preset,
            cfg_scale=1.3,
            diffusion_steps=10,
            max_new_tokens=cap_of(observed_generated),
            seed=42,
        )
        _assert_waveform(waveform_again, sample_rate_again)
        print(
            f"[e2e] repeat run: samples={waveform_again.shape[2]} "
            f"rms={float(waveform_again.float().pow(2).mean().sqrt()):.6f} "
            f"(bitwise equality is informational)"
        )

        # 3) Calibrated cap: below half the observed generated sequence length.
        cap = cap_of(observed_generated)
        capped_waveform, capped_rate = generate_realtime_audio(
            model=model,
            processor=processor,
            text=TEST_SCRIPT,
            voice_preset=preset,
            cfg_scale=1.3,
            diffusion_steps=10,
            max_new_tokens=cap,
            seed=42,
        )
        _assert_waveform(capped_waveform, capped_rate)
        assert bool(probe["output"].reach_max_step_sample[0]) is True
        assert capped_waveform.shape[2] <= auto_length

        # 4) Diffusion steps are independent of the generation length control.
        stepped_waveform, stepped_rate = generate_realtime_audio(
            model=model,
            processor=processor,
            text=TEST_SCRIPT,
            voice_preset=preset,
            cfg_scale=1.3,
            diffusion_steps=4,
            max_new_tokens=cap,
            seed=42,
        )
        _assert_waveform(stepped_waveform, stepped_rate)
        assert len(model.noise_scheduler.timesteps) == 4
    finally:
        probe["restore"]()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Realtime E2E requires CUDA.")
def test_canonical_node_realtime_run_with_force_offload(realtime_assets, real_vendored_modules):
    """Run real realtime generation, warm-offload it, then reattach it."""
    from ComfyUI_VibeVoice.nodes.tts_node import VibeVoiceTTSNode

    with _comfyui_preview_stub():
        result = VibeVoiceTTSNode.execute(
            model_name=TEST_MODEL_NAME,
            text=TEST_SCRIPT,
            quantize_llm_4bit=False,
            attention_mode="sdpa",
            cfg_scale=1.3,
            inference_steps=2,
            seed=42,
            do_sample=False,
            temperature=0.95,
            top_p=0.95,
            top_k=0,
            max_new_tokens=64,
            force_offload=True,
            device="cuda",
            dtype="auto",
            voice_preset=realtime_assets["preset_path"].stem,
        )
    _assert_waveform(result[0]["waveform"], result[0]["sample_rate"])
    assert not realtime_assets["patcher"].is_loaded

    reloaded_patcher, reloaded_model, reloaded_processor = real_vendored_modules[
        "generation"
    ].load_vibevoice_model(
        model_name=TEST_MODEL_NAME,
        device="cuda",
        dtype="auto",
        attention_mode="sdpa",
        quantize_4bit=False,
    )
    assert reloaded_patcher is realtime_assets["patcher"]
    assert reloaded_patcher.is_loaded
    assert reloaded_model is realtime_assets["model"]
    assert reloaded_processor is realtime_assets["processor"]


@pytest.fixture
def standard_model_env():
    """Return the explicitly supplied standard checkpoint or skip clearly."""
    model_dir_raw = os.environ.get("VIBEVOICE_STANDARD_MODEL_DIR", "").strip()
    if not model_dir_raw:
        pytest.skip(
            "Set VIBEVOICE_STANDARD_MODEL_DIR to a real standard VibeVoice "
            "checkpoint directory to run this acceptance test."
        )
    model_dir = Path(model_dir_raw).expanduser()
    if not model_dir.is_dir():
        pytest.fail(f"VIBEVOICE_STANDARD_MODEL_DIR is not a directory: {model_dir}")
    if not (model_dir / "config.json").is_file():
        pytest.fail(f"Standard checkpoint is missing config.json: {model_dir}")
    if not any(model_dir.glob("*.safetensors")) and not any(model_dir.glob("*.bin")):
        pytest.skip(f"No model weight files found in standard checkpoint: {model_dir}.")
    return model_dir


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Standard E2E requires CUDA.")
def test_canonical_node_standard_run_with_force_offload(
    standard_model_env, real_vendored_modules, monkeypatch
):
    """Run a real standard model and verify warm offload/reload when opted in."""
    model_dir = standard_model_env
    from ComfyUI_VibeVoice.nodes.tts_node import VibeVoiceTTSNode
    from ComfyUI_VibeVoice.modules.model_info import AVAILABLE_VIBEVOICE_MODELS
    from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_PATCHER_CACHE

    generation = real_vendored_modules["generation"]
    monkeypatch.setitem(
        AVAILABLE_VIBEVOICE_MODELS,
        TEST_STANDARD_MODEL_NAME,
        {
            "type": "local_dir",
            "path": str(model_dir),
            "tokenizer_repo": "Qwen/Qwen2.5-1.5B",
        },
    )
    patcher = model = processor = None
    try:
        patcher, model, processor = generation.load_vibevoice_model(
            model_name=TEST_STANDARD_MODEL_NAME,
            device="cuda",
            dtype="auto",
            attention_mode="sdpa",
            quantize_4bit=False,
        )
        with _comfyui_preview_stub():
            result = VibeVoiceTTSNode.execute(
                model_name=TEST_STANDARD_MODEL_NAME,
                text="[1] This is a standard forced-offload acceptance test.",
                quantize_llm_4bit=False,
                attention_mode="sdpa",
                cfg_scale=1.3,
                inference_steps=2,
                seed=42,
                do_sample=False,
                temperature=0.95,
                top_p=0.95,
                top_k=0,
                max_new_tokens=8,
                force_offload=True,
                device="cuda",
                dtype="auto",
                # A silent reference is rejected by audio_utils before it can
                # clone anything, so the RMS assertion below would be
                # meaningless. A low-frequency tone is a real, audible
                # reference voice instead.
                speaker_1_voice={
                    "waveform": _reference_tone(),
                    "sample_rate": 24000,
                },
            )
        _assert_waveform(result[0]["waveform"], result[0]["sample_rate"])
        assert not patcher.is_loaded
        reloaded_patcher, reloaded_model, reloaded_processor = generation.load_vibevoice_model(
            model_name=TEST_STANDARD_MODEL_NAME,
            device="cuda",
            dtype="auto",
            attention_mode="sdpa",
            quantize_4bit=False,
        )
        assert reloaded_patcher is patcher
        assert reloaded_patcher.is_loaded
        assert reloaded_model is model
        assert reloaded_processor is processor
    finally:
        VIBEVOICE_PATCHER_CACHE.pop(
            f"{TEST_STANDARD_MODEL_NAME}_attn_sdpa_q4_0", None
        )
        AVAILABLE_VIBEVOICE_MODELS.pop(TEST_STANDARD_MODEL_NAME, None)
        del patcher, model, processor
