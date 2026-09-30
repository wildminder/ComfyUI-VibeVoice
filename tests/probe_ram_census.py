r"""Standalone PROBE — real host-RAM census + SAMPLED PEAK for a real file.

Runs the production load path (no conftest mocks, fresh package alias) over a
real VibeVoice checkpoint and reports, per phase:

  * ``[vvcensus]`` — which storages the model holds after the load: aimdo
    file views vs mmap vs PRIVATE host bytes, the patcher's ``backup`` stash
    and the vbar ranges core allocated (modules/memory_census.py).
  * ``[vvrss]`` — the PROCESS working set sampled on a background thread for
    the whole phase, i.e. the PEAK. A post-hoc RSS delta cannot see the
    transient that users report (the 7B fp8 spike is gone before the
    post-H2D census runs), so the peak is sampled, never inferred.

Phase A = ``load_external_vibevoice_model`` (host-side materialisation).
Phase B = ``load_vibevoice_from_external`` (patcher construction + H2D).

Run (headless, no ComfyUI server):
    COMFYUI_ROOT='C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI' \
      "C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe" \
      tests/probe_ram_census.py

Which files: argv, else ``VV_PROBE_FILES`` (``os.pathsep``-separated), else
the conventional diffusion_models directory. Nothing is downloaded and
nothing is guessed: a missing file is reported, never substituted.
"""
import contextlib
import gc
import importlib
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.environ.get(
    "COMFYUI_ROOT", r"C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# package alias for modules.* (mirrors the node, WITHOUT conftest's mocks)
_ALIAS = "ComfyUI_VibeVoice"
_spec = importlib.util.spec_from_file_location(
    _ALIAS, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
sys.modules[_ALIAS] = importlib.util.module_from_spec(_spec)
with contextlib.suppress(Exception):
    _spec.loader.exec_module(sys.modules[_ALIAS])

import torch  # noqa: E402  (after the path bootstrap, as in production)

import comfy.memory_management  # noqa: E402
import comfy.model_patcher  # noqa: E402
from comfy_aimdo import control  # noqa: E402
from ComfyUI_VibeVoice.modules.config_detect import detect_config_name  # noqa: E402
from ComfyUI_VibeVoice.modules.convrot_quant import (  # noqa: E402
    scan_checkpoint_quantization)
from ComfyUI_VibeVoice.modules.external_loader import (  # noqa: E402
    load_external_vibevoice_model)
from ComfyUI_VibeVoice.modules.generation import (  # noqa: E402
    generate_audio, load_vibevoice_from_external)
from ComfyUI_VibeVoice.modules.memory_census import (  # noqa: E402
    RssSampler,
    census,
    census_enabled,
    format_census,
    memory_snapshot,
    peak_delta,
    rss_bytes,
)


def aimdo_ready() -> bool:
    """Initialise comfy-aimdo the way ComfyUI's main.py does, or say why not.

    Without this the probe measures the LEGACY route: ``dynamic_vram_available``
    is False, ``select_patcher_class`` returns the legacy patcher, and
    ``_load_state_dict_into_model_from_memory`` clones every tensor into private
    memory (modules/external_loader.py:1511-1566). That is a real, documented
    cost of the legacy path, not the user's dynamic-route report — so a probe
    that skips this bootstrap measures the wrong thing entirely.

    The ``host_buffer`` / ``model_mmap`` / ``model_vbar`` submodules snapshot
    ``lib = control.lib`` at IMPORT time, so they must be re-imported after
    ``control.init()`` has loaded the native library.
    """
    if not torch.cuda.is_available():
        print("probe: no CUDA — the dynamic route cannot be measured")
        return False
    try:
        if not control.init():
            print("probe: aimdo control.init() failed")
            return False
        import comfy_aimdo

        for name in ("host_buffer", "model_mmap", "model_vbar",
                     "vram_buffer", "storage"):
            sub = getattr(comfy_aimdo, name, None)
            if sub is not None and getattr(sub, "lib", None) is None:
                importlib.reload(sub)
        devices = list(comfy.model_management.get_all_torch_devices())
        if not bool(control.init_devices(
                (d.index, int(4 * 1024 ** 3)) for d in devices)):
            print("probe: aimdo init_devices() failed")
            return False
        # The two globals main.py flips when aimdo is available.
        comfy.memory_management.aimdo_enabled = True
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        print(f"probe: aimdo ready, devices={[str(d) for d in devices]}")
        return True
    except Exception as exc:
        print(f"probe: aimdo unavailable ({exc}) — the census would measure "
              "the LEGACY route, which clones into private memory")
        return False


def models_dir() -> Path:
    """Conventional diffusion_models tree, resolved at runtime.

    Never a baked-in path for one host: ``VV_PROBE_MODELS_DIR`` wins, then the
    ComfyUI root this script was bootstrapped against. Failing loudly is
    cheaper than probing the wrong tree.
    """
    for candidate in (
        os.environ.get("VV_PROBE_MODELS_DIR"),
        os.path.join(os.environ.get("COMFYUI_ROOT", ""), "models", "diffusion_models"),
    ):
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    raise SystemExit(
        "Cannot find a diffusion_models directory. Set VV_PROBE_MODELS_DIR, or "
        "COMFYUI_ROOT to a ComfyUI install containing models/diffusion_models."
    )


def targets():
    """Weight files to probe: argv > ``VV_PROBE_FILES`` > conventional dir."""
    from_argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    if from_argv:
        return [Path(p) for p in from_argv]
    env = os.environ.get("VV_PROBE_FILES", "")
    if env:
        return [Path(p) for p in env.split(os.pathsep) if p]
    return sorted(models_dir().glob("VibeVoice-*.safetensors"))


def header_summary(path: Path) -> dict:
    """File size + safetensors HEADER census (no tensor is materialised)."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        meta = [k for k in keys if k.endswith(".comfy_quant")]
        formats = {}
        for k in meta:
            # comfy_quant is a tiny uint8 JSON sidecar; reading it costs
            # kilobytes and the file's dtype histogram comes from the header.
            raw = bytes(f.get_tensor(k).numpy()).decode("utf-8")
            fmt = json.loads(raw).get("format", "?")
            formats[fmt] = formats.get(fmt, 0) + 1
    quant_map = scan_checkpoint_quantization(str(path))
    return {
        "bytes": path.stat().st_size,
        "keys": len(keys),
        "quant_metas": len(meta),
        "formats": formats,
        "resident_fp8": sum(1 for i in quant_map.values() if i.resident_fp8),
        "convrot": sum(1 for i in quant_map.values() if i.convrot),
    }


def reference_audio(seconds: float = 2.0, sample_rate: int = 24000) -> dict:
    """A synthetic ComfyUI AUDIO dict to drive the real generate path.

    Content is irrelevant — the probe measures HOST RAM, not audio quality, and
    the shipped ``voices/*.pt`` presets are cached-KV blobs for the realtime
    family (keys ``lm``/``tts_lm``/``neg_lm``/``neg_tts_lm``), not waveforms,
    so they cannot drive ``generate_audio`` at all. A tone exercises exactly
    the same weight-streaming path a user's script does: every forward pulls
    weights disk->VRAM through the vbar.
    """
    t = torch.linspace(0.0, seconds, int(seconds * sample_rate), dtype=torch.float32)
    waveform = 0.1 * torch.sin(2 * torch.pi * 220.0 * t).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sample_rate}


def attribute_residual(patcher, model, ws_delta: int) -> None:
    """Name WHICH object holds the host bytes generation just added.

    The census can only see the MODEL's storages, and after a generation it
    still reports 100% file views and zero private param bytes — so when the
    working set grows anyway, the bytes are in something the census does not
    walk. The one host allocation the dynamic route makes that is not a model
    storage is core's pinned staging: six ``HostBuffer``s created in
    ``ModelPatcherDynamic.load`` (comfy/model_patcher.py:1874-1881) and grown
    by ``extend()`` as weights are staged for the H2D. Each carries its own
    ``.size``, so the attribution is a direct read, not an inference.

    ``patcher.model`` — not the inner module — is where ``dynamic_pins`` lives:
    core's ``load()`` reads ``self.model.dynamic_pins[...]`` and ``self.model``
    is the node's handler, which is what owns the real model.
    """
    print("\n--- residual attribution ---")
    print(f"working set added by generation: {ws_delta / 2**30:.2f} GiB")
    mm = comfy.model_management
    model_size = patcher.model_size() if hasattr(patcher, "model_size") else 0
    hostbuf_size = mm.pinned_hostbuf_size(model_size)
    print(f"core MAX_PINNED_MEMORY = {mm.MAX_PINNED_MEMORY / 2**30:.2f} GiB")
    print(f"model_size = {model_size / 2**30:.2f} GiB -> "
          f"pinned_hostbuf_size = {hostbuf_size / 2**30:.2f} GiB "
          f"(the max_grow_size of each pin_state HostBuffer)")
    print(f"core TOTAL_PINNED_MEMORY = {mm.TOTAL_PINNED_MEMORY / 2**30:.2f} GiB")

    total_pins = 0
    # The handler owns the pins; fall back to the model for a bare patcher.
    owner = getattr(patcher, "model", None) or model
    pins = getattr(owner, "dynamic_pins", None) or {}
    if not pins:
        print("  (no dynamic_pins on the patcher's model — nothing staged)")
    for device, pin_state in pins.items():
        for subset in mm.PIN_SUBSETS + mm.LOADED_PIN_SUBSETS + mm.FAST_PIN_SUBSETS:
            entry = pin_state.get(subset)
            if not entry:
                continue
            buffer = entry[0]
            size = int(getattr(buffer, "size", 0) or 0)
            total_pins += size
            print(f"  pin_state[{device}][{subset!r}].size = {size / 2**30:.3f} GiB")
    print(f"  pinned host-buffer total = {total_pins / 2**30:.2f} GiB")
    unaccounted = ws_delta - total_pins
    print(f"  unaccounted by pins = {unaccounted / 2**30:+.2f} GiB "
          "(mapping page cache + CUDA/allocator overhead if positive)")


def probe(path: Path) -> None:
    """Load ``path`` through production and report census + sampled peaks."""
    print(f"\n{'=' * 78}\n=== {path.name}  ({path.stat().st_size / 2**30:.2f} GiB)\n{'=' * 78}")
    summary = header_summary(path)
    print("HEADER:", json.dumps(summary))

    config_name = detect_config_name(str(path)) or "VibeVoice-7B"
    print(f"config_name={config_name}  census_enabled={census_enabled()}")

    baseline = rss_bytes()[0]
    print(f"BASELINE working_set={baseline / 2**30:.2f} GiB")

    # ---- Phase A: host-side materialisation ------------------------
    with RssSampler(label="phase-a:load_external", series=True) as sampler:
        bundle = load_external_vibevoice_model(
            str(path), config_name,
            attention_mode="sdpa", use_llm_4bit=False, dtype_str="auto",
        )
        sampler.mark("bundle-returned")
    sampler.report("phase-a:load_external")
    print(" ", sampler.profile())
    print("  peak delta:", _delta_line(peak_delta(sampler, baseline), summary))
    print(format_census(census(bundle["model"]), phase=f"post-load:{config_name}"))
    print("  weight_family:", bundle.get("weight_family"),
          "quant_stats:", bundle.get("quant_stats"),
          "dynamic_vram_route:", bundle.get("dynamic_vram_route"))

    model = bundle["model"]
    fp8 = [m for m in model.modules() if type(m).__name__ == "FP8Linear"]
    print(f"FP8Linear modules: {len(fp8)}; "
          f"storage dtypes: {sorted({str(m.weight.dtype) for m in fp8}) or 'none'}")
    views = 0
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and getattr(
                weight.untyped_storage(), "_comfy_tensor_file_slice", None) is not None:
            views += 1
    print(f"params carrying an aimdo file view: {views} "
          "(0 is expected on a CPU/legacy-route probe; a CUDA+aimdo run "
          "preserves views on the dynamic route)")

    # ---- Phase B: patcher + H2D -------------------------------------
    if not torch.cuda.is_available():
        print("no CUDA — skipping phase B (patcher/H2D)")
        return
    before_b = rss_bytes()[0]
    with RssSampler(label="phase-b:h2d") as sampler_b:
        patcher, loaded_model, processor = load_vibevoice_from_external(
            bundle, device="auto", dtype="auto", attention_mode="sdpa")
        sampler_b.mark("loaded")
    sampler_b.report("phase-b:h2d")
    print("  peak delta:", _delta_line(peak_delta(sampler_b, before_b), summary))
    print(format_census(census(loaded_model, patcher), phase=f"post-h2d:{config_name}"))
    print("  patcher:", type(patcher).__name__, "loaded_size:", patcher.loaded_size())

    # ---- Phase C: a REAL generation ---------------------------------
    # Phases A and B measure the load. A user's "25.1 -> 32.2GB, STAYS" is
    # read AFTER generation, and generation is the only thing that reads the
    # whole checkpoint: every forward pulls weights disk->VRAM through the
    # vbar. If the residual were a private host allocation it would show up
    # here as private/USS growth; if it is the mapping's page cache it shows
    # up as working-set growth with private flat. That is the whole question,
    # and it cannot be answered by a load-only measurement.
    voice = reference_audio()
    before_c, before_c_private = rss_bytes()
    with RssSampler(label="phase-c:generate") as sampler_c:
        audio, sample_rate = generate_audio(
            model=loaded_model,
            processor=processor,
            text="Speaker 1: Hello, this is a short host memory probe.",
            voice_samples=[voice],
            speaker_ids=[1],
            cfg_scale=1.3,
            inference_steps=2,
            seed=42,
            do_sample=False,
        )
        sampler_c.mark("generated")
    sampler_c.report("phase-c:generate")
    print("  peak delta:", _delta_line(peak_delta(sampler_c, before_c), summary))
    print(f"  audio: {tuple(audio.shape)} @ {sample_rate}Hz")
    print(format_census(census(loaded_model, patcher), phase=f"post-generate:{config_name}"))
    after_ws, after_private = rss_bytes()
    print(f"  post-generate working_set={(after_ws - before_c) / 2**30:+.2f} GiB "
          f"private={(after_private - before_c_private) / 2**30:+.2f} GiB over pre-generate")
    attribute_residual(patcher, loaded_model, after_ws - before_c)

    try:
        patcher.unpatch_model(destroy=True)
    except Exception as e:  # probe cleanup must not hide the measurement
        print("  unpatch failed:", e)
    gc.collect()
    with contextlib.suppress(Exception):
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()


def _delta_line(delta: dict, summary: dict) -> str:
    """Human line naming the sampled peaks as multiples of the FILE size.

    ``peak_ws`` is what Task Manager showed (working set, file cache
    included). ``peak_uss`` is the private RESIDENT set — the difference
    between the two is mapping page cache, not host allocations, and it is
    what decides whether a spike is a defect or the file being read.
    """
    file_bytes = summary["bytes"]
    return (f"peak_ws_delta={delta['peak_ws_delta'] / 2**30:.2f} GiB "
            f"({delta['peak_ws_delta'] / file_bytes:.2f}x file) "
            f"peak_uss_delta={delta['peak_uss_delta'] / 2**30:.2f} GiB "
            f"({delta['peak_uss_delta'] / file_bytes:.2f}x file) "
            f"peak_commit={delta['peak_private'] / 2**30:.2f} GiB "
            f"end_ws={delta['end_ws'] / 2**30:.2f} GiB "
            f"end_uss={delta['end_uss'] / 2**30:.2f} GiB "
            f"samples={delta['samples']}")


def main():
    found = targets()
    if not found:
        print("No weight files to probe. Pass paths as argv or set VV_PROBE_FILES.")
        return
    if not aimdo_ready():
        # Measuring the legacy route would be measuring the wrong thing: it
        # clones every tensor to private memory by design, which is a
        # different (already-documented) cost from the dynamic-route defect
        # this probe exists to attribute.
        print("probe: refusing to measure without aimdo")
        return
    for path in found:
        if not path.exists():
            print(f"SKIP (missing): {path}")
            continue
        probe(path)


if __name__ == "__main__":
    main()
