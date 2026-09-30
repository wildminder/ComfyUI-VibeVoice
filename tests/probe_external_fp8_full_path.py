r"""Standalone PROBE — run the REAL external fp8 load path headless, no GPU load.

Live evidence (7B fp8, runs 2-4, 2026-09-30):

    [vvrss]    stream-apply-safetensors peak_private=16.09GB end_private=13.03GB
    [vvcensus] stream-end private=1.02GB (view=8.31GB 89.1%)
    [vvrss]    load-to-device (later)     peak_private=4.27GB

Mapping-only probes have now cleared every piece individually at full scale
(probe_dense_read_scale.py: 9.47GB fp8 file -> 22MB private for views; +9GB
ws/uss file-backed on touch; clean free). Yet the live run held ~10GB private
at stream end that the model census cannot see, released again by the next
phase. Source reading has run out of candidates — so reproduce the exact live
call path headless and let the loader's own instrumentation print.

This calls ONLY ``load_external_vibevoice_model`` — the load-NODE work. The
docstring contract: the model is built on CPU/meta and the host-to-device
transfer belongs to the patcher later. We never build the patcher, never call
load_models_gpu, never touch the weights: the model comes back holding aimdo
file views. Structural probe, not a model load.

Run (headless, no ComfyUI server):
    COMFYUI_ROOT='C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI' \
      "C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe" \
      tests/probe_external_fp8_full_path.py [path-to-fp8-file]
"""

import gc
import importlib.util
import logging
import os
import sys
from unittest.mock import MagicMock

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get(
    "COMFYUI_ROOT", r"C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI"))
sys.path.insert(0, ROOT_DIR)

# The loader package must be imported under its alias (modules/external_loader
# reaches the vendored source via ``..src.vibevoice``), with the REAL vendored
# model code — no conftest mocks; this embedded python is the live one.
_PKG_ALIAS = "ComfyUI_VibeVoice"
if _PKG_ALIAS not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        _PKG_ALIAS, os.path.join(ROOT_DIR, "__init__.py"),
        submodule_search_locations=[ROOT_DIR])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[_PKG_ALIAS] = pkg
    try:
        spec.loader.exec_module(pkg)
    except Exception:
        pass

# Network-server mocks (same as conftest) so node imports stay inert.
_mock_server = MagicMock()
_mock_server.PromptServer.instance = MagicMock()
sys.modules.setdefault("server", _mock_server)
sys.modules.setdefault("aiohttp", MagicMock())
sys.modules.setdefault("aiohttp.web", MagicMock())

import comfy.memory_management  # noqa: E402

DEFAULT_FILE = (
    r"C:\AI\ComfyUI\ComfyUI\models\diffusion_models\VibeVoice-7B-fp8_e4m3.safetensors")


def _init_aimdo() -> bool:
    """Same sequence main.py uses, including the import-order reload."""
    import importlib

    try:
        from comfy_aimdo import control
        if not control.init():
            return False
        import comfy.model_management as mm
        if not bool(control.init_devices(
                (d.index, int(512 * 1024 ** 2))
                for d in mm.get_all_torch_devices())):
            return False
        import comfy_aimdo
        for name in ("host_buffer", "model_mmap", "model_vbar",
                     "vram_buffer", "storage"):
            sub = getattr(comfy_aimdo, name, None)
            if sub is not None and getattr(sub, "lib", None) is None:
                importlib.reload(sub)
        comfy.memory_management.aimdo_enabled = True
        return True
    except Exception as exc:
        print(f"probe: aimdo unavailable ({exc})")
        return False


def _counters() -> dict:
    from modules.memory_census import memory_snapshot
    return memory_snapshot()


def _mb(value) -> str:
    try:
        return f"{value / (1024 ** 2):9.1f} MB"
    except TypeError:
        return f"{str(value):>12}"


def _line(label: str, c: dict) -> None:
    print(f"PROBE {label:<40} ws={_mb(c['ws']).strip()} "
          f"uss={_mb(c['uss']).strip()} private={_mb(c['private']).strip()}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    weight_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    if not os.path.isfile(weight_path):
        print(f"probe: file not found: {weight_path}")
        return 1
    file_bytes = os.path.getsize(weight_path)

    if not torch.cuda.is_available():
        print("probe: no CUDA — the aimdo runtime cannot initialise")
        return 0
    if not _init_aimdo():
        print("probe: aimdo did not initialise")
        return 0

    device = torch.device("cuda")

    # main.py:300-301 does exactly this rebinding once aimdo initialises; the
    # loader's route probe (dynamic_vram_available -> resolve_core_patcher_class)
    # reads this attribute at call time. Without it the probe silently runs the
    # legacy clone route instead of the live dynamic route.
    import comfy.model_patcher as _mp
    if not hasattr(_mp, "ModelPatcherDynamic"):
        print("probe: this comfy has no ModelPatcherDynamic — cannot force the "
              "dynamic route")
        return 1
    _mp.CoreModelPatcher = _mp.ModelPatcherDynamic

    from ComfyUI_VibeVoice.modules import external_loader as _el
    from ComfyUI_VibeVoice.modules.loader import VibeVoiceLoader
    from modules import quant_common as _qc
    from modules import convrot_quant as _cq
    from modules.memory_census import report_census

    # ---- BISECT: print private after every loader-internal step ----------
    def _wrap(obj, name, label):
        real = getattr(obj, name)

        def wrapped(*a, **kw):
            result = real(*a, **kw)
            c = _counters()
            print(f"BISECT {label:<44} private={_mb(c['private']).strip()}")
            return result

        setattr(obj, name, wrapped)
        return real

    # Function-local ``from X import Y`` re-resolves the module attribute at
    # call time, so patching the SOURCE module is enough for those; class
    # staticmethods are patched on the class.
    _wrap(_cq, "scan_checkpoint_quantization", "scan_checkpoint_quantization")
    _wrap(_el, "resolve_sidecar_config", "resolve_sidecar_config")
    _wrap(VibeVoiceLoader, "_load_config", "VibeVoiceLoader._load_config")
    _wrap(VibeVoiceLoader, "_load_tokenizer", "VibeVoiceLoader._load_tokenizer")
    _wrap(VibeVoiceLoader, "_load_processor", "VibeVoiceLoader._load_processor")
    _wrap(VibeVoiceLoader, "_instantiate_model", "VibeVoiceLoader._instantiate_model")
    _wrap(_qc, "replace_linears_for_quant", "replace_linears_for_quant")
    _wrap(_el, "_read_safetensors_tensor", "_read_safetensors_tensor (pass-1)")
    _wrap(_el, "_stream_apply_safetensors", "_stream_apply_safetensors")
    _wrap(VibeVoiceLoader, "_post_assign_fixups", "VibeVoiceLoader._post_assign_fixups")

    from ComfyUI_VibeVoice.modules.external_loader import (
        load_external_vibevoice_model,)

    c0 = _counters()
    _line(f"baseline (file={file_bytes / 1024**2:.0f}MB)", c0)

    bundle = load_external_vibevoice_model(
        weight_path=weight_path,
        config_name="VibeVoice-7B",
        attention_mode="sage",
        use_llm_4bit=False,
        dtype_str="auto",
        device=device,
    )

    gc.collect()
    c1 = _counters()
    _line("after load (bundle held)", c1)
    print(f"PROBE d_private over whole load: "
          f"{_mb(c1['private'] - c0['private']).strip()} "
          f"({(c1['private'] - c0['private']) / file_bytes:.2f}x file)")

    model = bundle.get("model")
    if model is not None:
        report_census(model, phase="probe-post-load")

    # Release everything the bundle holds (model first — views die with it).
    bundle.clear()
    del model
    gc.collect()
    c2 = _counters()
    _line("after bundle freed", c2)
    print(f"PROBE private retained after free: "
          f"{_mb(c2['private'] - c0['private']).strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
