r"""Standalone PROBE — does the aimdo read scale clean, or commit private at GB scale?

Live evidence (1.5B bf16, 2026-09-30, user console):

    [vvrss] dense-read-state-dict peak_private=3.12GB  (start_ws=1.09GB end_ws=1.10GB)
    [vvrss] dense-bind-state-dict end_private=3.17GB
    [vvcensus] pre-h2d view=5.04GB(100.0%) private=0B

The read is exactly ``comfy.utils.load_torch_file`` (external_loader.py:684),
whose aimdo arm is core's ``load_safetensors`` (comfy/utils.py:97) — the same
code the 256MB probe (probe_aimdo_view_private_cost.py) measured at 0.02x
file size in private. The FLAT ws during the live read is the open question:
either the 3.12GB private was mostly the process baseline (the [vvrss] line
does not print start_private), or aimdo's ModelMMAP behaves differently at
multi-GB scale / for this specific file (a native-side threshold we cannot
read from Python).

This probe measures the SAME phases against, in order:

  * a synthetic ~1 GB file (8 tensors, written to temp)
  * the REAL 1.5B bf16 checkpoint (5.04 GiB, ~1.2k tensors) — READ-ONLY:
    mapping, header parse, per-tensor frombuffer views, one read-only sweep,
    then free. No model is built, nothing is instantiated.

Phases per file (deltas from the previous phase):

  1. baseline
  2. load_torch_file          — mapping + views created, nothing consumed
  3. touch every view (sum)   — the read-only fault pattern the assign loop hits
  4. dict freed               — nothing should retain the mapping

VERDICT TABLE:
  phase 2 d_private ~1x file  => aimdo's mapping commits private at this scale
                                 (threshold behavior; core-side, file-shape-
                                 dependent). The dense read is NOT free and the
                                 [vvrss] 3.12GB is real load cost.
  phase 2 flat, 3 flat        => the dense read is clean at scale; the live
                                 3.12GB end_private was process baseline plus
                                 small overhead, and the machine-RAM growth the
                                 user sees is NOT this process's private commit.

Run (headless, no ComfyUI server, no model load):
    COMFYUI_ROOT='C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI' \
      "C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe" \
      tests/probe_dense_read_scale.py
"""

import gc
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.environ.get(
    "COMFYUI_ROOT", r"C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import comfy.memory_management  # noqa: E402
import comfy.utils  # noqa: E402

REAL_FILE = r"C:\AI\ComfyUI\ComfyUI\models\diffusion_models\VibeVoice-1.5B-bf16.safetensors"


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


def _report(label: str, before: dict, after: dict) -> None:
    row = f"  {label:<36}"
    for key in ("ws", "uss", "private"):
        row += f" d_{key}={_mb(after[key] - before[key])}"
    print(row)


def _run_file(path: str, file_bytes: int, touch: bool = True) -> None:
    print(f"file: {os.path.basename(path)} "
          f"({_mb(file_bytes).strip()} = {file_bytes} bytes, "
          f"{os.path.exists(path) and 'exists'})")

    c0 = _counters()
    print(f"  {'1 baseline':<36} ws={_mb(c0['ws']).strip()} "
          f"private={_mb(c0['private']).strip()}")

    sd = comfy.utils.load_torch_file(path)
    gc.collect()
    c1 = _counters()
    _report("2 load_torch_file (views made)", c0, c1)
    n_views = len(sd)

    if touch:
        # Read-only sweep: faults every page the way a real consumer would.
        # fp8 storages have no CPU sum kernel, so touch one byte per 4 KiB
        # page through a uint8 view instead of a dtype reduction.
        touched = 0
        for tensor in sd.values():
            if tensor.numel():
                t_u8 = tensor.reshape(-1).view(torch.uint8)
                t_u8[::4096].sum()
            touched += 1
        gc.collect()
        c2 = _counters()
        _report(f"3 after touching {touched} views", c1, c2)

    del sd
    gc.collect()
    c3 = _counters()
    _report("4 dict freed", c2 if touch else c1, c3)
    print(f"  views={n_views}; total d_private={_mb(c3['private'] - c0['private'])}"
          f" ({(c3['private'] - c0['private']) / file_bytes:.3f}x file)")


def main() -> int:
    if not torch.cuda.is_available():
        print("probe: no CUDA — the aimdo runtime cannot initialise")
        return 0
    if not _init_aimdo():
        print("probe: aimdo did not initialise; cannot measure its mapping")
        return 0

    import psutil

    print(f"probe: machine RAM={_mb(psutil.virtual_memory().total).strip()}, "
          f"aimdo_enabled={comfy.memory_management.aimdo_enabled}")

    # -- bracket 1: synthetic ~1 GB, same shape as the clean 256MB probe ----
    tmp = tempfile.mkdtemp(prefix="vv_scale_probe_")
    synth = os.path.join(tmp, "probe_1g.safetensors")
    import safetensors.torch as st
    payload = {}
    n_rows, n_cols = 8192, 8192  # 128 MiB per bf16 tensor, 8 tensors = 1 GiB
    for i in range(8):
        payload[f"w{i}"] = torch.randn(n_rows, n_cols, dtype=torch.bfloat16)
    st.save_file(payload, synth)
    del payload
    gc.collect()
    _run_file(synth, os.path.getsize(synth))
    os.unlink(synth)
    os.rmdir(tmp)
    gc.collect()

    # -- bracket 2: the REAL checkpoints, read-only --------------------------
    # argv override: any number of .safetensors paths (e.g. the 7B fp8 file,
    # whose 9.47GB size crosses the 8GiB 32-bit-size boundary).
    targets = sys.argv[1:] or [REAL_FILE]
    for target in targets:
        if os.path.exists(target):
            _run_file(target, os.path.getsize(target))
        else:
            print(f"probe: real file not found, skipping: {target}")

    print(
        "VERDICT: phase-2 d_private ~1x file => aimdo mapping commits private\n"
        "  at this scale (core-side threshold); phase 2+3 flat => the dense\n"
        "  read is clean and the live 3.12GB end_private was baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
