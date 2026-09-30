r"""Standalone PROBE — does core's paging read pollute the Windows file cache?

Live evidence (user, 1.5B bf16, 2026-09-30): machine RAM 20.8 -> 27.3 GB during
the TTS node, while the ComfyUI PROCESS stays flat (private 3.12 -> 3.22GB,
ws 1.09 -> 1.16GB per the [vvrss] lines). +6.5GB machine RAM that no process
counter sees is either (a) the OS file cache absorbing the staged file reads
or (b) page faults on the aimdo mapping. Only (a)/(b) can grow machine RAM
invisible to per-process counters, and only (a) is avoidable in principle.

core's production arm for vbar-destined weights is `read_tensor_file_slice_into
(tensor, cuda_destination)` -> destination=None -> `comfy_aimdo.host_buffer.
read_file_to_device(..., mark_cold=False)` (comfy/memory_management.py:44-63):
a NATIVE file->device read. If aimdo opens the file unbuffered, machine
"used" will NOT move and the fill the user sees must be the mapping
arm. If it moves 1:1, core reads into the cache exactly like any reader -
which is the same accounting a Krea checkpoint load performs.

Method: fresh COLD offsets of a real checkpoint (never read by earlier
probes), one arm per offset; psutil.virtual_memory().used/available around
each arm, plus process ws/private for contrast.

Arms:
  1. core: read_tensor_file_slice_into(aimdo_view, cuda_tensor) — the
     production file->VRAM path.
  2. plain: file.readinto(preallocated cpu tensor) — the fallback arm /
     every "normal" reader (what our old safe_open pass-1 did).
  3. touch: touch every page of the mapping via frombuffer views — arm (b),
     machine "used" should NOT move (file-backed ws only).

Run (headless, no ComfyUI server, no model):
    COMFYUI_ROOT='C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI' \
      "C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe" \
      tests/probe_core_slice_read_cost.py
"""

import ctypes
import gc
import os
import sys
import time

import psutil
import torch

sys.path.insert(0, os.environ.get(
    "COMFYUI_ROOT", r"C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import comfy.memory_management  # noqa: E402
import comfy.utils  # noqa: E402

# 16.66GB bf16 ASR checkpoint: earlier probes only ever read HEADER bytes of
# it, so its data offsets are cold.
COLD_FILE = (
    r"C:\AI\ComfyUI\ComfyUI\models\diffusion_models\VibeVoice-ASR-HF-bf16.safetensors")
CHUNK_TARGET = 2 * 1024 ** 3  # 2 GiB per arm


def _init_aimdo() -> bool:
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


def _gb(value) -> str:
    return f"{value / (1024 ** 3):7.2f} GB"


def _sys_state() -> tuple:
    vm = psutil.virtual_memory()
    pr = psutil.Process()
    mi = pr.memory_info()
    return vm.used, vm.available, getattr(mi, "rss", 0)


def _report(label: str, before: tuple, after: tuple, nbytes: int, secs: float) -> None:
    print(f"  {label:<46} "
          f"sys_used+{_gb(after[0] - before[0])} "
          f"sys_avail{_gb(after[1] - before[1])} "
          f"proc_rss+{_gb(after[2] - before[2])} "
          f"({_gb(nbytes)} in {secs:.1f}s = {_gb(nbytes / max(secs, 1e-9))}/s)")


def _sleep_control() -> None:
    """Machine-state drift with no IO at all."""
    before = _sys_state()
    time.sleep(3.0)
    gc.collect()
    after = _sys_state()
    _report("control (3s idle, no IO)", before, after, 0, 3.0)


def main() -> int:
    if not torch.cuda.is_available():
        print("probe: no CUDA")
        return 0
    if not _init_aimdo():
        print("probe: aimdo did not initialise")
        return 0
    if not os.path.isfile(COLD_FILE):
        print(f"probe: cold file missing: {COLD_FILE}")
        return 1

    device = torch.device("cuda")
    views = comfy.utils.load_torch_file(COLD_FILE)
    items = [(k, v) for k, v in views.items()
             if v.is_contiguous() and v.numel() * v.element_size() > 0]
    print(f"probe: file={COLD_FILE}")
    print(f"probe: {len(items)} tensors, "
          f"{sum(v.numel() * v.element_size() for _, v in items) / 1024**3:.2f} GB "
          f"of views (mapping only so far)")

    # Order matters: each arm uses a disjoint, previously-unread region.
    order = sorted(items, key=lambda kv: kv[1].storage_offset())

    # ---- arm 1: core file -> device (the production vbar read) -----------
    region, total = [], 0
    for k, v in order:
        region.append((k, v))
        total += v.numel() * v.element_size()
        if total >= CHUNK_TARGET:
            break
    dest = torch.empty(total, dtype=torch.uint8, device=device)
    gc.collect()
    before = _sys_state()
    t0 = time.perf_counter()
    cursor = 0
    declined = 0
    for k, _ in region:
        tensor = views[k]
        nbytes = tensor.numel() * tensor.element_size()
        if not comfy.memory_management.read_tensor_file_slice_into(
                tensor, dest[cursor:cursor + nbytes]):
            declined += 1
        cursor += nbytes
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    after = _sys_state()
    _report(f"arm 1 core file->device ({len(region)} tensors, "
            f"{declined} declined)", before, after, total, secs)
    del dest
    torch.cuda.empty_cache()
    gc.collect()

    # ---- arm 2: plain buffered read into host memory ---------------------
    consumed = len(region)
    region, total = [], 0
    for k, v in order[consumed:]:
        region.append((k, v))
        total += v.numel() * v.element_size()
        if total >= CHUNK_TARGET:
            break
    consumed += len(region)
    host = torch.empty(total, dtype=torch.uint8)
    base_off = views[region[0][0]].untyped_storage()._comfy_tensor_file_slice.offset
    with open(COLD_FILE, "rb") as fh:
        before = _sys_state()
        t0 = time.perf_counter()
        view = memoryview((ctypes.c_ubyte * total).from_address(host.data_ptr()))
        for k, _ in region:
            info = views[k].untyped_storage()._comfy_tensor_file_slice
            off = info.offset - base_off
            size = views[k].numel() * views[k].element_size()
            fh.seek(off)
            got = 0
            while got < size:
                got += fh.readinto(view[off + got:off + size])
        secs = time.perf_counter() - t0
        after = _sys_state()
    _report(f"arm 2 plain readinto ({len(region)} tensors)", before, after, total, secs)
    del host
    gc.collect()

    # ---- arm 3: fault the mapping itself (ws/file-backed arm) ------------
    region, total = [], 0
    for k, v in order[consumed:]:
        region.append((k, v))
        total += v.numel() * v.element_size()
        if total >= CHUNK_TARGET:
            break
    before = _sys_state()
    t0 = time.perf_counter()
    for k, _ in region:
        t = views[k].reshape(-1).view(torch.uint8)
        t[::4096].sum()
    secs = time.perf_counter() - t0
    after = _sys_state()
    _report(f"arm 3 fault mapping pages ({len(region)} tensors)", before, after, total, secs)

    _sleep_control()

    views.clear()
    gc.collect()
    print("VERDICT: arm 1 sys_used ~1x bytes => core's paging read feeds the "
          "Windows\n  file cache (inherent, identical for Krea; mark_cold "
          "does not prevent it).\n  arm 1 sys_used ~0 => the read bypasses "
          "the cache; the fill the user sees is the mapping/arm-3\n  or a "
          "buffered read somewhere else on our path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
