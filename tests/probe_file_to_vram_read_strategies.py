r"""Standalone PROBE — which file->VRAM read shape is fast AND leaves RAM alone?

The user reports (2026-09-30, live): with the old pass-1 the 7B fp8 load read no
SSD but committed the whole 9.47 GB file to host RAM (19.7 -> 28 -> 20 GB);
after replacing that with header byte-range reads, the load now drives the SSD
and "fills RAM and VRAM". Both behaviours are real; the question is which read
shape gives a single sequential NVMe pass with no resident page cache.

Four arms, each in its own process so page-cache state is independent:

  A  safe_open the whole file, then per-tensor .to(cuda)
       -- what pass 1 used to do as a side effect, then the stream
  B  header byte-range reads, then per-tensor .to(cuda)
       -- what the loader does now
  C  core's own read_tensor_file_slice_into(view, preallocated_cuda)
       -- cache-clean file->VRAM DMA, no host staging at all
  D  plain buffered readinto into a preallocated CPU buffer, then .to(cuda)
       -- the "sequential streaming" baseline, no mapping

Metrics per arm: elapsed, machine RAM delta, process private, and
GetProcessIoCounters ReadTransferCount (bytes the FS layer was asked for).

Run (headless, no ComfyUI server, no model load):
    COMFYUI_ROOT='C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI' \
      "C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe" \
      tests/probe_file_to_vram_read_strategies.py --arm B --size-gb 2
"""

import argparse
import ctypes
import gc
import json
import os
import struct
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.environ.get(
    "COMFYUI_ROOT", r"C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI"))

import comfy.memory_management  # noqa: E402
import comfy.utils  # noqa: E402
from comfy.memory_management import read_tensor_file_slice_into  # noqa: E402

N_TENSORS = 64
DTYPE = torch.bfloat16


# ----------------------------------------------------------------- metrics
class _MemStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


_k32 = ctypes.windll.kernel32


def machine_used_gb():
    st = _MemStatus()
    st.dwLength = ctypes.sizeof(st)
    _k32.GlobalMemoryStatusEx(ctypes.byref(st))
    return (st.ullTotalPhys - st.ullAvailPhys) / 1024 ** 3


def io_counters():
    c = _IoCounters()
    _k32.GetProcessIoCounters(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c))
    return c.ReadTransferCount


def private_gb():
    try:
        import psutil
        return psutil.Process().memory_info().private / 1024 ** 3
    except Exception:
        return float("nan")


# ----------------------------------------------------------------- fixture
def build_file(tmpdir, size_gb):
    """A real .safetensors of `size_gb`, written once, then only read."""
    path = os.path.join(tmpdir, "big.safetensors")
    if os.path.exists(path) and os.path.getsize(path) > size_gb * 1024 ** 3 * 0.9:
        return path
    per = int(size_gb * 1024 ** 3 / N_TENSORS)
    per -= per % 4096
    header, offset, meta = {}, 0, {}
    for i in range(N_TENSORS):
        name = f"layer{i}.weight"
        header[name] = {
            "dtype": "BF16", "shape": [per // 2],
            "data_offsets": [offset, offset + per],
        }
        meta[name] = {"format": "float8_e4m3fn"}  # not used; keeps it a plain file
        offset += per
    hb = json.dumps(header).encode("utf-8")
    chunk = torch.zeros(per // 2, dtype=DTYPE).view(torch.uint8).numpy().tobytes()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for _ in range(N_TENSORS):
            f.write(chunk)
    return path


# ----------------------------------------------------------------- the arms
def arm_a(path, dev):
    """safe_open the whole file, then per-tensor .to(cuda)."""
    from safetensors import safe_open
    f = safe_open(path, framework="pt", device="cpu")
    keys = list(f.keys())
    held = [f.get_tensor(k) for k in keys]          # whole file mapped+committed
    out = []
    for t in held:
        out.append(t.to(dev))
    del held, out
    f = None
    gc.collect()


def arm_b(path, dev):
    """Header byte-range reads, then per-tensor .to(cuda) via the iterator."""
    from ComfyUI_VibeVoice.modules.external_loader import (
        read_safetensors_tensors_by_name,
    )
    read_safetensors_tensors_by_name(path, ["layer0.weight"])
    views = comfy.utils.load_torch_file(path)
    out = []
    for t in views.values():
        out.append(t.to(dev))
    del views, out
    gc.collect()


def arm_c(path, dev):
    """core's cache-clean file->VRAM DMA into preallocated CUDA tensors."""
    views = comfy.utils.load_torch_file(path)
    out = []
    for t in views.values():
        dst = torch.empty(t.shape, dtype=t.dtype, device=dev)
        if not read_tensor_file_slice_into(t, dst):
            dst.copy_(t)
        out.append(dst)
    del views, out
    gc.collect()


ARMS = {"A": arm_a, "B": arm_b, "C": arm_c}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--size-gb", type=float, default=2.0)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA on this box")
        return 2
    dev = torch.device("cuda", 0)
    print(f"aimdo_enabled={getattr(comfy.memory_management, 'aimdo_enabled', '?')}")

    tmp = tempfile.mkdtemp(prefix="vvread_")
    try:
        path = build_file(tmp, args.size_gb)
        size = os.path.getsize(path)
        print(f"arm {args.arm}: {path}  {size / 1024 ** 3:.2f} GiB  "
              f"{N_TENSORS} tensors")

        # Warm the *code* (CUDA context, imports) so t0 is a fair baseline.
        torch.zeros(1, device=dev).sum().item()
        gc.collect()

        m0, p0, i0 = machine_used_gb(), private_gb(), io_counters()
        t0 = time.perf_counter()
        ARMS[args.arm](path, dev)
        torch.cuda.synchronize(dev)
        dt = time.perf_counter() - t0
        m1, p1, i1 = machine_used_gb(), private_gb(), io_counters()

        print(f"  elapsed        {dt:6.2f} s  -> {size / 1024 ** 3 / dt:5.2f} GiB/s")
        print(f"  machine used   {m1 - m0:+6.2f} GB   (peak not sampled)")
        print(f"  process private{p1 - p0:+6.2f} GB")
        print(f"  fs read bytes  {(i1 - i0) / 1024 ** 3:6.2f} GiB  "
              f"({(i1 - i0) / size:.2f}x file)")
    finally:
        if not args.keep:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
