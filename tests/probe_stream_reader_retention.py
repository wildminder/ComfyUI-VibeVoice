r"""Standalone PROBE — who retains the SECOND copy during a streamed read.

The 7B fp8 report (docs/2026-09-29-dynamic-vram-port-and-int64-conv-regression.md)
measured a load-time peak against an 8.82 GiB file that reached ~2x, and the
extra ~1x needed a retainer. Two candidate shapes:

  H1 "get_tensor is an mmap VIEW"  -> the per-tensor ``clone()`` at
     modules/external_loader.py:1047 is a real second copy sitting beside the
     read's own ~1x, and that is the ~2x.
  H2 "get_tensor is an OWNED COPY" -> the mapping is never a factor and the
     extra ~1x must come from somewhere else entirely.

An earlier revision of this probe settled VIEW vs COPY by trying to delete
the checkpoint and treating a refusal as proof of a live mapping. That
evidence is INVALID: on this host ``os.remove`` fails with a plain access
denial (WinError 5) even after every ``get_tensor`` result has been dropped,
so a refusal cannot be distinguished from an ordinary permission failure.
It is replaced here by :func:`is_view`, which asks the Win32 region API what
kind of region the tensor's bytes actually live in — ``MEM_MAPPED`` backed by
the checkpoint file, against a ``MEM_PRIVATE`` heap control. That is a direct
answer, not an inference from byte counts or from file-delete semantics.

An earlier revision also gave a contradictory reading (ws flat while private
commit grew) because the read finished inside one sample interval, so the
only reading taken was the endpoint. This revision removes that ambiguity
three ways:

* the read is SLOWED (a fixed sleep per tensor) so the sampler sees the
  interior, not just the endpoints;
* VIEW vs COPY is settled STRUCTURALLY, see above;
* a WARM-UP read runs first, so the allocator is not handing out virgin pages
  that the first-touch cost would attribute to the measurement.

Run (headless, no ComfyUI server):
    COMFYUI_ROOT='C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI' \
      "C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe" \
      tests/probe_stream_reader_retention.py

Nothing is downloaded and no host path is baked in: the probe BUILDS its own
synthetic safetensors file in a temp dir.
"""

import ctypes
import gc
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.environ.get(
    "COMFYUI_ROOT", r"C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import psutil  # noqa: E402

from safetensors import safe_open  # noqa: E402
import safetensors.torch as st  # noqa: E402

MEM_PRIVATE = 0x20000
MEM_MAPPED = 0x40000
_REGION_TYPES = {MEM_PRIVATE: "MEM_PRIVATE", MEM_MAPPED: "MEM_MAPPED",
                 0x1000000: "MEM_IMAGE"}


class _MEMORY_BASIC_INFORMATION(ctypes.Structure):  # noqa: N801 - win32 name
    _fields_ = [
        ("BaseAddress", ctypes.c_void_p),
        ("AllocationBase", ctypes.c_void_p),
        ("AllocationProtect", ctypes.c_ulong),
        ("PartitionId", ctypes.c_ushort),
        ("RegionSize", ctypes.c_size_t),
        ("State", ctypes.c_ulong),
        ("Protect", ctypes.c_ulong),
        ("Type", ctypes.c_ulong),
    ]


MB = 1024 * 1024
GIB = 1 << 30
# ~0.5 GiB of bf16 weights in 32 tensors, shaped like a real checkpoint: a few
# large matrices plus small ones, so per-tensor size VARIES.
N_TENSORS = 32
TENSOR_MB = 16
SLOW_READ_S = 0.05


def snapshot() -> dict:
    proc = psutil.Process(os.getpid())
    info = proc.memory_info()
    return {
        "ws": int(info.rss),
        "uss": int(proc.memory_full_info().uss),
        "private": int(getattr(info, "private", info.rss)),
    }


class Peak:
    """Background peak sampler, the same shape as the shipped RssSampler."""

    def __init__(self, interval: float = 0.02):
        self.interval = interval
        self.peak = {"ws": 0, "uss": 0, "private": 0}
        self.start = snapshot()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            self._take()
            self._stop.wait(self.interval)

    def _take(self) -> dict:
        reading = snapshot()
        for key in self.peak:
            self.peak[key] = max(self.peak[key], reading[key])
        return reading

    def close(self) -> "Peak":
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.end = self._take()
        return self


def build_file(path: str) -> int:
    tensors = {}
    for i in range(N_TENSORS):
        n = TENSOR_MB * MB // 2  # bf16 => half the bytes
        # Vary the size so the allocator cannot serve every tensor from one
        # warm block: a uniform size would let a fragmentation-free reading
        # of "no extra bytes" hide behind reuse.
        n -= i * 16 * 1024
        tensors[f"layer{i}.weight"] = torch.zeros(
            n // 4096, 4096, dtype=torch.bfloat16)
    tensors["layer0.bias"] = torch.zeros(4096, dtype=torch.bfloat16)
    st.save_file(tensors, path)
    return os.path.getsize(path)


def _region_of(address: int) -> tuple:
    """``(region type name, backing file)`` for ``address`` via Win32.

    ``VirtualQuery`` says what KIND of region the bytes are — ``MEM_MAPPED``
    for a mapping, ``MEM_PRIVATE`` for a heap allocation — and
    ``GetMappedFileNameW`` names the file behind a mapped region. This is the
    replacement for the old delete-based test, which could not tell a live
    mapping from an ordinary access denial.
    """
    if sys.platform != "win32":  # pragma: no cover - this host is Windows
        return ("unsupported", "")
    mbi = _MEMORY_BASIC_INFORMATION()
    kernel32 = ctypes.windll.kernel32
    if not kernel32.VirtualQuery(
        ctypes.c_void_p(address), ctypes.byref(mbi), ctypes.sizeof(mbi)
    ):
        return ("unknown", "")
    name = _REGION_TYPES.get(int(mbi.Type), f"0x{int(mbi.Type):x}")
    buf = ctypes.create_unicode_buffer(1024)
    fn = ctypes.windll.psapi.GetMappedFileNameW
    fn.restype = ctypes.c_ulong
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_wchar_p,
                   ctypes.c_ulong]
    copied = fn(kernel32.GetCurrentProcess(), ctypes.c_void_p(address), buf, 1024)
    return (name, buf.value if copied else "")


def is_view(path: str, key: str) -> bool:
    """True iff ``get_tensor`` handed back a live mapping of ``path``.

    Decided by the region the tensor's bytes occupy, cross-checked against a
    heap allocation's region. The backing-file comparison is done on the file
    NAME because ``GetMappedFileNameW`` answers in NT namespace
    (``\\Device\\HarddiskVolume3\\...``) while ``os.path.realpath`` answers in
    DOS namespace (``C:\\...``); comparing the full strings would report a
    mismatch for the file itself.
    """
    with safe_open(path, framework="pt", device="cpu") as f:
        tensor = f.get_tensor(key)
    kind, backing = _region_of(tensor.data_ptr())
    control_kind, _ = _region_of(torch.zeros(1024, dtype=torch.bfloat16).data_ptr())
    name = os.path.basename(path).lower()
    is_mapping = kind == "MEM_MAPPED" and name in backing.lower()
    print(
        f"[vvprobe]   region({key!r}) = {kind} backing={backing or '-'} | "
        f"heap control = {control_kind} -> "
        f"{'VIEW' if is_mapping else 'OWNED COPY'}"
    )
    return is_mapping


def read_retaining(path: str, clone: bool, slow: bool) -> dict:
    """Stream every tensor, RETAIN it (and optionally clone it), return peaks.

    Retention is a first-class term: the loaded model holds every assigned
    tensor, so the list is the model, not scratch space.
    """
    held = []
    peak = Peak()
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            held.append(tensor.clone() if clone else tensor)
            if slow:
                time.sleep(SLOW_READ_S)
    peak.close()
    retained = sum(t.untyped_storage().nbytes() for t in held)
    return {"peak": peak, "retained": retained, "n": len(held)}


def report(name: str, result: dict, file_bytes: int) -> None:
    peak, start = result["peak"].peak, result["peak"].start
    print(
        f"[vvprobe] {name:20s} file={file_bytes / GIB:.2f}GiB "
        f"retained={result['retained'] / GIB:.2f}GiB n={result['n']} "
        f"peak_ws={(peak['ws'] - start['ws']) / GIB:+.2f}GiB "
        f"peak_uss={(peak['uss'] - start['uss']) / GIB:+.2f}GiB "
        f"peak_priv={(peak['private'] - start['private']) / GIB:+.2f}GiB"
    )


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="vv_stream_probe_")
    path = os.path.join(tmp, "synthetic.safetensors")
    file_bytes = build_file(path)
    print(f"[vvprobe] built {file_bytes} B at {path}")
    print(f"[vvprobe] safetensors.get_tensor returns a VIEW: "
          f"{is_view(path, 'layer0.weight')}")

    # WARM-UP: one full read, discarded, so the allocator and the page cache
    # are not in their virgin state during the measured reads.
    read_retaining(path, clone=True, slow=False)
    gc.collect()
    time.sleep(0.5)

    for clone in (False, True):
        gc.collect()
        time.sleep(0.3)
        result = read_retaining(path, clone=clone, slow=True)
        report("read+retain CLONE" if clone else "read+retain VIEW", result,
               file_bytes)
        del result
        gc.collect()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
