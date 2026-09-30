r"""Standalone PROBE — does ``safetensors.get_tensor`` ALIAS the file mapping?

This settles H1/H2 structurally, without any reliance on Windows delete
semantics. The previous probe (``probe_stream_reader_retention.py``) inferred a
live mapping from ``os.remove`` raising, but a plain access denial (WinError
5) is indistinguishable from a real mapping, so it could not decide.

Method: the Win32 region API, not delete semantics. ``VirtualQuery`` on the
tensor's ``data_ptr`` reports what KIND of region the bytes live in
(``MEM_MAPPED`` for a file mapping, ``MEM_PRIVATE`` for a heap allocation),
and ``GetMappedFileNameW`` names the backing FILE of a mapped region. Comparing
the reported file path with the checkpoint path decides it exactly, and it does
not depend on which mapping instance the address came from — a comparison of
``data_ptr`` against a mapping this process opened itself would be meaningless,
because ``safe_open`` opens its own mapping at its own base.

* ``MEM_MAPPED`` + this file's path -> the tensor IS a VIEW of the mapping.
* ``MEM_PRIVATE``                    -> safetensors deserialised an OWNED COPY.

Run (headless, no ComfyUI server):
    COMFYUI_ROOT='C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI' \
      "C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe" \
      tests/probe_safetensors_aliases_mapping.py

No host path is baked in and nothing is downloaded: the probe builds its own
synthetic file in a temp dir.
"""

import ctypes
import gc
import mmap
import os
import sys
import tempfile
import time

sys.path.insert(0, os.environ.get(
    "COMFYUI_ROOT", r"C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import psutil  # noqa: E402

from safetensors import safe_open  # noqa: E402
import safetensors.torch as st  # noqa: E402

MB = 1024 * 1024
GIB = 1 << 30

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


def region_of(address: int) -> tuple:
    """``(region type name, backing file path or "")`` for ``address``."""
    mbi = _MEMORY_BASIC_INFORMATION()
    kernel32 = ctypes.windll.kernel32
    if not kernel32.VirtualQuery(
        ctypes.c_void_p(address), ctypes.byref(mbi), ctypes.sizeof(mbi)
    ):
        return ("unknown", "")
    name = _REGION_TYPES.get(int(mbi.Type), f"0x{int(mbi.Type):x}")
    buf = ctypes.create_unicode_buffer(1024)
    try:
        fn = ctypes.windll.psapi.GetMappedFileNameW
        fn.restype = ctypes.c_ulong
        fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_ulong,
        ]
        copied = fn(kernel32.GetCurrentProcess(), ctypes.c_void_p(address),
                    buf, 1024)
    except Exception:
        copied = 0
    return (name, buf.value if copied else "")


def snapshot() -> dict:
    proc = psutil.Process(os.getpid())
    info = proc.memory_info()
    return {
        "ws": int(info.rss),
        "uss": int(proc.memory_full_info().uss),
        "private": int(getattr(info, "private", info.rss)),
    }


def build_file(path: str) -> None:
    st.save_file({
        "small.weight": torch.zeros(64, dtype=torch.bfloat16),
        "big.weight": torch.zeros(4 * MB // 2, dtype=torch.bfloat16),
    }, path)


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="vv_alias_probe_")
    path = os.path.join(tmp, "synthetic.safetensors")
    build_file(path)
    size = os.path.getsize(path)
    real = os.path.realpath(path)
    print(f"[vvprobe] file = {size} B at {real}")

    def same_file(backing: str) -> bool:
        """Compare a mapping's backing path with ``real``.

        ``GetMappedFileNameW`` answers in NT namespace
        (``\\Device\\HarddiskVolume3\\...``) while ``os.path.realpath``
        answers in DOS namespace (``C:\\...``), so the two are compared on
        their tail after the drive/volume prefix is stripped — a prefix
        mismatch must not be mistaken for a different file.
        """
        if not backing:
            return False
        tail = backing.rsplit("\\", 1)[-1].lower()
        return real.rsplit("\\", 1)[-1].lower() == tail and tail in backing

    # Arm 1 — the structural question, asked while the file mapping exists.
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            kind, backing = region_of(tensor.data_ptr())
            verdict = (
                "VIEW of this checkpoint"
                if kind == "MEM_MAPPED" and same_file(backing)
                else f"OWNED COPY ({kind}"
                     f"{', ' + backing if backing else ''})"
            )
            print(f"[vvprobe] get_tensor({key!r}) region={kind} "
                  f"backing={backing or '-'} -> {verdict}")
            del tensor
    gc.collect()

    # Arm 1b — a real file mapping whose kind is known by construction, to
    # prove the region API reads a mapping as MEM_MAPPED and names its file on
    # this host at all. numpy.memmap is the one mapping whose address is
    # obtainable through the plain Python API (a read-only mmap.mmap object
    # refuses ctypes.from_buffer).
    import numpy as np
    mapped_array = np.memmap(path, dtype=np.uint8, mode="r")
    kind, backing = region_of(mapped_array.ctypes.data)
    print(f"[vvprobe] control np.memmap region={kind} "
          f"backing={backing or '-'}")
    del mapped_array
    gc.collect()

    # Arm 1c — a heap allocation, the other known-by-construction kind.
    block = torch.zeros(4 * MB // 2, dtype=torch.bfloat16)
    kind, backing = region_of(block.data_ptr())
    print(f"[vvprobe] control torch.empty region={kind} backing={backing or '-'}")
    del block
    gc.collect()

    # ------------------------------------------------------------------
    # Byte arms. The structural arms above answer the ownership question; these
    # answer the OTHER question the shipped guide got wrong — whether a
    # safetensors read and a plain mmap read cost the same in `ws`/`uss`.
    # A 4 MiB file is far too small for a working-set delta to mean anything,
    # so the byte arms use a file big enough that a mapping cannot hide.
    # ------------------------------------------------------------------
    big_path = os.path.join(tmp, "big.safetensors")
    st.save_file(
        {f"layer{i}.weight": torch.zeros(32 * MB // 2, dtype=torch.bfloat16)
         for i in range(16)},
        big_path,
    )
    big_size = os.path.getsize(big_path)
    print(f"\n[vvprobe] byte arms, file = {big_size} B "
          f"({big_size / GIB:.2f} GiB)")

    def arm(name: str, body):
        """Run ``body()``, snapshot while its result is STILL ALIVE, then drop.

        The retention has to outlive the reading: a body whose return value is
        discarded before the snapshot measures a freed working set, which is
        how an arm reads +0.00 for everything.
        """
        gc.collect()
        time.sleep(0.4)
        before = snapshot()
        held = body()
        after = snapshot()
        print(
            f"[vvprobe] {name:30s} d_ws={(after['ws'] - before['ws']) / GIB:+.2f}GiB "
            f"d_uss={(after['uss'] - before['uss']) / GIB:+.2f}GiB "
            f"d_private={(after['private'] - before['private']) / GIB:+.2f}GiB"
        )
        del held
        gc.collect()
        time.sleep(0.4)

    def read_safetensors():
        held = []
        with safe_open(big_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                flat = tensor.reshape(-1)
                for i in range(0, flat.numel(), 4096):
                    flat[i]
                held.append(tensor)
        return held

    def read_plain_mmap():
        fh = open(big_path, "rb")
        mapped = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        for offset in range(0, mapped.size(), 4096):
            mapped[offset]
        return (fh, mapped)

    def read_bytearray():
        block = bytearray(big_size)
        for offset in range(0, big_size, 4096):
            block[offset] = 1
        return block

    # Warm-up so the page cache is not charged to the arm under test.
    # Warm-up so the page cache is not charged to the arm under test.
    held = read_safetensors()
    del held
    gc.collect()
    time.sleep(0.5)
    arm("safetensors get_tensor (touched)", read_safetensors)
    arm("plain mmap read (touched)", read_plain_mmap)
    arm("bytearray (private control)", read_bytearray)

    # Arm 5 — THE question for the loader. The views above were read while the
    # ``safe_open`` handle was still open. ``iter_safetensors_tensors`` reads
    # the WHOLE file inside ONE ``with safe_open``, so the loader's peak is
    # reached with every touched page still charged. Does closing the handle
    # release those bytes while the tensors are still held?
    #
    # YES -> the ~1x is the handle's lifetime, not ownership, and reading one
    #        tensor per handle bounds the loader's private peak at ~1x.
    # NO  -> the private bytes are retained by the tensors themselves and the
    #        loader's route has an irreducible ~1x, whatever the mapping says.
    gc.collect()
    time.sleep(0.4)
    base = snapshot()
    held = []
    with safe_open(big_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for key in keys:
            tensor = f.get_tensor(key)
            flat = tensor.reshape(-1)
            for i in range(0, flat.numel(), 4096):
                flat[i]
            held.append(tensor)
        open_snap = snapshot()
    closed_snap = snapshot()
    print(
        f"[vvprobe] whole-file read in ONE safe_open, {len(held)} views held:\n"
        f"[vvprobe]   while the handle is OPEN   "
        f"d_ws={(open_snap['ws'] - base['ws']) / GIB:+.2f}GiB "
        f"d_uss={(open_snap['uss'] - base['uss']) / GIB:+.2f}GiB "
        f"d_private={(open_snap['private'] - base['private']) / GIB:+.2f}GiB\n"
        f"[vvprobe]   after the handle CLOSED   "
        f"d_ws={(closed_snap['ws'] - base['ws']) / GIB:+.2f}GiB "
        f"d_uss={(closed_snap['uss'] - base['uss']) / GIB:+.2f}GiB "
        f"d_private={(closed_snap['private'] - base['private']) / GIB:+.2f}GiB"
    )
    # Same again, but cloning each tensor as the loader does.
    cloned = [t.clone() for t in held]
    cloned_snap = snapshot()
    print(
        f"[vvprobe]   + clone() of every view    "
        f"d_ws={(cloned_snap['ws'] - base['ws']) / GIB:+.2f}GiB "
        f"d_uss={(cloned_snap['uss'] - base['uss']) / GIB:+.2f}GiB "
        f"d_private={(cloned_snap['private'] - base['private']) / GIB:+.2f}GiB"
    )
    del cloned, held
    gc.collect()
    time.sleep(0.4)


if __name__ == "__main__":
    main()
