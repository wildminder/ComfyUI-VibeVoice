r"""Standalone PROBE — does Windows count file-backed MAPPED pages as `uss`?

This is the measurement that decides what the 7B fp8 report's "~2x file at
load time" actually IS.

``modules/memory_census.py`` quotes three process quantities and says of
``uss`` that it is "the number that separates a real host-RAM defect from page
cache" (memory_census.py:389-391). That claim is only true if the ``uss``
reader excludes file-backed pages. On Windows the readers are:

* ``psutil.Process.memory_info().rss``  -> ``WorkingSetSize``
* ``psutil.Process.memory_full_info().uss`` -> psutil's own computation, which
  on Windows enumerates the working set with ``QueryWorkingSet`` and subtracts
  shareable pages;
* ``PROCESS_MEMORY_COUNTERS_EX.PrivateUsage`` -> private COMMIT, which is a
  different quantity again (reserved, touched or not).

If ``uss`` counts mapped file pages, then a load that STREAMS a checkpoint
through an mmap reads ~1x file into the working set and the instrument calls
it "private", which is exactly the misreading that makes a streaming reader
look like a 2x host-RAM defect. If ``uss`` excludes them, the same load
correctly reports ~1x.

The probe is unambiguous by construction: a file is mapped and fully touched
(a) through ``mmap`` and (b) through ``safe_open``'s views, and a bytearray of
the same size is allocated as the private control.

THE VERDICT IS DECIDED PER ARM, and that is the correction. An earlier
revision asked one question of both arms at once and generalised the answer,
which this same run refuted: the two arms are charged DIFFERENTLY on this
host (measured 2026-09-30 — ``mmap`` moves ``ws``/``uss`` and not
``private``; ``safe_open``, the shape the loader actually uses, moves
``private`` and neither ``ws`` nor ``uss``). A single yes/no cannot cover
both, and shipping one told a user reading the 7B fp8 line that a real
private spike was page cache.

Note also what this probe does NOT decide: whether ``get_tensor`` returns a
view or an owned copy is a structural question, settled in
``tests/probe_safetensors_aliases_mapping.py`` with the Win32 region API.
Byte counts cannot settle it — an earlier revision of that probe tried, via a
file-delete test that cannot distinguish a live mapping from an ordinary
access denial on this host.

Run (headless, no ComfyUI server):
    COMFYUI_ROOT='C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI' \
      "C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe" \
      tests/probe_mapped_pages_uss.py
"""

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

GIB = 1 << 30
SIZE = 512 * (1 << 20)  # 512 MiB, big enough that a mapping cannot hide


def snapshot() -> dict:
    proc = psutil.Process(os.getpid())
    info = proc.memory_info()
    return {
        "ws": int(info.rss),
        "uss": int(proc.memory_full_info().uss),
        "private": int(getattr(info, "private", info.rss)),
    }


def build_file(path: str) -> int:
    chunk = torch.zeros(32 * (1 << 20) // 2, dtype=torch.bfloat16)
    tensors = {f"t{i}": chunk.clone() for i in range(16)}
    st.save_file(tensors, path)
    del tensors, chunk
    gc.collect()
    return os.path.getsize(path)


def settle() -> dict:
    gc.collect()
    time.sleep(0.4)
    return snapshot()


def delta(before: dict, after: dict) -> dict:
    return {k: after[k] - before[k] for k in before}


def show(name: str, before: dict, after: dict) -> dict:
    d = delta(before, after)
    print(
        f"[vvprobe] {name:26s} "
        f"d_ws={d['ws'] / GIB:+.2f}GiB "
        f"d_uss={d['uss'] / GIB:+.2f}GiB "
        f"d_private={d['private'] / GIB:+.2f}GiB"
    )
    return d


def arm_mmap(path: str) -> dict:
    """MAP + fully touch a file. Resident working set, zero private bytes."""
    before = settle()
    with open(path, "rb") as fh:
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            step = 4096
            for offset in range(0, SIZE, step):
                mapped[offset]
            after = snapshot()
            d = show("mmap ACCESS_READ (touched)", before, after)
    settle()
    return d


def arm_safe_open(path: str) -> dict:
    """The loader's actual read shape: safe_open views, fully touched."""
    before = settle()
    with safe_open(path, framework="pt", device="cpu") as f:
        held = [f.get_tensor(key) for key in f.keys()]
        total = 0
        for tensor in held:
            flat = tensor.reshape(-1)
            for i in range(0, flat.numel(), 4096):
                flat[i]
            total += tensor.numel()
        after = snapshot()
        d = show(f"safe_open views (n={len(held)})", before, after)
    del held
    settle()
    return d


def arm_private() -> dict:
    """Control: the same bytes as a genuine private allocation."""
    before = settle()
    block = bytearray(SIZE)
    step = 4096
    for offset in range(0, SIZE, step):
        block[offset] = 1
    after = snapshot()
    d = show("bytearray (private control)", before, after)
    del block
    settle()
    return d


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="vv_uss_probe_")
    path = os.path.join(tmp, "mapped.safetensors")
    size = build_file(path)
    print(f"[vvprobe] mapped file = {size} B ({size / GIB:.2f} GiB)")

    # Warm-up: touch the file once so the page cache is not counted as a
    # first-touch cost of the arm under test.
    arm_mmap(path)
    arm_safe_open(path)
    arm_private()

    mapped = arm_mmap(path)
    viewed = arm_safe_open(path)
    private = arm_private()

    print()
    print(f"[vvprobe] VERDICT  mapped d_uss={mapped['uss'] / GIB:+.2f}GiB  "
          f"safe_open d_uss={viewed['uss'] / GIB:+.2f}GiB  "
          f"private d_uss={private['uss'] / GIB:+.2f}GiB")
    # Decided PER ARM, not once for both. The two arms are charged differently
    # on this host (measured, 2026-09-30: the generic mmap arm moves ws/uss
    # and not private; the safe_open arm the LOADER actually uses moves
    # neither ws nor uss and does move private), so a single verdict over
    # both is a generalisation the same run refutes — and shipping it told a
    # user reading the 7B fp8 line that a real private spike was page cache.
    moved = lambda arm: arm > 64 * (1 << 20)  # noqa: E731 - one-line predicate
    if moved(mapped["uss"]):
        print("[vvprobe] mmap arm: uss COUNTS file-backed mapped pages "
              "(ws/uss include ~1x file, private ~0).")
    else:
        print("[vvprobe] mmap arm: uss EXCLUDES file-backed mapped pages.")
    if moved(viewed["uss"]):
        print("[vvprobe] safe_open arm (the loader's read): uss COUNTS "
              "file-backed mapped pages (~1x file).")
    else:
        print("[vvprobe] safe_open arm (the loader's read): uss does NOT "
              "move — a safetensors read charges 'private' instead "
              f"(d_private={viewed['private'] / GIB:+.2f}GiB), so it is "
              "indistinguishable from a heap allocation by these counters.")
    if moved(viewed["uss"]) != moved(mapped["uss"]):
        print("[vvprobe] ARMS DISAGREE: no platform-wide rule about uss "
              "generalises across read shapes. Do not read ws/uss/private as "
              "a private-RAM figure on their own; cross-check the "
              "[vvcensus] private= bucket, which walks storages.")
    print("[vvprobe] NOTE: whether get_tensor returns a VIEW or an OWNED "
          "COPY is a separate question this probe does not answer — see "
          "tests/probe_safetensors_aliases_mapping.py, which decides it "
          "with the Win32 region API rather than from byte counts.")


if __name__ == "__main__":
    main()
