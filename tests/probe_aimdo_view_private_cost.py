r"""Standalone PROBE — what does HOLDING the aimdo view dict cost in private commit?

Live evidence (7B fp8, 2026-09-30, run 2/3):

    [vvcensus] stream-end  model holds private=1.02GB   (view=8.31GB 89.1%)
    [vvrss]    stream-apply end_private=13.03GB

so ~12GB is committed OUTSIDE the model's storages during the assign pass and
released before the H2D. The only sizeable thing in that window is the aimdo
view dict: ``iter_safetensors_tensors`` -> ``comfy.utils.load_torch_file``
-> core ``load_safetensors``, which maps the whole file and hands back
per-tensor ``torch.frombuffer`` views.

Two candidate explanations, and they need opposite fixes:

  A. The MAPPING is charged private commit on this host (aimdo's ModelMMAP
     may map with SEC_IMAGE / copy-on-write semantics, which count as
     private). Then the ~12GB is core's own accounting — the same thing
     Krea holds — and there is nothing for this pack to remove.
  B. Something COPIES during the pass (e.g. core's set_attr_param clones an
     inference tensor, comfy/utils.py:975-979). Then the spike is ours and
     fixable.

The already-measured arms do NOT settle it: probe_mapped_pages_uss.py found
plain ``mmap`` moves ws/uss and NOT private, while ``safe_open`` moves private
— but the aimdo path uses NEITHER shape (it is frombuffer over a ModelMMAP
mapping, tagged ``_comfy_tensor_file_slice``).

This probe measures the aimdo path itself, per phase, at ~256 MB scale:

  1. baseline
  2. after ``load_torch_file`` (mapping + all views created, nothing consumed)
  3. after touching every page (simulating the assign loop reading the bytes)
  4. after assigning the views into parameters via ``comfy.utils.set_attr_param``
  5. after releasing the dict (params keep the views — the production shape)

VERDICT: phase 2 or 3 already at ~file size => A (core's accounting, accept).
Flat until phase 4 => B (a copy in assign; ours to fix).

Run (headless, no ComfyUI server):
    COMFYUI_ROOT='C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI' \
      "C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe" \
      tests/probe_aimdo_view_private_cost.py
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
    """The three process counters the census quotes, in one reading."""
    # The ``ComfyUI_VibeVoice`` package alias is a conftest construct, so a
    # standalone script must import the module by its real path.
    from modules.memory_census import memory_snapshot
    return memory_snapshot()


def _mb(value) -> str:
    try:
        return f"{value / (1024 ** 2):8.1f} MB"
    except TypeError:
        return f"{str(value):>12}"


def _report(label: str, before: dict, after: dict, file_bytes: int) -> None:
    row = f"{label:<34}"
    for key in ("ws", "uss", "private"):
        row += f" d_{key}={_mb(after[key] - before[key])}"
    print(row)


def main() -> int:
    if not torch.cuda.is_available():
        print("probe: no CUDA — the aimdo runtime cannot initialise")
        return 0
    if not _init_aimdo():
        print("probe: aimdo did not initialise; cannot measure its mapping")
        return 0

    import safetensors.torch as st

    tmp = tempfile.mkdtemp(prefix="vv_aimdo_probe_")
    path = os.path.join(tmp, "probe.safetensors")
    n_rows, n_cols = 4096, 4096  # 4096*4096*2 = 32 MiB per tensor, 8 tensors
    payload = {}
    for i in range(8):
        payload[f"w{i}"] = torch.randn(n_rows, n_cols, dtype=torch.bfloat16)
    st.save_file(payload, path)
    file_bytes = os.path.getsize(path)
    print(f"probe: file={_mb(file_bytes).strip()} ({file_bytes} bytes), "
          f"aimdo_enabled={comfy.memory_management.aimdo_enabled}")
    del payload
    gc.collect()

    # -- phase 1: baseline ------------------------------------------------
    c0 = _counters()
    print(f"probe: baseline ws={_mb(c0['ws']).strip()} "
          f"private={_mb(c0['private']).strip()}")

    # -- phase 2: the view dict, nothing touched ---------------------------
    sd = comfy.utils.load_torch_file(path)
    gc.collect()
    c1 = _counters()
    _report("2 load_torch_file (views made)", c0, c1, file_bytes)

    # -- phase 3: touch every page (what the assign loop does) -------------
    for tensor in sd.values():
        tensor.sum()
    gc.collect()
    c2 = _counters()
    _report("3 after touching every view", c1, c2, file_bytes)

    # -- phase 4: assign into parameters -----------------------------------
    class Tree(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for i in range(8):
                self.register_parameter(
                    f"w{i}", torch.nn.Parameter(
                        torch.empty(n_rows, n_cols, dtype=torch.bfloat16),
                        requires_grad=False))

    tree = Tree()
    for name, tensor in sd.items():
        comfy.utils.set_attr_param(tree, name, tensor)
    del sd
    gc.collect()
    c3 = _counters()
    _report("4 after set_attr_param (+dict gone)", c2, c3, file_bytes)

    # -- phase 5: production shape — params hold the views ------------------
    tagged = sum(
        1 for t in tree.parameters()
        if hasattr(t.untyped_storage(), "_comfy_tensor_file_slice")
    )
    print(f"probe: parameters still carrying a file slice: {tagged}/8")
    gc.collect()
    c4 = _counters()
    _report("5 steady state (params hold views)", c3, c4, file_bytes)

    print()
    total = c3["private"] - c0["private"]
    print(f"probe: private committed by the whole load: "
          f"{_mb(total).strip()} for a {_mb(file_bytes).strip()} file "
          f"({total / file_bytes:.2f}x)")
    print("VERDICT: ~1x at phase 2 or 3 => A (core's mapping accounting, "
          "nothing to fix here); flat until phase 4 => B (a copy in assign).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
