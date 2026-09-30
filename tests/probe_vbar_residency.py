r"""Standalone PROBE — does our streaming tree STAY resident in the vbar?

The open finding (F7 in .dev/docs/2026-09-30-vibevoice-load-investigation) is
that a live 1.5B run re-served the whole model every autoregressive step:

    [vvpull] vbar/_ComfyStreamLinear = 123,568 pulls (1,562,753 MB)
    [vvrss] tts-generate start_sys=24.60GB end_sys=44.34GB   prompt 566.97s

One full model pass per AR step is NORMAL for a streamed forward — every
forward touches every weight, so the pull COUNT is not the anomaly. The anomaly
is the COST: those pulls resolve through ``comfy.ops.cast_bias_weight`` ->
``cast_modules_with_vbar`` -> ``resolve_cast_module_with_vbar``. That path is
cheap ONLY when the module's vbar range is still resident:

    signature = vbar_fault(s._v)
    resident  = vbar_signature_compare(signature, s._v_signature)

``resident`` short-circuits to ``s._v_weight`` (a view into the arena). When
it is False core runs ``materialize_meta_param`` and transfers the weight —
and because our weights are aimdo FILE SLICES, that transfer is a disk read,
not a RAM read (measured: host-side view reads are 1:1 and 0.55 GB/s,
tests/probe_core_slice_read_cost.py). That is the user's exact symptom.

So the question this probe answers is binary and it is NOT answerable by
counting pulls: **does ``resident`` stay True across repeated forwards, and if
not, what makes it go False?**

Three arms, each a synthetic tree (no real checkpoint, no model download):

  A  baseline      small tree, nothing competing  -> protocol must be stable
  B  big tree      staged tree large enough to press the arena
  C  B + churn     a large transient allocation between forwards, which is
                   what the speech-diffusion decoder does mid-prompt

Arms B and C exist because the live machine had ~12 GB of VRAM already in use
while only ~5.5 GB was staged: if the arena cannot hold the tree, eviction is
expected and unavoidable, and the fix is budget/shape, not the wrapper.

Run (headless, synthetic tensors only, no user checkpoint touched):

    COMFYUI_ROOT='C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI' \
      "C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe" \
      tests/probe_vbar_residency.py

Verdict table printed at the end:

  A faulted == 0            -> our wrappers speak the core protocol correctly
                               and residency is stable in isolation
  B faulted == 0, C > 0     -> eviction under allocation pressure (budget
                               problem); quantify fault_rate and bytes/re-fault
  B faulted > 0             -> the tree does not fit the arena at all; the
                               size the patcher reports is wrong or too big
"""

import gc
import importlib
import os
import sys
import tempfile
import time
from unittest.mock import MagicMock

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get(
    "COMFYUI_ROOT", r"C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI"))
sys.path.insert(0, ROOT_DIR)

# The package imports itself as ``ComfyUI_VibeVoice`` (conftest does the same);
# with the alias registered, ``modules.*`` imports below resolve to the same
# module objects the shipped code uses.
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

_mock_server = MagicMock()
_mock_server.PromptServer.instance = MagicMock()
sys.modules.setdefault("server", _mock_server)
sys.modules.setdefault("aiohttp", MagicMock())
sys.modules.setdefault("aiohttp.web", MagicMock())

import comfy.memory_management  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.utils  # noqa: E402

WIDTH = 2048
DTYPE = torch.bfloat16
PER_LINEAR_MB = WIDTH * WIDTH * 2 / (1024 ** 2)


# ---------------------------------------------------------------- aimdo


def _init_aimdo(budget_bytes: int) -> bool:
    """Same sequence main.py uses, including the import-order reload.

    ``budget_bytes`` is the per-device vbar arena main.py would ask for; it is
    a parameter here because arm C's whole point is varying it.
    """
    try:
        from comfy_aimdo import control
        if not control.init():
            return False
        if not bool(control.init_devices(
                (d.index, int(budget_bytes))
                for d in comfy.model_management.get_all_torch_devices())):
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


# ---------------------------------------------------------------- counters

STATS = {"calls": 0, "resident": 0, "faulted": 0,
         "xfer_calls": 0, "xfer_bytes": 0, "file_reads": 0}

_orig_resolve = comfy.ops.resolve_cast_module_with_vbar
_orig_gathered = comfy.model_management.cast_to_gathered
_orig_slice_read = comfy.memory_management.read_tensor_file_slice_into


def _counting_resolve(s, *args, **kwargs):
    """Classify the branch core is about to take, without changing it.

    ``resolve_cast_module_with_vbar`` is the only place ``prefetch["resident"]``
    is read, and ``cast_bias_weight`` deletes ``_prefetch`` immediately after —
    so wrapping the resolver is the one hook that sees the decision.
    """
    prefetch = getattr(s, "_prefetch", None)
    STATS["calls"] += 1
    if prefetch is not None and prefetch.get("resident"):
        STATS["resident"] += 1
    else:
        STATS["faulted"] += 1
    return _orig_resolve(s, *args, **kwargs)


def _counting_gathered(source, destination, *args, **kwargs):
    STATS["xfer_calls"] += 1
    try:
        STATS["xfer_bytes"] += sum(t.numel() * t.element_size()
                                   for t in source
                                   if isinstance(t, torch.Tensor))
    except Exception:
        pass
    return _orig_gathered(source, destination, *args, **kwargs)


def _counting_slice_read(*args, **kwargs):
    STATS["file_reads"] += 1
    return _orig_slice_read(*args, **kwargs)


def _install_hooks():
    comfy.ops.resolve_cast_module_with_vbar = _counting_resolve
    comfy.model_management.cast_to_gathered = _counting_gathered
    comfy.memory_management.read_tensor_file_slice_into = _counting_slice_read


def _prod_split():
    """The SHIPPED counters (modules/comfy_stream.py), read the same way the
    live ``[vvpull]`` line reads them. Arm D proves the observer the pack
    installs on core's resolver produces the same verdict as this probe's own
    independent hook — the probe's hook is installed first, so the pack's
    observer wraps it and both see every call."""
    from ComfyUI_VibeVoice.modules.comfy_stream import (
        _VBAR_SPLIT, pull_stats_line,
    )

    return dict(_VBAR_SPLIT), pull_stats_line()


def _reset():
    for key in STATS:
        STATS[key] = 0


def _mb(value) -> str:
    return f"{value / (1024 ** 2):10.1f}MB"


# ---------------------------------------------------------------- tree


class Net(torch.nn.Module):
    """A flat stack of Linears: the shape core's vbar path is built for.

    No bias and no norm so every module clears core's 16 KB force-load
    threshold (``comfy/model_patcher.py:1975-1989``) and therefore takes the
    vbar.alloc branch — the branch under test.
    """

    def __init__(self, layers: int):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(WIDTH, WIDTH, bias=False, dtype=DTYPE)
             for _ in range(layers)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Handler(torch.nn.Module):
    """Stand-in for the pack's model handler (an nn.Module owning the model,
    which is how core's ``model_size()`` and ``load_models_gpu`` see it)."""

    def __init__(self, model, size):
        super().__init__()
        self.model = model
        self.processor = None
        self.size = size
        self.model_pack_name = "probe-vbar"
        self.attention_mode = "eager"
        self.cache_key = "probe-vbar"

    def load_model(self, device, attention_mode="eager"):
        pass


def _write_checkpoint(layers: int, path: str) -> None:
    import safetensors.torch as st

    tensors = {
        f"blocks.{i}.weight": torch.randn(WIDTH, WIDTH, dtype=DTYPE)
        for i in range(layers)
    }
    st.save_file(tensors, path)


def _build(path: str):
    """The real loader route: aimdo views -> assign -> streaming convert."""
    from ComfyUI_VibeVoice.modules.comfy_stream import convert_tree_for_streaming
    from ComfyUI_VibeVoice.modules.external_loader import (
        _load_state_dict_into_model_from_memory,
    )

    state_dict = comfy.utils.load_torch_file(path)
    with torch.device("meta"):
        model = Net(len(state_dict))
    loaded = _load_state_dict_into_model_from_memory(
        model, state_dict, preserve_file_views=True,
    )
    convert_tree_for_streaming(loaded)
    return loaded


# ---------------------------------------------------------------- arms


def _arm(label: str, layers: int, budget_mb: int, steps: int,
         churn_mb: int = 0, ballast_mb: int = 0, force_fast_disk=None):
    tmp = tempfile.mkdtemp(prefix="vibevoice_vbar_")
    path = os.path.join(tmp, "probe.safetensors")
    _write_checkpoint(layers, path)

    loaded = _build(path)
    size_bytes = layers * WIDTH * WIDTH * 2

    from ComfyUI_VibeVoice.modules.patcher import (
        load_to_device, select_patcher_class,
    )

    cls = select_patcher_class(None, torch.device("cuda"))

    patcher = cls(
        Handler(loaded, size_bytes),
        load_device=comfy.model_management.get_torch_device(),
        offload_device=torch.device("cpu"),
        size=size_bytes,
        dtype=DTYPE,
    )
    if force_fast_disk is not None:
        # Set after construction so the stated policy WINS over the
        # checkpoint-derived one -- that is what _adopt_storage_policy's
        # _fast_disk_explicit guard means, and the comparison needs it.
        patcher._fast_disk_explicit = True
        patcher.fast_disk = bool(force_fast_disk)

    print(f"\n=== ARM {label} ===")
    print(f"  tree={layers} x {PER_LINEAR_MB:.1f}MB = "
          f"{_mb(size_bytes).strip()}   vbar budget={budget_mb}MB   "
          f"churn={churn_mb}MB   ballast={ballast_mb}MB   "
          f"force_fast_disk={force_fast_disk}")

    _reset()
    load_to_device(patcher)
    load_stats = dict(STATS)
    from ComfyUI_VibeVoice.modules.patcher import resolve_fast_disk
    adopted = resolve_fast_disk(loaded)
    print(f"  patcher.fast_disk = {patcher.fast_disk}   "
          f"resolve_fast_disk(model) = {adopted}")
    print(f"  load      resolve={load_stats['calls']} "
          f"faulted={load_stats['faulted']} "
          f"xfer={_mb(load_stats['xfer_bytes']).strip()}")

    device = comfy.model_management.get_torch_device()
    x = torch.randn(1, 16, WIDTH, dtype=DTYPE, device=device)

    ballast = None
    if ballast_mb:
        # Simulate the live condition: another resident model eating VRAM, so
        # the arena competes for what is genuinely left. Held for the whole
        # arm — this is the difference between "the arena budget parameter is
        # small" (arm D, which is NOT a cap) and "there is no room left".
        ballast = torch.empty(int(ballast_mb * 1024 ** 2 // 2),
                              dtype=DTYPE, device=device)
        ballast.fill_(1.0)
        torch.cuda.synchronize()
        free_now, _ = torch.cuda.mem_get_info()
        print(f"  ballast={ballast_mb}MB -> vram_free="
              f"{free_now / 1024 ** 3:.1f}GB")

    # First forward: expected to fault (nothing was ever resolved yet).
    _reset()
    loaded(x)
    torch.cuda.synchronize()
    warm = dict(STATS)
    print(f"  forward-0 resolve={warm['calls']} faulted={warm['faulted']}")

    _reset()
    from ComfyUI_VibeVoice.modules.comfy_stream import reset_pull_stats
    from modules.memory_census import memory_snapshot

    reset_pull_stats()
    host_before = memory_snapshot()
    t0 = time.time()
    for _ in range(steps):
        loaded(x)
        if churn_mb:
            # What the speech-diffusion decoder does between AR steps: a big
            # short-lived allocation through the same allocator.
            scratch = torch.empty(int(churn_mb * 1024 ** 2 // 2),
                                  dtype=DTYPE, device=device)
            scratch.fill_(1.0)
            del scratch
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    run = dict(STATS)
    host_after = memory_snapshot()
    # Pinned host staging under comfy.pinned_memory is PRIVATE commit that
    # outlives the run -- this is the "RAM fills and stays" signature.
    print(f"            host private {host_before['private'] / 1024**3:.2f}GB"
          f" -> {host_after['private'] / 1024**3:.2f}GB"
          f" (+{(host_after['private'] - host_before['private']) / 1024**3:.2f}GB)"
          f"  machine used {host_before['sys_used'] / 1024**3:.2f}GB"
          f" -> {host_after['sys_used'] / 1024**3:.2f}GB"
          f" (+{(host_after['sys_used'] - host_before['sys_used']) / 1024**3:.2f}GB)")

    fault_rate = run["faulted"] / max(1, run["calls"])
    print(f"  forwards  resolve={run['calls']} resident={run['resident']} "
          f"faulted={run['faulted']} ({fault_rate:.1%})")
    print(f"            xfer={_mb(run['xfer_bytes']).strip()} in "
          f"{run['xfer_calls']} calls, file_reads={run['file_reads']}")
    print(f"            {steps} steps in {elapsed:.2f}s "
          f"({elapsed / steps * 1000:.1f} ms/step)")

    verdict = ("STABLE" if run["faulted"] == 0
               else "RE-READS" if run["faulted"] == run["calls"]
               else "PARTIAL")
    print(f"  VERDICT {label}: {verdict}")

    split, line = _prod_split()
    print(f"  SHIPPED [vvpull] {line}")
    if split["resident_calls"] + split["reread_calls"]:
        agree = (split["reread_calls"] == run["faulted"]
                 and split["resident_calls"] == run["resident"])
        print(f"  observer agrees with probe hook: {agree}")
        if not agree:
            print("  !! observer/probe disagreement — the shipped counters "
                  "cannot be trusted")

    del x, patcher, loaded, ballast
    gc.collect()
    comfy.model_management.soft_empty_cache()
    return {"label": label, **run, "verdict": verdict,
            "fault_rate": fault_rate, "size_mb": size_bytes / 1024 ** 2}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-mb", type=int, default=8192,
                        help="per-device vbar arena main.py would ask for")
    parser.add_argument("--layers", type=int, default=96)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--churn-mb", type=int, default=0)
    parser.add_argument("--ballast-mb", type=int, default=0,
                        help="VRAM held by a stand-in 'other resident model'")
    parser.add_argument("--force-fast-disk", choices=["true", "false"],
                        default=None,
                        help="pin the patcher's storage policy; compares the "
                             "two core policies under identical pressure")
    parser.add_argument("--overload", action="store_true",
                        help="single arm that deliberately stages more than "
                             "the arena holds, to render the re-read counters")
    args = parser.parse_args()

    print("PROBE vbar residency across repeated forwards")
    free, total = torch.cuda.mem_get_info()
    print(f"  device: {torch.cuda.get_device_name(0)}  "
          f"free={free / 1024 ** 3:.1f}GB/{total / 1024 ** 3:.1f}GB  "
          f"budget={args.budget_mb}MB")
    if not _init_aimdo(args.budget_mb * 1024 ** 2):
        return 1
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    _install_hooks()

    results = []
    if args.overload:
        results.append(_arm("D overloaded", layers=args.layers,
                            budget_mb=args.budget_mb, steps=args.steps,
                            churn_mb=args.churn_mb,
                            ballast_mb=args.ballast_mb,
                            force_fast_disk=(None if args.force_fast_disk
                                             is None
                                             else args.force_fast_disk == "true")))
    else:
        results.append(_arm("A small/no-churn", layers=8,
                            budget_mb=args.budget_mb, steps=20))
        results.append(_arm("B big/no-churn", layers=args.layers,
                            budget_mb=args.budget_mb, steps=args.steps))
        results.append(_arm("C big/churn", layers=args.layers,
                            budget_mb=args.budget_mb, steps=args.steps,
                            churn_mb=2048))

    print("\nVERDICT TABLE")
    print(f"  {'arm':<20}{'faulted/calls':>18}{'fault%':>9}"
          f"{'xfer MB':>12}{'verdict':>12}")
    for r in results:
        print(f"  {r['label']:<20}{r['faulted']:>10}/{r['calls']:<7}"
              f"{r['fault_rate']:>8.1%}{r['xfer_bytes'] / 1024 ** 2:>12.1f}"
              f"{r['verdict']:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())