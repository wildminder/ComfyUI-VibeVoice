r"""Standalone PROBE — is every weight route now ONE streaming assign?

The external agent's change made the FP8 route fast by streaming each tensor
straight onto the load device, but left the dense (BF16/FP16) route on the old
shape: build a full CPU state dict, assign it, then ``model.to(cuda)``. The
user still sees SSD traffic and RAM growth on BF16 (fix-3.txt diagnoses it as
the whole-tree ``.to()`` page-faulting the file mapping at ~0.55 GB/s).

This probe does NOT load a real checkpoint. It builds a ~2 MB synthetic
safetensors file and a matching tiny module, then checks the three properties
the unification has to hold:

  1. ``read_safetensors_tensors_by_name`` returns exact values for the scale
     keys, maps nothing, and works for every supported dtype incl. scalars.
  2. ``VibeVoiceLoader._stream_apply_dense(..., target_device=cuda)`` lands
     every parameter AND buffer on the GPU with bit-exact values.
  3. The same call with ``target_device=None`` still assigns file views
     (the internal/official-model path's contract is unchanged).

Run (headless, no ComfyUI server, no model load):
    COMFYUI_ROOT='C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI' \
      "C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe" \
      tests/probe_unified_dense_route.py
"""

import importlib.util
import os
import sys
import tempfile
from unittest.mock import MagicMock

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get(
    "COMFYUI_ROOT", r"C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI"))
sys.path.insert(0, ROOT_DIR)

# The package imports itself as ``ComfyUI_VibeVoice``; with the alias
# registered, ``modules.*`` resolve to the same objects the shipped code uses.
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

import comfy.utils  # noqa: E402

from ComfyUI_VibeVoice.modules.base_loader import iter_safetensors_tensors  # noqa: E402
from ComfyUI_VibeVoice.modules.external_loader import (  # noqa: E402
    read_safetensors_tensors_by_name,
)
from ComfyUI_VibeVoice.modules.loader import VibeVoiceLoader  # noqa: E402

FAILURES = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


class Tiny(torch.nn.Module):
    """Deliberately mixed: params, a buffer, bf16 + fp32, and a 0-dim scale."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(64, 32, bias=True)
        self.b = torch.nn.Linear(32, 8, bias=False)
        self.register_buffer("scale", torch.tensor(0.25, dtype=torch.float32))


def build_fixture(tmpdir):
    """Two files: a clean dense one, and a quant-shaped one carrying scales."""
    torch.manual_seed(0)
    model = Tiny()
    tensors = {k: v.detach().clone() for k, v in model.state_dict().items()}
    from safetensors.torch import save_file

    dense_path = os.path.join(tmpdir, "tiny-dense.safetensors")
    save_file(tensors, dense_path)

    # Stand-ins for the FP8 checkpoint's per-tensor scales, in a bf16 weight file
    # so the reader's dtype table is exercised on every entry it claims.
    quant = dict(tensors)
    quant["a.weight"] = tensors["a.weight"].to(torch.bfloat16)
    quant["a.weight_scale"] = torch.tensor(0.03125, dtype=torch.float32)
    quant["b.weight_scale"] = torch.tensor(0.125, dtype=torch.float16)
    quant_path = os.path.join(tmpdir, "tiny-quant.safetensors")
    save_file(quant, quant_path)
    return dense_path, quant_path, tensors, quant


def arm_1_scale_reader(path, tensors):
    print("\n1. header-range scale reader (no mapping, no full read)")
    wanted = ["a.weight_scale", "b.weight_scale", "a.weight", "not.a.key"]
    got = read_safetensors_tensors_by_name(path, wanted)

    check("absent keys are simply not returned", "not.a.key" not in got)
    check("only the requested keys come back", set(got) == set(wanted) - {"not.a.key"},
          f"got {sorted(got)}")
    for key in ("a.weight_scale", "b.weight_scale", "a.weight"):
        check(f"{key} is bit-exact", torch.equal(got[key], tensors[key]))
        check(f"{key} keeps its dtype", got[key].dtype == tensors[key].dtype,
              f"{got[key].dtype} vs {tensors[key].dtype}")
    check("0-dim scale keeps its shape", got["a.weight_scale"].shape == torch.Size([]))
    check("bf16 weight is read as bf16", got["a.weight"].dtype == torch.bfloat16)
    check("fp16 scale is read as fp16", got["b.weight_scale"].dtype == torch.float16)
    check("empty request is a no-op", read_safetensors_tensors_by_name(path, []) == {})


def arm_2_streaming_to_device(path, tensors):
    print("\n2. streaming dense assign -> CUDA (the unified production route)")
    if not torch.cuda.is_available():
        check("CUDA available", False, "skipped — no GPU in this environment")
        return
    dev = torch.device("cuda", 0)
    model = Tiny()
    before = torch.cuda.memory_allocated(dev)

    missing, unexpected = VibeVoiceLoader._stream_apply_dense(
        model, iter_safetensors_tensors(path), target_device=dev
    )
    check("no missing keys", list(missing) == [], f"{list(missing)[:5]}")
    check("no unexpected keys", list(unexpected) == [], f"{list(unexpected)[:5]}")

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    on_cuda = [n for n, p in list(params.items()) + list(buffers.items())
               if p.device.type != "cuda"]
    check("every parameter and buffer landed on cuda", not on_cuda, f"cpu: {on_cuda}")
    check("no meta left behind", not any(p.is_meta for p in model.parameters()))

    for key, want in tensors.items():
        got = params.get(key, buffers.get(key))
        check(f"{key} is bit-exact on gpu", got is not None and torch.equal(got.cpu(), want))

    grew = torch.cuda.memory_allocated(dev) - before
    check("weights really occupy VRAM", grew > 0, f"+{grew / 1024:.0f} KiB")


def arm_3_streaming_to_cpu(path, tensors):
    print("\n3. streaming dense assign with no device (internal loader contract)")
    model = Tiny()
    missing, unexpected = VibeVoiceLoader._stream_apply_dense(
        model, iter_safetensors_tensors(path)
    )
    check("assigns on cpu when no device is given",
          all(p.device.type == "cpu" for p in model.parameters()))
    check("a.weight is bit-exact",
          torch.equal(model.a.weight.data, tensors["a.weight"]))
    check("no missing keys", list(missing) == [])
    check("no unexpected keys", list(unexpected) == [])


def arm_4_no_batch_route():
    print("\n4. the batch (CPU state dict) route is gone from the loaders")
    import inspect
    from ComfyUI_VibeVoice.modules import external_loader as EL

    for fn in (EL.load_external_vibevoice_model, EL.load_external_vibevoice_asr_model):
        src = inspect.getsource(fn)
        dense_at = src.index("_stream_apply_dense_safetensors(")
        rest = src[dense_at:]
        check(f"{fn.__name__}: no _load_weight_state_dict after the dense arm",
              "_load_weight_state_dict(" not in rest.split("cast_model_to_dtype_if_needed")[0])
    check("comfy.utils is still the only assign primitive used",
          "set_attr_param" in inspect.getsource(VibeVoiceLoader._stream_apply_dense))


def main():
    print("=" * 72)
    print("PROBE: unified dense weight route")
    print("=" * 72)
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    print(f"comfy root: {os.environ.get('COMFYUI_ROOT', '(default)')}")
    with tempfile.TemporaryDirectory() as tmp:
        dense_path, quant_path, tensors, quant = build_fixture(tmp)
        print(f"dense fixture: {os.path.basename(dense_path)} "
              f"({os.path.getsize(dense_path) / 1024:.0f} KiB, {len(tensors)} tensors)")
        print(f"quant fixture: {os.path.basename(quant_path)} "
              f"({os.path.getsize(quant_path) / 1024:.0f} KiB, {len(quant)} tensors)")
        arm_1_scale_reader(quant_path, quant)
        arm_2_streaming_to_device(dense_path, tensors)
        arm_3_streaming_to_cpu(dense_path, tensors)
    arm_4_no_batch_route()

    print("\n" + "=" * 72)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILED -> {FAILURES}")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
