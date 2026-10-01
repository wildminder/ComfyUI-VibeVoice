"""Native ComfyUI lowvram streaming for foreign (transformers-style) trees.

Core's partial-load/partial-unload machinery only STREAMS modules that carry
``comfy_cast_weights`` (+ ``weight_function``/``bias_function`` hooks); foreign
modules are silently left on CPU by ``partially_unload`` and have no way back,
producing "Input type CUDA... weight type CPU..." crashes (see plan
2026-08-26-native-lowvram-streaming-transformers-tree.md §1).

This module makes our tree fluent in that protocol. See
``docs/2026-08-26-native-lowvram-streaming-transformers-tree.md`` for the
design:

- ``make_streaming(cls, compute_fn)`` builds a subclass whose forward acquires
  weights through ``comfy.ops.cast_bias_weight(..., offloadable=True)``;
- ``convert_tree_for_streaming(root)`` swaps ``module.__class__`` IN PLACE
  (no tensor copies, meta-safe, idempotent);
- quant residents (GGUFLinear/ConvRotInt8Linear) implement streaming natively
  and pin their raw storage dtype via the core-supported
  ``<param>_comfy_model_dtype`` attribute instead of being wrapped here.

When everything resides on one device and no weight functions are attached,
the wrappers take a zero-overhead fast path (identical to the original
module), so fully-resident models pay nothing.
"""

from __future__ import annotations

import logging
import traceback
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F

from .diagnostics import diagnostics_enabled, vbar_observer_enabled

logger = logging.getLogger(__name__)

# Last device on which a streaming forward actually computed. Core streams
# leaf weights via a cast buffer and never moves the underlying Parameter, so
# parameter-derived `.device` queries lie under partial offload; the
# activation device seen here is the only reliable compute-device signal.
# Containers with direct params (see make_streaming_container) relocate those
# params to this device on attribute access.
_LAST_DEVICE: torch.device | None = None

# PULL STATS (2026-09-30). Per-generate counters for the weight pulls the
# streaming leaves perform, so a live run can say WHICH path served a weight
# and how many bytes moved disk/VRAM per step:
#   vbar_calls / vbar_bytes     — module had a `_v` vbar allocation: core's
#       file->VRAM paging read (measured cache-clean, tests/
#       probe_core_slice_read_cost.py arm 1).
#   nonvbar_calls / nonvbar_bytes — no `_v`: core's cast-buffer path, which
#       falls back to a HOST-side copy of a file view (arm 3: 1:1 RAM at
#       0.55 GB/s) whenever read_tensor_file_slice_into declines.
#   colocated_calls             — fast path, weight already beside the input.
# Per-kind breakdown keyed by module class name, because "the embedding is
# re-read 300 times" and "every linear is re-read" are different bugs.
_PULL_STATS: dict = {}

#: Residency split for vbar-served pulls. Every pull "serves" a weight either
#: way, so the byte totals above cannot express the one distinction that
#: decides whether inference is fast: a RESIDENT pull returns ``s._v_weight``,
#: a pointer into the VRAM arena (comfy/ops.py:274-276), while a RE-READ runs
#: the transfer branch — and because the module's parameter still aliases the
#: aimdo file slice, that transfer is disk traffic (comfy/ops.py:181-237).
_VBAR_SPLIT = {"resident_calls": 0, "reread_calls": 0, "reread_bytes": 0}
_VBAR_OBSERVER: object = None


def reset_pull_stats() -> None:
    """Clear the counters before a measured run (one generate, one report)."""
    _PULL_STATS.clear()
    for key in _VBAR_SPLIT:
        _VBAR_SPLIT[key] = 0


def pull_stats_line() -> str:
    """One paste-ready ``[vvpull]`` summary of the counters since the last reset."""
    if not _PULL_STATS:
        return "[vvpull] no streaming leaf pulls recorded"
    parts = []
    for key in sorted(_PULL_STATS):
        stats = _PULL_STATS[key]
        mb = stats["bytes"] / (1024 ** 2)
        parts.append(f"{key}={stats['calls']}({mb:.0f}MB)")
    line = "[vvpull] " + " ".join(parts)
    total = _VBAR_SPLIT["resident_calls"] + _VBAR_SPLIT["reread_calls"]
    if total:
        pct = 100.0 * _VBAR_SPLIT["reread_calls"] / total
        line += (
            f" | resident={_VBAR_SPLIT['resident_calls']}"
            f" reread={_VBAR_SPLIT['reread_calls']}({pct:.1f}%"
            f",{_VBAR_SPLIT['reread_bytes'] / (1024 ** 2):.0f}MB)"
        )
    try:
        import torch as _torch

        if _torch.cuda.is_available():
            free, total_bytes = _torch.cuda.mem_get_info()
            line += f" | vram_free={free / (1024 ** 3):.1f}GB"
    except Exception:
        pass
    return line


def _record_pull(kind: str, path: str, nbytes: int) -> None:
    key = f"{path}/{kind}"
    entry = _PULL_STATS.get(key)
    if entry is None:
        entry = _PULL_STATS[key] = {"calls": 0, "bytes": 0}
    entry["calls"] += 1
    entry["bytes"] += nbytes


def _install_vbar_observer() -> None:
    """Observe — never alter — the residency verdict core already computed."""
    global _VBAR_OBSERVER
    if _VBAR_OBSERVER is not None:
        return

    if not vbar_observer_enabled():
        _VBAR_OBSERVER = False
        return
    try:
        import comfy.ops

        original = comfy.ops.resolve_cast_module_with_vbar
    except Exception:
        return
    if getattr(original, "_vv_observer", False):
        _VBAR_OBSERVER = original
        return
    if _VBAR_OBSERVER is False:
        return

    def observed(s, *args, **kwargs):
        prefetch = getattr(s, "_prefetch", None)
        if prefetch is not None:
            if prefetch.get("resident"):
                _VBAR_SPLIT["resident_calls"] += 1
            else:
                _VBAR_SPLIT["reread_calls"] += 1
                weight = getattr(s, "weight", None)
                if isinstance(weight, torch.Tensor):
                    _VBAR_SPLIT["reread_bytes"] += (
                        weight.numel() * weight.element_size())
        return original(s, *args, **kwargs)

    observed._vv_observer = True
    observed._vv_original = original
    comfy.ops.resolve_cast_module_with_vbar = observed
    _VBAR_OBSERVER = observed


def _current_compute_device() -> torch.device | None:
    if _LAST_DEVICE is not None:
        return _LAST_DEVICE
    try:
        import comfy.model_management as mm

        return mm.get_torch_device()
    except Exception:
        return None


# ====================================================================
# Cast plumbing
# ====================================================================


@contextmanager
def _acquire(module: nn.Module, x: torch.Tensor, dtype_from_input: bool = True):
    """Yield ``(weight, bias)`` guaranteed on ``x.device``."""
    global _LAST_DEVICE
    _LAST_DEVICE = x.device
    wf = getattr(module, "weight_function", None)
    bf = getattr(module, "bias_function", None)
    w = getattr(module, "weight", None)
    b = getattr(module, "bias", None)
    coLocated = (
        (wf is None or len(wf) == 0)
        and (bf is None or len(bf) == 0)
        and (not isinstance(w, torch.Tensor) or w.device == x.device)
        and (not isinstance(b, torch.Tensor) or b.device == x.device)
    )
    if coLocated:
        _record_pull(type(module).__name__, "colocated", 0)
        yield w, b
        return

    import comfy.ops

    if dtype_from_input:
        weight, bias, stream = comfy.ops.cast_bias_weight(
            module, x, offloadable=True
        )
    else:
        weight, bias, stream = comfy.ops.cast_bias_weight(
            module, None, device=x.device, offloadable=True
        )
    served = weight.numel() * weight.element_size() if isinstance(weight, torch.Tensor) else 0
    _record_pull(
        type(module).__name__,
        "vbar" if hasattr(module, "_v") else "nonvbar",
        served,
    )
    try:
        yield weight, bias
    finally:
        comfy.ops.uncast_bias_weight(module, weight, bias, stream)


# ====================================================================
# Per-kind compute functions (signature: (self, x, weight, bias) -> Tensor)
# ====================================================================


_DTYPE_MISMATCH_LOGGED: set = set()


def _log_conv_dtype_mismatch(module: nn.Module, x: torch.Tensor, w, b) -> None:
    """One-shot diagnostic for dtype-mismatched conv inputs."""
    if not diagnostics_enabled():
        return
    key = (id(module), str(x.dtype))
    if key in _DTYPE_MISMATCH_LOGGED:
        return
    _DTYPE_MISMATCH_LOGGED.add(key)
    chain = " <- ".join(
        f"{fr.filename.replace(chr(92), '/').rsplit('/', 1)[-1]}:{fr.lineno} {fr.name}"
        for fr in reversed(traceback.extract_stack()[-15:-1])
    )
    logger.warning(
        "[vvdtype] conv dtype mismatch on %s(id=%x, layer_id=%s): x %s on %s "
        "dtype=%s | w %s dtype=%s | b %s dtype=%s | call chain (innermost last): %s",
        type(module).__name__, id(module) & 0xFFFFFF,
        getattr(module, "_layer_id", None),
        tuple(x.shape), x.device, x.dtype,
        tuple(w.shape) if w is not None else None,
        None if w is None else w.dtype,
        None if b is None else "-",
        None if b is None else b.dtype,
        chain,
    )


def _compute_linear(self, x, w, b):
    return F.linear(x, w, b)


def _compute_embedding(self, x, w, b):
    return F.embedding(
        x,
        w,
        padding_idx=getattr(self, "padding_idx", None),
        max_norm=getattr(self, "max_norm", None),
        norm_type=getattr(self, "norm_type", 2),
        scale_grad_by_freq=getattr(self, "scale_grad_by_freq", False),
        sparse=getattr(self, "sparse", False),
    )


def _compute_conv1d(self, x, w, b):
    if x.dtype != w.dtype:
        _log_conv_dtype_mismatch(self, x, w, b)
    return self._conv_forward(x, w, b)


def _compute_conv_transpose1d(self, x, w, b):
    if x.dtype != w.dtype:
        _log_conv_dtype_mismatch(self, x, w, b)
    return F.conv_transpose1d(
        x, w, b,
        stride=self.stride,
        padding=self.padding,
        output_padding=getattr(self, "output_padding", 0),
        dilation=self.dilation,
        groups=self.groups,
    )


def _compute_layernorm(self, x, w, b):
    return F.layer_norm(
        x, self.normalized_shape, w, b,
        getattr(self, "eps", 1e-5),
    )


def _compute_convlayernorm(self, x, w, b):
    x = x.transpose(1, 2)
    x = F.layer_norm(
        x.float(), self.normalized_shape,
        w.float() if w is not None else None,
        b.float() if b is not None else None,
        self.eps,
    ).type_as(x)
    return x.transpose(1, 2)


def _rmsnorm_math(self, x, w):
    out = self._norm(x.float()).type_as(x)
    if w is not None:
        out = out * w
    return out


def _compute_rmsnorm(self, x, w, b):
    return _rmsnorm_math(self, x, w)


def _compute_convrmsnorm(self, x, w, b):
    x = x.transpose(1, 2)
    out = _rmsnorm_math(self, x, w)
    return out.transpose(1, 2)


def _compute_qwen2rmsnorm(self, x, w, b):
    eps = getattr(self, "eps", None)
    if eps is None:
        eps = self.variance_epsilon
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return w * x.to(input_dtype)


_BUILTIN_COMPUTE = {
    nn.Linear: _compute_linear,
    nn.Embedding: _compute_embedding,
    nn.Conv1d: _compute_conv1d,
    nn.ConvTranspose1d: _compute_conv_transpose1d,
    nn.LayerNorm: _compute_layernorm,
}

_EXTRA_COMPUTE: dict = {}


def register_streaming_type(base_cls, compute_fn) -> None:
    """Teach the converter how to stream an additional module class."""
    _EXTRA_COMPUTE[base_cls] = compute_fn


def _resolve_compute(base_cls):
    if base_cls in _EXTRA_COMPUTE:
        return _EXTRA_COMPUTE[base_cls]
    for kind in _BUILTIN_COMPUTE:
        try:
            if issubclass(base_cls, kind):
                return _BUILTIN_COMPUTE[kind]
        except TypeError:
            continue
    return None


_SUBCLASS_CACHE: dict = {}


def _streaming_weight(self):
    return self._parameters.get("weight")


def _streaming_bias(self):
    return self._parameters.get("bias")


def make_streaming(base_cls, compute_fn, index_input: bool = False):
    """Build (once per base class) a streaming subclass of ``base_cls``."""
    index_input = index_input or issubclass(base_cls, nn.Embedding)
    key = ("leaf", base_cls, index_input)
    if key in _SUBCLASS_CACHE:
        return _SUBCLASS_CACHE[key]

    if index_input:
        def forward(self, x):
            with _acquire(self, x, dtype_from_input=False) as (w, b):
                return compute_fn(self, x, w, b)
    else:
        def forward(self, x):
            with _acquire(self, x) as (w, b):
                return compute_fn(self, x, w, b)

    namespace = {
        "comfy_cast_weights": True,
        "weight_function": [],
        "bias_function": [],
        "_streaming_compute": staticmethod(compute_fn),
        "forward": forward,
        "weight": property(_streaming_weight),
        "bias": property(_streaming_bias),
        "__module__": base_cls.__module__,
    }
    streaming = type(f"_ComfyStream{base_cls.__name__}", (base_cls,), namespace)
    _SUBCLASS_CACHE[key] = streaming
    return streaming


def make_streaming_container(base_cls):
    """Build a streaming subclass for a module that owns DIRECT parameters."""
    key = ("container", base_cls)
    if key in _SUBCLASS_CACHE:
        return _SUBCLASS_CACHE[key]

    base_forward = base_cls.forward

    def forward(self, x, *args, **kwargs):
        global _LAST_DEVICE
        _LAST_DEVICE = x.device
        _relocate_direct_params(self, x.device)
        return base_forward(self, x, *args, **kwargs)

    def __getattr__(self, name):
        params = self.__dict__.get("_parameters")
        if params is not None and name in params:
            p = params[name]
            if isinstance(p, torch.Tensor) and not p.is_meta:
                dev = _current_compute_device()
                if dev is not None and p.device != dev:
                    p.data = p.data.to(dev)
            return p
        return nn.Module.__getattr__(self, name)

    namespace = {
        "comfy_cast_weights": True,
        "weight_function": [],
        "bias_function": [],
        "forward": forward,
        "__getattr__": __getattr__,
        "weight": property(_streaming_weight),
        "bias": property(_streaming_bias),
        "__module__": base_cls.__module__,
    }
    streaming = type(f"_ComfyStream{base_cls.__name__}", (base_cls,), namespace)
    _SUBCLASS_CACHE[key] = streaming
    return streaming


def _relocate_direct_params(module, dev):
    for name, p in module.named_parameters(recurse=False):
        if p.device != dev and not p.is_meta:
            p.data = p.data.to(dev)


def _is_quant_resident(module) -> bool:
    return bool(getattr(module, "_quant_resident", False))


# Gate for ``convert_tree_for_streaming``. Default is False so models that fit
# in VRAM run at 100% native PyTorch GPU speed without per-op streaming interceptors
# or Aimdo VBAR paging overhead.
_STREAMING_CONVERSION_ENABLED = False


@contextmanager
def streaming_conversion(enabled: bool):
    """Run the enclosed block with streaming conversion on or off."""
    global _STREAMING_CONVERSION_ENABLED
    previous = _STREAMING_CONVERSION_ENABLED
    _STREAMING_CONVERSION_ENABLED = enabled
    try:
        yield
    finally:
        _STREAMING_CONVERSION_ENABLED = previous


def convert_tree_for_streaming(root: nn.Module, skip=()) -> dict:
    """Swap eligible leaf-module classes for streaming subclasses in place."""
    if not _STREAMING_CONVERSION_ENABLED:
        logger.debug(
            "Streaming conversion bypassed: model will execute with native "
            "resident PyTorch CUDA tensors."
        )
        return {}

    _install_vbar_observer()

    try:
        from .vendor_streaming_types import register_vendored_types

        register_vendored_types(register_streaming_type)
    except Exception as e:
        logger.debug(f"vendored streaming types unavailable: {e}")

    skip_set = [root]
    skip_set.extend(skip)

    census: dict = {}
    unknown_with_params = []
    for name, module in root.named_modules():
        if module is root or any(module is s for s in skip_set):
            continue
        base = type(module)
        if getattr(module, "comfy_cast_weights", False):
            continue
        if _is_quant_resident(module):
            continue
        compute = _resolve_compute(base)
        if compute is None:
            direct_params = list(
                name for name, _ in module.named_parameters(recurse=False)
            )
            if direct_params:
                if getattr(base, "forward", None) is not nn.Module.forward:
                    module.__class__ = make_streaming_container(base)
                    kind = base.__name__
                    census[kind] = census.get(kind, 0) + 1
                else:
                    unknown_with_params.append(f"{name} ({base.__name__})")
            continue
        module.__class__ = make_streaming(base, compute)
        kind = base.__name__
        census[kind] = census.get(kind, 0) + 1

    if unknown_with_params:
        logger.warning(
            "Streaming conversion skipped %d parameter-bearing module "
            "kind(s) without a registered streaming forward: %s",
            len(unknown_with_params), unknown_with_params[:8],
        )
    if census:
        logger.debug(
            "Streaming-enabled %d module(s): %s",
            sum(census.values()),
            ", ".join(f"{k}={v}" for k, v in sorted(census.items())),
        )
    return census