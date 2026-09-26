"""Native ComfyUI lowvram streaming for foreign (transformers-style) trees.

Core's partial-load/partial-unload machinery only STREAMS modules that carry
``comfy_cast_weights`` (+ ``weight_function``/``bias_function`` hooks); foreign
modules are silently left on CPU by ``partially_unload`` and have no way back,
producing "Input type CUDA... weight type CPU..." crashes (see plan
2026-08-26-native-lowvram-streaming-transformers-tree.md §1).

This module makes our tree fluent in that protocol, mirroring the proven
ComfyUI-Raon-OpenTTS approach (native.py `_ComfyLinear/_ComfyEmbedding/
_ComfyConv1d`):

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
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Last device on which a streaming forward actually computed. Core streams
# leaf weights via a cast buffer and never moves the underlying Parameter, so
# parameter-derived `.device` queries lie under partial offload; the
# activation device seen here is the only reliable compute-device signal.
# Containers with direct params (see make_streaming_container) relocate those
# params to this device on attribute access.
_LAST_DEVICE: torch.device | None = None


def _current_compute_device() -> torch.device | None:
    # _LAST_DEVICE (the last activation device seen by a streaming forward)
    # is the freshest signal and stays correct for CPU/--cpu runs; every
    # model entry runs a leaf forward before the first direct-param read,
    # so it is set in time. ComfyUI's current device is the fallback for
    # reads that happen before any forward (standalone tokenizer calls).
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
def _acquire(module: nn.Module, x: torch.Tensor):
    """Yield ``(weight, bias)`` guaranteed on ``x.device``.

    Fast path: params already co-located and no streaming functions attached
    (nothing was offloaded) -> yield the module's own tensors untouched.
    Slow path: core's ``cast_bias_weight`` pulls offloaded weights back
    (honoring LowVramPatch weight_functions) and pins them for async offload.
    """
    global _LAST_DEVICE
    _LAST_DEVICE = x.device
    wf = getattr(module, "weight_function", None)
    bf = getattr(module, "bias_function", None)
    # NOTE: parameters live in module._parameters, not __dict__; getattr
    # resolves the Parameter (or None for elementwise_affine=False norms).
    w = getattr(module, "weight", None)
    b = getattr(module, "bias", None)
    coLocated = (
        (wf is None or len(wf) == 0)
        and (bf is None or len(bf) == 0)
        and (not isinstance(w, torch.Tensor) or w.device == x.device)
        and (not isinstance(b, torch.Tensor) or b.device == x.device)
    )
    if coLocated:
        yield w, b
        return

    import comfy.ops

    weight, bias, stream = comfy.ops.cast_bias_weight(
        module, x, offloadable=True
    )
    try:
        yield weight, bias
    finally:
        comfy.ops.uncast_bias_weight(module, weight, bias, stream)


# ====================================================================
# Per-kind compute functions (signature: (self, x, weight, bias) -> Tensor)
# ====================================================================


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
    return self._conv_forward(x, w, b)


def _compute_conv_transpose1d(self, x, w, b):
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
    # Vendored ConvLayerNorm: channels-last transpose + fp32 layer_norm.
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
    # Vendored ConvRMSNorm fallback path (APEX fusion deliberately unused:
    # deterministic math, identical results).
    x = x.transpose(1, 2)
    out = _rmsnorm_math(self, x, w)
    return out.transpose(1, 2)


def _compute_qwen2rmsnorm(self, x, w, b):
    # transformers Qwen2RMSNorm: fp32 stats, cast back, THEN scale.
    # Older transformers name the epsilon `variance_epsilon`.
    eps = getattr(self, "eps", None)
    if eps is None:
        eps = self.variance_epsilon
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return w * x.to(input_dtype)


# Base kinds covered without any vendored imports.
_BUILTIN_COMPUTE = {
    nn.Linear: _compute_linear,
    nn.Embedding: _compute_embedding,
    nn.Conv1d: _compute_conv1d,
    nn.ConvTranspose1d: _compute_conv_transpose1d,
    nn.LayerNorm: _compute_layernorm,
}

# Registered lazily by callers for vendored/external classes.
_EXTRA_COMPUTE: dict = {}


def register_streaming_type(base_cls, compute_fn) -> None:
    """Teach the converter how to stream an additional module class."""
    _EXTRA_COMPUTE[base_cls] = compute_fn


def _resolve_compute(base_cls):
    """Exact registered match first, then builtin-kind isinstance fallback
    (covers subclasses such as vendored or HF Linear variants)."""
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
    # Core's `_load_list`/`check_module_offload_mem` does `getattr(op,
    # "weight")` (and `"bias"`) on EVERY module flagged `comfy_cast_weights`,
    # assuming Linear-like ops expose those names. Modules without a native
    # `weight`/`bias` (norms, containers, embeddings) would raise
    # AttributeError and abort placement. Route through the real
    # `_parameters` store so the actual Parameter is returned when present
    # and ``None`` otherwise. A plain class attribute would shadow a real
    # Parameter on biased modules, so a property is required.
    return self._parameters.get("weight")


def _streaming_bias(self):
    return self._parameters.get("bias")


def make_streaming(base_cls, compute_fn):
    """Build (once per base class) a streaming subclass of ``base_cls``.

    Used for leaf ops (Linear/Conv/Embedding/norm/Layernorm) whose compute
    is a single ``(self, x, weight, bias) -> Tensor`` function. The forward
    acquires weights through core's cast path (so they can be streamed), and
    the subclass exposes ``weight``/``bias`` so core's placement bookkeeping
    (``get_key_weight``) never raises on biasless modules.
    """
    key = ("leaf", base_cls)
    if key in _SUBCLASS_CACHE:
        return _SUBCLASS_CACHE[key]

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
    """Build a streaming subclass for a module that owns DIRECT parameters
    (e.g. layer-scale ``gamma`` in ``Block1D``) but no registered leaf
    compute.

    Core only streams params named ``weight``/``bias``; modules with nested
    params are excluded from the load list entirely (``_load_list`` marks
    them ``default=True``), so arbitrary direct parameters on a foreign
    module are NEVER placed by core and strand wherever state-dict loading
    left them (CPU), raising "Expected all tensors to be on the same device"
    mid-inference. Marking the module ``comfy_cast_weights`` keeps core's
    bookkeeping (``get_key_weight``) happy via the ``weight``/``bias``
    properties, and the class's ``__getattr__`` relocates direct params to
    the current compute device ON ACCESS. Access-time relocation is required
    because the tokenizer's ``forward_features`` loop reads ``block.gamma``
    directly WITHOUT invoking ``block.forward`` - so no forward hook can see
    the read. The compute device comes from :data:`_LAST_DEVICE` (the last
    activation device seen by any streaming forward); parameter-derived
    ``.device`` queries cannot be used because core streams leaf weights via
    cast buffers and never moves the underlying Parameters. The module's own
    ``forward`` also relocates (and refreshes ``_LAST_DEVICE``) as a
    belt-and-suspenders measure for callers that DO go through it.
    """
    key = ("container", base_cls)
    if key in _SUBCLASS_CACHE:
        return _SUBCLASS_CACHE[key]

    base_forward = base_cls.forward

    def forward(self, x, *args, **kwargs):
        # Transparent passthrough of extra args/kwargs: the wrapped module
        # keeps its base forward's full signature (the native ASR encoders
        # receive padding_cache=/use_cache=; a bare (self, x) wrapper raised
        # "unexpected keyword argument 'padding_cache'" there).
        global _LAST_DEVICE
        _LAST_DEVICE = x.device
        _relocate_direct_params(self, x.device)
        return base_forward(self, x, *args, **kwargs)

    def __getattr__(self, name):
        # nn.Module resolves direct params through its own __getattr__ from
        # the _parameters store; intercept that path so reading a param
        # directly (block.gamma) relocates it to the compute device first.
        # Use self.__dict__ to avoid recursing back into __getattr__ before
        # nn.Module.__init__ has populated the stores.
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


# Quant-resident modules stream natively (Phase 3) and are never rewrapped.
def _is_quant_resident(module) -> bool:
    return bool(getattr(module, "_quant_resident", False))


def convert_tree_for_streaming(root: nn.Module, skip=()) -> dict:
    """Swap eligible leaf-module classes for streaming subclasses in place.

    Args:
        root: Model tree to mutate (typically the heavy model inside a
            handler).
        skip: Module objects to leave untouched (e.g. handler shells).

    Returns:
        Census dict ``{kind_name: count}`` for the modules actually
        converted. Idempotent: already-streaming modules are skipped.
    """
    # Vendored/external kinds (lazy: avoids import cycles at module load).
    try:
        from .vendor_streaming_types import register_vendored_types

        register_vendored_types(register_streaming_type)
    except Exception as e:  # pragma: no cover - defensive
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
            continue  # already streaming (idempotent re-entry)
        if _is_quant_resident(module):
            continue  # native streaming implemented on the class itself
        compute = _resolve_compute(base)
        if compute is None:
            direct_params = list(
                name for name, _ in module.named_parameters(recurse=False)
            )
            if direct_params:
                if getattr(base, "forward", None) is not nn.Module.forward:
                    # Module owns direct params (e.g. layer-scale gammas)
                    # that core never places (nested-param modules are
                    # excluded from the load list). Convert it to a streaming
                    # container: its __getattr__ relocates each direct param
                    # to the compute device on access, so params read outside
                    # their own forward (tokenizer forward_features reads
                    # block.gamma directly) never strand on CPU.
                    module.__class__ = make_streaming_container(base)
                    kind = base.__name__
                    census[kind] = census.get(kind, 0) + 1
                else:
                    # No forward to delegate to -> cannot stream. Keep the
                    # legacy skip-and-warn so we don't inject a forward-time
                    # crash. Such modules are rare dead param holders whose
                    # params are placed via the parent.
                    unknown_with_params.append(f"{name} ({base.__name__})")
            # Else: a pure container with only child parameters - those
            # children are converted independently, so there is nothing to
            # relocate here and no stranding risk.
            continue
        module.__class__ = make_streaming(base, compute)
        kind = base.__name__
        census[kind] = census.get(kind, 0) + 1

    if unknown_with_params:
        logger.warning(
            "Streaming conversion skipped %d parameter-bearing module "
            "kind(s) without a registered streaming forward; they will not "
            "survive partial offload: %s",
            len(unknown_with_params), unknown_with_params[:8],
        )
    if census:
        logger.debug(
            "Streaming-enabled %d module(s): %s",
            sum(census.values()),
            ", ".join(f"{k}={v}" for k, v in sorted(census.items())),
        )
    return census
