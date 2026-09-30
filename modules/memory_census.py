"""Host-RAM census for a loaded VibeVoice model.

MEASURE FIRST. Two RAM defects were reported on external single-file loads
(docs/2026-09-29-dynamic-vram-port-and-int64-conv-regression.md, "UPDATE 4"):
the 1.5B bf16 file leaves ~7GB resident, and the 7B fp8 file spikes ~17GB
transiently. Both were explained by guesswork until this module existed. Every
hypothesis about where host bytes live is now a number instead: this walks the
model's storages once and splits their bytes into the four states that matter,
then prints exactly one ``[vvcensus]`` line.

The four states, straight from how ComfyUI hands out weights:

* **view** — the storage carries ``_comfy_tensor_file_slice``
  (``comfy/utils.py:145-150``), i.e. aimdo's ``ModelMMAP`` file view. These
  bytes are page-cache backed: they cost working set, not private RAM, and
  core re-reads them disk->VRAM at every forward
  (``comfy/memory_management.py:57-63``).
* **mmap** — the storage carries ``_comfy_tensor_mmap_refs`` but no slice tag
  (a raw ``torch.load(mmap=True)`` mapping, or a view whose slice attribute
  was severed). Mapped, not private.
* **private** — a real HOST allocation: a clone, a dequant, a meta-straggler
  materialisation, or the raw safetensors read that the iterator's
  aimdo-off fallback performs (the aimdo arm yields zero-copy file views).
  This is the bucket the two defects
  actually live in, and the only one a fix should move. Note that it is a
  bucket and not a claim about the process: on this host the PROCESS counters
  do not agree with it for every read shape (see the section header further
  down), which is why the census walks storages instead of reading RSS.
* **offhost** — the parameter/buffer now lives on an accelerator. Not host RAM
  at all, and a separate bucket for a concrete reason: core force-loads every
  module under 16KB (``comfy/model_patcher.py:1975-1989``) to keep the stream
  buffer from going lopsided, so a small ``LayerNorm`` legitimately ends up as
  a device copy after ``load_to_device`` while its file view is stashed in
  ``patcher.backup``. Calling that "private host bytes" would blame the host
  for VRAM.

Plus the patcher's own host-side stash — ``ModelPatcherDynamic.load`` keeps
``self.backup`` for every module it did not take the vbar branch for
(comfy/model_patcher.py:2000-2009) and ``self.backup_buffers`` for every
buffer (:2010-2016) — and the vbar ranges allocated at :1993, so a reader can
tell "weights are paged" from "weights are stashed". A stashed VIEW costs no
new RAM (the mapping was already counted); a stashed PRIVATE tensor is a live
host copy, which is why the stash is split the same way.

Design notes:

* :func:`census` is PURE: no logging, no env reads, no mutation. It takes a
  model and an optional patcher and returns byte totals. Every call site logs
  it through :func:`report_census`, the only place that touches the environment
  or the logger.
* :func:`census` is an INVENTORY of the storages the model still HOLDS, and
  an inventory cannot see a transient. The reported 7B fp8 spike (+17GB) is
  gone by the time the post-H2D census runs, which is exactly why a post-hoc
  RSS delta is not a measurement of a spike. :class:`RssSampler` samples the
  PROCESS for the whole load and reports the PEAK of three distinct
  quantities, so "which storages" and "how many bytes at the worst instant"
  are answered by two different instruments.
* It is total, not partial: a ``MagicMock`` model, a half-built module, a meta
  tensor or a hostile ``__getattr__`` yields zeros rather than an exception.
  Instrumentation must never be the thing that breaks a load.
* Storages are de-duplicated by their ``c10::Storage`` identity
  (``storage._cdata``), so tied weights, views of a view and — importantly for
  a structural census — the all-zero-pointer storages of a meta model are each
  counted exactly once. The patcher stash is walked with its OWN de-dup set:
  a stashed tensor is a separate lifetime from the live parameter and must be
  reported even when the two share a storage.

Gated on the ``VIBEVOICE_RAM_CENSUS`` environment variable; default ON (one
line per load is acceptable noise, and the whole point is to have the number
when a user reports a spike). Set it to ``0``/``false``/``no``/``off`` to
silence it.
"""

import contextlib
import ctypes
import gc
import logging
import os
import time
import threading

import torch

logger = logging.getLogger(__name__)

#: Storage attribute aimdo's file views carry (comfy/utils.py:145-150).
_FILE_SLICE_ATTR = "_comfy_tensor_file_slice"
#: Storage attribute carrying the ``(ModelMMAP, memoryview)`` refs
#: (comfy/utils.py:149) — i.e. this storage is backed by the mapping.
_MMAP_REFS_ATTR = "_comfy_tensor_mmap_refs"

#: The byte states every group (params, buffers, patcher stash) is split into.
STATES = ("view", "mmap", "private", "offhost")

#: The four tensor populations a census reports, and the two that come from
#: the patcher rather than the model.
_MODEL_GROUPS = ("param", "buffer")
_STASH_GROUPS = (("backup", "backup"), ("backup_buffers", "backup_buffer"))

#: How many module families the single log line enumerates before it collapses
#: the tail into a count. The full rollup is always in the returned dict.
_MAX_RENDERED_FAMILIES = 8

#: Prefix ``make_streaming`` gives every class it synthesises
#: (modules/comfy_stream.py:335/:405). A census that reported
#: ``_ComfyStreamLinear`` would hide the very families a reader is looking for,
#: so the label is the original class name.
_STREAMING_PREFIX = "_ComfyStream"

_FALSEY_ENV = frozenset({"0", "false", "no", "off"})


class _Seen:
    """De-dup bookkeeping for one walk: storages counted, mappings pinned."""

    __slots__ = ("storages", "mmap_refs")

    def __init__(self, mmap_refs=None):
        self.storages = set()
        self.mmap_refs = set() if mmap_refs is None else mmap_refs


def census_enabled() -> bool:
    """True unless ``VIBEVOICE_RAM_CENSUS`` is explicitly switched off."""
    return os.environ.get("VIBEVOICE_RAM_CENSUS", "1").strip().lower() not in _FALSEY_ENV


def census(model, patcher=None) -> dict:
    """Byte census of ``model``'s parameters and buffers, plus ``patcher``'s stash.

    Pure and total: it never raises, never mutates, and returns zeros for
    anything it cannot introspect (a mock, a meta model, a parameterless
    container).

    Returns a dict with one group per tensor population — ``param``,
    ``buffer``, ``backup``, ``backup_buffer`` — where each group carries
    ``<group>_bytes``/``<group>_count`` totals and a ``<group>_<state>_bytes``
    / ``<group>_<state>_count`` split over :data:`STATES`, plus:

    ``families``
        ``{"<module class>/params": bytes, "<module class>/buffers": bytes}``.
        Splitting the two keeps a family's label honest: a family can be fully
        paged for its weights and still hold a private buffer.
    ``vbar_bytes`` / ``vbar_count``
        The vbar ranges core allocated for this model (:1993).
    ``mmap_refs_storages``
        Distinct storages holding ``_comfy_tensor_mmap_refs``, including those
        that also carry a slice tag — i.e. the live ``ModelMMAP`` references
        this model and the patcher stash are pinning between them.
    """
    report = {"families": {}}
    for group in _MODEL_GROUPS + tuple(g for _, g in _STASH_GROUPS):
        report[f"{group}_bytes"] = 0
        report[f"{group}_count"] = 0
        for state in STATES:
            report[f"{group}_{state}_bytes"] = 0
            report[f"{group}_{state}_count"] = 0
    report["vbar_bytes"] = 0
    report["vbar_count"] = 0

    seen = _Seen()
    for _module_name, module, family in _iter_modules(model):
        direct_params = _direct_tensors(module, "_parameters")
        direct_buffers = _direct_tensors(module, "_buffers")
        if not direct_params and not direct_buffers:
            continue
        _accumulate(report, direct_params, "param", family, seen)
        _accumulate(report, direct_buffers, "buffer", family, seen)
        _accumulate_vbar(report, module)

    _accumulate_stash(report, patcher, seen)
    report["mmap_refs_storages"] = len(seen.mmap_refs)
    return report


def report_census(model, patcher=None, phase: str = "") -> dict:
    """Log one ``[vvcensus]`` line for ``model`` and return the census.

    The single side-effecting entry point: it reads the env gate and, when
    enabled, emits exactly one ``logger.info`` line. Returns the report either
    way (callers and tests use it regardless of the gate) — the walk is cheap
    and side-effect free.
    """
    report = census(model, patcher)
    if census_enabled():
        logger.info(format_census(report, phase=phase))
    return report


def format_census(report: dict, phase: str = "") -> str:
    """Render ``report`` as the single paste-ready ``[vvcensus]`` log line."""
    families = ", ".join(
        f"{name}={_human(size)}"
        for name, size in sorted(
            report["families"].items(), key=lambda kv: (-kv[1], kv[0])
        )[:_MAX_RENDERED_FAMILIES]
    )
    n_families = len(report["families"])
    if n_families > _MAX_RENDERED_FAMILIES:
        families += f", +{n_families - _MAX_RENDERED_FAMILIES} more"
    return (
        f"[vvcensus] {phase} "
        f"params={_split(report, 'param')} "
        f"buffers={_split(report, 'buffer')} "
        f"stash={_split(report, 'backup')} "
        f"stash_buffers={_split(report, 'backup_buffer')} "
        f"families[{n_families}] {families or 'none'} "
        f"vbar={_human(report['vbar_bytes'])}/{report['vbar_count']}m "
        f"mmap_refs_storages={report['mmap_refs_storages']}"
    )


def _split(report: dict, group: str) -> str:
    """``total=1.00GB view=..(..%) mmap=.. private=.. offhost=..`` for one group."""
    total = report[f"{group}_bytes"]
    view = report[f"{group}_view_bytes"]
    view_pct = 100.0 * view / total if total else 0.0
    return (
        f"total={_human(total)}"
        f" view={_human(view)}({view_pct:.1f}%)"
        f" mmap={_human(report[f'{group}_mmap_bytes'])}"
        f" private={_human(report[f'{group}_private_bytes'])}"
        f" offhost={_human(report[f'{group}_offhost_bytes'])}"
    )


def _human(n_bytes: int) -> str:
    """Byte count as a compact fixed-point string (units of 1024)."""
    value = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TB"  # pragma: no cover - the loop always returns above


def _iter_modules(model):
    """``(qualified name, module, class name)`` for every module in ``model``."""
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return
    try:
        for name, module in named_modules():
            yield name, module, _family_name(module)
    except Exception:  # pragma: no cover - defensive
        return


def _family_name(module) -> str:
    """The module's class name, with the synthesised streaming prefix removed.

    The dynamic route rewrites every converted leaf to
    ``_ComfyStream<Original>`` (modules/comfy_stream.py:335/:405), so the class
    name is otherwise unreadable in a per-family rollup.
    """
    name = type(module).__name__
    return name[len(_STREAMING_PREFIX):] if name.startswith(_STREAMING_PREFIX) else name


def _direct_tensors(module, attr: str):
    """The module's OWN parameters/buffers (no recursion) as ``(name, tensor)``."""
    holder = getattr(module, attr, None)
    if not isinstance(holder, dict):
        return ()
    return tuple(
        (name, value)
        for name, value in holder.items()
        if isinstance(value, torch.Tensor)
    )


def _classify(tensor, storage) -> str:
    """Which of :data:`STATES` this tensor's bytes belong to."""
    if tensor.device.type != "cpu":
        return "offhost"
    if getattr(storage, _FILE_SLICE_ATTR, None) is not None:
        return "view"
    if getattr(storage, _MMAP_REFS_ATTR, None) is not None:
        return "mmap"
    return "private"


def _accumulate(report, tensors, group: str, family: str, seen: _Seen) -> None:
    """Add ``tensors``' bytes to ``report`` under ``group`` and their family."""
    for _name, tensor in tensors:
        try:
            storage = tensor.untyped_storage()
            nbytes = storage.nbytes()
            # _cdata is the address of the underlying c10::Storage: a true
            # identity. data_ptr would collide for every meta storage (all
            # pointers are 0) and would miss a view into a shared storage.
            key = getattr(storage, "_cdata", None)
            if key is None:  # pragma: no cover - every torch storage has it
                key = (storage.data_ptr(), nbytes)
            state = _classify(tensor, storage)
        except Exception:
            # Exotic tensor types (a GGUF/quant resident whose storage the
            # generic torch API cannot describe) are skipped, never fatal:
            # instrumentation must not be able to fail a load.
            logger.debug("census: unreadable storage, skipped", exc_info=True)
            continue
        if key in seen.storages:
            continue
        seen.storages.add(key)
        if getattr(storage, _MMAP_REFS_ATTR, None) is not None:
            seen.mmap_refs.add(key)

        report[f"{group}_bytes"] += nbytes
        report[f"{group}_count"] += 1
        report[f"{group}_{state}_bytes"] += nbytes
        report[f"{group}_{state}_count"] += 1

        if nbytes and group in _MODEL_GROUPS:
            # Zero-byte entries are dropped: a tied weight sharing a storage
            # with an already-counted family contributes nothing here, and
            # listing it as "0B" would only pad the single log line.
            split = "params" if group == "param" else "buffers"
            rollup_key = f"{family}/{split}"
            report["families"][rollup_key] = (
                report["families"].get(rollup_key, 0) + nbytes
            )


def _accumulate_vbar(report, module) -> None:
    """Credit the vbar range core allocated for ``module``, if any.

    ``_v`` is the range tuple core stores on the module at
    comfy/model_patcher.py:1993; its third element is the byte size (see the
    block arithmetic at :2018-2019). Anything else is ignored — this is a
    measurement, not a contract.
    """
    v = getattr(module, "_v", None)
    if v is None:
        return
    try:
        size = int(v[2])
    except Exception:
        return
    report["vbar_bytes"] += size
    report["vbar_count"] += 1


def _accumulate_stash(report, patcher, seen: _Seen) -> None:
    """Credit ``patcher.backup`` / ``patcher.backup_buffers``.

    The dynamic patcher stashes the pre-load tensor for every module it did
    not take the vbar branch for (comfy/model_patcher.py:2000-2009) and for
    every buffer (:2010-2016). The stash gets its own de-dup set: its entries
    are separate lifetimes from the live parameters and must be reported even
    when a storage is shared, while the mapping references it holds still count
    once against ``mmap_refs_storages``.
    """
    for attr, group in _STASH_GROUPS:
        stash = getattr(patcher, attr, None)
        if not isinstance(stash, dict):
            continue
        tensors = tuple(
            tensor
            for tensor in (
                # Core stores either a raw tensor or a
                # namedtuple('Dimension', ['weight', 'inplace_update']) (:2001).
                getattr(entry, "weight", entry)
                for entry in stash.values()
            )
            if isinstance(tensor, torch.Tensor)
        )
        _accumulate(
            report,
            ((None, tensor) for tensor in tensors),
            group,
            family="stash",
            seen=_Seen(seen.mmap_refs),
        )
        # The entry COUNT is the patcher's own bookkeeping, not a storage
        # count: keep it faithful even if an entry held no usable tensor.
        report[f"{group}_count"] = len(stash)


# ====================================================================
# Process-level peak sampling (the transient half of the measurement)
# ====================================================================
#
# The census above answers "which storages does the model hold". It cannot
# answer "how many bytes were live at the worst instant of the load": by the
# time a post-H2D census runs, the transient is already freed, which is
# precisely why a post-hoc RSS delta is not a measurement of a spike. This
# half samples the PROCESS instead, on a background thread, for the whole
# load, and reports the sampled maximum of three distinct quantities:
#
# * ``ws`` — working set: what Task Manager shows, and what a user's
#   "25 -> 42GB" report IS. It INCLUDES file-backed pages, so reading a
#   multi-GB checkpoint through a mapping inflates it by up to 1x file with
#   bytes the process does not own and the OS can reclaim under pressure.
# * ``uss`` — the "unique"/private set as this platform's reader computes it.
#   :func:`uss_counts_mapped_pages` reports what that means HERE, and the
#   answer decides whether ``uss`` can be read as private RAM at all.
# * ``private`` — private COMMIT (reserved, touched or not). Reported because
#   it is what a pinned/CUDA allocation costs, but torch reserves arenas it
#   never fills, so it over-counts: never quote it as residency.
#
# MEASURED, 2026-09-29, and MEASURED PER READ SHAPE. The previous revision of
# this note generalised from ONE arm of a single probe, and the generalisation
# was refuted by that same probe's other arm. Both arms, same file, same host
# (0.50 GiB, ``tests/probe_safetensors_aliases_mapping.py``, reproduced by
# ``tests/probe_mapped_pages_uss.py``):
#
#   plain ``mmap.ACCESS_READ``, every page touched
#       ws +0.50GiB   uss +0.50GiB   private +0.00GiB
#   ``safe_open.get_tensor`` views, every page touched   <- the LOADER's shape
#       ws +0.00GiB   uss +0.00GiB   private +0.50GiB
#   ``bytearray`` (private control)
#       ws +0.50GiB   uss +0.50GiB   private +0.50GiB
#
# So ``ws`` and ``uss`` count file-backed pages for a plain ``mmap`` read and
# NOT for the safetensors read this loader actually performs, while ``private``
# moves for the safetensors read and NOT for the plain ``mmap``. "ws and uss
# include ~1x file for ANY streamed read" is therefore FALSE for the route the
# numbers are collected on, and the corollary that "private is the only
# private-RAM figure" does not follow either. None of the three settles it.
#
# WHAT DOES SETTLE IT: :func:`census` walks storages and separates aimdo file
# views from real host allocations, so the ``[vvcensus] private=`` bucket is a
# private-RAM figure while all three process counters are, on this host, a
# mixture of both. The sampler answers the OTHER half — how many bytes at the
# worst instant — and the two must be read together.
#
# CONSEQUENCE for the 7B fp8 report (8.82GiB file, peak uss 18.06GiB, settling
# to 10.26GiB): the earlier note re-labelled that peak as "~1x parameters +
# ~1x page cache, not a defect". That re-labelling rested on the plain-``mmap``
# arm above, which is NOT the shape this loader reads in, so it is WITHDRAWN.
# The measured ~2x private peak stands as an unattributed residual and is
# still open. What is now established about the read is that
# ``safe_open.get_tensor`` returns a VIEW of the file mapping —
# ``VirtualQuery``/``GetMappedFileNameW`` report ``MEM_MAPPED`` backed by the
# checkpoint itself, against a ``MEM_PRIVATE`` ``torch.empty`` control — that
# the ~1x is NOT released by closing the ``safe_open`` handle, and that the
# per-tensor ``clone()`` (``modules/external_loader.py:1047``) adds a further
# ~1x beside it. See :func:`ram_measurement_guide` for what to quote.



def memory_snapshot() -> dict:
    """``{"ws", "uss", "private"}`` in bytes; zeros for whatever is unreadable.

    Three readers, in order of fidelity: ``psutil`` (a ComfyUI dependency;
    ``memory_full_info().uss`` is where Windows exposes the private resident
    set), the Win32 ``PROCESS_MEMORY_COUNTERS_EX`` struct (for the embedded
    interpreter when it ships without psutil), and ``/proc/self/statm`` for
    Linux. Never raises: instrumentation that can break a load is not
    instrumentation.
    """
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        info = proc.memory_info()
        snapshot = {
            "ws": int(info.rss),
            "private": int(getattr(info, "private", info.rss)),
        }
        try:
            snapshot["uss"] = int(proc.memory_full_info().uss)
        except Exception:
            snapshot["uss"] = snapshot["ws"]
        # Machine-level used RAM. A phase can grow system memory WITHOUT any
        # process counter moving (the 2026-09-30 live report: ComfyUI flat at
        # 3.2GB private / 1.16GB ws while machine RAM climbed 20.8 -> 27.3GB),
        # so the machine counter is the only one that can attribute that.
        try:
            snapshot["sys_used"] = int(psutil.virtual_memory().used)
        except Exception:
            snapshot["sys_used"] = 0
        return snapshot
    except Exception:
        pass
    if os.name == "nt":
        try:
            counters = _PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(counters)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(counters), counters.cb)
            if ok:
                ws = int(counters.WorkingSetSize)
                return {"ws": ws, "uss": ws,
                        "private": int(counters.PrivateUsage), "sys_used": 0}
        except Exception:
            pass
        return {"ws": 0, "uss": 0, "private": 0, "sys_used": 0}
    try:
        with open("/proc/self/statm", "rb") as fh:
            fields = fh.read().split()
        ws = int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
        return {"ws": ws, "uss": ws, "private": 0, "sys_used": 0}
    except Exception:
        return {"ws": 0, "uss": 0, "private": 0, "sys_used": 0}


def rss_bytes() -> tuple:
    """``(working_set, private_commit)`` — the two-tuple shorthand."""
    snapshot = memory_snapshot()
    return snapshot["ws"], snapshot["private"]


#: Result of :func:`read_shape_profile`, computed once. ``None`` until the
#: probe has run.
_USS_COUNTS_MAPPED: bool | None = None
_READ_SHAPE_PROFILE: dict | None = None
#: Size the cached profile was measured at, so every threshold is relative to
#: the same bytes the arms moved.
_READ_SHAPE_PROBE_BYTES = 0

#: The read shapes the probe separates. ``mmap`` is the generic mapping read;
#: ``safetensors`` is the shape the iterator's aimdo-off fallback performs
#: (the aimdo arm reads the ``mmap`` shape); ``private_control`` is a heap
#: allocation, and is the arm that says whether
#: any of the readings can be trusted at all.
_READ_SHAPES = ("mmap", "safetensors", "private_control")


def _touch_safetensors(path: str, probe_bytes: int):
    """Yield every tensor of ``path`` with every page touched, keeping them.

    Retention matters: a body whose results are dropped before the reading is
    taken measures a freed working set, which is how an arm reports +0.00 for
    all three quantities and looks like a measurement.
    """
    from safetensors import safe_open

    held = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            flat = tensor.reshape(-1)
            for index in range(0, flat.numel(), 4096):
                flat[index]
            held.append(tensor)
    return held


def _touch_mmap(path: str, probe_bytes: int):
    """Map ``path`` read-only, touch every page, and KEEP the mapping open."""
    import mmap

    handle = open(path, "rb")
    mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    for offset in range(0, mapped.size(), 4096):
        mapped[offset]
    return (handle, mapped)


def _touch_bytearray(path: str, probe_bytes: int):
    """A genuine private allocation of the same size: the control arm."""
    block = bytearray(probe_bytes)
    for offset in range(0, probe_bytes, 4096):
        block[offset] = 1
    return block


def _write_probe_safetensors(path: str, probe_bytes: int) -> None:
    """Write a REAL safetensors file of about ``probe_bytes``.

    The safetensors arm has to open an actual checkpoint: a file of raw zeros
    makes ``safe_open`` raise, and an arm that silently drops out is how this
    probe ended up reporting one read shape while the shipped prose spoke for
    two. The tensors are several, not one, so per-tensor behaviour is
    represented rather than a single degenerate blob.
    """
    import safetensors.torch as st

    chunk = max(4096, (probe_bytes // 8) // 2)  # bf16 => 2 bytes per element
    tensors = {
        f"probe{i}.weight": torch.zeros(chunk, dtype=torch.bfloat16)
        for i in range(8)
    }
    st.save_file(tensors, path)


def read_shape_profile(probe_bytes: int = 64 * 1024 * 1024) -> dict:
    """Measure, PER READ SHAPE, what each process counter does on this host.

    Returns ``{shape: {"ws": delta, "uss": delta, "private": delta}}`` — the
    growth each counter showed while that arm's bytes were held.

    The shape distinction is the whole point and was what the earlier note got
    wrong. A generic ``mmap`` read and the safetensors read this loader
    performs are charged DIFFERENTLY (measured on this host: the mapping arm
    moves ``ws``/``uss`` and not ``private``; the safetensors arm moves
    ``private`` and not ``ws``/``uss``), so a single platform-level yes/no
    about whether ``uss`` counts mapped pages cannot be generalised across
    them, and neither can the claim that ``private`` is thereby the
    private-RAM figure.

    The control arm is what makes the other two readable: if a private
    allocation does not move the counters, this host's readings are noise and
    the caller is told so.

    Cached after the first call — it touches the filesystem and the page cache,
    and it is a property of the platform, not of the load being measured.
    Never raises: any failure returns an empty profile, and every consumer
    degrades to the conservative "quote nothing, cross-check the census".
    """
    global _READ_SHAPE_PROFILE, _USS_COUNTS_MAPPED, _READ_SHAPE_PROBE_BYTES
    if _READ_SHAPE_PROFILE is not None:
        return _READ_SHAPE_PROFILE
    _READ_SHAPE_PROBE_BYTES = int(probe_bytes)

    profile: dict = {}
    path = None
    try:
        import tempfile

        handle, path = tempfile.mkstemp(prefix="vv_read_shape_probe_", suffix=".safetensors")
        os.close(handle)
        _write_probe_safetensors(path, probe_bytes)
        gc.collect()
        time.sleep(0.05)

        for shape, body in (
            ("mmap", _touch_mmap),
            ("safetensors", _touch_safetensors),
            ("private_control", _touch_bytearray),
        ):
            try:
                gc.collect()
                time.sleep(0.05)
                before = memory_snapshot()
                held = body(path, probe_bytes)
                after = memory_snapshot()
                profile[shape] = {
                    metric: after[metric] - before[metric]
                    for metric in ("ws", "uss", "private")
                }
                # Released only AFTER the reading, for the reason above.
                del held
                gc.collect()
                time.sleep(0.05)
            except Exception:
                # One broken arm must not discard the others: the whole
                # finding is that the arms differ, so a profile with a
                # missing arm is still evidence, and the guide says which
                # arm it could not measure.
                logger.debug("read-shape probe arm failed: %s", shape,
                             exc_info=True)
                profile.setdefault(shape, {})
    except Exception:
        logger.debug("read-shape probe failed", exc_info=True)
        profile = {}
    finally:
        if path is not None:
            try:
                os.remove(path)
            except OSError:
                pass
    _READ_SHAPE_PROFILE = profile
    _USS_COUNTS_MAPPED = _arm_moved(profile, "mmap", "uss")
    return profile


def _arm_moved(profile: dict, shape: str, metric: str) -> bool:
    """Did ``metric`` move by more than half the probe size for ``shape``?

    Relative to the probe size on purpose: an absolute byte threshold would
    either miss a small probe or read allocator noise as a verdict on a large
    one. A missing arm counts as "did not move", which is the safe direction —
    it keeps ``uss`` from being quoted as private RAM.
    """
    arm = profile.get(shape)
    if not arm or _READ_SHAPE_PROBE_BYTES <= 0:
        return False
    return abs(int(arm.get(metric, 0))) > _READ_SHAPE_PROBE_BYTES // 2


def uss_counts_mapped_pages(probe_bytes: int = 64 * 1024 * 1024) -> bool:
    """Does this platform's ``uss`` reader count file-backed mapped pages?

    True/False for the GENERIC ``mmap`` read only — which is the arm the
    answer is well defined for. It is deliberately NOT a statement about every
    read: :func:`read_shape_profile` measures the safetensors shape the loader
    actually performs, and on this host the two disagree. Callers that need to
    reason about a loader peak must use the profile, not this flag.

    Fails CLOSED: any error returns False, so "unknown" never becomes "uss is
    private". Backed by the cached profile, so it costs nothing after the
    first call.
    """
    global _USS_COUNTS_MAPPED
    if _USS_COUNTS_MAPPED is None:
        read_shape_profile(probe_bytes)
    return bool(_USS_COUNTS_MAPPED)


def ram_measurement_guide() -> str:
    """One line naming WHICH quantity answers WHICH question, for this host.

    Reports what was MEASURED for each read shape rather than asserting a
    platform-level rule. The earlier revision generalised the ``mmap`` arm's
    result to "any streamed read", which its own probe refuted: the
    safetensors read the loader performs moves ``private``, not ``ws``/``uss``.
    Shipping that generalisation was actively harmful — it told a user
    reading the 7B fp8 line that a real private spike was page cache.

    The guidance that survives: none of the three process counters settles the
    question on its own, so quote the sampled peak as the load's growth AND
    cross-check ``[vvcensus] private=``, which walks storages and is the only
    figure here that separates host allocations from aimdo file views.
    """
    profile = read_shape_profile()
    if not profile:
        return (
            "read-shape probe unavailable on this host, so no process counter "
            "here is a proven private-RAM figure; trust the [vvcensus] "
            "private= bucket, which walks storages"
        )
    control = profile.get("private_control")
    if not _arm_moved(profile, "private_control", "private") and not \
            _arm_moved(profile, "private_control", "ws"):
        return (
            "read-shape probe found this host's counters do not even move for "
            "a private allocation, so none of them can be read as RAM here; "
            "trust the [vvcensus] private= bucket, which walks storages"
        )

    def _say(shape: str) -> str:
        if not profile.get(shape):
            return "not measured"
        movers = [
            name for name in ("ws", "uss", "private")
            if _arm_moved(profile, shape, name)
        ]
        return f"+{'/'.join(movers) if movers else 'nothing'}"

    counted = uss_counts_mapped_pages()
    mapped_arm = (
        "COUNTS file-backed mapped pages" if counted
        else "EXCLUDES file-backed mapped pages"
    )
    return (
        f"measured on this host — a plain mmap read {mapped_arm} "
        f"({_say('mmap')}), while the loader's safetensors read is charged "
        f"differently ({_say('safetensors')}), so no single counter settles "
        f"it: quote the sampled peak as the load's growth and cross-check "
        f"[vvcensus] private=, which walks storages"
    )


class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):  # noqa: N801 - win32 name
    """``PROCESS_MEMORY_COUNTERS_EX``: WorkingSetSize + PrivateUsage."""

    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def _marks_digest(marks) -> str:
    """One-line digest of the labelled marks: label@private.

    A mark that shows private already at file-size before the loop starts is
    the whole answer (the mapping's own accounting); a mark still flat there
    means the growth is inside the pass. Kept short so the paste-ready line
    stays readable.
    """
    if not marks:
        return "-"
    return ",".join(
        f"{label}@{_human(reading.get('private', 0))}"
        for label, reading, _peak in marks
    )


    # -- reporting ----------------------------------------------------


class RssSampler:
    """Background peak sampler for one load: working set, USS and commit.

    A context manager::

        with RssSampler() as sampler:
            load_model()
        print(sampler.line())

    The sampling thread is a daemon and is always stopped by ``__exit__``,
    which takes one final synchronous reading — so the peak includes the
    instant the block returned. :meth:`mark` records a labelled reading plus
    the running peak, so a multi-phase load is attributed phase by phase
    instead of collapsing into one number.

    ``interval`` defaults to 20ms: fine enough that a multi-second
    materialisation is sampled hundreds of times, coarse enough that the
    sampler never perturbs what it measures.
    """

    #: The four quantities tracked, in the order they are rendered.
    #: ``sys_used`` is MACHINE-level used RAM (psutil.virtual_memory().used):
    #: a phase can grow system memory with every process counter flat (file
    #: cache, other processes), and only a machine counter can attribute that.
    METRICS = ("ws", "uss", "private", "sys_used")

    def __init__(self, interval: float = 0.02, label: str = "", series: bool = False):
        self.interval = interval
        self.label = label
        self.samples = 0
        self.marks = []
        self.start = {metric: 0 for metric in self.METRICS}
        self.end = {metric: 0 for metric in self.METRICS}
        self.peak = {metric: 0 for metric in self.METRICS}
        # Opt-in timeline. A load is hundreds of samples, so keeping them is
        # only worth it when the SHAPE of the transient is the question (which
        # second did it happen in?), never by default.
        self.series = [] if series else None
        self._t0 = time.monotonic()
        self._stop = threading.Event()
        self._thread = None

    # -- sampling -----------------------------------------------------
    def sample(self) -> dict:
        """Take one reading, fold it into the peaks, return the snapshot."""
        snapshot = memory_snapshot()
        self.samples += 1
        for metric in self.METRICS:
            value = snapshot.get(metric, 0)
            if value > self.peak[metric]:
                self.peak[metric] = value
        if self.series is not None:
            self.series.append(
                (time.monotonic() - self._t0, snapshot["ws"], snapshot["uss"]))
        return snapshot

    def profile(self, buckets: int = 16) -> str:
        """Coarse timeline of the sampled working set, as one log line.

        Only meaningful when the sampler was built with ``series=True``. The
        SHAPE is the attribution: a plateau that grows with the file is a
        read-driven accumulation, one tall isolated bar is a single large
        temporary, and a drop is the end of a phase. Bucket MAXIMA, so a
        short-lived spike cannot hide between two samples.
        """
        if not self.series:
            return f"[vvrss] {self.label} profile=off (construct with series=True)"
        buckets = max(1, buckets)
        span = max((t for t, _, _ in self.series), default=0.0) or 1.0
        peaks = [0] * buckets
        for t, ws, _uss in self.series:
            index = min(buckets - 1, int(buckets * t / span))
            peaks[index] = max(peaks[index], ws)
        top = max(peaks) or 1
        bar = "".join(" ▁▂▃▄▅▆▇█"[min(7, int(8 * value / top))] for value in peaks)
        return (f"[vvrss] {self.label} timeline(ws {span:.1f}s in {buckets} buckets) "
                f"{bar} peak={_human(top)}")

    def mark(self, label: str) -> dict:
        """Record a labelled reading plus the running peak at that moment."""
        snapshot = self.sample()
        self.marks.append((label, dict(snapshot), dict(self.peak)))
        return snapshot

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    # -- context manager ----------------------------------------------
    def __enter__(self) -> "RssSampler":
        self.start = self.sample()
        self._thread = threading.Thread(
            target=self._run, name="vv-rss-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.end = self.sample()
        return False

    def line(self, phase: str = "") -> str:
        """One paste-ready ``[vvrss]`` line: start, sampled peak, end.

        Carries all three sampled peaks plus the host's measured read-shape
        guide, because they are NOT interchangeable: on this host a plain
        ``mmap`` read and the safetensors read the loader performs are charged
        to different counters (measured, see the section header), so no one
        of them settles attribution on its own and the line says so.
        """
        return (
            f"[vvrss] {phase or self.label} "
            f"peak_ws={_human(self.peak['ws'])} "
            f"peak_uss={_human(self.peak['uss'])} "
            f"peak_private={_human(self.peak['private'])} "
            f"start_ws={_human(self.start['ws'])} "
            f"end_ws={_human(self.end['ws'])} "
            f"end_uss={_human(self.end['uss'])} "
            f"end_private={_human(self.end['private'])} "
            f"start_sys={_human(self.start.get('sys_used', 0))} "
            f"end_sys={_human(self.end.get('sys_used', 0))} "
            f"samples={self.samples} "
            f"marks={_marks_digest(self.marks)} "
            f"note=[{ram_measurement_guide()}]"
        )

    def report(self, phase: str = "") -> str:
        """Log the sampled peak at INFO and return the line (env-gated)."""
        line = self.line(phase)
        if census_enabled():
            logger.info(line)
        return line


def peak_delta(sampler: RssSampler, baseline: int) -> dict:
    """Sampled peak growth over a ``baseline`` working set.

    ``baseline`` is what the caller measured BEFORE the load started. A peak
    is a sampled maximum, so these deltas are upper bounds on what the load's
    own pages contributed — which is the honest way to quote a spike that is
    over before the process settles. ``peak_uss_delta`` is the number that
    says whether those pages were private allocations or file cache — ON A
    HOST WHERE ``uss`` EXCLUDES file-backed pages. On Windows it does not
    (measured: :func:`uss_counts_mapped_pages`), so ``peak_private_delta``
    is the one to read there, and ``uss_counts_mapped_pages`` is reported
    alongside so a reader is never left guessing which platform they are on.
    """
    return {
        "baseline": baseline,
        "peak_ws": sampler.peak["ws"],
        "peak_uss": sampler.peak["uss"],
        "peak_private": sampler.peak["private"],
        "peak_ws_delta": max(0, sampler.peak["ws"] - baseline),
        "peak_uss_delta": max(0, sampler.peak["uss"] - baseline),
        "peak_private_delta": max(0, sampler.peak["private"] - baseline),
        "end_ws": sampler.end["ws"],
        "end_uss": sampler.end["uss"],
        "end_private": sampler.end["private"],
        "start_sys_used": sampler.start.get("sys_used", 0),
        "end_sys_used": sampler.end.get("sys_used", 0),
        "samples": sampler.samples,
        "uss_counts_mapped_pages": uss_counts_mapped_pages(),
    }


# ====================================================================
# Production wiring (why a census alone is not a measurement)
# ====================================================================
#
# The census above is an inventory of storages a model still HOLDS, and the
# sampler above is the only instrument that can see a TRANSIENT. Until this
# existed, the sampler was reachable only from a standalone probe: grep found
# RssSampler/peak_delta in this module and in tests, and in NO production
# call site, so a user with a real 7B fp8 spike had no shipped way to obtain
# the one number the ask asks for. A measurement nobody can take is not a
# measurement.
#
# The rule this enforces: EVERY phase that can move host RAM is wrapped, and
# each wrapper reports against a baseline taken immediately before it. A
# per-phase line is the only form in which a transient can be attributed — one
# number for the whole load cannot distinguish "the loader held 2x" from "the
# reader mapped 1x file and the loader held 1x".


@contextlib.contextmanager
def measured_load(phase: str, series: bool = False):
    """Sample the process for one load phase and log the peak on exit.

    ::

        with measured_load("external-load"):
            model = load_external_vibevoice_model(...)

    Yields the :class:`RssSampler` so the body can add :meth:`~RssSampler.mark`
    boundaries around sub-phases (e.g. ``stream-begin`` / ``stream-end``), which
    is how a spike gets attributed to a specific step rather than to "the
    load". The line is logged on exit, including when the body raises, so a
    failed load still yields the peak it reached.

    Gated on the same ``VIBEVOICE_RAM_CENSUS`` switch as the census. The
    sampler is always constructed (it is ~20ms of a background thread and it
    is what makes ``__exit__``'s final reading meaningful), but nothing is
    logged when the gate is off. Instrumentation must not be able to fail a
    load, so a sampler error degrades to no measurement, never to an
    exception in the caller's load path.
    """
    sampler = RssSampler(label=phase, series=series)
    try:
        with sampler:
            yield sampler
    finally:
        try:
            sampler.report(phase)
            if sampler.series is not None:
                logger.info(sampler.profile())
        except Exception:  # pragma: no cover - never break a load
            logger.debug("measured_load report failed", exc_info=True)
