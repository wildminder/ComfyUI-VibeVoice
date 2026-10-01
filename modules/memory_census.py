"""Host-RAM census for a loaded VibeVoice model.

MEASURE FIRST. Two RAM defects were reported on external single-file loads:
the 1.5B bf16 file leaves ~7GB resident, and the 7B fp8 file spikes ~17GB
transiently. Every hypothesis about where host bytes live is now a number
instead: this walks the model's storages once and splits their bytes into the
four states that matter, then prints exactly one ``[vvcensus]`` line.

The four states:

* **view** — the storage carries ``_comfy_tensor_file_slice``, i.e. aimdo's
  ``ModelMMAP`` file view. These bytes are page-cache backed: they cost working
  set, not private RAM.
* **mmap** — the storage carries ``_comfy_tensor_mmap_refs`` but no slice tag
  (a raw ``torch.load(mmap=True)`` mapping, or a view whose slice attribute
  was severed).
* **private** — a real HOST allocation: a clone, a dequant, or a meta-straggler
  materialisation.
* **offhost** — the parameter/buffer lives on an accelerator (CUDA VRAM). Not
  host RAM at all.

Plus the patcher's own host-side stash (``backup`` / ``backup_buffers``) and
the vbar ranges allocated.
"""

import contextlib
import ctypes
import gc
import logging
import os
import time
import threading

import torch

from .diagnostics import census_enabled

logger = logging.getLogger(__name__)

_FILE_SLICE_ATTR = "_comfy_tensor_file_slice"
_MMAP_REFS_ATTR = "_comfy_tensor_mmap_refs"

STATES = ("view", "mmap", "private", "offhost")

_MODEL_GROUPS = ("param", "buffer")
_STASH_GROUPS = (("backup", "backup"), ("backup_buffers", "backup_buffer"))

_MAX_RENDERED_FAMILIES = 8
_STREAMING_PREFIX = "_ComfyStream"


class _Seen:
    """De-dup bookkeeping for one walk: storages counted, mappings pinned."""

    __slots__ = ("storages", "mmap_refs")

    def __init__(self, mmap_refs=None):
        self.storages = set()
        self.mmap_refs = set() if mmap_refs is None else mmap_refs


def census(model, patcher=None) -> dict:
    """Byte census of ``model``'s parameters and buffers, plus ``patcher``'s stash."""
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
    """Log one ``[vvcensus]`` line for ``model`` and return the census."""
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
    value = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TB"


def _iter_modules(model):
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return
    try:
        for name, module in named_modules():
            yield name, module, _family_name(module)
    except Exception:
        return


def _family_name(module) -> str:
    name = type(module).__name__
    return name[len(_STREAMING_PREFIX):] if name.startswith(_STREAMING_PREFIX) else name


def _direct_tensors(module, attr: str):
    holder = getattr(module, attr, None)
    if not isinstance(holder, dict):
        return ()
    return tuple(
        (name, value)
        for name, value in holder.items()
        if isinstance(value, torch.Tensor)
    )


def _classify(tensor, storage) -> str:
    if tensor.device.type != "cpu":
        return "offhost"
    if getattr(storage, _FILE_SLICE_ATTR, None) is not None:
        return "view"
    if getattr(storage, _MMAP_REFS_ATTR, None) is not None:
        return "mmap"
    return "private"


def _accumulate(report, tensors, group: str, family: str, seen: _Seen) -> None:
    for _name, tensor in tensors:
        try:
            storage = tensor.untyped_storage()
            nbytes = storage.nbytes()
            key = getattr(storage, "_cdata", None)
            if key is None:
                key = (storage.data_ptr(), nbytes)
            state = _classify(tensor, storage)
        except Exception:
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
            split = "params" if group == "param" else "buffers"
            rollup_key = f"{family}/{split}"
            report["families"][rollup_key] = (
                report["families"].get(rollup_key, 0) + nbytes
            )


def _accumulate_vbar(report, module) -> None:
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
    for attr, group in _STASH_GROUPS:
        stash = getattr(patcher, attr, None)
        if not isinstance(stash, dict):
            continue
        tensors = tuple(
            tensor
            for tensor in (
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
        report[f"{group}_count"] = len(stash)


# ====================================================================
# Process-level peak sampling
# ====================================================================

def memory_snapshot() -> dict:
    """``{"ws", "uss", "private"}`` in bytes; zeros for whatever is unreadable."""
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


_USS_COUNTS_MAPPED: bool | None = None
_READ_SHAPE_PROFILE: dict | None = None
_READ_SHAPE_PROBE_BYTES = 0
_READ_SHAPES = ("mmap", "safetensors", "private_control")


def _touch_safetensors(path: str, probe_bytes: int):
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
    import mmap

    handle = open(path, "rb")
    mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    for offset in range(0, mapped.size(), 4096):
        mapped[offset]
    return (handle, mapped)


def _touch_bytearray(path: str, probe_bytes: int):
    block = bytearray(probe_bytes)
    for offset in range(0, probe_bytes, 4096):
        block[offset] = 1
    return block


def _write_probe_safetensors(path: str, probe_bytes: int) -> None:
    import safetensors.torch as st

    chunk = max(4096, (probe_bytes // 8) // 2)
    tensors = {
        f"probe{i}.weight": torch.zeros(chunk, dtype=torch.bfloat16)
        for i in range(8)
    }
    st.save_file(tensors, path)


def read_shape_profile(probe_bytes: int = 64 * 1024 * 1024) -> dict:
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
                del held
                gc.collect()
                time.sleep(0.05)
            except Exception:
                logger.debug("read-shape probe arm failed: %s", shape, exc_info=True)
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
    arm = profile.get(shape)
    if not arm or _READ_SHAPE_PROBE_BYTES <= 0:
        return False
    return abs(int(arm.get(metric, 0))) > _READ_SHAPE_PROBE_BYTES // 2


def uss_counts_mapped_pages(probe_bytes: int = 64 * 1024 * 1024) -> bool:
    global _USS_COUNTS_MAPPED
    if _USS_COUNTS_MAPPED is None:
        read_shape_profile(probe_bytes)
    return bool(_USS_COUNTS_MAPPED)


def ram_measurement_guide() -> str:
    profile = read_shape_profile()
    if not profile:
        return (
            "read-shape probe unavailable on this host; trust the [vvcensus] "
            "private= bucket, which walks storages"
        )
    control = profile.get("private_control")
    if not _arm_moved(profile, "private_control", "private") and not \
            _arm_moved(profile, "private_control", "ws"):
        return (
            "read-shape probe found this host's counters do not move for a "
            "private allocation; trust the [vvcensus] private= bucket"
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


class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):  # noqa: N801
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
    if not marks:
        return "-"
    return ",".join(
        f"{label}@{_human(reading.get('private', 0))}"
        for label, reading, _peak in marks
    )


class RssSampler:
    """Background peak sampler for one load: working set, USS and commit."""

    METRICS = ("ws", "uss", "private", "sys_used")

    def __init__(self, interval: float = 0.02, label: str = "", series: bool = False):
        self.interval = interval
        self.label = label
        self.samples = 0
        self.marks = []
        self.start = {metric: 0 for metric in self.METRICS}
        self.end = {metric: 0 for metric in self.METRICS}
        self.peak = {metric: 0 for metric in self.METRICS}
        self.series = [] if series else None
        self._t0 = time.monotonic()
        self._stop = threading.Event()
        self._thread = None

    def sample(self) -> dict:
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
        snapshot = self.sample()
        self.marks.append((label, dict(snapshot), dict(self.peak)))
        return snapshot

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

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
        line = self.line(phase)
        if census_enabled():
            logger.info(line)
        return line


def peak_delta(sampler: RssSampler, baseline: int) -> dict:
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


@contextlib.contextmanager
def measured_load(phase: str, series: bool = False):
    """Sample the process for one load phase and log the peak on exit."""
    sampler = RssSampler(label=phase, series=series)
    try:
        with sampler:
            yield sampler
    finally:
        try:
            sampler.report(phase)
            if sampler.series is not None:
                logger.info(sampler.profile())
        except Exception:
            logger.debug("measured_load report failed", exc_info=True)