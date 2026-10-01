"""Tests for the host-RAM census (``modules/memory_census.py``).

Two halves:

1. **Pure** (no GPU, no aimdo): a KB-scale synthetic model whose storages are
   tagged exactly the way ``comfy.utils.load_safetensors`` tags them
   (``comfy/utils.py:145-150``), plus deliberately mis-tagged ones. These pin
   the classification itself — which bytes are views, which are mappings,
   which are private host allocations, which are simply on the device — on
   numbers small enough to assert exactly.

2. **Real** (CUDA + the REAL comfy-aimdo stack, KB-scale synthetic checkpoint):
   the acceptance probe. After a real ``load_to_device`` on the dynamic route
   the census must report ZERO private host param bytes, 100% tagged-view
   coverage, ZERO device-resident bytes and an EMPTY patcher stash — i.e. the
   weights are still file views paged through vbar, not copied anywhere. A
   regression in the loader, the conversion or the patcher family gate shows
   up here as a non-zero number instead of as a user-reported RAM spike.

   Every leaf in the probe is sized ABOVE core's 16KB force-load threshold
   (``comfy/model_patcher.py:1975-1989``) on purpose: under it core
   deliberately force-loads the module to the device and stashes the view
   (``LayerNorm(256)`` does exactly that — measured, not assumed), which is
   correct behaviour but would mask the property under test.

The aimdo bootstrap (``_aimdo_ready`` / ``aimdo_runtime``) is copied from
``tests/test_dynamic_vram_mechanism.py`` — including its repair of
comfy_aimdo's import-order trap — because a fixture cannot be shared across
test modules without becoming a conftest-wide GPU dependency.
"""

import collections
import contextlib
import gc
import logging
import os
import tempfile
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

import comfy.memory_management
import comfy.model_management
import comfy.model_patcher
import comfy.utils
from comfy_aimdo import control

from ComfyUI_VibeVoice.modules.comfy_stream import convert_tree_for_streaming
from ComfyUI_VibeVoice.modules.external_loader import _load_state_dict_into_model_from_memory
from ComfyUI_VibeVoice.modules.memory_census import (
    census,
    census_enabled,
    format_census,
    report_census,
)
from ComfyUI_VibeVoice.modules.patcher import load_to_device, select_patcher_class

# Concrete byte sizes of the synthetic trees below (bf16 => 2 bytes/element).
LIN_W = 256 * 256 * 2
LIN_B = 256 * 2
EMB_W = 64 * 256 * 2
BUF = 256 * 2


def _tag(tensor, slice_attr=True, mmap_attr=True):
    """Tag a tensor's storage the way aimdo's file views are tagged."""
    storage = tensor.untyped_storage()
    if slice_attr:
        setattr(storage, "_comfy_tensor_file_slice", ("fake", "lock", 0, 8))
    if mmap_attr:
        setattr(storage, "_comfy_tensor_mmap_refs", ("fake-mmap", "fake-mv"))
    return storage


class Synthetic(nn.Module):
    """KB-scale: 256x256 bf16 linear + a 64x256 bf16 embedding + a buffer."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(256, 256, dtype=torch.bfloat16)
        self.emb = nn.Embedding(64, 256, dtype=torch.bfloat16)
        self.register_buffer("scale", torch.ones(256, dtype=torch.bfloat16))


@pytest.fixture
def synthetic():
    return Synthetic()


class TestCensusClassification:
    def test_all_tagged_storages_report_zero_private_bytes(self, synthetic):
        for tensor in list(synthetic.parameters()) + list(synthetic.buffers()):
            _tag(tensor)
        report = census(synthetic)

        assert report["param_bytes"] == LIN_W + LIN_B + EMB_W
        assert report["param_private_bytes"] == 0
        assert report["param_view_bytes"] == report["param_bytes"]
        assert report["param_view_count"] == 3  # lin.weight, lin.bias, emb.weight
        assert report["mmap_refs_storages"] == 4  # + the buffer

    def test_view_coverage_is_total_for_tagged_params(self, synthetic):
        for tensor in synthetic.parameters():
            _tag(tensor)
        report = census(synthetic)
        assert report["param_view_bytes"] / report["param_bytes"] == 1.0

    def test_untagged_param_is_private(self, synthetic):
        _tag(synthetic.lin.weight)
        report = census(synthetic)

        assert report["param_private_bytes"] == LIN_B + EMB_W
        assert report["param_private_count"] == 2
        assert report["param_view_count"] == 1

    def test_mmap_only_storage_is_mapped_not_private(self, synthetic):
        # A raw torch.load(mmap=True) mapping: refs present, no slice tag.
        _tag(synthetic.emb.weight, slice_attr=False)
        report = census(synthetic)
        assert report["param_mmap_bytes"] == EMB_W
        assert report["param_mmap_count"] == 1
        assert report["param_private_bytes"] == LIN_W + LIN_B
        # mmap refs ride on the storage, so this one counts too.
        assert report["mmap_refs_storages"] == 1

    def test_device_tensor_is_offhost_not_private(self, synthetic):
        """Core force-loads sub-16KB modules to the device; that is VRAM."""
        _tag(synthetic.lin.weight)
        report = census(synthetic.cuda())

        assert report["param_private_bytes"] == 0
        assert report["param_offhost_bytes"] == LIN_W + LIN_B + EMB_W
        assert report["param_view_bytes"] == 0

    def test_buffers_are_reported_separately_from_params(self, synthetic):
        _tag(synthetic.scale)
        report = census(synthetic)
        assert report["buffer_bytes"] == BUF
        assert report["buffer_view_bytes"] == BUF
        assert report["buffer_private_bytes"] == 0

    def test_family_rollup_splits_params_and_buffers(self, synthetic):
        for tensor in list(synthetic.parameters()) + list(synthetic.buffers()):
            _tag(tensor)
        report = census(synthetic)

        assert report["families"]["Linear/params"] == LIN_W + LIN_B
        assert report["families"]["Embedding/params"] == EMB_W
        assert report["families"]["Synthetic/buffers"] == BUF
        # The stash is not a module family.
        assert not any(k.startswith("stash/") for k in report["families"])

    def test_shared_storage_counted_once(self, synthetic):
        """A row view of a tied weight must not double the byte total."""
        _tag(synthetic.emb.weight)
        synthetic.lin.bias = nn.Parameter(synthetic.emb.weight.detach()[0])
        report = census(synthetic)

        assert report["param_view_bytes"] == EMB_W
        assert report["param_view_count"] == 1  # the row shares emb's storage
        assert report["param_bytes"] == LIN_W + EMB_W

    def test_streaming_class_names_report_the_original_family(self):
        """The dynamic route renames every leaf; the rollup must not."""
        class _ComfyStreamLinear(nn.Linear):
            pass

        model = nn.Module()
        layer = _ComfyStreamLinear(256, 256, dtype=torch.bfloat16)
        _tag(layer.weight)
        model.add_module("lin", layer)

        report = census(model)
        # The rollup is per family regardless of state: the bias is untagged
        # (private) here and is still this family's bytes.
        assert list(report["families"]) == ["Linear/params"]
        assert report["families"]["Linear/params"] == LIN_W + LIN_B

    def test_vbar_ranges_are_summed(self, synthetic):
        synthetic.lin._v = (0, 0, 4096)
        synthetic.emb._v = (0, 4096, 8192)
        report = census(synthetic)
        assert report["vbar_bytes"] == 12288
        assert report["vbar_count"] == 2

    def test_malformed_vbar_is_ignored(self, synthetic):
        synthetic.lin._v = object()
        assert census(synthetic)["vbar_count"] == 0


class TestPatcherStash:
    def test_backup_and_backup_buffers_totals(self, synthetic):
        for tensor in synthetic.parameters():
            _tag(tensor)
        patcher = MagicMock()
        # Core's two backup shapes: a raw tensor, and the Dimension
        # namedtuple (comfy/model_patcher.py:2001).
        patcher.backup = {
            "lin.weight": synthetic.lin.weight,
            "norm.bias": collections.namedtuple(
                "Dimension", ["weight", "inplace_update"]
            )(synthetic.emb.weight, False),
        }
        patcher.backup_buffers = {"scale": synthetic.scale}

        report = census(synthetic, patcher)
        assert report["backup_bytes"] == LIN_W + EMB_W
        assert report["backup_view_bytes"] == LIN_W + EMB_W
        assert report["backup_private_bytes"] == 0
        assert report["backup_count"] == 2
        assert report["backup_buffer_bytes"] == BUF
        assert report["backup_buffer_count"] == 1

    def test_stashed_private_copy_is_reported_as_private(self):
        """A stashed private tensor is a live host copy — the defect shape."""
        model = Synthetic()
        stashed = torch.zeros(64, 256, dtype=torch.bfloat16)
        patcher = MagicMock()
        patcher.backup = {"emb.weight": stashed}
        patcher.backup_buffers = {}

        report = census(model, patcher)
        assert report["backup_private_bytes"] == EMB_W
        assert report["backup_view_bytes"] == 0

    def test_stash_sharing_a_live_storage_is_still_reported(self, synthetic):
        """The stash has its own lifetime: it is counted, not de-duped away."""
        for tensor in list(synthetic.parameters()) + list(synthetic.buffers()):
            _tag(tensor)
        patcher = MagicMock()
        patcher.backup = {"lin.weight": synthetic.lin.weight}
        patcher.backup_buffers = {}

        report = census(synthetic, patcher)
        assert report["param_view_bytes"] == report["param_bytes"]
        assert report["backup_view_bytes"] == LIN_W
        # One mapping, referenced from both places.
        assert report["mmap_refs_storages"] == 4

    def test_patchless_census_is_all_zero(self, synthetic):
        report = census(synthetic, None)
        assert report["backup_bytes"] == 0
        assert report["backup_count"] == 0
        assert report["backup_buffer_bytes"] == 0

    def test_magicmock_patcher_is_not_a_stash(self, synthetic):
        """A MagicMock patcher (every unit test in this repo) reads as zero."""
        report = census(synthetic, MagicMock())
        assert report["backup_bytes"] == 0
        assert report["backup_count"] == 0


class TestTotality:
    @pytest.mark.parametrize("bad", [None, MagicMock(), object(), 42, "model"])
    def test_non_model_inputs_are_a_noop(self, bad):
        report = census(bad, MagicMock())
        assert report["param_bytes"] == 0
        assert report["families"] == {}
        # Formatting must survive too: the line is emitted whatever the model
        # turned out to be.
        assert format_census(report).startswith("[vvcensus]")

    def test_meta_model_is_counted_without_touching_data(self):
        with torch.device("meta"):
            model = Synthetic()
        report = census(model)
        assert report["param_offhost_bytes"] == LIN_W + LIN_B + EMB_W
        assert report["buffer_offhost_bytes"] == BUF
        # Meta storages all report data_ptr 0: the census must not collapse
        # them into one (that is what storage identity, not pointer, buys).
        assert report["param_count"] == 3
        assert report["buffer_count"] == 1


class TestRendering:
    def test_line_is_single_and_prefixed(self, synthetic, caplog, monkeypatch):
        monkeypatch.setenv("VIBEVOICE_DIAGNOSTICS", "1")
        for tensor in list(synthetic.parameters()) + list(synthetic.buffers()):
            _tag(tensor)
        logger_name = "ComfyUI_VibeVoice.modules.memory_census"
        with caplog.at_level(logging.INFO, logger=logger_name):
            report_census(synthetic, phase="pre-h2d:synthetic")
        lines = [
            r for r in caplog.records if r.getMessage().startswith("[vvcensus]")
        ]
        assert len(lines) == 1
        message = lines[0].getMessage()
        assert "\n" not in message
        assert "pre-h2d:synthetic" in message
        assert "private=0B" in message
        assert "100.0%" in message

    def test_line_is_silent_in_production(self, synthetic, caplog, monkeypatch):
        """Default (no env var set) must publish nothing."""
        monkeypatch.delenv("VIBEVOICE_DIAGNOSTICS", raising=False)
        monkeypatch.delenv("VIBEVOICE_RAM_CENSUS", raising=False)
        for tensor in list(synthetic.parameters()) + list(synthetic.buffers()):
            _tag(tensor)
        with caplog.at_level(
            logging.INFO, logger="ComfyUI_VibeVoice.modules.memory_census"
        ):
            report_census(synthetic, phase="pre-h2d:synthetic")
        assert not [r for r in caplog.records if "[vvcensus]" in r.getMessage()]

    def test_families_are_capped_but_counted(self):
        report = census(Synthetic())
        report["families"] = {f"Kind{i}/params": 1024 * (i + 1) for i in range(10)}
        message = format_census(report, phase="many-families")
        assert "families[10]" in message
        assert "+2 more" in message

    def test_gate_defaults_off_and_honours_env(self, monkeypatch):
        monkeypatch.delenv("VIBEVOICE_DIAGNOSTICS", raising=False)
        monkeypatch.delenv("VIBEVOICE_RAM_CENSUS", raising=False)
        assert census_enabled() is False, "production must be silent"
        for value in ("0", "false", "No", "OFF", " off "):
            monkeypatch.setenv("VIBEVOICE_RAM_CENSUS", value)
            assert census_enabled() is False
        monkeypatch.setenv("VIBEVOICE_RAM_CENSUS", "1")
        assert census_enabled() is True

    def test_master_switch_overrides_the_sub_gate(self, monkeypatch):
        monkeypatch.setenv("VIBEVOICE_RAM_CENSUS", "0")
        monkeypatch.setenv("VIBEVOICE_DIAGNOSTICS", "1")
        assert census_enabled() is True
        monkeypatch.setenv("VIBEVOICE_DIAGNOSTICS", "0")
        assert census_enabled() is False

    def test_disabled_gate_still_returns_the_report(self, synthetic, caplog, monkeypatch):
        monkeypatch.delenv("VIBEVOICE_DIAGNOSTICS", raising=False)
        monkeypatch.setenv("VIBEVOICE_RAM_CENSUS", "0")
        with caplog.at_level(
            logging.INFO, logger="ComfyUI_VibeVoice.modules.memory_census"
        ):
            report = report_census(synthetic)
        assert not [r for r in caplog.records if "[vvcensus]" in r.getMessage()]
        assert report["param_bytes"] > 0


# ====================================================================
# Real aimdo route: the acceptance probe
# ====================================================================

def _aimdo_ready() -> bool:
    """True when control.init + init_devices both succeed (main.py:74/:278).

    Copied from tests/test_dynamic_vram_mechanism.py — including its repair
    of comfy_aimdo's import-order trap: ``host_buffer`` / ``model_mmap`` /
    ``model_vbar`` / ``vram_buffer`` / ``storage`` snapshot ``lib =
    control.lib`` AT IMPORT TIME. Under pytest those modules are imported (via
    ``comfy.memory_management``) before ``control.init()`` loads the native
    library, so their ``lib`` is None forever and ``register_load_device``
    would crash on ``hostbuf_allocate``. A reload re-runs each module's
    top-level binding after the library exists, and callers resolve through the
    module at call time, so the reload is transparent.
    """
    import importlib

    try:
        if not control.init():
            return False
        if not bool(
            control.init_devices(
                (d.index, int(2 * 1024 ** 3))
                for d in comfy.model_management.get_all_torch_devices()
            )
        ):
            return False
        import comfy_aimdo

        for name in ("host_buffer", "model_mmap", "model_vbar", "vram_buffer", "storage"):
            submodule = getattr(comfy_aimdo, name, None)
            if submodule is not None and getattr(submodule, "lib", None) is None:
                importlib.reload(submodule)
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def aimdo_runtime():
    """Flip the two globals main.py:300-301 sets, for real, then undo them.

    Restores ``aimdo_enabled`` and the ``CoreModelPatcher`` alias in
    ``finally``: other tests in this process read both, and
    ``comfy.utils.load_torch_file`` branches on the flag.
    """
    if not torch.cuda.is_available() or not _aimdo_ready():
        pytest.skip("comfy-aimdo native stack unavailable headless")
    old_enabled = comfy.memory_management.aimdo_enabled
    old_alias = comfy.model_patcher.CoreModelPatcher
    comfy.memory_management.aimdo_enabled = True
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    yield
    comfy.memory_management.aimdo_enabled = old_enabled
    comfy.model_patcher.CoreModelPatcher = old_alias
    gc.collect()
    with contextlib.suppress(Exception):
        control.deinit()


def _tiny_handler(module):
    """Minimal stand-in for ExternalVibeVoiceModelHandler (same shape as the
    dynamic-mechanism test): the pack's handler is an ``nn.Module`` that owns
    the real model, which is how core's ``model_size()`` sees it.
    """

    class TinyHandler(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = module
            self.processor = None
            self.model_pack_name = "tiny-census"
            self.attention_mode = "eager"
            self.cache_key = "tiny-census"
            self.source_path = ""

        def load_model(self, device, attention_mode: str = "eager"):
            pass

    return TinyHandler()


#: LayerNorm width chosen so its weights clear core's 16KB force-load
#: threshold (comfy/model_patcher.py:1975-1989) by 2x.
NORM = 16384


class Tiny(nn.Module):
    """Every leaf here converts (``nn.Linear``/``nn.LayerNorm``/``nn.Embedding``
    all have streaming forwards, modules/comfy_stream.py:245-251) and every
    leaf is above the force-load threshold, so all three take the vbar path.
    """

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(256, 256, dtype=torch.bfloat16)
        self.norm = nn.LayerNorm(NORM, dtype=torch.bfloat16)
        self.emb = nn.Embedding(64, 256, dtype=torch.bfloat16)


@pytest.fixture
def tiny_checkpoint():
    d = tempfile.mkdtemp(prefix="vibevoice_census_")
    path = os.path.join(d, "tiny.safetensors")
    import safetensors.torch as st

    st.save_file(
        {
            "lin.weight": torch.randn(256, 256, dtype=torch.bfloat16),
            "lin.bias": torch.randn(256, dtype=torch.bfloat16),
            "norm.weight": torch.ones(NORM, dtype=torch.bfloat16),
            "norm.bias": torch.zeros(NORM, dtype=torch.bfloat16),
            "emb.weight": torch.randn(64, 256, dtype=torch.bfloat16),
        },
        path,
    )
    yield path


def _load_tiny(tiny_checkpoint):
    """The real loader route: aimdo file views -> assign -> streaming convert."""
    sd = comfy.utils.load_torch_file(tiny_checkpoint)
    with torch.device("meta"):
        model = Tiny()
    loaded = _load_state_dict_into_model_from_memory(
        model, sd, preserve_file_views=True,
    )
    convert_tree_for_streaming(loaded)
    return loaded


def _dynamic_patcher(loaded):
    cls = select_patcher_class("dense", torch.device("cuda"))
    return cls(
        _tiny_handler(loaded),
        load_device=comfy.model_management.get_torch_device(),
        offload_device=torch.device("cpu"),
    )


@pytest.mark.usefixtures("aimdo_runtime")
class TestCensusOnRealDynamicRoute:
    def test_pre_h2d_all_param_bytes_are_file_views(self, tiny_checkpoint):
        """Before the H2D: every param byte is an aimdo file view, 0 private."""
        report = census(_load_tiny(tiny_checkpoint))

        assert report["param_bytes"] == LIN_W + LIN_B + EMB_W + 2 * NORM * 2
        assert report["param_private_bytes"] == 0
        assert report["param_view_bytes"] == report["param_bytes"]
        assert report["mmap_refs_storages"] == 5

    def test_post_load_to_device_nothing_copied_anywhere(self, tiny_checkpoint):
        """The acceptance probe.

        After a real ``load_to_device`` on the dynamic route the model must
        still hold file views (core paged it through vbar), nothing may sit on
        the device, and the patcher must have stashed nothing. A private byte,
        an offhost byte or a backup entry here is the host-RAM defect this
        census exists to catch.
        """
        loaded = _load_tiny(tiny_checkpoint)
        patcher = _dynamic_patcher(loaded)
        try:
            load_to_device(patcher)
            report = report_census(loaded, patcher, phase="post-h2d:tiny")

            assert report["param_private_bytes"] == 0
            assert report["param_view_bytes"] == report["param_bytes"]
            assert report["param_view_bytes"] / report["param_bytes"] == 1.0
            assert report["param_offhost_bytes"] == 0
            assert report["backup_bytes"] == 0
            assert report["backup_count"] == 0
            assert report["backup_buffer_bytes"] == 0
            # The vbar ranges core allocated at comfy/model_patcher.py:1993
            # cover the whole model (plus per-range alignment slack from
            # ``vram_aligned_size``): it really took the paged path.
            assert report["vbar_count"] == 3
            assert report["param_bytes"] <= report["vbar_bytes"]
            assert report["vbar_bytes"] - report["param_bytes"] <= 4096
        finally:
            with contextlib.suppress(Exception):
                patcher.unpatch_model(destroy=True)
            gc.collect()
            comfy.model_management.soft_empty_cache()

    def test_sub_threshold_module_is_force_loaded_and_stashed(self, tiny_checkpoint):
        """The <16KB case, measured: a device copy, not a host copy.

        Core force-loads small modules on purpose
        (comfy/model_patcher.py:1975-1989). This pins what that costs: the
        device owns a copy, the stash owns the VIEW, and the host gains no
        private bytes — the reason ``offhost`` is its own census state.
        """
        sd = comfy.utils.load_torch_file(tiny_checkpoint)
        with torch.device("meta"):
            model = Tiny()
        loaded = _load_state_dict_into_model_from_memory(
            model, sd, preserve_file_views=True,
        )
        loaded.norm = nn.LayerNorm(256, dtype=torch.bfloat16)
        loaded.norm.weight = nn.Parameter(torch.zeros(256, dtype=torch.bfloat16))
        loaded.norm.bias = nn.Parameter(torch.zeros(256, dtype=torch.bfloat16))
        for tensor in (loaded.norm.weight, loaded.norm.bias):
            _tag(tensor)
        convert_tree_for_streaming(loaded)

        patcher = _dynamic_patcher(loaded)
        try:
            load_to_device(patcher)
            report = census(loaded, patcher)

            assert report["param_offhost_bytes"] == 512 * 2
            assert report["param_private_bytes"] == 0
            assert report["backup_count"] == 2
            assert report["backup_view_bytes"] == 512 * 2
            assert report["backup_private_bytes"] == 0
        finally:
            with contextlib.suppress(Exception):
                patcher.unpatch_model(destroy=True)
            gc.collect()
            comfy.model_management.soft_empty_cache()


class TestRssSamplerMarksArePrinted:
    """Regression: a mark nobody can see attributes nothing.

    2026-09-30: `RssSampler.mark()` appended to `self.marks`, but the only
    printer was `profile()`, gated on `series=True`. A mark added specifically
    to attribute the fp8 load's host-RAM spike therefore never reached the
    console, and a live measurement round was spent on an inert probe. The
    paste-ready line must now carry every mark.
    """

    def test_line_carries_every_mark_with_its_private_reading(self):
        from ComfyUI_VibeVoice.modules.memory_census import RssSampler

        sampler = RssSampler(label="probe")
        sampler.start = sampler.sample()
        sampler.end = sampler.start
        sampler.mark("stream-begin")
        sampler.mark("first-view")
        sampler.mark("stream-end")

        line = sampler.line("probe")
        assert "marks=" in line, line
        digest = line.split("marks=")[1].split(" note=")[0]
        labels = [part.split("@")[0] for part in digest.split(",")]
        assert labels == ["stream-begin", "first-view", "stream-end"], digest
        for part in digest.split(","):
            assert "@" in part and part.split("@")[1], part

    def test_line_without_marks_is_still_wellformed(self):
        from ComfyUI_VibeVoice.modules.memory_census import RssSampler

        sampler = RssSampler(label="probe")
        sampler.start = sampler.sample()
        sampler.end = sampler.start
        line = sampler.line("probe")
        assert "marks=-" in line, line
        assert " note=[" in line, line
