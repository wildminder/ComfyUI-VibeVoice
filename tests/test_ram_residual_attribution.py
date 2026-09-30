"""Regression: WHERE the external dense load's host bytes are (measured 2026-09-29).

The 1.5B bf16 file was reported as leaving ~7GB resident
(docs/2026-09-29-dynamic-vram-port-and-int64-conv-regression.md, "UPDATE 4").
``tests/probe_ram_census.py`` measured it on the real file with aimdo up:

    phase-a  load_external      peak_ws_delta = 0.05 GiB (0.01x file)
    phase-b  load_to_device     peak_ws_delta = 0.00 GiB
    phase-c  generate           peak_ws_delta = 6.33 GiB (1.26x file), STAYS
    residual: pin_state['weights-loaded'].size = 5.47 GiB  (unaccounted +0.86)

and, at every one of those three points, the model's own storages read
``params view=5.04GB(100.0%) private=0B offhost=1.12MB stash=883.63KB(100% view)``.

So the retained quantity the loader controls is ZERO private bytes, and the
bytes a user sees are core's pinned staging buffer created in
``ModelPatcherDynamic.load`` (comfy/model_patcher.py:1874-1881) — not a copy
the node made. These tests pin that contract so a future change cannot
silently reintroduce a host copy without one of them going red:

* :class:`TestDenseRouteRetainsNoPrivateBytes` — the dynamic route must bind
  file views (0 private) after a real ``load_to_device``.
* :class:`TestTheAssertionHasTeeth` — the SAME measurement on the legacy
  ``preserve_file_views=False`` branch reports the whole state dict as
  private. Without this, "0 private" could pass for a census that simply
  cannot see anything, and the positive test would prove nothing.
* :class:`TestResidualIsNotModelStorage` — the census sees none of the
  residual, so attributing a spike to the model from the census alone would
  be wrong; this test states that explicitly.

aimdo bootstrap is copied from tests/test_ram_census.py on purpose: a fixture
cannot be shared across test modules without becoming a conftest-wide GPU
dependency.
"""

import contextlib
import gc
import importlib
import inspect
import os
import re
import tempfile

import pytest
import torch
from torch import nn

import comfy.memory_management
import comfy.model_management
import comfy.model_patcher
import comfy.utils
from comfy_aimdo import control

from ComfyUI_VibeVoice.modules.comfy_stream import convert_tree_for_streaming
from ComfyUI_VibeVoice.modules.external_loader import (
    _load_state_dict_into_model_from_memory,
)
from ComfyUI_VibeVoice.modules.memory_census import census, rss_bytes
from ComfyUI_VibeVoice.modules.patcher import load_to_device, select_patcher_class

# Concrete KB-scale quantities for the fixture below.
LIN_W = 256 * 256 * 2
LIN_B = 256 * 2
EMB_W = 64 * 256 * 2
NORM = 256 * 2
TOTAL_BYTES = LIN_W + LIN_B + EMB_W + 2 * NORM


def _aimdo_ready() -> bool:
    """control.init + init_devices, with comfy_aimdo's import-order repair.

    ``host_buffer``/``model_mmap``/``model_vbar`` bind ``lib = control.lib`` at
    import time; under pytest they are imported before ``control.init()``, so
    their ``lib`` is None forever unless reloaded afterwards.
    """
    try:
        if not control.init():
            return False
        devices = list(comfy.model_management.get_all_torch_devices())
        if not bool(control.init_devices(
                (d.index, int(2 * 1024 ** 3)) for d in devices)):
            return False
        import comfy_aimdo

        for name in ("host_buffer", "model_mmap", "model_vbar",
                     "vram_buffer", "storage"):
            submodule = getattr(comfy_aimdo, name, None)
            if submodule is not None and getattr(submodule, "lib", None) is None:
                importlib.reload(submodule)
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def aimdo_runtime():
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


class Tiny(nn.Module):
    """Every leaf converts (Linear/LayerNorm/Embedding all stream)."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(256, 256, dtype=torch.bfloat16)
        self.norm = nn.LayerNorm(256, dtype=torch.bfloat16)
        self.emb = nn.Embedding(64, 256, dtype=torch.bfloat16)


class _Handler(nn.Module):
    """Stand-in for ExternalVibeVoiceModelHandler: owns the real model, so
    core's ``model_size()`` and ``dynamic_pins`` see it, exactly as the node's
    handler does."""

    def __init__(self, module):
        super().__init__()
        self.model = module
        self.processor = None
        self.model_pack_name = "tiny-residual"
        self.attention_mode = "eager"
        self.cache_key = "tiny-residual"
        self.source_path = ""

    def load_model(self, device, attention_mode: str = "eager"):
        pass


@pytest.fixture
def tiny_checkpoint():
    d = tempfile.mkdtemp(prefix="vibevoice_residual_")
    path = os.path.join(d, "tiny.safetensors")
    import safetensors.torch as st

    st.save_file(
        {
            "lin.weight": torch.randn(256, 256, dtype=torch.bfloat16),
            "lin.bias": torch.randn(256, dtype=torch.bfloat16),
            "norm.weight": torch.ones(256, dtype=torch.bfloat16),
            "norm.bias": torch.zeros(256, dtype=torch.bfloat16),
            "emb.weight": torch.randn(64, 256, dtype=torch.bfloat16),
        },
        path,
    )
    yield path


def _loaded(tiny_checkpoint, preserve_file_views: bool):
    sd = comfy.utils.load_torch_file(tiny_checkpoint)
    with torch.device("meta"):
        model = Tiny()
    return _load_state_dict_into_model_from_memory(
        model, sd, preserve_file_views=preserve_file_views,
    )


class TestLoaderChoosesFileViewsOnTheDynamicRoute:
    """The one line that decides views-vs-clone must stay wired to the route.

    ``_load_state_dict_into_model_from_memory`` is called from every load
    path (dense AND quant — finding F2 moved the quant families onto the
    dynamic route, so they carry the same argument), and its
    ``preserve_file_views`` value is the ONLY thing standing between "file
    views" (0 private bytes) and "clone the whole checkpoint" (1x file in
    private memory — measured at 2.01x peak working set in this session's
    first probe run, which took exactly that branch).

    Every other test here calls the helper directly, so none of them can see
    the loader passing the wrong value. These pin the call sites themselves.
    Verified to fail when the argument is replaced by a literal.
    """

    def _dense_call_sites(self):
        from ComfyUI_VibeVoice.modules import external_loader

        source = inspect.getsource(external_loader)
        return re.findall(
            r"_load_state_dict_into_model_from_memory\(\s*"
            r"model,\s*state_dict,\s*preserve_file_views=([^,)]+)",
            source,
        )

    def test_both_dense_sites_pass_the_resolved_route_flag(self):
        arguments = [a.strip() for a in self._dense_call_sites()]
        assert len(arguments) >= 2, (
            "expected the dense AND quant load paths to bind weights through "
            f"the helper; found {len(arguments)} call site(s)"
        )
        resolved = ("dynamic_route",)
        for argument in arguments:
            is_resolved = (
                argument in resolved
                # quant sites resolve inline rather than via a local
                or argument.startswith("dynamic_vram_available(")
            )
            assert is_resolved, (
                f"a load site passes preserve_file_views={argument!r}; it must "
                "pass the route's own resolved flag. A literal False clones "
                "every tensor into private host memory — 5.04GB on the 1.5B, "
                "measured — and a literal True would pin file views on the "
                "legacy route, which cannot stream from them."
            )
        # Both spellings must still be exercised somewhere: the local the
        # dense paths compute once, and the inline form the quant sites use.
        assert "dynamic_route" in arguments
        assert any(a.startswith("dynamic_vram_available(") for a in arguments)

    def test_the_route_flag_is_the_dynamic_selector_not_a_constant(self):
        from ComfyUI_VibeVoice.modules import external_loader

        source = inspect.getsource(external_loader)
        assert re.search(
            r"dynamic_route\s*=\s*\(\s*select_patcher_class\(", source
        ), ("dynamic_route must be resolved from select_patcher_class, so the "
            "loader and the patcher construction sites cannot disagree")


@pytest.mark.usefixtures("aimdo_runtime")
class TestDenseRouteRetainsNoPrivateBytes:
    """The measured contract: the dynamic route hands core FILE VIEWS."""

    def test_binding_the_state_dict_copies_nothing(self, tiny_checkpoint):
        report = census(_loaded(tiny_checkpoint, preserve_file_views=True))

        assert report["param_bytes"] == TOTAL_BYTES
        assert report["param_private_bytes"] == 0, (
            "every param byte must be an aimdo file view on the dynamic "
            "route; a private byte here is a host copy the load made"
        )
        assert report["param_view_bytes"] == TOTAL_BYTES

    def test_load_to_device_leaves_no_private_bytes_and_no_private_stash(
        self, tiny_checkpoint
    ):
        loaded = _loaded(tiny_checkpoint, preserve_file_views=True)
        convert_tree_for_streaming(loaded)

        cls = select_patcher_class("dense", torch.device("cuda"))
        assert cls.__name__.startswith("VibeVoiceDynamic"), (
            "the dense family must still resolve to the dynamic patcher; if "
            "this regressed, the measurements below describe a different "
            "route than the one users run"
        )
        patcher = cls(
            _Handler(loaded),
            load_device=comfy.model_management.get_torch_device(),
            offload_device=torch.device("cpu"),
        )
        try:
            load_to_device(patcher)
            report = census(loaded, patcher)

            assert report["param_private_bytes"] == 0
            assert report["buffer_private_bytes"] == 0
            # A stashed VIEW is the mapping core keeps for its <=16KB
            # force-load path (comfy/model_patcher.py:1975-1989); a stashed
            # PRIVATE tensor would be a live host copy.
            assert report["backup_private_bytes"] == 0
            assert report["backup_buffer_private_bytes"] == 0
        finally:
            with contextlib.suppress(Exception):
                patcher.unpatch_model(destroy=True)
            gc.collect()
            comfy.model_management.soft_empty_cache()


class TestTheAssertionHasTeeth:
    """Non-vacuity: the same measurement on the legacy branch DOES report bytes.

    This is the "verified to fail before the fix" half. ``preserve_file_views``
    is the one line that decides whether the route keeps file views or clones
    every tensor into private memory (modules/external_loader.py:1560-1566);
    when that line is reverted, the assertions above go red on real numbers
    rather than passing vacuously.
    """

    def test_the_legacy_clone_branch_reports_the_whole_file_as_private(
        self, tiny_checkpoint
    ):
        report = census(_loaded(tiny_checkpoint, preserve_file_views=False))

        assert report["param_private_bytes"] == TOTAL_BYTES, (
            "the legacy branch clones every tensor; if this ever reads 0, the "
            "census cannot see host allocations and the dynamic-route tests "
            "prove nothing"
        )
        assert report["param_view_bytes"] == 0


class TestResidualIsNotModelStorage:
    """Why a post-hoc census is not enough to attribute a spike.

    The measured 1.5B residual was 6.33 GiB of working set that stays, of
    which 5.47 GiB is core's pinned ``weights-loaded`` HostBuffer
    (comfy/model_patcher.py:1874-1881) — an object no amount of walking the
    model's parameters will ever see. These tests state that boundary so the
    next person measuring a spike checks the pin buffers before blaming the
    loader.
    """

    def test_census_reports_no_private_bytes_while_the_process_holds_more(
        self, tiny_checkpoint
    ):
        loaded = _loaded(tiny_checkpoint, preserve_file_views=True)
        report = census(loaded)
        before_ws, _ = rss_bytes()

        # A host allocation the census cannot see — the shape of core's
        # pinned staging, standing in for it at KB scale. It must be WRITTEN:
        # a freshly ``empty`` allocation reserves address space without
        # committing pages, which is exactly why the earlier commit-vs-
        # resident gap in the 1.5B numbers needed private-commit figures.
        #
        # The measurement is PROCESS-private memory, deliberately NOT torch
        # memory: torch's CPU allocator hands back a warm block when one is
        # available, so an N-byte torch allocation adds anywhere from 0 to N
        # bytes to the working set depending on what the process already
        # holds. In a full-suite run that made the delta assertion below pass
        # or fail on allocator state alone. A bytearray of the same size is a
        # plain heap allocation, and the census blindness — which is what this
        # test is about — is identical either way: the point is that the
        # census walks the MODEL's storages, not the process's.
        ballast = bytearray(8 * 1024 * 1024)
        for offset in range(0, len(ballast), 4096):
            ballast[offset] = 1
        try:
            after_ws, _ = rss_bytes()
            assert report["param_private_bytes"] == 0
            assert report["param_bytes"] == TOTAL_BYTES
            assert after_ws - before_ws > len(ballast) // 2, (
                "the ballast must actually land in the working set, or this "
                "test is not demonstrating the blind spot"
            )
        finally:
            del ballast
            gc.collect()

    def test_pinned_pin_buffers_are_a_separate_quantity_from_the_census(
        self, tiny_checkpoint
    ):
        """The pins live on the patcher's model, not on the census subject."""
        loaded = _loaded(tiny_checkpoint, preserve_file_views=True)
        convert_tree_for_streaming(loaded)
        handler = _Handler(loaded)
        # Core's pin_state shape (comfy/model_patcher.py:1874-1881), with a
        # stand-in for the six HostBuffers.
        handler.dynamic_pins = {
            torch.device("cuda"): {"weights-loaded": (object(), [], [-1], [0], [0], {})}
        }

        report = census(loaded, handler)
        assert report["param_private_bytes"] == 0
        assert report["param_bytes"] == TOTAL_BYTES
        # Nothing in the census is derived from dynamic_pins: that is the
        # point. The attribution has to read them directly.
        assert "pin_bytes" not in report
