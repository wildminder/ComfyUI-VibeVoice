"""Tests for the SHIPPED half of the RAM instrumentation (task fp8-peak).

The QA round found three things wrong, and these pin the fixes:

1. ``RssSampler``/``peak_delta`` were wired into NOTHING in production. A
   sampler reachable only from a probe is not a measurement a user can take,
   so :class:`TestTheSamplerIsActuallyWired` asserts the loader itself enters
   a sampled block. This is the "verified to fail without the fix" test: with
   the ``measured_load(...)`` wrapper removed from
   ``_stream_apply_safetensors`` it fails, because the source no longer
   contains the call.

2. The two flaky assertions in the previous round were order-dependent on
   live process RSS. :class:`TestLiveReadingsAreNotExact` states the corrected
   contract (a tolerance, not equality) so neither is reintroduced.

3. ``memory_census`` claimed ``uss`` "is the number that separates a real
   host-RAM defect from page cache". On Windows that is FALSE — measured, see
   ``tests/probe_mapped_pages_uss.py`` — and believing it turns every streamed
   checkpoint into a phantom 2x defect.
   :class:`TestUssIsNotClaimedToSeparatePageCache` pins the correction, and
   :func:`uss_counts_mapped_pages` is exercised for real at MB scale.

4. THE CORRECTION WAS ITSELF OVERWRITTEN, and shipped. The note generalised
   the ``mmap`` arm's result to "ws and uss both include ~1x file for ANY
   streamed read", on a route where the same probe measured the safetensors
   read — the one this loader performs — moving ``private`` and neither
   ``ws`` nor ``uss``. That sentence told a user reading the 7B fp8 line that
   a real private spike was page cache.
   :class:`TestReadShapeIsMeasuredPerArmNotAsOnePlatformRule` pins the
   per-shape replacement: :func:`read_shape_profile` measures each read shape
   and a control separately, and the shipped prose may not claim a rule it did
   not measure.

5. The dense (bf16) route had NO sampled block at all — the packet was wired
   for the quantized read only, so the 1.5B retention could not be observed
   in live ComfyUI. :class:`TestTheDenseRouteIsSampled` pins the wiring.

Everything here is KB/MB-scale. No test in this file loads a model.
"""

import gc
import inspect
import logging
import os
import tempfile

import pytest
import torch

from ComfyUI_VibeVoice.modules import base_loader
from ComfyUI_VibeVoice.modules import external_loader
from ComfyUI_VibeVoice.modules import generation
from ComfyUI_VibeVoice.modules import memory_census
from ComfyUI_VibeVoice.modules import memory_census as _census_module
from ComfyUI_VibeVoice.modules import asr_generation
from ComfyUI_VibeVoice.modules.memory_census import (
    RssSampler,
    measured_load,
    peak_delta,
    ram_measurement_guide,
    read_shape_profile,
    rss_bytes,
    uss_counts_mapped_pages,
)


class TestTheSamplerIsActuallyWired:
    """The load path must SAMPLE, not merely be measurable in principle.

    The QA round's central complaint: ``grep`` for the sampler found it only
    in ``modules/memory_census.py``, a probe, and a test. A user loading the
    7B fp8 file got no ``[vvrss]`` line, so the load-time peak the ask asks
    for simply could not be obtained from a shipped build.

    Asserted against the SOURCE rather than by running a load, because a real
    load is a multi-GB event and the wiring is a static property: the wrapper
    is a call the loader makes, so the call being absent is the regression.
    """

    def test_the_streamed_quant_read_is_wrapped_in_a_sampled_block(self):
        source = inspect.getsource(external_loader._stream_apply_safetensors)
        assert "measured_load(" in source, (
            "_stream_apply_safetensors is where a quantized load's host-RAM "
            "TRANSIENT happens, and it is sampled end to end so the peak can "
            "be attributed to the read rather than to 'the load'. Without "
            "this wrapper the only RAM numbers a user gets are post-hoc "
            "censuses, which by construction cannot see a spike that has "
            "already resolved."
        )

    def test_the_wrapper_brackets_the_read_with_marks(self):
        source = inspect.getsource(external_loader._stream_apply_safetensors)
        assert 'mark("stream-begin")' in source
        assert 'mark("stream-end")' in source, (
            "the two marks are what turn 'the load peaked at N' into 'the read "
            "peaked at N'; one mark is a phase, zero is unattributable"
        )

    def test_measured_load_is_reachable_from_the_loader_module(self):
        assert hasattr(external_loader, "measured_load"), (
            "the loader imports measured_load; if this is gone the wrapper "
            "above is dead code and the sampler is unreachable again"
        )


class TestTheDenseRouteIsSampled:
    """The packet covered the quantized read ONLY.

    ``grep`` found exactly one ``measured_load`` call site — the quant read in
    ``_stream_apply_safetensors`` — so a user loading the dense bf16 1.5B file
    got no ``[vvrss]`` line at all. That is precisely the file whose residual
    (phase-c growth to 6.33 GiB, core's 5.47 GiB pin buffer) had to be measured
    in a standalone probe because the shipped build could not report it.

    Asserted against the SOURCE: a real load is a multi-GB event and the
    wiring is a static property — the wrapper is a call the loader makes, so
    the call being absent IS the regression.

    The dense route used to be two phases (``dense-read-state-dict`` then
    ``dense-bind-state-dict``). Streaming merged them into one: the bind now
    happens per-tensor, so a single wrapper around the stream is both the read
    and the bind.
    """

    def test_the_dense_stream_is_sampled(self):
        source = inspect.getsource(external_loader)
        assert 'measured_load("dense-stream-apply")' in source, (
            "streaming a dense checkpoint onto the device is a whole-file "
            "event that resolves before the post-H2D census runs; without a "
            "sampled block the dense 1.5B route has no observable peak"
        )

    def test_the_dense_stream_is_the_only_dense_read(self):
        """The batch read/bind pair is gone from both loader branches.

        ``dense-read-state-dict`` / ``dense-bind-state-dict`` only survive in
        the unreachable ConvRot batch fallback; a live dense load must not go
        through it, or the whole-tree ``model.to()`` comes back with it.
        """
        from ComfyUI_VibeVoice.modules import external_loader as EL

        for fn in (EL.load_external_vibevoice_model,
                   EL.load_external_vibevoice_asr_model):
            src = inspect.getsource(fn)
            dense = src[src.index("_stream_apply_dense_safetensors("):]
            dense = dense.split("cast_model_to_dtype_if_needed")[0]
            assert "dense-read-state-dict" not in dense
            assert "dense-bind-state-dict" not in dense
            assert "_load_weight_state_dict(" not in dense

    @pytest.mark.parametrize("module", [generation, asr_generation])
    def test_the_h2d_phase_that_grows_core_pin_buffers_is_sampled(self, module):
        """``load_to_device`` is where core's dynamic ``load()`` runs.

        On the dynamic route that is where the pinned host staging buffer is
        grown (``comfy/model_patcher.py:1874-1881``), which is the residual
        the 1.5B diagnosis blames. A post-hoc census cannot see it.
        """
        source = inspect.getsource(module)
        assert 'measured_load("load-to-device")' in source, (
            f"{module.__name__} does not sample load_to_device, so the bytes "
            "core's pinned staging keeps resident are unobservable in live "
            "ComfyUI"
        )
        assert "load_to_device(patcher)" in source


class TestMeasuredLoadContract:
    """``measured_load`` must never be able to break a load, and must report."""

    def test_yields_a_sampler_and_logs_on_exit(self, caplog, monkeypatch):
        monkeypatch.setenv("VIBEVOICE_DIAGNOSTICS", "1")
        with caplog.at_level(logging.INFO):
            with measured_load("phase-under-test") as sampler:
                assert isinstance(sampler, RssSampler)
                sampler.mark("mid")
        assert any("[vvrss] phase-under-test" in r.message
                   for r in caplog.records)

    def test_reports_even_when_the_body_raises(self, caplog, monkeypatch):
        """A failed load still reached a peak; that peak is the evidence."""
        monkeypatch.setenv("VIBEVOICE_DIAGNOSTICS", "1")
        with caplog.at_level(logging.INFO):
            with pytest.raises(ValueError):
                with measured_load("phase-that-fails"):
                    raise ValueError("boom")
        assert any("[vvrss] phase-that-fails" in r.message
                   for r in caplog.records)

    def test_respects_the_env_gate(self, monkeypatch, caplog):
        monkeypatch.delenv("VIBEVOICE_DIAGNOSTICS", raising=False)
        monkeypatch.setenv("VIBEVOICE_RAM_CENSUS", "0")
        with caplog.at_level(logging.INFO):
            with measured_load("silenced"):
                pass
        assert "[vvrss]" not in caplog.text

    def test_the_line_carries_all_three_peaks_and_the_platform_guide(self):
        """All three peaks and the guide, or the line misleads.

        The three counters are charged DIFFERENTLY for different read shapes
        on this host (measured: a plain ``mmap`` read moves ws/uss, the
        loader's safetensors read moves private), so a line that carried only
        one of them would attribute a read to whichever counter happened to
        move, and a user would report the wrong defect.
        """
        with measured_load("phase-line") as sampler:
            sampler.mark("x")
        line = sampler.line("phase-line")
        for token in ("peak_ws=", "peak_uss=", "peak_private=", "end_private=",
                      "note=["):
            assert token in line, f"a [vvrss] line without {token} is not quotable"
        assert ("COUNTS file-backed" in line) or ("EXCLUDES file-backed" in line), (
            "the line must state which platform reading applies to the mmap "
            "arm, so a reader is never left guessing whether uss means "
            "private RAM"
        )

    def test_the_line_does_not_generalise_the_mmap_arm_to_every_read(self):
        """The exact sentence QA refuted, guarded against coming back.

        It told a user reading the 7B fp8 line that ws and uss include ~1x
        file for ANY streamed read — on the one route where the session's own
        probe measured the opposite, so a real private spike could be
        dismissed as page cache.
        """
        with measured_load("phase-refuted") as sampler:
            pass
        line = sampler.line("phase-refuted")
        assert "for any streamed read" not in line, (
            "'ws and uss include ~1x file for any streamed read' was measured "
            "false for the loader's own safetensors read; shipping it again "
            "is what let a real private spike be dismissed as page cache"
        )


class TestReadShapeIsMeasuredPerArmNotAsOnePlatformRule:
    """The generalisation the QA round refuted, and its replacement."""

    def test_the_probe_separates_the_two_read_shapes_and_a_control(self):
        _census_module._USS_COUNTS_MAPPED = None
        _census_module._READ_SHAPE_PROFILE = None
        _census_module._READ_SHAPE_PROBE_BYTES = 0
        try:
            profile = read_shape_profile(probe_bytes=8 * 1024 * 1024)
        finally:
            _census_module._USS_COUNTS_MAPPED = None
            _census_module._READ_SHAPE_PROFILE = None
            _census_module._READ_SHAPE_PROBE_BYTES = 0
            gc.collect()
        assert set(profile) >= {"mmap", "safetensors", "private_control"}, (
            "one platform-level yes/no cannot cover both read shapes: the "
            "plain mmap arm and the safetensors arm the loader actually "
            "performs are charged to different counters on this host, and the "
            "control arm is what makes either reading trustworthy"
        )
        for shape, arm in profile.items():
            assert set(arm) == {"ws", "uss", "private"}, (
                f"{shape} must report all three counters, got {sorted(arm)}"
            )

    def test_the_guide_reports_each_shape_separately(self):
        """The shipped note must not speak for a shape it did not measure."""
        guide = ram_measurement_guide()
        assert "mmap" in guide and "safetensors" in guide, (
            "the note must name both read shapes: that they are charged "
            "differently is the whole finding"
        )
        assert "any streamed read" not in guide

    def test_the_guide_does_not_claim_private_is_the_only_private_figure(self):
        """Also refuted by the same run: the safetensors arm moves private."""
        guide = ram_measurement_guide()
        assert "only private-RAM figure" not in guide, (
            "'private and the census are the only private-RAM figures' does "
            "not follow: on this host private moves by ~1x file for a "
            "safetensors read that moves neither ws nor uss"
        )

    def test_the_shipped_source_no_longer_states_the_refuted_rule(self):
        """Guarded at the source, so it cannot return via a later edit."""
        for module in (memory_census, external_loader, base_loader):
            text = inspect.getsource(module)
            assert "for any streamed read" not in text, (
                f"{module.__name__} re-states the refuted 'ws and uss include "
                "~1x file for any streamed read' rule"
            )
        assert "only private-RAM figure" not in inspect.getsource(memory_census)

    def test_the_loader_no_longer_excuses_the_peak_as_page_cache(self):
        """The 7B fp8 ~2x peak is an open residual, not a re-labelled one."""
        source = inspect.getsource(external_loader)
        assert "NOT a 2x host-RAM defect" not in source, (
            "that re-labelling was the consequence QA refuted; the reading it "
            "rested on came from the plain-mmap arm, which is not the shape "
            "this loader reads in"
        )

    def test_the_clone_line_is_cited_at_its_current_location(self):
        """The clone moved; the evidence that cites it must move with it."""
        for module in (memory_census, external_loader):
            text = inspect.getsource(module)
            assert "external_loader.py:1029" not in text, (
                "external_loader.py:1029 is no longer the clone(); the clone "
                "is at :1047 and a citation to the old line points a reader at "
                "the ValueError instead"
            )


class TestUssIsNotClaimedToSeparatePageCache:
    """The corrected claim, and the probe that establishes it."""

    def test_the_probe_returns_a_decision_and_caches_it(self):
        _census_module._USS_COUNTS_MAPPED = None
        _census_module._READ_SHAPE_PROFILE = None
        _census_module._READ_SHAPE_PROBE_BYTES = 0
        try:
            verdict = uss_counts_mapped_pages(probe_bytes=8 * 1024 * 1024)
            assert isinstance(verdict, bool)
            # Second call must be the cached answer, not a second probe.
            assert uss_counts_mapped_pages() is verdict
        finally:
            _census_module._USS_COUNTS_MAPPED = None
            _census_module._READ_SHAPE_PROFILE = None
            _census_module._READ_SHAPE_PROBE_BYTES = 0
            gc.collect()

    def test_the_probe_fails_closed_when_the_platform_cannot_be_measured(self):
        """Unknown must read as "do not quote uss as private".

        The probe touches the filesystem and the page cache, so it can fail
        on a locked-down host. The failure mode that matters is a WRONG
        answer, not a missing one: if it returned True without evidence, a
        streamed checkpoint would again be reported as a private-RAM defect.
        Break the file creation — the first thing the probe does — and the
        answer must be False.
        """
        import tempfile as _tempfile

        _census_module._USS_COUNTS_MAPPED = None
        _census_module._READ_SHAPE_PROFILE = None
        _census_module._READ_SHAPE_PROBE_BYTES = 0
        real_mkstemp = _tempfile.mkstemp

        def boom(*a, **k):
            raise OSError("cannot create probe file")

        _tempfile.mkstemp = boom
        try:
            assert uss_counts_mapped_pages(probe_bytes=4 * 1024 * 1024) is False
        finally:
            _tempfile.mkstemp = real_mkstemp
            _census_module._USS_COUNTS_MAPPED = None
            _census_module._READ_SHAPE_PROFILE = None
            _census_module._READ_SHAPE_PROBE_BYTES = 0
            gc.collect()

    def test_peak_delta_reports_which_platform_reading_applies(self):
        with RssSampler() as sampler:
            sampler.mark("held")
        delta = peak_delta(sampler, baseline=0)
        assert "uss_counts_mapped_pages" in delta, (
            "a bare uss_delta invites the exact misreading this corrects; the "
            "reader must be told whether it separates page cache here"
        )
        assert "peak_private_delta" in delta
        assert delta["peak_private_delta"] >= 0


class TestLiveReadingsAreNotExact:
    """The two flakes the QA round found, stated as the corrected contract.

    Both asserted something about a MOVING process that cannot be true:
    ``rss_bytes()`` and ``memory_snapshot()`` are two separate live reads of
    the same quantity, and a torch allocation may be served from a warm block
    and add nothing to the working set. Asserting equality / a fixed delta
    made two tests fail on roughly one run in two, depending only on what
    else the process had touched.
    """

    def test_consecutive_readings_agree_within_drift_not_exactly(self):
        ws_a, private_a = rss_bytes()
        ws_b, private_b = rss_bytes()
        # Two reads microseconds apart must be close, not identical.
        assert abs(ws_b - ws_a) <= 64 * 1024 * 1024
        assert abs(private_b - private_a) <= 64 * 1024 * 1024

    def test_a_plain_heap_allocation_lands_in_the_working_set(self):
        """The property the ballast test actually needs, without torch.

        A ``bytearray`` cannot be served from torch's allocator cache, so the
        working set must grow. This is the non-flaky form of the assertion
        that failed.
        """
        before, _ = rss_bytes()
        block = bytearray(16 * 1024 * 1024)
        for offset in range(0, len(block), 4096):
            block[offset] = 1
        try:
            after, _ = rss_bytes()
            assert after - before > len(block) // 2
        finally:
            del block
            gc.collect()

    def test_a_torch_allocation_is_NOT_guaranteed_to_move_the_working_set(self):
        """The reason the old assertion was flaky, pinned so it stays fixed.

        torch's CPU allocator hands back a warm block when the process
        already holds one, so an N-byte torch tensor can add as little as 0
        bytes to the working set. That is what made
        ``assert after_ws - before_ws > len(ballast) // 2`` fail on an 8MB
        ballast in a full-suite process: the ballast was real, the ALLOCATOR
        simply had a block to reuse, and the test measured the allocator's
        state rather than the code under test.

        The correct statement is a BOUND, not a floor: the working set
        cannot grow by more than the allocation, whatever the allocator
        decides. No lower bound is asserted, because none holds.
        """
        warm = torch.empty(8 * 1024 * 1024, dtype=torch.uint8)
        warm[::4096] = 1
        before, _ = rss_bytes()
        again = torch.empty(8 * 1024 * 1024, dtype=torch.uint8)
        again[::4096] = 1
        after, _ = rss_bytes()
        # A small allowance for process-wide drift (other threads, allocator
        # bookkeeping); what is excluded is the multi-MB growth the old
        # floor demanded.
        assert after - before <= len(again) + 64 * 1024 * 1024
        del warm, again
        gc.collect()
