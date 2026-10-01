"""Tests for the sampled-PEAK half of the RAM census (task fp8-peak).

``census`` inventories the storages a model HOLDS; it cannot see a transient
— and the 7B fp8 report is a transient (+17GB, gone before the post-H2D
census). These tests cover the instrument that does see it: a background
sampler over the process working set, private resident set and commit.

Scale note: the real event is 8.82GiB, so nothing here loads a model. The
sampler is exercised with a few MiB of Python buffers held across a
synchronous ``sample()``, which is the same mechanism at 1/1000 the size —
the assertions are about WHAT is tracked, never about the magnitudes.
"""

import time

import pytest

from ComfyUI_VibeVoice.modules.memory_census import (
    RssSampler,
    memory_snapshot,
    peak_delta,
    rss_bytes,
)


def _touch_mib(mib: int) -> list:
    """Touch ``mib`` MiB of real, touched memory and keep it alive."""
    blocks = [bytearray(1024 * 1024) for _ in range(mib)]
    for block in blocks:
        block[0] = 1
        block[-1] = 1
    return blocks


#: How far two consecutive live process readings may drift before the test
#: treats it as a real disagreement. A full-suite pytest process has other
#: threads (torch's autotuner, ComfyUI's samplers) touching pages, and the
#: observed drift is a few hundred KB; 64 MiB is far above that noise and far
#: below the 96 MiB blocks these tests hold, so it cannot mask a genuine
#: divergence between the two readers.
_READING_DRIFT_TOLERANCE = 64 * 1024 * 1024


class TestMemorySnapshot:
    def test_reports_all_three_quantities(self):
        snapshot = memory_snapshot()
        # The three PROCESS counters are the contract. ``sys_used``
        # (machine-wide RAM) rides along so a [vvrss] line can attribute a
        # jump to this process or to the box, so assert the required keys are
        # present rather than pinning the whole key set.
        assert {"ws", "uss", "private"} <= set(snapshot)
        assert snapshot["ws"] > 0, "this platform must be able to read a working set"
        assert snapshot["uss"] > 0
        assert snapshot["private"] > 0

    def test_rss_bytes_is_the_two_tuple_shorthand(self):
        # These are TWO SEPARATE live readings of a MOVING process: a
        # background thread in a full-suite run can add or drop pages between
        # the calls. The contract under test is that ``rss_bytes`` is the
        # two-tuple shorthand over the SAME reader, not that the OS returns an
        # identical number twice — asserting equality made this test fail
        # ~1 run in 2 on process-wide drift of a few hundred KB.
        ws, private = rss_bytes()
        snapshot = memory_snapshot()
        for name, before, after in (
            ("ws", ws, snapshot["ws"]),
            ("private", private, snapshot["private"]),
        ):
            drift = abs(after - before)
            assert drift <= _READING_DRIFT_TOLERANCE, (
                f"{name} drifted {drift} B between two consecutive readings; "
                "if this is a real disagreement rather than process noise, "
                "rss_bytes and memory_snapshot have diverged"
            )


class TestRssSampler:
    def test_catches_a_transient_a_post_hoc_reading_cannot(self):
        """The whole reason the sampler exists, in miniature.

        Hold memory across a synchronous sample, then release it BEFORE the
        final reading. A post-hoc delta sees nothing; the peak must still
        report it.
        """
        before = rss_bytes()[0]
        with RssSampler(interval=0.01) as sampler:
            held = _touch_mib(96)
            sampler.mark("held")
            del held
        assert sampler.peak["ws"] > before, "peak must include the freed block"
        # ...and the END reading is back near the start, which is exactly the
        # shape of the reported defect: a spike that resolves.
        assert sampler.end["ws"] < sampler.peak["ws"]

    def test_stops_its_thread_and_keeps_sampling(self):
        with RssSampler(interval=0.01) as sampler:
            held = _touch_mib(32)
            time.sleep(0.05)
            del held
        first = sampler.samples
        time.sleep(0.05)
        assert sampler.samples == first, "__exit__ must stop the sampling thread"
        assert first > 1

    def test_marks_record_reading_and_running_peak(self):
        with RssSampler() as sampler:
            sampler.mark("start")
            held = _touch_mib(48)
            sampler.mark("held")
            del held
        labels = [label for label, _snapshot, _peak in sampler.marks]
        assert labels == ["start", "held"]
        held_reading = dict(sampler.marks[1][1])
        held_peak = dict(sampler.marks[1][2])
        assert held_peak["ws"] >= held_reading["ws"]
        assert held_peak["ws"] == sampler.peak["ws"]

    def test_line_names_every_quantity_it_tracks(self):
        with RssSampler() as sampler:
            pass
        line = sampler.line("phase-test")
        assert line.startswith("[vvrss] phase-test")
        for token in ("peak_ws=", "peak_uss=", "peak_private=",
                      "start_ws=", "end_ws=", "samples="):
            assert token in line

    def test_report_respects_the_env_gate(self, monkeypatch, caplog):
        import logging

        with RssSampler() as sampler:
            pass
        monkeypatch.setenv("VIBEVOICE_RAM_CENSUS", "0")
        with caplog.at_level(logging.INFO):
            line = sampler.report("silenced")
        assert "[vvrss] silenced" in line
        assert "[vvrss]" not in caplog.text

    def test_profile_is_off_without_a_series(self):
        assert "profile=off" in RssSampler().profile()

    def test_profile_renders_one_bucket_per_column(self):
        with RssSampler(interval=0.01, series=True) as sampler:
            held = _touch_mib(64)
            time.sleep(0.05)
            del held
        line = sampler.profile(buckets=8)
        assert "timeline" in line
        # One block character per bucket, between the ")" and the peak.
        blocks = line.split(") ", 1)[1].split(" peak=", 1)[0]
        assert len(blocks) == 8
        assert all(char in " ▁▂▃▄▅▆▇█" for char in blocks)


class TestPeakDelta:
    def test_reports_ws_and_private_deltas_against_a_baseline(self):
        with RssSampler() as sampler:
            held = _touch_mib(64)
            sampler.mark("held")
            del held
        delta = peak_delta(sampler, baseline=sampler.peak["ws"] - (16 << 20))
        assert delta["baseline"] > 0
        assert delta["peak_ws_delta"] == pytest.approx(16 << 20, rel=0.01)
        assert delta["peak_uss_delta"] >= 0
        assert delta["samples"] == sampler.samples

    def test_never_reports_a_negative_delta(self):
        with RssSampler() as sampler:
            pass
        delta = peak_delta(sampler, baseline=sampler.peak["ws"] + (1 << 30))
        assert delta["peak_ws_delta"] == 0
        assert delta["peak_uss_delta"] == 0
