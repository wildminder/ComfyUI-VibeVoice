"""Unit tests for the ``--accept`` audio oracle (plan step S0.2).

The plan's own words: "a test that asserts 'generation completes' or 'RMS > 0'
proved worthless here — it passed on audio that was pure garbage." These tests
exist so the oracle is pinned on *structure* — a duration band, a way the clip
ended, no long silence, a spectrum that is neither a flat noise band nor one
that wanders, and a level floor — and so that the failure the plan is chasing
(a 0.13 s near-silent clip from an EOS head that fires on the first latent) is
caught by construction.

Everything is synthetic and in-memory: no checkpoint, no GPU, no wav fixtures
committed to the repo. The clips are built from a harmonic stack shaped like
voiced speech, which is what the gates are calibrated against, and from the two
known-bad shapes the gates exist to catch.
"""

import importlib.util
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

DIAG_PATH = Path(__file__).parent.parent / "diag_realtime_quality.py"
SR = 24000


@pytest.fixture(scope="module")
def oracle():
    """The diagnostic module, loaded from disk.

    Same by-path load ``tests/realtime_e2e_support.py`` uses, so
    ``import diag_realtime_quality`` never enters the default suite's
    collection. The module's top-level imports pull in the ComfyUI path
    bootstrap, which conftest has already prepared.
    """
    spec = importlib.util.spec_from_file_location("diag_realtime_quality", DIAG_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------- clip builders --
def speech_shaped(seconds: float, f0: float = 140.0, rate_hz: float = 3.0) -> np.ndarray:
    """A voiced-speech stand-in: a harmonic stack under a syllabic envelope.

    Voiced speech is energy-dominated well below 4 kHz and its spectrum holds
    still, which is exactly the pair of properties the spectral gate reads. The
    harmonics are 1/k so the stack falls off like a real glottal source, and
    the envelope opens and closes so the clip has no silent run.
    """
    t = np.arange(int(SR * seconds)) / SR
    envelope = 0.5 * (1.0 + np.sin(2.0 * np.pi * rate_hz * t)) * np.sin(np.pi * t / seconds) ** 0.5
    voice = sum(
        np.sin(2.0 * np.pi * f0 * (k + 1) * t) / (k + 1)
        for k in range(6)
    )
    return (0.3 * envelope * voice).astype(np.float32)


def _write_wav(path: Path, x: np.ndarray, sr: int = SR) -> Path:
    """Write mono int16 PCM, the same shape ``save_wav`` produces."""
    clipped = np.clip(x, -1.0, 1.0)
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((clipped * 32767).astype("<i2").tobytes())
    return path


def _statuses(oracle, tmp_path, x, latents=None) -> dict[str, str]:
    """Gate the array in memory (no wav round trip) and return name -> status."""
    return {g.name: g.status for g in oracle._run_gates(x, SR, latents)}


# --------------------------------------------------------------------- tests --
def test_synthetic_good_clip_passes_every_gate(oracle):
    """A clip with the structure of speech is accepted, gates included.

    Deliberately asserts the duration and termination gates are not SKIPPED, so
    the two gates that can be skipped by omission are exercised as real verdicts
    here.
    """
    x = speech_shaped(3.0)
    latents = round(x.size / SR / oracle.LATENT_SECONDS)
    results = oracle._run_gates(x, SR, latents, oracle.TERMINATION_SELF)

    assert [g.status for g in results] == ["PASS"] * 5, [
        (g.name, g.status, g.detail) for g in results
    ]


def test_short_silent_clip_fails_duration_and_rms(oracle):
    """The F9 shape: the EOS head fired on the first latent, so a ~3 s
    utterance came back as 0.13 s of digital silence.

    Both failures the plan names are asserted: duration, because 0.13 s is far
    outside the band for the 23 latents the utterance needed, and rms, because
    the clip carries no signal at all.
    """
    x = np.zeros(int(0.133 * SR), dtype=np.float32)

    results = oracle._run_gates(x, SR, latents=23, termination=oracle.TERMINATION_SELF)
    statuses = {g.name: g.status for g in results}
    assert statuses["duration"] == "FAIL"
    assert statuses["level"] == "FAIL"
    assert "rms 0.0000" in next(g.detail for g in results if g.name == "level")

    # The duration gate measures length, not loudness, so against a latent
    # count of 1 the same clip is exactly on time and only the rms gate fires.
    # Pinned because it is the difference between the two gates, and because a
    # future change that made duration silently cover level would be a bug.
    assert {
        g.name: g.status
        for g in oracle._run_gates(x, SR, latents=1, termination=oracle.TERMINATION_SELF)
    }["duration"] == "PASS"


def test_long_silent_run_fails_the_silence_gate(oracle):
    """A 30 s clip that is 5 s of digital silence fails on the run, not on rms.

    The clip as a whole is loud and long, so only the run-length rule can catch
    this: the decoder stalled rather than breathed.
    """
    rng = np.random.default_rng(7)
    x = speech_shaped(30.0)
    x[int(12.0 * SR):int(17.0 * SR)] = 0.0  # 5 s of digital silence
    # int16 quantisation is not involved here, so make the stall strictly below
    # the floor rather than exactly zero.
    x[int(12.0 * SR):int(17.0 * SR)] = rng.uniform(-1e-6, 1e-6, int(5.0 * SR))

    statuses = _statuses(oracle, None, x)
    assert statuses["silence"] == "FAIL"
    detail = next(g.detail for g in oracle._run_gates(x, SR, None) if g.name == "silence")
    assert "5.00s" in detail


def test_white_noise_fails_spectral_stability(oracle):
    """White noise is rejected by the spectral gate.

    Measured note: the plan's clause — "centroid stable across the clip's
    thirds" — does NOT reject white noise, because white noise is perfectly
    stable (measured relative spread 0.006, versus 0.000 for speech-shaped
    audio). What rejects it is the flat noise band itself: a flat power spectrum
    has its centroid at sr/4, i.e. 0.5 of Nyquist (measured 0.5012), far above
    the voiced-speech ceiling. Both clauses live in the one ``spectral`` gate.
    """
    rng = np.random.default_rng(11)
    x = (0.3 * rng.standard_normal(int(SR * 3.0))).astype(np.float32)

    statuses = _statuses(oracle, None, x)
    assert statuses["spectral"] == "FAIL"
    detail = next(g.detail for g in oracle._run_gates(x, SR, None) if g.name == "spectral")
    assert "FLAT NOISE BAND" in detail


# ------------------------------------------------------ oracle plumbing tests --
def test_zero_length_clip_is_rejected_by_name(oracle, tmp_path):
    """A zero-length clip fails, and the level gate says so in words."""
    path = _write_wav(tmp_path / "empty.wav", np.zeros(0, dtype=np.float32))

    result = next(g for g in oracle.accept_wav(path) if g.name == "level")
    assert result.status == "FAIL"
    assert "zero-length" in result.detail


def test_unknown_latent_count_skips_the_duration_gate(oracle, tmp_path):
    """No latent count means the duration gate is SKIPPED with a reason.

    It must never be an unannounced pass: the number is not in a wav, and
    inferring it from the duration would make the gate unfalsifiable.
    """
    path = _write_wav(tmp_path / "good.wav", speech_shaped(3.0))

    duration = next(g for g in oracle.accept_wav(path) if g.name == "duration")
    assert duration.status == "SKIPPED"
    assert "latent count unknown" in duration.detail


def test_latent_count_comes_from_a_sidecar_when_not_given(oracle, tmp_path):
    """``<file>.latents`` supplies the count, and the duration gate then runs."""
    path = _write_wav(tmp_path / "good.wav", speech_shaped(3.0))
    path.with_name(path.name + ".latents").write_text("23\n", encoding="utf-8")

    count, origin = oracle.resolve_latents(path, None)
    assert count == 23
    assert origin.endswith("good.wav.latents")
    duration = next(g for g in oracle.accept_wav(path) if g.name == "duration")
    assert duration.status == "PASS"


def test_stored_golden_passes_every_gate(oracle):
    """The trusted reference clip must be accepted — this is the load-bearing one.

    A gate that rejects the golden is worse than no gate, because it teaches
    you to ignore the oracle. The golden was produced by the OFFICIAL
    microsoft/VibeVoice realtime demo on transformers 4.57.6; see
    ``docs/plans/evidence/golden/PROVENANCE.md``. It is only ~100 KB and needs
    no model to read, so it lives in the default suite and keeps the spectral
    tolerance honest against real speech rather than a synthetic stand-in.
    """
    golden = Path(__file__).parent.parent / "docs" / "plans" / "evidence" / "golden" / (
        "en-Carter_man-hello.wav"
    )
    if not golden.is_file():
        pytest.skip(f"golden not present: {golden}")

    results = oracle.accept_wav(golden)
    statuses = [g.status for g in results]
    # The golden ships a .latents sidecar, so the duration gate is a real PASS;
    # how the clip ended is not recorded in a wav, so that gate is SKIPPED
    # rather than assumed.
    assert statuses == ["PASS", "SKIPPED", "PASS", "PASS", "PASS"], [
        (g.name, g.status, g.detail) for g in results
    ]


def test_golden_is_accepted_when_it_is_recorded_as_self_terminated(oracle, tmp_path):
    """The golden's PROVENANCE records it stopping on its own EOS at 1.07 s.

    Pinned so the new gate is not a trap for a known-good clip: with the fact
    supplied, the golden must come out fully green.
    """
    golden = Path(__file__).parent.parent / "docs" / "plans" / "evidence" / "golden" / (
        "en-Carter_man-hello.wav"
    )
    if not golden.is_file():
        pytest.skip(f"golden not present: {golden}")

    results = oracle.accept_wav(golden, termination=oracle.TERMINATION_SELF)
    assert [g.status for g in results] == ["PASS", "PASS", "PASS", "PASS", "PASS"], [
        (g.name, g.status, g.detail) for g in results
    ]


def test_termination_gate_rejects_a_clip_that_ran_to_the_budget(oracle):
    """A structurally perfect clip that was truncated must FAIL.

    This is the gate the 2026-09-26 acceptance run needed and did not have: all
    six clips passed duration/silence/spectral/level while the model was still
    running to the latents budget with first-latent EOS at 1.7e-5, i.e. never
    signalling end of speech. The waveform of such a clip is indistinguishable
    from a complete one, so the only place the fact exists is the generation
    loop's flag — and that is what the gate reads.
    """
    x = speech_shaped(3.0)
    latents = round(x.size / SR / oracle.LATENT_SECONDS)

    ok = {g.name: g.status for g in oracle._run_gates(x, SR, latents, oracle.TERMINATION_SELF)}
    truncated = {
        g.name: g.status for g in oracle._run_gates(x, SR, latents, oracle.TERMINATION_BUDGET)
    }
    assert ok == {k: "PASS" for k in ok}
    assert truncated["termination"] == "FAIL"
    # Everything else about the clip is unchanged — that is the point.
    assert {k: v for k, v in truncated.items() if k != "termination"} == {
        k: v for k, v in ok.items() if k != "termination"
    }
    assert "truncated" in oracle.check_termination(oracle.TERMINATION_BUDGET).detail


def test_unknown_termination_is_skipped_not_assumed(oracle, tmp_path):
    """Nothing in a wav says how it ended, so the gate says so.

    It must never be an unannounced pass: without the fact, "the model stopped
    on its own" is exactly the claim this run cannot make.
    """
    path = _write_wav(tmp_path / "good.wav", speech_shaped(3.0))

    result = next(g for g in oracle.accept_wav(path) if g.name == "termination")
    assert result.status == "SKIPPED"
    assert "unknown" in result.detail


def test_termination_comes_from_a_sidecar_and_an_explicit_value_wins(oracle, tmp_path):
    """``<file>.termination`` supplies the fact; ``--termination`` overrides it."""
    path = _write_wav(tmp_path / "good.wav", speech_shaped(3.0))
    sidecar = path.with_name(path.name + ".termination")
    sidecar.write_text("budget\n", encoding="utf-8")

    value, origin = oracle.resolve_termination(path, None)
    assert value == "budget"
    assert origin.endswith("good.wav.termination")
    assert next(g for g in oracle.accept_wav(path) if g.name == "termination").status == "FAIL"

    value, origin = oracle.resolve_termination(path, "self")
    assert (value, origin) == ("self", "--termination")
    assert next(
        g for g in oracle.accept_wav(path, termination="self")
        if g.name == "termination"
    ).status == "PASS"


def test_an_unrecognised_termination_token_is_refused(oracle, tmp_path):
    """A sidecar saying "fine" is a broken contract, not a pass."""
    path = _write_wav(tmp_path / "good.wav", speech_shaped(3.0))
    path.with_name(path.name + ".termination").write_text("fine\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        oracle.accept_wav(path)
    assert "self" in str(excinfo.value) and "budget" in str(excinfo.value)


def test_accept_exits_non_zero_on_a_broken_clip_and_zero_on_a_good_one(
    oracle, tmp_path, capsys, monkeypatch
):
    """The CLI contract the plan states: non-zero on broken, zero on good."""
    good = _write_wav(tmp_path / "good.wav", speech_shaped(3.0))
    broken = _write_wav(
        tmp_path / "broken.wav", np.zeros(int(0.133 * SR), dtype=np.float32)
    )

    monkeypatch.setattr(sys, "argv", ["diag_realtime_quality.py", "--accept", str(broken)])
    assert oracle.main() == 1
    assert "REJECTED" in capsys.readouterr().out

    monkeypatch.setattr(
        sys, "argv",
        ["diag_realtime_quality.py", "--accept", str(good), "--latents", "23"],
    )
    assert oracle.main() == 0
    assert "ACCEPTED" in capsys.readouterr().out


def test_accept_exits_non_zero_when_the_clip_ran_to_the_budget(
    oracle, tmp_path, monkeypatch, capsys
):
    """The CLI contract for the new gate: good structure + truncated -> exit 1.

    This is the command that would have rejected today's acceptance grid, which
    passed every other gate while the model never signalled end of speech.
    """
    good = _write_wav(tmp_path / "good.wav", speech_shaped(3.0))

    monkeypatch.setattr(
        sys, "argv",
        [
            "diag_realtime_quality.py", "--accept", str(good),
            "--latents", "23", "--termination", "budget",
        ],
    )
    assert oracle.main() == 1
    out = capsys.readouterr().out
    assert "REJECTED" in out and "termination" in out
