"""BUG-006 Step 6 — End-to-end intelligibility gate (MANUAL / user-run).

This is NOT a fast unit test. It requires the real checkpoint + a GPU and is the
actual acceptance gate that proves the spoken content is no longer gibberish.

Run it (for example, from the ComfyUI embedded Python) with:

    RUN_E2E=1 pytest tests/test_e2e_intelligibility.py -s

By default it is SKIPPED so the normal CI suite stays fast and GPU-free. A more
convenient path is the standalone script ``e2e_smoke_test.py``, which writes .wav
files you can actually listen to.

What these tests assert automatically:
  * the waveform is finite (no NaN/Inf),
  * it is non-silent (RMS above floor) and longer than ~0.5s,
  * a 2-speaker script produces meaningfully longer audio than a 1-speaker one
    (i.e. target-length generation, not a fixed reference-length blob).

The final "is the content intelligible?" judgement is made by listening to the
output (see e2e_smoke_test.py). When a speech-to-text model is available, add a
WER assertion to ``test_e2e_matches_target_text_manual``.
"""

import os

import pytest

RUN_E2E = os.environ.get("RUN_E2E") == "1"


@pytest.mark.skipif(
    not RUN_E2E,
    reason="e2e intelligibility gate requires GPU + checkpoint; run with RUN_E2E=1",
)
class TestE2EIntelligibility:
    def _run(self, script, voice_count, model_name="VibeVoice-1.5B"):
        from ComfyUI_VibeVoice.modules.generation import (
            load_vibevoice_model,
            generate_audio,
        )
        import numpy as np
        import torch

        patcher, model, processor = load_vibevoice_model(
            model_name=model_name, device="auto", dtype="auto"
        )
        # Synthetic reference tones as placeholders; real ears need real voice clips.
        sr = 24000
        voices = []
        for _ in range(voice_count):
            t = np.arange(0, 3.0, 1.0 / sr)
            voices.append((0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32))
        voice_samples = [
            {"waveform": torch.from_numpy(v).unsqueeze(0), "sample_rate": sr} for v in voices
        ]
        speaker_ids = list(range(1, voice_count + 1))
        wav, rate = generate_audio(
            model=model,
            processor=processor,
            text=script,
            voice_samples=voice_samples,
            speaker_ids=speaker_ids,
            cfg_scale=1.3,
            inference_steps=10,
            seed=42,
            do_sample=True,
            temperature=0.95,
            top_p=0.95,
            top_k=0,
            max_new_tokens=None,
        )
        return wav, rate

    def _as_np(self, wav):
        import numpy as np

        if hasattr(wav, "cpu"):
            wav = wav.cpu().numpy()
        return np.asarray(wav, dtype=np.float32).reshape(-1)

    def test_e2e_1speaker_non_silent_and_finite(self):
        wav, rate = self._run(
            "Speaker 1: Hello, this is a test of the text to speech system.", 1
        )
        w = self._as_np(wav)
        assert np.isfinite(w).all(), "waveform contains NaN/Inf"
        assert w.shape[0] > rate * 0.5, "audio too short (<0.5s) — not generating target speech"
        rms = float(np.sqrt(np.mean(w**2)))
        assert rms > 1e-3, "audio is (near) silent — generation failed"

    def test_e2e_2speaker_longer_than_1speaker(self):
        w1, _ = self._run(
            "Speaker 1: Hello, this is a test of the text to speech system.", 2
        )
        w2, _ = self._run(
            "Speaker 1: Hello, this is a test of the text to speech system.\n"
            "Speaker 2: And I am the second speaker, confirming the cloning works.",
            2,
        )
        d1 = self._as_np(w1).shape[0]
        d2 = self._as_np(w2).shape[0]
        assert d2 > d1 * 1.3, "2-speaker output not meaningfully longer than 1-speaker (length bug?)"

    def test_e2e_matches_target_text_manual(self):
        # Placeholder for an automatic WER gate. Without a speech-to-text model in this
        # environment we rely on manual listening (see e2e_smoke_test.py). When an ASR is
        # available, decode e2e_1speaker.wav and assert WER below threshold here.
        pytest.skip(
            "Automatic WER gate requires a speech-to-text model; use manual listening "
            "(e2e_smoke_test.py) for now."
        )
