"""S4.2 — generate the audio-acceptance grid and gate every clip.

The plan's tier-A step: 3 scripts (1 sentence / 4 sentences / 200+ words) x 2
voice presets, each clip checked by the S0.2 oracle in ``diag_realtime_quality.py``
and then judged by a human for intelligibility and timbre.

This script does the machine half: it generates the grid through the production
adapter (``modules.realtime_generation.generate_realtime_audio``) driven by the
canonical node's loader, records the **true** latent count behind every clip so
the oracle's duration gate is not circular, records **how each clip ended** (the
model's own end-of-speech, or the length budget) so the oracle can reject a
truncated clip, and prints the oracle's verdict per clip. The human half is a
listening pass over the wavs it leaves behind.

    python.exe audio_acceptance.py --outdir docs/plans/evidence/audio-acceptance

The oracle is imported from ``diag_realtime_quality`` rather than reimplemented,
so the acceptance verdict and ``diag_realtime_quality.py --accept`` can never
disagree. **No transcription-based scoring is used anywhere** — VibeVoice-ASR
mis-scores degraded audio as a clean transcript and is not an oracle for this.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Register this folder under the stable ``ComfyUI_VibeVoice`` alias, the same
# bootstrap every entry-point script in this repo uses.
import e2e_smoke_test  # noqa: E402,F401  (package-alias bootstrap)

SCRIPTS = {
    "1-sentence": "The quick brown fox jumps over the lazy dog.",
    "4-sentences": (
        "The harbour was quiet that morning. "
        "A single gull circled the mast before settling on the rail. "
        "Somewhere below, the engines turned over twice and then went still. "
        "By noon the tide had come in and the quay was empty again."
    ),
    "200-plus-words": (
        "Good morning, and welcome to the weekly engineering briefing. "
        "I will start with the status of the model pipeline, because there is "
        "more to report this week than there has been in a month. The training "
        "run that had been failing on the cluster finished on Tuesday evening, "
        "and the resulting checkpoint is already uploaded to shared storage. "
        "Its evaluation scores are slightly better than the previous release on "
        "every benchmark we track, which is encouraging but not yet decisive. "
        "The inference side is where most of the work happened. We cut the cold "
        "start cost roughly in half by caching the tokenizer output, and we "
        "removed a redundant copy that was doubling peak memory during export. "
        "Neither change alters the output, and both are covered by regression "
        "tests that run on every merge. On the operational side, the nightly job "
        "that rebuilds the documentation search index has been failing since the "
        "fourth, because a link checker rejects two of the older example pages. "
        "That is a documentation problem rather than a platform one, and it is "
        "assigned for this week. Finally, a reminder that the freeze for the next "
        "release begins on Friday, so anything that still needs to land should be "
        "reviewed before the end of the day on Thursday. Thank you, and please "
        "put questions in the channel rather than in direct messages, so that "
        "everyone can find the answers later. That is all for today."
    ),
}

VOICES = ("en-Carter_man", "en-Emma_woman")


def _model_dir() -> Path:
    return Path(
        os.environ.get(
            "VIBEVOICE_REALTIME_MODEL_DIR",
            r"C:\AI\ComfyUI\ComfyUI\models\tts\VibeVoice\VibeVoice-Realtime-0.5B",
        )
    )


def _voices_dir() -> Path:
    return _model_dir().parent / "voices"


def _register_paths() -> None:
    """Register the model and voice folders the way ComfyUI's main does."""
    import folder_paths

    from ComfyUI_VibeVoice.modules.folder_registration import VOICE_PRESET_FOLDER_KEY
    from ComfyUI_VibeVoice.modules.model_info import AVAILABLE_VIBEVOICE_MODELS

    model_dir = _model_dir()
    if not model_dir.is_dir():
        raise SystemExit(f"realtime checkpoint not found: {model_dir}")
    voices = _voices_dir()
    if not voices.is_dir():
        raise SystemExit(f"voice preset folder not found: {voices}")

    AVAILABLE_VIBEVOICE_MODELS["VibeVoice-Realtime-0.5B"] = {
        "type": "local_dir",
        "path": str(model_dir),
        "tokenizer_repo": "Qwen/Qwen2.5-1.5B",
    }
    folder_paths.folder_names_and_paths.setdefault(
        VOICE_PRESET_FOLDER_KEY, ([str(voices)], {".pt"})
    )


def _generate(model, processor, text: str, preset, out_wav: Path) -> tuple[int, str]:
    """Run one clip, write it and its two sidecars. Returns (latents, ending).

    The latent count is measured by counting how many times the acoustic
    decoder runs, which the generation loop does exactly once per produced
    latent. It is deliberately *not* derived from the resulting duration, which
    would make the oracle's duration gate check the number against itself, and
    not from ``sequences.shape[1]``, which counts the interleaved text window
    as well. (Counting ``tts_eos_classifier`` calls is wrong too: the loop
    evaluates the head roughly three times per latent.)

    The ending is read from the generation loop's own ``reach_max_step_sample``
    flag, which the loop sets when it exits on the length budget instead of on
    the model's end-of-speech signal. It is written as a ``.termination``
    sidecar because nothing inside the wav records it, and without it the
    oracle's other four gates cannot tell a complete clip from a truncated one.
    """
    from ComfyUI_VibeVoice.modules.realtime_generation import generate_realtime_audio

    decoder = model.model.acoustic_tokenizer
    original_decode = decoder.decode
    counter = {"latents": 0}
    captured: dict = {}

    def _counting_decode(*args, **kwargs):
        counter["latents"] += 1
        return original_decode(*args, **kwargs)

    original_generate = model.generate

    def _capturing_generate(**kwargs):
        outputs = original_generate(**kwargs)
        captured["outputs"] = outputs
        return outputs

    decoder.decode = _counting_decode
    model.generate = _capturing_generate
    try:
        waveform, sample_rate = generate_realtime_audio(
            model=model,
            processor=processor,
            text=text,
            voice_preset=preset,
            cfg_scale=1.5,
            diffusion_steps=10,
            max_new_tokens=0,
            seed=42,
        )
    finally:
        decoder.decode = original_decode
        model.generate = original_generate

    latents = counter["latents"]
    reach_max = getattr(captured.get("outputs"), "reach_max_step_sample", None)
    ending = "budget" if reach_max is not None and bool(reach_max.any()) else "self"

    import numpy as np

    from diag_realtime_quality import save_wav

    flat = waveform.detach().float().cpu().reshape(-1).numpy()
    save_wav(out_wav, flat.astype(np.float32), sample_rate)
    out_wav.with_name(out_wav.name + ".latents").write_text(f"{latents}\n", encoding="utf-8")
    out_wav.with_name(out_wav.name + ".termination").write_text(
        f"{ending}\n", encoding="utf-8"
    )
    return latents, ending


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", default="docs/plans/evidence/audio-acceptance")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _register_paths()

    from diag_realtime_quality import accept_wav  # the S0.2 oracle, not a copy
    from ComfyUI_VibeVoice.modules.generation import load_vibevoice_model
    from ComfyUI_VibeVoice.modules.voice_presets import load_voice_preset
    from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_PATCHER_CACHE

    patcher, model, processor = load_vibevoice_model(
        model_name="VibeVoice-Realtime-0.5B", device="cuda", dtype="auto"
    )

    rows = []
    try:
        for voice in VOICES:
            preset = load_voice_preset(str(_voices_dir() / f"{voice}.pt"), model.device)
            for label, text in SCRIPTS.items():
                name = f"{voice}__{label}"
                wav = outdir / f"{name}.wav"
                latents, ending = _generate(model, processor, text, preset, wav)
                gates = accept_wav(wav, latents, ending)
                failed = [g.name for g in gates if g.failed]
                rows.append((name, latents, ending, gates, failed))
                print(f"\n=== {name}: {len(text.split())} words, {latents} latents, "
                      f"ended on {ending} ===")
                for gate in gates:
                    print(f"    {gate.status:<7} {gate.name}: {gate.detail}")
            del preset
    finally:
        VIBEVOICE_PATCHER_CACHE.pop("VibeVoice-Realtime-0.5B_attn_sdpa_q4_0", None)

    print("\n\n=== ORACLE SUMMARY ===")
    for name, latents, ending, _gates, failed in rows:
        print(f"  {'PASS' if not failed else 'FAIL'}  {name}  "
              f"({latents} latents, ended on {ending})")
    failures = [name for name, _l, _e, _g, failed in rows if failed]
    print(f"\n{len(rows) - len(failures)}/{len(rows)} clips passed the oracle.")
    if failures:
        print("failing: " + ", ".join(failures))
    truncated = [name for name, _l, ending, _g, _f in rows if ending == "budget"]
    if truncated:
        print(
            "\nNOTE: " + str(len(truncated)) + " clip(s) stopped on the length "
            "budget rather than on the model's own end-of-speech signal, so they "
            "are truncated: " + ", ".join(truncated)
        )
    print("\nHuman half (S4.2): listen to each wav in", outdir.resolve(),
          "and record intelligibility + timbre in the evidence doc.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
