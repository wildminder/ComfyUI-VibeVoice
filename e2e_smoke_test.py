#!/usr/bin/env python
"""End-to-end intelligibility smoke test for the canonical VibeVoice TTS node.

BUG-006 acceptance gate (Step 6 of the fix plan). The unit tests only prove the
*algorithm* is correct; the real question — "is the spoken content still gibberish?" —
can only be answered on the actual checkpoint + GPU. This script runs the same code paths
the ComfyUI node uses and writes `.wav` files for you to LISTEN to.

Standard (reference-audio) branch, using ``modules.generation.generate_audio``::

    C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe e2e_smoke_test.py \
        --model VibeVoice-1.5B --voice C:/path/voice1.wav --voice C:/path/voice2.wav

If you have no reference wavs handy, the script falls back to a synthetic tone so the
pipeline still executes — but for a real intelligibility check, pass real voice clips
(3–10 s of clean speech per speaker).

Realtime branch, using the official cached-prompt adapter. The realtime checkpoint is a
separate single-speaker architecture: it needs an official ``.pt`` voice prompt and does
NOT accept reference wavs::

    C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe e2e_smoke_test.py \
        --realtime-model VibeVoice-Realtime-0.5B \
        --voice-preset models/tts/VibeVoice/voices/en-Carter_man.pt

``--realtime-model`` and ``--voice-preset`` must be supplied together.

What it checks automatically (and prints):
  * generation completes without error,
  * the waveform is non-empty and finite,
  * duration is sane (longer scripts => longer audio, i.e. target-length, not a
    fixed reference-length blob),
  * RMS energy is above the silence floor.

The final verdict is YOUR EARS: play the .wav files and confirm the words match the script.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import wave
from pathlib import Path

import numpy as np


def _bootstrap_package_alias() -> None:
    """Register this custom-node folder under the stable ``ComfyUI_VibeVoice``
    alias so ``from ComfyUI_VibeVoice.modules...`` works when the script is run
    standalone (outside pytest, which does this in conftest.py).

    The on-disk directory is ``ComfyUI-VibeVoice`` (hyphen), which is not a valid
    Python identifier, so a plain PYTHONPATH entry is not enough — we must build
    the package spec explicitly, mirroring conftest.py §3.
    """
    alias = "ComfyUI_VibeVoice"
    if alias in sys.modules:
        return
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location(
        alias,
        os.path.join(root, "__init__.py"),
        submodule_search_locations=[root],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[alias] = pkg
    try:
        spec.loader.exec_module(pkg)
    except Exception:
        # __init__.py may exit early under a pytest/ComfyUI guard; the alias is
        # still registered so submodule imports resolve.
        pass
    for sub in ("modules", "nodes", "src"):
        sub_path = os.path.join(root, sub)
        sub_alias = f"{alias}.{sub}"
        if os.path.isdir(sub_path) and sub_alias not in sys.modules:
            sub_spec = importlib.util.spec_from_file_location(
                sub_alias,
                os.path.join(sub_path, "__init__.py"),
                submodule_search_locations=[sub_path],
            )
            sub_pkg = importlib.util.module_from_spec(sub_spec)
            sys.modules[sub_alias] = sub_pkg
            try:
                sub_spec.loader.exec_module(sub_pkg)
            except Exception:
                pass


_bootstrap_package_alias()


def _load_voice(path: str, target_sr: int = 24000):
    """Load a wav into a mono float32 numpy array at target_sr. Falls back to a
    synthetic 220 Hz tone if the file cannot be read (so the pipeline still runs).

    Uses the node's audio backend (PyAV primary — the ComfyUI-core decoder —
    with soundfile/torchaudio/librosa fallbacks) and torchaudio-quality
    resampling, so the smoke test exercises the exact production code path.
    """
    try:
        from ComfyUI_VibeVoice.modules import audio_backend

        data, sr = audio_backend.load_audio_file(path)
        data = np.asarray(data, dtype=np.float32)
        if sr != target_sr:
            data = audio_backend.resample_audio(data, sr, target_sr)
        if np.abs(data).max() > 1.0:
            data = data / np.abs(data).max()
        return data.astype(np.float32)
    except Exception as e:  # pragma: no cover - defensive fallback
        print(f"  [warn] could not read '{path}' ({e}); using synthetic tone instead")
        t = np.arange(0, 3.0, 1.0 / target_sr)
        return (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _synth_tone(target_sr: int = 24000, seconds: float = 3.0):
    t = np.arange(0, seconds, 1.0 / target_sr)
    return (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _save_wav(path: str, waveform: np.ndarray, sample_rate: int = 24000):
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 3:
        waveform = waveform[0, 0]
    elif waveform.ndim == 2:
        waveform = waveform[0]
    # clip to [-1, 1] to avoid PCM overflow
    waveform = np.clip(waveform, -1.0, 1.0)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((waveform * 32767).astype("<i2").tobytes())


def _report(name: str, waveform: np.ndarray, sample_rate: int) -> dict:
    w = np.asarray(waveform).reshape(-1).astype(np.float32)
    dur = len(w) / sample_rate
    rms = float(np.sqrt(np.mean(w**2))) if len(w) else 0.0
    finite = bool(np.isfinite(w).all())
    print(f"  [{name}] duration={dur:.2f}s rms={rms:.4f} finite={finite} numel={len(w)}")
    return {"duration": dur, "rms": rms, "finite": finite, "numel": len(w)}


def _run_realtime(args) -> int:
    """Generate one realtime sample from an official cached .pt voice prompt.

    The realtime branch uses the same cached-prompt adapter as the canonical
    ``VibeVoice TTS`` node. It never falls back to standard reference-audio
    generation: a realtime checkpoint with raw reference wavs is not a valid
    input.
    """
    # Imported only in the realtime branch so --help and the standard path keep
    # their lightweight import behavior.
    try:
        from ComfyUI_VibeVoice.modules.generation import load_vibevoice_model
        from ComfyUI_VibeVoice.modules.realtime_generation import generate_realtime_audio
        from ComfyUI_VibeVoice.modules.voice_presets import load_voice_preset
    except Exception as e:  # pragma: no cover
        print("ERROR: could not import the realtime modules. Run this from the "
              "ComfyUI embedded Python with the custom-node folder on PYTHONPATH.", file=sys.stderr)
        print(f"  ({e})", file=sys.stderr)
        return 2

    preset_path = Path(args.voice_preset)
    if not preset_path.is_file():
        print(f"ERROR: voice preset not found: {preset_path}", file=sys.stderr)
        return 2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading realtime model '{args.realtime_model}' on device={args.device} "
          f"dtype={args.dtype} ...")
    patcher, model, processor = load_vibevoice_model(
        model_name=args.realtime_model, device=args.device, dtype=args.dtype
    )

    import comfy.model_management as model_management

    print(f"Loading voice preset '{preset_path.name}' ...")
    preset = load_voice_preset(str(preset_path.resolve()), model_management.get_torch_device())

    print("\nGenerating realtime sample ...")
    wav, rate = generate_realtime_audio(
        model=model,
        processor=processor,
        text=args.realtime_script,
        voice_preset=preset,
        cfg_scale=args.cfg_scale,
        diffusion_steps=args.inference_steps,
        max_new_tokens=args.max_new_tokens or None,
        seed=args.seed,
    )
    report = _report("realtime", wav, rate)
    _save_wav(str(outdir / "e2e_realtime.wav"), wav, rate)

    print("\n=== VERDICT (manual) ===")
    print(f"  Wrote wav to: {(outdir / 'e2e_realtime.wav').resolve()}")
    print("  Listen to it and confirm the spoken words match the realtime script")
    print("  in the timbre of the selected voice preset.")
    print(f"  duration={report['duration']:.2f}s rms={report['rms']:.4f} "
          f"finite={report['finite']}")
    return 0 if report["finite"] and report["rms"] > 1e-4 else 1


def main():
    ap = argparse.ArgumentParser(description="BUG-006 e2e intelligibility smoke test")
    ap.add_argument("--model", default="VibeVoice-1.5B", help="TTS model name from MODEL_CONFIGS")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--dtype", default="auto", help="auto | bf16 | fp16 | fp32")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inference_steps", type=int, default=10)
    ap.add_argument("--cfg_scale", type=float, default=1.3)
    ap.add_argument("--max_new_tokens", type=int, default=0, help="0 = auto")
    ap.add_argument("--voice", action="append", default=[], help="reference wav (repeat per speaker)")
    ap.add_argument("--outdir", default="e2e_output")
    ap.add_argument(
        "--realtime-model",
        default=None,
        help=(
            "Realtime checkpoint name (e.g. VibeVoice-Realtime-0.5B). Enables the "
            "cached-voice-prompt branch and requires --voice-preset."
        ),
    )
    ap.add_argument(
        "--voice-preset",
        default=None,
        help=(
            "Official cached .pt voice prompt for the realtime branch. "
            "Requires --realtime-model."
        ),
    )
    ap.add_argument(
        "--realtime-script",
        default="This is a realtime acceptance test for the VibeVoice realtime model.",
        help="Single-speaker script used by the realtime branch.",
    )
    ap.add_argument(
        "--script1",
        default="Speaker 1: Hello, this is a test of the text to speech system.",
    )
    ap.add_argument(
        "--script2",
        default="Speaker 1: Hello, this is a test of the text to speech system.\n"
        "Speaker 2: And I am the second speaker, confirming the cloning works.",
    )
    args = ap.parse_args()

    if bool(args.realtime_model) != bool(args.voice_preset):
        print(
            "ERROR: --realtime-model and --voice-preset must be supplied together.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.realtime_model:
        return _run_realtime(args)

    # Late imports so the script fails with a helpful message outside ComfyUI.
    try:
        from ComfyUI_VibeVoice.modules.generation import load_vibevoice_model, generate_audio
    except Exception as e:  # pragma: no cover
        print("ERROR: could not import ComfyUI_VibeVoice modules. Run this from the "
              "ComfyUI embedded Python with the custom-node folder on PYTHONPATH.", file=sys.stderr)
        print(f"  ({e})", file=sys.stderr)
        sys.exit(2)

    sr = 24000
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model '{args.model}' on device={args.device} dtype={args.dtype} ...")
    patcher, model, processor = load_vibevoice_model(
        model_name=args.model, device=args.device, dtype=args.dtype
    )

    # Build reference voice samples.
    if args.voice:
        voices = [_load_voice(p, sr) for p in args.voice]
    else:
        print("  No --voice provided; using synthetic tones (pass real clips for a real check).")
        voices = [_synth_tone(sr)]
    voice_samples = [{"waveform": __import__("torch").from_numpy(v).unsqueeze(0), "sample_rate": sr}
                     for v in voices]
    speaker_ids = list(range(1, len(voices) + 1))

    print(f"\nGenerating 1-speaker sample ...")
    try:
        wav1, rate1 = generate_audio(
            model=model, processor=processor, text=args.script1,
            voice_samples=voice_samples[:1], speaker_ids=[1],
            cfg_scale=args.cfg_scale, inference_steps=args.inference_steps,
            seed=args.seed, do_sample=True, temperature=0.95, top_p=0.95, top_k=0,
            max_new_tokens=args.max_new_tokens or None,
        )
        r1 = _report("1-speaker", wav1, rate1)
        _save_wav(str(outdir / "e2e_1speaker.wav"), wav1, rate1)
    except Exception as e:
        print(f"  [FAIL] 1-speaker generation raised: {e}")
        r1 = None

    need_two = len(voices) >= 2
    r2 = None
    if need_two:
        print(f"\nGenerating 2-speaker sample ...")
        try:
            wav2, rate2 = generate_audio(
                model=model, processor=processor, text=args.script2,
                voice_samples=voice_samples[:2], speaker_ids=[1, 2],
                cfg_scale=args.cfg_scale, inference_steps=args.inference_steps,
                seed=args.seed, do_sample=True, temperature=0.95, top_p=0.95, top_k=0,
                max_new_tokens=args.max_new_tokens or None,
            )
            r2 = _report("2-speaker", wav2, rate2)
            _save_wav(str(outdir / "e2e_2speaker.wav"), wav2, rate2)
        except Exception as e:
            print(f"  [FAIL] 2-speaker generation raised: {e}")
    else:
        print("\nSkipping 2-speaker sample (need >= 2 --voice clips; only 1 available).")

    print("\n=== VERDICT (manual) ===")
    print(f"  Wrote wavs to: {outdir.resolve()}")
    print("  Listen to e2e_1speaker.wav and e2e_2speaker.wav. The spoken WORDS must match the")
    print("  script text (not a stream of unrelated syllables). If the content is correct and")
    print("  the two speakers' voices are distinct, BUG-006 is fixed.")
    if r1 and r2:
        ratio = r2["duration"] / r1["duration"] if r1["duration"] else 0
        print(f"  2-speaker/1-speaker duration ratio = {ratio:.2f}x (expect > 1.5x if target-length).")
    print("  If the standard output is still wrong, verify the reference wavs and")
    print("  script first. The realtime checkpoint is a separate architecture: run it")
    print("  through --realtime-model with --voice-preset, never as a standard fallback.")


if __name__ == "__main__":
    sys.exit(main() or 0)
