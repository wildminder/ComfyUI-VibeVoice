<a id="readme-top"></a>

<div align="center">

# ComfyUI-VibeVoice

<img alt="ComfyUI-VibeVoice logo" src="https://github.com/user-attachments/assets/0b9663a1-01ab-4ad8-812b-2b60169e236c" />


Long-form, multi-speaker conversational **TTS** and **ASR** for ComfyUI.<br>
Microsoft's VibeVoice, with model download, VRAM management and audio handling handled for you.

<a href="https://github.com/wildminder/ComfyUI-VibeVoice/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
&nbsp;&middot;&nbsp;
<a href="https://github.com/wildminder/ComfyUI-VibeVoice/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>

[![License: MIT][license-shield]][license-url]
[![Last commit][commit-shield]][commit-url]
[![ComfyUI][comfy-shield]][comfy-url]
[![Model weights on Hugging Face][hf-shield]][hf-url]

</div>

> [!NOTE]
> **ComfyUI-VibeVoice** wraps the Microsoft [VibeVoice](https://github.com/microsoft/VibeVoice) family — 1.5B / 7B long-form conversational TTS, a 0.5B realtime model, and ASR — as native ComfyUI nodes. Models download on first run into `ComfyUI/models/tts/VibeVoice/` and load under ComfyUI's VRAM manager. Nodes appear under `WMNodes/sound/tts` and `WMNodes/sound/asr`.

<details>
<summary><b>Table of Contents</b></summary>

* [Features](#features)
* [Models](#models)
* [Getting started](#getting-started)
* [Usage](#usage)
  * [Script format](#script-format)
  * [Voice modes](#voice-modes)
  * [Realtime models](#realtime)
  * [Node inputs](#node-inputs)
  * [Loading external models](#external-models)
    * [Host RAM during an external load](#host-ram)
* [Performance &amp; stability](#performance)
  * [Feature compatibility &amp; VRAM matrix](#vram-matrix)
  * [`transformers` version support](#transformers-support)
  * [Attention backends](#attention-backends)
* [Changelog](#changelog)
* [License](#license)
* [Acknowledgments](#acknowledgments)
* [Star history](#star-history)

</details>

<p id="features" align="center">◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆</p>

## ❖ Features

<div align="center">
  <img alt="nodes" src="https://github.com/user-attachments/assets/a7e80bbf-7006-422a-aa81-45ae5001a3b2" />
  <br/><br/>
</div>


| Capability | Detail |
| :--- | :--- |
| **Multi-speaker TTS** | ![TTS][task-tts] - Up to 4 distinct voices in one output |
| **Voice cloning** | ![TTS][task-tts] - Any `.wav` / `.mp3` as a speaker reference |
| **Hybrid scripts** | ![TTS][task-tts] - Cloned voices plus speakers that reuse a provided reference, in the same script |
| **Speech recognition** | ![ASR][task-asr] - `VibeVoice ASR` returns timestamped speaker segments |
| **Quantization** | ![GGUF][badge-gguf] ![fp8][badge-fp8] ![int8][badge-int8] ![int4][badge-int4] - GGUF, fp8, int8 and 4-bit, plus safetensors |
| **Attention backends** | `eager` / `sdpa` / `flash_attention_2` / `sage` |
| **Memory** | Models download and load under ComfyUI's VRAM manager |


<p id="models" align="center">◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆</p>

## ⌬ Models

Downloaded on first run into `ComfyUI/models/tts/VibeVoice/`.

| Model | Type | Size | Weights |
| :--- | :---: | :---: | :---: |
| **VibeVoice-1.5B** | ![TTS][task-tts] | 3.0 GB | [![][hf-microsoft]](https://huggingface.co/microsoft/VibeVoice-1.5B) |
| **VibeVoice-7B** | ![TTS][task-tts] | 17.4 GB | [![][hf-vibevoice]](https://huggingface.co/vibevoice/VibeVoice-7B) |
| **VibeVoice-Realtime-0.5B** | ![realtime][task-realtime] | 1.5 GB | [![][hf-microsoft]](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) |
| **VibeVoice-ASR-HF** | ![ASR][task-asr] | 17.4 GB | [![][hf-microsoft]](https://huggingface.co/microsoft/VibeVoice-ASR-HF) |

> [!TIP]
> Quantized weights are optional. Point `Load VibeVoice Model` at a `.gguf` or a
> quantized safetensors file; leave the node unconnected to use the standard model.

<p align="right"><a href="#readme-top">⟔ ▲ ⟓ back to top</a></p>

<p id="getting-started" align="center">◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆</p>

## ⇩ Getting started

**ComfyUI Manager** — `Manager` → `Install Custom Nodes` → search
`ComfyUI-VibeVoice` → `Install` → restart.

<details>
<summary><b>Manual installation</b></summary>

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/wildminder/ComfyUI-VibeVoice.git
cd ComfyUI-VibeVoice
pip install -r requirements.txt
```

Restart ComfyUI. Nodes appear under `WMNodes/sound/tts`.

| Want | Install |
| :--- | :--- |
| `sage` attention | `sageattention` — [Windows wheels](https://github.com/wildminder/AI-windows-whl) |
| 4-bit quantization | `bitsandbytes` |
| Audio fallbacks | `pip install ComfyUI-VibeVoice[audio-extra]` |

</details>

<p id="usage" align="center">◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆</p>

## ❯ Usage

1. Add `VibeVoice TTS` to the graph.
2. Connect a `Load Audio` `AUDIO` output to a `speaker_*_voice` input to clone
   that voice. Leave an input empty to reuse a provided reference.
3. Write the dialogue in the `text` input.
4. Queue the prompt. One audio file is returned for the whole conversation.

Defaults are sane for most scripts; every widget is listed in
[Node inputs](#node-inputs).

<p id="script-format" align="center">· · · · · · · · · · · · · ·</p>

### ▣ Script format

Assign lines to speakers in either of two formats. Both are treated identically.

```text
[1] Hello, this is the first speaker.
[2] And this is the second.
```

```text
Speaker 1: Hello, this is the first speaker.
Speaker 2: And this is the second.
```

You can also add an optional colon to the modern format (e.g., `[1]: ...`).
The node handles all variations consistently.

<p id="voice-modes" align="center">· · · · · · · · · · · · · ·</p>

### ▣ Voice modes

**Speaker tagging.**

*   **Modern Format (Recommended):** `[1] This is the first speaker.`
*   **Classic Format:** `Speaker 1: This is the first speaker.`

**Hybrid voice generation.** This is a powerful feature that lets you mix
cloned voices in the same script.

> [!WARNING]
> **At least one `speaker_*_voice` reference audio is required** — VibeVoice
> anchors the timbre of every speaker to a provided reference, so you cannot
> generate a fully reference-free ("zero-shot") voice for any speaker.

*   **To Clone a Voice:** Connect a `Load Audio` node to the speaker's input (e.g., `speaker_1_voice`).
*   **To Reuse a Cloned Voice:** Any other speaker may be left empty. When a speaker has no reference, its voice is cloned from the provided reference(s) rather than generated from scratch.

**Example hybrid script:**

```text
[1] This line will use the audio from speaker_1_voice.
[2] This line will reuse the cloned voice from speaker 1.
[1] I'm back with my cloned voice.
```

In this example, you would connect an audio source to `speaker_1_voice`;
speakers `[2]` are cloned from it.

<p id="realtime" align="center">· · · · · · · · · · · · · ·</p>

### ▣ Realtime models (single-speaker voice prompts)

Selecting a realtime checkpoint in `model_name` switches `VibeVoice TTS` to the
official windowed realtime path: **single-speaker**, voiced from a cached `.pt`
preset instead of reference audio.

<details>
<summary><b>Realtime behavior, setup and length budget</b></summary>

| | Realtime models |
| :--- | :--- |
| **Speakers** | Single-speaker — speaker tags are stripped and lines joined into one utterance |
| **Voice source** | Official cached `.pt` prompt via `voice_preset`; reference-audio cloning is not supported (connected `speaker_*_voice` inputs are ignored with a warning) |
| **`cfg_scale`** | Use `1.5`–`1.8`. Below `1.5` the model collapses into syllable repetition; the node raises low values to `1.5` and logs a warning. `2.0`+ starts drifting |
| **`inference_steps` vs `max_new_tokens`** | Independent: diffusion quality/time vs generated length |
| **Sampling widgets** | `do_sample` / `temperature` / `top_p` / `top_k` are not used; model defaults apply |
| **Output** | A completed ComfyUI `AUDIO` — ComfyUI has no live PCM streaming, so the realtime model still delivers a finished clip |

**Setup:**

1.  Copy an official voice prompt (e.g. `en-Carter_man.pt`) to
    `ComfyUI/models/tts/VibeVoice/voices/` — or any folder registered under
    ComfyUI's `vibevoice_voices` key. Prompts are discovered recursively by file
    stem, so the official `voices/streaming_model/*.pt` layout can be copied intact.
2.  Add `VibeVoice TTS`; set `model_name` to `VibeVoice-Realtime-0.5B` and
    `voice_preset` to the prompt stem (e.g. `en-Carter_man`).
3.  Write a single-speaker script, leave `speaker_*_voice` unconnected, queue.

**Length budget (`max_new_tokens`).** `0` means *auto*, not "model maximum".
The auto budget charges prefilled text tokens and generated speech latents
against one allowance (each window = 5 text + 6 latent units) and adds a tail
so the model can finish its last clause and reach end-of-speech; the chosen
budget is logged and an explicit positive value always wins. If the console
reports *"stopped on the length budget … without the model signalling end of
speech"*, the clip was cut off mid-script — **raise** `max_new_tokens` or
shorten the script.

> [!IMPORTANT]
> `VibeVoice TTS` is the only TTS node — it serves standard and realtime model
> families alike. The former `VibeVoice Realtime TTS` node has been removed;
> workflows saved against it only need the node swapped for `VibeVoice TTS`,
> with the model selection and script carried over unchanged.

</details>

<p id="node-inputs" align="center">· · · · · · · · · · · · · ·</p>

### ▣ Node inputs

Every widget on `VibeVoice TTS`, with shipped defaults:

<details>
<summary><b>Parameter reference</b></summary>

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `model_name` | first standard | Model checkpoint. ASR models live on the `VibeVoice ASR` node instead |
| `text` | sample dialogue | The script — `[1] …` or `Speaker 1: …`, see [Script format](#script-format) |
| `speaker_1_voice` … `speaker_4_voice` | empty | Reference `AUDIO` (e.g. from `Load Audio`) cloning that speaker; empty speakers reuse a provided reference |
| `external_model` | empty | Optional `VIBEVOICE_MODEL` from `Load VibeVoice Model`; overrides `model_name` when connected — see [Loading external models](#external-models) |
| `voice_preset` | `None` | Official `.pt` voice prompt; required by realtime models, ignored by standard ones |
| `quantize_llm_4bit` | `False` | LLM in 4-bit NF4 (needs `bitsandbytes`); the diffusion head stays full precision |
| `attention_mode` | `sdpa` | `eager` (safest) · `sdpa` (balanced) · `flash_attention_2` (fastest) · `sage` — `sage` is excluded for realtime models, see [Attention backends](#attention-backends) |
| `cfg_scale` | `1.3` | Adherence to the reference voice; higher is stricter. Realtime: `1.5`–`1.8` |
| `inference_steps` | `10` | Diffusion steps — quality vs time, never speech length |
| `max_new_tokens` | `0` (auto) | Utterance length budget; raise if cut off, lower if it runs long. Realtime budgets count text and speech units together, see [Realtime models](#realtime) |
| `do_sample` | `True` | Sampling on/off (`False` = greedy). Not used by realtime |
| `temperature` | `0.95` | Randomness (`do_sample` only) |
| `top_p` | `0.95` | Nucleus sampling (`do_sample` only) |
| `top_k` | `0` | Top-K sampling; `0` disables (`do_sample` only) |
| `seed` | `42` | Fix it to compare runs; `0` = random each run |
| `device` | `auto` | Compute device; `auto` follows ComfyUI's default |
| `dtype` | `auto` | Model precision; `auto` picks the optimal type for the device. bf16 and fp32 both produce correct speech |
| `force_offload` | `False` | Fully offload the model from VRAM after generation |

</details>

<p id="external-models" align="center">· · · · · · · · · · · · · ·</p>

### ▣ Loading external models

To run VibeVoice weights you already have on disk (safetensors / `.bin` /
`.gguf`) instead of the official download, wire the **`Load VibeVoice Model`**
node into the `external_model` input of the TTS or ASR node — connected, it
**overrides** `model_name`.

<details>
<summary><b>Setup, sidecar files and loader notes</b></summary>

1.  Place the weight file in `models/diffusion_models/` (a.k.a. `models/unet/`).
2.  *(Optional but recommended)* Bind configs with sidecar JSONs next to the weights:

| Sidecar | Binds | Fallback when absent |
| :--- | :--- | :--- |
| `<weights>.config.json` | Architecture config | `config.json` in the same directory, then the packaged default (`1.5B` / `7B` only) |
| `<weights>.preprocessor.json` | Audio preprocessor config | `preprocessor_config.json` in the same directory |
| `tokenizer.json` | Qwen2.5 text tokenizer | Packaged tokenizer, then a Hugging Face download |

3.  Select the file in `model_file`; keep `config_name` on `Auto-detect` unless
    needed — it reads the weight file's embedding fingerprint, and an explicit
    choice contradicting the weights is auto-corrected with a warning.
4.  Connect `VIBEVOICE_MODEL` → `external_model`.

**Notes:**

*   The model is built entirely on **CPU**; the single host-to-device transfer is
    owned by ComfyUI's VRAM arbitration.
*   Type guards reject mis-wiring (an ASR model on the TTS node and vice versa)
    with an error naming the correct node; a realtime bundle on the TTS node is
    accepted and routed through the realtime path.
*   `quantize_llm_4bit` applies to TTS / realtime models only; ASR always loads
    at full precision.
*   **GGUF:** `.gguf` files are listed in `model_file` and dequantized via the
    `gguf` package (`pip install gguf`) — ComfyUI's stock diffusion loader
    cannot parse GGUF. Both `models/diffusion_models/` and the ComfyUI-GGUF
    `unet_gguf` folder are discovered.
*   **Mismatch guard:** a `config_name` contradicting the weights fails fast
    with the offending tensor names and the correct `config_name` suggested —
    no raw `size mismatch` stack trace.
*   **Windows `Pin error.`:** under RAM pressure ComfyUI core can log a harmless
    flood of `[WARNING] Pin error.` while pinning partially offloaded weights.
    Quant-resident loads (GGUF / ConvRot INT8 / fp8) never offload, so they
    avoid it; otherwise start ComfyUI with `--disable-pinned-memory`.

</details>

<p id="host-ram" align="center">· · · · · · · · · · · · · ·</p>

#### ▣ Host RAM during an external load

Loading a single-file dense checkpoint costs roughly **one copy of the file in
host RAM**. That is measured on a *synthetic* dense bf16 `.safetensors` driven
through the real loader: synchronous per-phase counters on a warm process give
`load_torch_file +1.00× file`, the clone loop `+0.00×`, and assign +
`_post_assign_fixups` `−0.02×` — the file, held once.

**That synthetic number was not the reported failure.** A live load of the
16.6 GB `VibeVoice-ASR-HF-bf16.safetensors` was observed peaking well above
the file — about **1.26×** on the 5.41 GB `VibeVoice-1.5B` and **1.74×** on the
16.66 GB ASR checkpoint. The earlier explanation of that number (a second full
build on an identical re-execution) was itself real and is fixed, but it was
**not** the whole cause: the remainder is simply the **entire model resident in
host RAM**, which is exactly what the legacy patcher does — it has no virtual
address space to page weights out of, so a dense bf16 state dict must be held
whole until the weights are moved to the device. The 1× figure above is what
the *load path* costs; the ~1.3–1.7× is what the *patcher* costs on top of it.

**Status: mechanism wired, outcome NOT achieved and NOT measured.** The dense
external single-file route now *selects* ComfyUI's DynamicVRAM patcher and loads
through `load_models_gpu()`. That is the whole of what is done. The weights are
**not** being demand-paged, and the RAM fix is **not working** — see "Known
defect" below. No host-RAM number has been observed on a real checkpoint.

<details>
<summary><b>Known defect in the DynamicVRAM port (the RAM fix does not work)</b></summary>

ComfyUI's `ModelPatcherDynamic` gates its entire demand-paging mechanism on a
single per-module attribute, `comfy_cast_weights`
(`comfy/model_patcher.py:1967`); only modules carrying it get an `m._v =
vbar.alloc(...)` handle (`:1993`) and are paged. Modules without it take the
branch at `:2000-2009`, which stashes the host tensor in `self.backup` and makes
a full eager device copy.

This route **suppresses** `convert_tree_for_streaming`
(`modules/external_loader.py:1904`, `:2239`) to avoid two owners of one weight —
but `convert_tree_for_streaming` is the *only* thing in this pack that sets
`comfy_cast_weights` on these modules. Core sets it solely on
`comfy.ops.manual_cast.*` (`comfy/ops.py:821-853`), classes this pack's
transformers trees never use. So the suppression removes the attribute that
unlocks paging, and the load lands entirely on the eager branch.

Consequences, all in core's source, none measured here:

*   Every original host tensor is retained **by reference** in
    `ModelPatcher.backup` for the patcher's lifetime — the full 16.66 GB stays
    in host RAM. This is the spike the work set out to remove, unchanged.
*   A full device copy is made on top, and `model_loaded_weight_memory` reports
    full residency, so the low-VRAM budget is defeated rather than honoured.
*   Core's per-module lazy H2D (driven from `hasattr(s, "_v")` in
    `comfy/ops.py:374`) never happens, because no `_v` is ever allocated.

A second, deeper obstacle is visible in the same source: even with
`comfy_cast_weights` set, `ModelPatcherDynamic` **never releases the host
parameter** — `setup_param` allocates `_v` but does not replace the parameter
(`comfy/model_patcher.py:1921-1953`). Core's host-RAM economy for its own
models comes from the *loader* (`comfy/utils.py:165-166`, aimdo
`load_safetensors` against a meta-initialised tree), not from the patcher. This
pack instead materialises the whole model in ordinary host memory first
(`modules/loader.py:895-903`: `safe_open.get_tensor` returns an owned copy under
`MMAP_TORCH_FILES=False`). So the patcher port alone is unlikely to remove the
spike even once paging is reachable.

The port therefore needs re-planning; it is not a defect that can be closed by
adjusting the existing steps. Until it is re-done and measured, treat the dense
route's host-RAM behaviour as **unchanged from the legacy patcher**.

</details>

The family split below is real and holds; only the dense row's *outcome* is
unproven:

*   **Dense external single-file** checkpoints — the plain-BF16 family — select
    `ModelPatcherDynamic` and load via `load_models_gpu()`. This is the only
    family that moved. When ComfyUI's DynamicVRAM is unavailable (aimdo did not
    initialise, or the load device is not CUDA) the selector **silently falls
    back to the legacy patcher** — the previous, known-good behaviour — so
    nothing breaks on a machine without it. The decision lives in one function,
    `select_patcher_class` in `modules/patcher.py`; it is the whole scope of
    this change.
*   **GGUF**, **ConvRot INT8** and **fp8-resident** stay on the legacy
    `comfy.model_patcher.ModelPatcher`, byte-identical. They install quant
    residents that already stream natively, so a dynamic patcher would gain
    nothing and would change their offload order.
*   The **standard-directory dropdown loaders** (no external bundle) also stay
    on the legacy patcher.

To see which class actually served a load — and whether the dynamic route or
the silent fallback was taken — the scratch probe below reports
`PATCHER SELECTION` alongside the RAM numbers.

<details>
<summary><b>Measuring it on your machine</b></summary>

*   Turn on `DEBUG` logging for the loader and read the dtype line it now
    emits, e.g. `Model cast torch.float32 -> torch.bfloat16 (812
    mismatched params)`. A cast is the one thing that can double the
    footprint: it allocates a second copy of every mismatched parameter before
    the old one is released. Both source dtype and parameter count are on the
    line for exactly that reason.
*   For a timeline of a full external ASR load — peak private commit, peak
    working set, CUDA allocation, per-phase timings and **how many times each
    load phase ran** — use the scratch probe
    `.dev/scratch/probe_external_asr_ram.py` (Windows; needs `COMFYUI_ROOT`
    set; it loads the real checkpoint, so run it yourself):

    ```
    python .dev/scratch/probe_external_asr_ram.py "<path-to-checkpoint>" --config-name VibeVoice-ASR
    ```

    It loads the checkpoint **twice by default** (`--passes`), keeping the
    first copy alive across the second — that is the situation a live graph is
    in between two generations, and a second full load is the usual reason a
    load costs twice the file size. Pass `--passes 1` for a single load.
    Add `--drive-consumer` to also push the bundle through the ASR node's cache
    layer, which is where a re-load caused by a cache-key mismatch shows up. It
    is also **required** for the `PATCHER SELECTION` block: only the consumer
    path constructs a patcher, so without it that block reports `NONE OBSERVED`.

*   **Paste the timeline back** — the `TIMELINE`, `PEAK`, `PHASE COUNTS` and
    `PATCHER SELECTION` blocks — when reporting a load that uses far more RAM
    than the file size. The counts are the diagnosis: with `--passes N`, each
    phase should show exactly `N`. More than `N` means the checkpoint is being
    loaded more times than you asked for, which is a different bug from a slow
    or cast-heavy single load. `PATCHER SELECTION` answers the separate
    question of *which* patcher served the load: `is_dynamic=True` means the
    weights are demand-paged, and `is_dynamic=False` on a machine that should
    have DynamicVRAM means the silent fallback to legacy fired.

</details>

<p align="right"><a href="#readme-top">⟔ ▲ ⟓ back to top</a></p>

<p id="performance" align="center">◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆</p>

## ⌁ Performance &amp; stability

This node features a sophisticated system for managing performance, memory, and stability.

<p id="vram-matrix" align="center">· · · · · · · · · · · · · ·</p>

### ▣ Feature compatibility &amp; VRAM matrix

| Quantize LLM | Attention Mode      | Behavior / Notes                                                                                                                                | Relative VRAM |
| :--- | :--- | :--- | :--- |
| **OFF**      | `eager`             | Full Precision. Most compatible baseline.                                                                                                       | High          |
| **OFF**      | `sdpa`              | Full Precision. Recommended for balanced performance.                                                                                           | High          |
| **OFF**      | `flash_attention_2` | Full Precision. High performance on compatible GPUs.                                                                                            | High          |
| **OFF**      | `sage`              | Full Precision. Uses high-performance mixed-precision kernels. **Realtime models do not use `sage`** — see [Attention backends](#attention-backends). | High          |
| **ON**       | `eager`             | **Falls back to `sdpa`** with `bfloat16` compute. Warns user.                                                                                   | **Low**       |
| **ON**       | `sdpa`              | **Recommended for memory savings.** Uses `bfloat16` compute.                                                                                    | **Low**       |
| **ON**       | `flash_attention_2` | **Falls back to `sdpa`** with `bfloat16` compute. Warns user.                                                                                   | **Low**       |
| **ON**       | `sage`              | **Recommended for stability on standard models.** Uses `fp32` compute to ensure numerical stability with quantization, resulting in slightly higher VRAM usage. **Realtime models do not use `sage`** — see [Attention backends](#attention-backends). | **Medium**    |

<p id="transformers-support" align="center">· · · · · · · · · · · · · ·</p>

### ▣ `transformers` version support

This node is developed, tested and shipped against `transformers>=5.3.0,<5.4` —
the one supported line for all three model families. Two families have a hard
floor at 5.3.0:

*   **ASR (`VibeVoice-ASR-HF`)** requires `transformers >= 5.3.0` — the ASR model
    classes do not exist in any 4.x release. On an older `transformers` the node
    fails at load time with a clear message.
*   **Realtime TTS (`VibeVoice-Realtime-0.5B`)** requires `transformers >= 5.3.0` —
    the official `.pt` voice prompts do not work correctly on 4.57 and earlier.

Standard TTS is less sensitive, but only the declared range is supported;
`pip install -r requirements.txt` pulls the right version.

<p id="attention-backends" align="center">· · · · · · · · · · · · · ·</p>

### ▣ Attention backends

| Backend | Standard TTS | Realtime TTS |
| :------ | :---: | :---: |
| `eager` | ✓ | ✓ |
| `sdpa` | ✓ | ✓ |
| `flash_attention_2` | ✓ | ✓ |
| `sage` | ✓ | ✗ — falls back to `sdpa` with a warning |

`sage` is excluded for realtime models: its kernel ignores the attention mask,
so a text window placed on top of the cached voice prompt attends ahead of its
own positions and the conditioned voice degrades. Standard TTS models are
unaffected and still benefit from `sage`.

> [!NOTE]
> **No dtype caveat.** bf16 (the `auto` default) and fp32 both produce correct
> speech — there is no precision you need to select.

<p align="right"><a href="#readme-top">⟔ ▲ ⟓ back to top</a></p>

<p id="changelog" align="center">◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆</p>

## § Changelog

<details>
<summary><strong>v2.10.0 - One TTS node, WMNodes categories, self-contained tests</strong></summary>

**◆ Changes**

*   **One TTS node.** `VibeVoice TTS` now handles realtime models as well: just
    pick `VibeVoice-Realtime-0.5B` in `model_name`. The separate realtime TTS
    node is gone; existing saved workflows load unchanged.
*   **All nodes moved under `WMNodes`.** `WMNodes/sound/tts` for the TTS and
    loader nodes, `WMNodes/sound/asr` for ASR. Node IDs are unchanged, so saved
    workflows still load — only the menu location moved.

**• Fixes**

*   The bundled test suite now works from a fresh clone; it no longer references
    development-only files.

Full release history: [CHANGELOG.md](CHANGELOG.md).

</details>


## § License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for more information. The VibeVoice model and its components are subject to the licenses provided by Microsoft. Please use responsibly.


## ❦ Acknowledgments

*   **Microsoft** for creating and open-sourcing the [VibeVoice](https://github.com/microsoft/VibeVoice) project.
*   **The ComfyUI team** for their incredible and extensible platform.

<p align="right"><a href="#readme-top">⟔ ▲ ⟓ back to top</a></p>


## ★ Star history

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=wildminder/ComfyUI-VibeVoice&type=Timeline)](https://www.star-history.com/#wildminder/ComfyUI-VibeVoice&Timeline)

</div>


<p align="center"><b>ComfyUI-VibeVoice</b></p>

<!-- OWNER BADGES -->
[hf-microsoft]: https://img.shields.io/badge/microsoft-lightgrey?style=flat-square&logo=huggingface&logoColor=white
[hf-vibevoice]: https://img.shields.io/badge/vibevoice-lightgrey?style=flat-square&logo=huggingface&logoColor=white

<!-- TASK BADGES -->
[task-tts]: https://img.shields.io/badge/TTS-6f42c1?style=flat-square
[task-asr]: https://img.shields.io/badge/ASR-17a2b8?style=flat-square
[task-realtime]: https://img.shields.io/badge/TTS%20realtime-28a745?style=flat-square

<!-- FORMAT BADGES -->
[badge-gguf]: https://img.shields.io/badge/GGUF-dfb317?style=flat-square
[badge-fp8]: https://img.shields.io/badge/fp8-28a745?style=flat-square
[badge-int8]: https://img.shields.io/badge/int8-17a2b8?style=flat-square
[badge-int4]: https://img.shields.io/badge/int4-ffc107?style=flat-square

<!-- HEADER BADGES -->
[contributors-shield]: https://img.shields.io/github/contributors/wildminder/ComfyUI-VibeVoice.svg?style=for-the-badge
[contributors-url]: https://github.com/wildminder/ComfyUI-VibeVoice/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/wildminder/ComfyUI-VibeVoice.svg?style=for-the-badge
[forks-url]: https://github.com/wildminder/ComfyUI-VibeVoice/network/members
[stars-shield]: https://img.shields.io/github/stars/wildminder/ComfyUI-VibeVoice.svg?style=for-the-badge
[stars-url]: https://github.com/wildminder/ComfyUI-VibeVoice/stargazers
[issues-shield]: https://img.shields.io/github/issues/wildminder/ComfyUI-VibeVoice.svg?style=for-the-badge
[issues-url]: https://github.com/wildminder/ComfyUI-VibeVoice/issues
[license-shield]: https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge
[license-url]: https://github.com/wildminder/ComfyUI-VibeVoice/blob/master/LICENSE
[commit-shield]: https://img.shields.io/github/last-commit/wildminder/ComfyUI-VibeVoice?style=for-the-badge
[commit-url]: https://github.com/wildminder/ComfyUI-VibeVoice/commits/master
[comfy-shield]: https://img.shields.io/badge/ComfyUI-D96E3F?style=for-the-badge
[comfy-url]: https://github.com/Comfy-Org/ComfyUI
[hf-shield]: https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black
[hf-url]: https://huggingface.co/microsoft/VibeVoice-1.5B

<p align="right">(<a href="#readme-top">back to top</a>)</p>
