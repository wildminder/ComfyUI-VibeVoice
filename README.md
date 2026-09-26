<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

<div align="center">
  <h1 align="center">ComfyUI-VibeVoice</h1>

<img src="https://github.com/user-attachments/assets/ef2af626-efd6-4ce9-a3bf-87e4d51ca82d" alt="ComfyUI-VibeVoice Nodes" width="70%">

  <p align="center">
    A custom node for ComfyUI that integrates Microsoft's VibeVoice, a frontier model for generating expressive, long-form, multi-speaker conversational audio.
    <br />
    <br />
    <a href="https://github.com/wildminder/ComfyUI-VibeVoice/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/wildminder/ComfyUI-VibeVoice/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>

<!-- PROJECT SHIELDS -->
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

VibeVoice is a novel framework by Microsoft for generating expressive, long-form, multi-speaker conversational audio. It excels at creating natural-sounding dialogue, podcasts, and more, with consistent voices for up to 4 speakers.

<div align="center">
      <img src="./example_workflows/VibeVoice_example.png" alt="ComfyUI-VibeVoice example workflow" width="70%">
  </div>

The custom node handles everything from model downloading and memory management to audio processing, allowing you to generate high-quality speech directly from a text script and reference audio files.

**✨ Key Features:**
*   **One Canonical TTS Node:** `VibeVoice TTS` handles both standard VibeVoice models and the realtime `VibeVoice-Realtime-0.5B` checkpoint.
*   **Multi-Speaker TTS:** Generate conversations with up to 4 distinct voices in a single audio output.
*   **High-Fidelity Voice Cloning:** Use any audio file (`.wav`, `.mp3`) as a reference for a speaker's voice.
*   **Hybrid Voice Cloning:** Mix and match cloned speakers in the same script — at least one `speaker_*_voice` reference audio is required; other speakers are cloned from the provided reference(s).
*   **Realtime Model Support:** Realtime checkpoints are single-speaker and use official cached `.pt` voice prompts instead of reference audio.
*   **Flexible Scripting:** Use simple `[1]` tags or the classic `Speaker 1:` format to write your dialogue.
*   **Advanced Attention Mechanisms:** Choose between `eager`, `sdpa`, `flash_attention_2`, and the high-performance `sage` attention for fine-tuned control over speed and compatibility.
*   **Robust 4-Bit Quantization:** Run the large language model component in 4-bit mode to significantly reduce VRAM usage.
*   **Automatic Model Management:** Models are downloaded automatically and managed efficiently by ComfyUI to save VRAM.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## 🚀 Getting Started

The easiest way to install is through the **ComfyUI Manager:**
1.  Go to `Manager` -> `Install Custom Nodes`.
2.  Search for `ComfyUI-VibeVoice` and click "Install".
3.  Restart ComfyUI.

Alternatively, to install manually:

1.  **Clone the Repository:**
    Navigate to your `ComfyUI/custom_nodes/` directory and clone this repository:
    ```sh
    git clone https://github.com/wildminder/ComfyUI-VibeVoice.git
    ```

2.  **Install Dependencies:**
    Open a terminal or command prompt, navigate into the cloned directory, and install the required Python packages. **For quantization support, you must install `bitsandbytes`**.
    ```sh
    cd ComfyUI-VibeVoice
    pip install -r requirements.txt
    ```

3.  **Optional: Install SageAttention**
    To enable the `sage` attention mode, you must install the `sageattention` library. For Windows users, a pre-compiled wheel is available at [AI-windows-whl](https://github.com/wildminder/AI-windows-whl).
    > **Note:** This is only required if you intend to use the `sage` attention mode.

> **Audio backend:** Audio resampling uses **`torchaudio`** (the ComfyUI-core idiom) as the primary
> library; file decoding uses **PyAV** (`av`, bundled with ComfyUI) with `soundfile` as fallback.
> **`librosa` is not required** — it is only an optional last-resort fallback, and the node
> degrades gracefully when it (or `scipy`) is absent. To install the optional fallbacks:
> `pip install scipy librosa` (or `pip install ComfyUI-VibeVoice[audio-extra]`).

4.  **Start/Restart ComfyUI:**
    Launch ComfyUI. The "VibeVoice TTS" node will appear under the `audio/tts` category. The first time you use the node, it will automatically download the selected model to your `ComfyUI/models/tts/VibeVoice/` folder.

## Models
| Model | Context Length | Generation Length |  Weight |
|-------|----------------|----------|----------|
| VibeVoice-1.5B | 64K | ~90 min | [HF link](https://huggingface.co/microsoft/VibeVoice-1.5B) |
| VibeVoice-Large| 32K | ~45 min | [HF link](https://huggingface.co/aoi-ot/VibeVoice-Large) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## 🛠️ Usage

The node is designed for maximum flexibility within your ComfyUI workflow.

1.  **Add Nodes:** Add the `VibeVoice TTS` node to your graph. Use ComfyUI's built-in `Load Audio` node to load your reference voice files.
2.  **Connect Voices (Optional):** Connect the `AUDIO` output from each `Load Audio` node to the corresponding `speaker_*_voice` input.
3.  **Write Your Script:** In the `text` input, write your dialogue using one of the supported formats.
4.  **Generate:** Queue the prompt. The node will process the script and generate a single audio file containing the full conversation.

> **Tip:** For a complete workflow, you can drag the example image from the `example_workflows` folder onto your ComfyUI canvas.

### Scripting and Voice Modes

#### Speaker Tagging
You can assign lines to speakers in two ways. Both are treated identically.

*   **Modern Format (Recommended):** `[1] This is the first speaker.`
*   **Classic Format:** `Speaker 1: This is the first speaker.`

You can also add an optional colon to the modern format (e.g., `[1]: ...`). The node handles all variations consistently.

#### Hybrid Voice Generation
This is a powerful feature that lets you mix cloned voices in the same script. **At least one `speaker_*_voice` reference audio is required** — VibeVoice anchors the timbre of every speaker to a provided reference, so you cannot generate a fully reference-free ("zero-shot") voice for any speaker.

*   **To Clone a Voice:** Connect a `Load Audio` node to the speaker's input (e.g., `speaker_1_voice`).
*   **To Reuse a Cloned Voice:** Any other speaker may be left empty. When a speaker has no reference, its voice is cloned from the provided reference(s) rather than generated from scratch.

**Example Hybrid Script:**
```
[1] This line will use the audio from speaker_1_voice.
[2] This line will reuse the cloned voice from speaker 1.
[1] I'm back with my cloned voice.
```
In this example, you would connect an audio source to `speaker_1_voice`; speakers `[2]` are cloned from it.

#### Realtime Models (Single-Speaker Voice Prompts)

`VibeVoice TTS` is the single canonical TTS node. When you select a realtime checkpoint
(e.g. `VibeVoice-Realtime-0.5B`) in `model_name`, the node switches to the official
windowed realtime inference path:

*   Realtime models are **single-speaker**. Speaker tags in the script are stripped and the
    lines are joined into one utterance.
*   Realtime **reference-audio cloning is not supported** — connect no `speaker_*_voice`
    inputs. Connected reference audio is ignored with a warning.
*   Realtime models require an official cached `.pt` voice prompt in the `voice_preset`
    input. Put the prompts in `models/tts/VibeVoice/voices` (or any folder registered under
    ComfyUI's `vibevoice_voices` key). Prompts are discovered recursively by file stem, so
    the official `voices/streaming_model/*.pt` layout can be copied intact.
*   **`cfg_scale` behaves differently for realtime models.** The realtime architecture
    generates a continuous acoustic latent by diffusion rather than sampling discrete
    speech tokens, so its guidance response is nothing like the standard model's. Below
    roughly `1.5` it stops transcribing the script and collapses into syllable repetition
    ("it's a, it's a, ..."). `1.5`–`1.8` speak the script correctly; `2.0` and above start
    drifting. Because the node shares one `cfg_scale` widget with the standard models
    (whose default of `1.3` is correct there), the realtime path raises any value below
    `1.5` to `1.5` and logs a warning. Set the widget to `1.5`–`1.8` to silence it.
*   `inference_steps` controls **diffusion quality/time**; `max_new_tokens` controls
    **generated length**. They are independent.
*   **`max_new_tokens = 0` means auto, not "the model maximum."** For realtime models this
    is a *combined* budget: the generation loop charges prefilled **text tokens** and
    generated **speech latents** against the same allowance — each window consumes 5 text
    tokens *and* 6 speech latents, i.e. 11 units. The auto budget therefore covers the
    script window by window at 11 units per 5-token window, plus a **tail** of two windows
    per text window, so the model has room to finish the last clause and reach its own
    end of speech instead of being cut off the moment the last text window is fed, then
    clips to whatever context the prompt left. The chosen budget is logged. An explicit
    positive `max_new_tokens` always wins.
*   **If the console reports "stopped on the length budget ... without the model signalling
    end of speech"**, the clip was cut off mid-script. Raise `max_new_tokens` or shorten
    the script — lowering it makes the truncation worse, not better. This should not
    happen with `max_new_tokens = 0`; if it does, the budget was overridden explicitly.
*   `do_sample` / `temperature` / `top_p` / `top_k` are not used by the current realtime
    loop; model defaults apply and a one-time warning is logged.
*   Output is a normal, completed ComfyUI `AUDIO` object. There is **no live PCM
    streaming** in ComfyUI; incremental playback would require a custom transport and is
    out of scope.

Example realtime setup:

1.  Download an official voice prompt (for example `en-Carter_man.pt`) and copy it to
    `ComfyUI/models/tts/VibeVoice/voices/`.
2.  Add the `VibeVoice TTS` node.
3.  Set `model_name` to `VibeVoice-Realtime-0.5B` and `voice_preset` to the prompt stem
    (e.g. `en-Carter_man`).
4.  Write a single-speaker script, leave the `speaker_*_voice` inputs unconnected, and
    queue the prompt.

> **Deprecated:** the old `VibeVoice Realtime TTS` node ID (`VibeVoiceRealtime`) still loads
> and forwards to `VibeVoice TTS` so saved workflows keep working, but it is deprecated and
> will be removed in the next major release. Build new realtime workflows with
> `VibeVoice TTS`.

### Node Inputs

*   **`model_name`**: Select the VibeVoice model to use. Standard TTS models are listed
    first, then realtime TTS models. ASR models are not listed here — they belong to the
    VibeVoice ASR node.
*   **`text`**: The conversational script. See "Scripting and Voice Modes" above for formatting.
*   **`quantize_llm_4bit`**: Enable to run the LLM component in 4-bit (NF4) mode, dramatically reducing VRAM usage.
*   **`attention_mode`**: Select the attention implementation: `eager` (safest), `sdpa` (balanced), `flash_attention_2` (fastest), or `sage` (quantized high-performance). **Realtime models do not use `sage`** — see the support matrix below.
*   **`cfg_scale`**: Controls how strongly the model adheres to the reference voice's timbre. Higher values are stricter. Recommended: `1.3`.
*   **`inference_steps`**: Number of diffusion steps for audio generation. Recommended: `10`. This controls diffusion quality/time only, never the speech length.
*   **`seed`**: A seed for reproducibility. Set to 0 for a random seed on each run.
*   **`do_sample`, `temperature`, `top_p`, `top_k`**: Standard sampling parameters for controlling the creativity and determinism of the speech generation. Not used by realtime models.
*   **`cfg_scale`**: Classifier-Free Guidance scale. Recommended: `1.3` for standard models, `1.5`-`1.8` for realtime models. Values below `1.5` make a realtime checkpoint emit syllable repetition instead of speech; the realtime path raises them to `1.5` and logs a warning.
*   **`max_new_tokens`**: Speech-token / generated-length budget. `0` = auto. Raise it if output is cut off, lower it if output is too long. For realtime models this budget pays for both prefilled text tokens and generated speech latents — see the realtime section above.
*   **`voice_preset`**: Official cached `.pt` voice prompt for realtime models. Ignored by standard models. Realtime models require a non-`None` value.
*   **`force_offload`**: Forces the model to be completely offloaded from VRAM after generation.

### Loading External Models

By default the nodes download / load official VibeVoice checkpoints from the `models/tts/VibeVoice` folder. You can instead load **any VibeVoice weight file you already have on disk** (safetensors / `.bin` / `.gguf`) via the dedicated **`Load VibeVoice Model`** node, which outputs a `VIBEVOICE_MODEL` that plugs into the optional `external_model` input of the **TTS** and **ASR** nodes.

**Setup:**

1.  Place your VibeVoice weight file in ComfyUI's `models/diffusion_models/` (a.k.a. `models/unet/`) folder.
2.  *(Optional but recommended)* Place sidecar JSON files next to the weight file to bind the architecture config, audio preprocessor, and tokenizer:
    *   `<weight_file>.config.json` — architecture config (preferred), or a `config.json` in the same directory.
    *   `<weight_file>.preprocessor.json` — audio preprocessor config (preferred), or a `preprocessor_config.json` in the same directory.
    *   `tokenizer.json` — Qwen2.5 text tokenizer (same directory). Falls back to the packaged tokenizer or a HuggingFace download if absent.
3.  Add the **`Load VibeVoice Model`** node, select your file in `model_file`, and pick the architecture in `config_name` (`Auto-detect`, `VibeVoice-1.5B`, `VibeVoice-7B`, `VibeVoice-Realtime-0.5B`, or `VibeVoice-ASR`). `Auto-detect` (the default) reads the weight file's embedding fingerprint and selects the matching family; an explicit selection that contradicts the weights is auto-corrected with a warning. When no sidecar config is present, `config_name` selects the packaged default config (available for `1.5B` and `7B`).
4.  Connect the node's `VIBEVOICE_MODEL` output to the `external_model` input of the TTS / ASR node. When connected, `external_model` **overrides** the `model_name` dropdown.

**Notes:**

*   The model is built entirely on **CPU**; the single host-to-device transfer is owned by ComfyUI's VRAM arbitration (same contract as the standard loader path).
*   Type guards prevent mis-wiring: an ASR model on the TTS node, a TTS model on the ASR node, etc. raise a clear error pointing to the correct node. A realtime bundle connected to the TTS node is accepted and routed through the canonical realtime path.
*   4-bit LLM quantization (`quantize_llm_4bit`) applies to TTS / realtime models only; ASR models are always loaded at full precision.
*   **GGUF support:** `.gguf` files are listed in the `model_file` dropdown and dequantized via the `gguf` Python package (`pip install gguf`). ComfyUI's stock `Load Diffusion Model` cannot parse GGUF, so this node handles it directly. Files in both `models/diffusion_models/` and the ComfyUI-GGUF `unet_gguf` folder are discovered.
*   **Config/weight mismatch guard:** if the selected `config_name` does not match the weight file's architecture, loading fails fast with a clear error naming the offending tensors and suggesting the correct `config_name` (or `Auto-detect`) — instead of a raw `size mismatch` stack trace.
*   **Windows RAM / `Pin error.` note:** when a model is partially offloaded to CPU, ComfyUI core tries to pin the offloaded weights (`cudaHostRegister`) as a best-effort, non-fatal optimization. Under RAM pressure this can log a flood of `[WARNING] Pin error.` messages — harmless, but noisy. Quant-resident loads (GGUF / ConvRot INT8 / fp8) avoid the situation entirely because their VRAM residency is roughly the file size, so nothing gets offloaded. If you still see pin errors with other oversized models, start ComfyUI with `--disable-pinned-memory` to opt out of the pinning attempt.

<!-- PERFORMANCE SECTION -->
## ⚙️ Performance & Advanced Features

This node features a sophisticated system for managing performance, memory, and stability.

### Feature Compatibility & VRAM Matrix

| Quantize LLM | Attention Mode      | Behavior / Notes                                                                                                                                | Relative VRAM |
| :----------- | :------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------- | :------------ |
| **OFF**      | `eager`             | Full Precision. Most compatible baseline.                                                                                                       | High          |
| **OFF**      | `sdpa`              | Full Precision. Recommended for balanced performance.                                                                                           | High          |
| **OFF**      | `flash_attention_2` | Full Precision. High performance on compatible GPUs.                                                                                            | High          |
| **OFF**      | `sage`              | Full Precision. Uses high-performance mixed-precision kernels. **Realtime models do not use `sage`** — see the support matrix below. | High          |
| **ON**       | `eager`             | **Falls back to `sdpa`** with `bfloat16` compute. Warns user.                                                                                   | **Low**       |
| **ON**       | `sdpa`              | **Recommended for memory savings.** Uses `bfloat16` compute.                                                                                    | **Low**       |
| **ON**       | `flash_attention_2` | **Falls back to `sdpa`** with `bfloat16` compute. Warns user.                                                                                   | **Low**       |
| **ON**       | `sage`              | **Recommended for stability on standard models.** Uses `fp32` compute to ensure numerical stability with quantization, resulting in slightly higher VRAM usage. **Realtime models do not use `sage`** — see the support matrix below. | **Medium**    |

### Support Matrix

#### `transformers` version support

The three model families do **not** share a supported `transformers` range. The
declared dependency is `transformers>=5.3.0,<5.4` — the one line this node is
developed, tested and shipped against, and the only line with a green realtime
row in the recorded matrix. 4.57.6 appears below as a *measured but not admitted*
row, not as a supported one; the ASR node additionally requires 5.3.0 and says so
at load time.

| Model family | `transformers` 4.5x | `transformers` 5.3.0 | Notes |
| :----------- | :------------------ | :-------------------- | :---- |
| Standard TTS (`VibeVoice-1.5B`, `VibeVoice-7B`) | ✅ measured, not admitted | ✅ | Unaffected either way — it never loads a cached KV prompt. |
| Realtime TTS (`VibeVoice-Realtime-0.5B`) | ⚠️ 10/11 acceptance tests pass | ✅ 11/11 | The realtime path is the only one that carries a pickled KV prompt across a 4.x→5.x cache refactor, so it is the only one that needs the shim. |
| ASR (`VibeVoice-ASR-HF`) | ⛔ | ✅ | **Requires `transformers >= 5.3.0`.** `VibeVoiceAsrForConditionalGeneration` does not exist in any 4.x release. On an older `transformers` the node fails with a clear message rather than an `ImportError`. |

The 4.5x row is not fully green: `test_realtime_checkpoint_generation_and_controls`
fails on the auto-length assertion (on 4.57 the model reaches end-of-speech
before exhausting the latents budget; on 5.3 it does not), which is why 4.x is
not the declared range. The recorded matrix, the exact commands and the two
reproduction steps for the isolated venv are in
[`docs/plans/2026-09-26-two-version-matrix.md`](docs/plans/2026-09-26-two-version-matrix.md).
`tests/test_pyproject.py` fails if the declared range is widened past a version
that matrix has not measured, so this table cannot silently go stale.

#### Attention backends

`sage` is **excluded for realtime models** and falls back to `sdpa` with a
warning. Measured on `VibeVoice-Realtime-0.5B` (RTX SM89, bf16, one fixed voice
prompt, cosine of the first-window conditioning against `eager`, gate `0.999`):

| Backend | cosine vs `eager` | Realtime |
| :------ | :--------------- | :------- |
| `sdpa` | 0.999962 | ✅ |
| `flash_attention_2` | 0.999940 | ✅ |
| `sage` | 0.994651 | ❌ excluded |

The sage kernel ignores the attention mask, so a text window placed on top of the
316-token voice prefill attends ahead of its own positions; its int8/fp8
quantisation adds a further relative L2 error of 0.043. The standard TTS family
is unaffected and still uses `sage`.

#### Precision

**No dtype caveat.** bf16 (the `auto` default) and fp32 were measured on the same
seed through the node path and agree: first-latent EOS `0.000017` vs `0.000016`
(neither stops generation), and the conditioning vectors have cosine `0.999968`.
Precision does not shift the end-of-speech decision, so there is no dtype you
need to select to get correct speech.

#### Why the realtime path needs a compatibility shim

The official `.pt` voice prompts are pickled with a legacy `DynamicCache` whose
KV lives in `key_cache` / `value_cache` lists. transformers 4.57+ expects cache
*layer objects* in a `layers` list. Getting this wrong does not raise: 5.x
silently builds the attention mask as if the prefill did not exist, and you get
a fraction of a second of noise instead of a voice. The shim and its reasoning
are documented at the top of
`src/vibevoice/modular/modeling_vibevoice_streaming_inference.py`.

A second 5.x change breaks the same path more quietly, and it is worth knowing
about if you read the loop: `cache_position` used to name *only* the tokens
about to be fed, and 5.x returns the entire history with the new positions
appended. Since the loop slices its inputs by `cache_position.shape[0]`, an
unadapted 5.x re-feeds the whole cached prefix on every step — the KV cache
grows super-linearly, the model never reaches its own end of speech, and every
clip runs to the length budget. The shim normalises it back to the 4.x contract.
The symptom to recognise is narrow and diagnostic: a one-word script still
works, anything longer never terminates on its own.


<!-- CHANGELOG -->
## Changelog

<details open>
<summary><strong>v2.9.0 - One canonical TTS node with correct realtime-model support</strong></summary>

### ✨ New Features
*   **`VibeVoice TTS` is now the single canonical TTS node.** Its `model_name` dropdown
    lists standard TTS models first, then realtime TTS models (ASR stays on the ASR node),
    and the node routes each family to the right generation path. The pre-existing input
    order is preserved exactly; the only new input is the appended `voice_preset` combo.
*   **Official realtime generation path.** `modules/realtime_generation.py` calls the
    vendored streaming processor's `process_input_with_cached_prompt()` and the official
    windowed `model.generate()` loop, deep-copying the cached prompt per run, seeding before
    generation, driving ComfyUI progress, and honoring user interruption. The old
    pad-token placeholder prefill (`prefill_voice_prompt()`) and the old
    `generate_streaming_audio()` wrapper are removed; they are no longer referenced by any
    production file.
*   **Realtime voice-prompt assets.** `modules/voice_presets.py` discovers `.pt` cached
    voice prompts in registered `vibevoice_voices` roots and in
    `models/tts/VibeVoice/voices`, resolves names case-insensitively (first registration
    wins on collisions), validates the four required cached branches (`lm`, `tts_lm`,
    `neg_lm`, `neg_tts_lm`), and caches by path/mtime/size/device. Loading uses
    `weights_only`-equivalent restricted deserialization inside
    `torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache])`, so official
    prompts load safely on transformers 5.3 while arbitrary globals stay rejected.
*   **Independent realtime controls.** `inference_steps` maps to the model's
    `set_ddpm_inference_steps()` (diffusion quality/time) and `max_new_tokens` maps to the
    total streaming sequence-length budget (`0` = model default). They are no longer
    conflated.
*   **Renamed/local realtime checkpoints are safe.** `modules/model_info.py` infers the
    `streaming_tts` family for local names containing `realtime`/`stream` (ASR names always
    win), model scanning skips `voices` directories and excludes `.pt` from weight
    discovery, and the node applies a loaded-class safety net so a renamed realtime
    checkpoint is re-routed correctly while a genuine mismatch fails with a clear error.
*   **Opt-in real-checkpoint acceptance.** `tests/test_realtime_e2e_gpu.py` (skipped by
    default) loads the real `VibeVoice-Realtime-0.5B` plus an official `.pt` voice prompt
    supplied only through environment variables, then asserts finite non-silent 24 kHz audio,
    cache reuse, a calibrated length cap that trips `reach_max_step_sample`, diffusion-step
    independence, and one canonical-node forced-offload run.
*   **`e2e_smoke_test.py` realtime branch.** New `--realtime-model` and `--voice-preset`
    arguments run the cached-prompt adapter (both required together). Realtime production
    modules are imported only inside that branch, so `--help` and the standard path keep
    their lightweight imports. The stale advice to use a realtime checkpoint as a generic
    standard fallback is removed.

### 🔧 Changes
*   **Deprecated forwarding shim.** `nodes/realtime_node.py` keeps the `VibeVoiceRealtime`
    node ID, its exact legacy input order (including the no-op `stream` widget), and one
    `AUDIO` output, but now holds no loading or generation logic: validation and execution
    delegate to `VibeVoiceTTSNode`, `stream` is stripped, and one process-level deprecation
    warning is logged. The class is targeted for removal in the next major release.
*   **Shared lifecycle.** Output dictionary, audio preview, force-offload (warm re-attach),
    and the cancellation fallback are implemented once and shared by both families.
*   **Validation.** Named realtime models require a non-`None` `voice_preset` at queue time
    with an actionable message naming the input and `models/tts/VibeVoice/voices`; an old
    saved prompt without the new key is treated as missing. Connected `external_model`
    inputs keep the existing queue-time bypass and are re-checked at execution time. The
    validator declares only `model_name`, `voice_preset`, and `external_model`, so the
    message is attributed to those inputs instead of being repeated once per widget
    (ComfyUI core emits one error per validated input).
*   **Unified model options.** A combined TTS-family selector orders standard models before
    realtime models, and `modules/custom_types.py` documents that `VibeVoiceModel` carries
    standard, realtime, and ASR bundles.
*   **Version metadata** bumped to `2.9.0`. The 2.8.2 changelog entry and the standard
    `example_workflows/VibeVoice_example.json` are unchanged and remain valid.

### 🧪 Tests
*   `tests/test_unified_tts_node.py` covers schema/append-only ordering, preset discovery
    fallback, the full validation truth table, mutually exclusive routing, external
    realtime/ASR handling, the loaded-pair safety net, independent steps/length, ignored-input
    warnings, and the shared output/offload/cancel paths.
*   `tests/test_realtime_node.py` was rewritten as shim tests: legacy widget order, one-time
    deprecation warning, `stream` stripping, delegation-only source assertions, unique node
    IDs, and a `tests/fixtures/legacy_realtime_workflow.json` compatibility fixture.
*   Both serialization `xfail`s are **resolved**: real `BaseModelOutputWithPast` /
    `DynamicCache` round trips and the real `en-Carter_man.pt` prompt now load on
    transformers 5.3 with safe loading retained.
*   `tests/test_integration.py` realtime progress/cancellation coverage retargeted to the
    canonical node, and new `tests/test_e2e_smoke_contract.py` pins the smoke script's CLI
    and realtime-branch wiring.

</details>

<details>
<summary><strong>v2.8.2 - Native VibeVoice-ASR-HF: correct, working transcription</strong></summary>

### ✨ New Features
*   **The ASR node now uses `VibeVoice-ASR-HF`** (≈17.4 GB) — the
    transformers-native conversion of the ASR-7B model
    (`microsoft/VibeVoice-ASR-HF`, requires transformers ≥ 5.3.0). It loads
    through the transformers builtins
    (`VibeVoiceAsrForConditionalGeneration` + `AutoProcessor`) with no
    vendored compat shims, transcribes in a single pass (matching ComfyUI's
    batch pipeline semantics — up to 60 minutes of audio), and decodes
    speaker/timestamp segments natively
    (`Start/End/Speaker/Content`). Saved workflows that still select the
    retired `VibeVoice-ASR` / `VibeVoice-ASR-Streaming-1.5B` /
    `VibeVoice-ASR-Streaming-7B` names resolve to `VibeVoice-ASR-HF`;
    locally downloaded streaming checkpoints keep working through the
    chunked streaming protocol.

### 🐛 Bug Fixes
*   **Switching attention modes no longer leaks the previous model.** The
    ASR loader registered every built model in a second cache under its own
    key format; when a setting changed (e.g. sdpa → sage), the eviction
    destroyed the patcher entry but the second entry survived and pinned the
    superseded 17 GB tree in RAM/VRAM — the next load's VRAM budget
    collapsed ("0.00 MB loaded, 15888 MB offloaded") and inference ran at a
    crawl. The ASR loader is now a pure builder: the live model has exactly
    one owner (the patcher key, evicted on model change), matching the main
    TTS node's technique.
*   **SageAttention now actually applies on ASR models.** The main loader
    patches `Qwen2Attention` post-load when sage is selected; the ASR
    loaders never did, so "sage" silently ran sdpa. Both ASR branches now
    apply the same post-load patch after the streaming conversion (and
    raise the actionable "Incompatible hardware/setup" error when sage is
    unavailable).
*   **No more `max_new_tokens`/`max_length` console warning on every ASR
    run.** The ASR-HF checkpoint ships a generation config presetting both
    values; the loader now clears the preset at load time since the node
    drives generation via `max_new_tokens` only.
*   **Any input sample rate now works (44.1 kHz etc.) — auto-resampled to
    the model's rate.** The native ASR feature extractor refuses audio
    sampled at anything but its declared 24 kHz ("was trained using a
    sampling rate of 24000 ... not 44100"); the vendored ASR processor only
    resampled file-path inputs, silently mis-timing in-memory numpy/tensor
    audio from the node. Both ASR paths now resample incoming audio to the
    processor's declared rate — the same inbuilt conversion the main
    VibeVoice TTS routine performs.
*   **ASR models no longer force-place onto the GPU mid-load (OOM fix).**
    The ASR loader and handler moved the whole tree to the target device
    (`.to(device)` / `device_map`) inside `patch_model` — before ComfyUI's
    VRAM arbitration ran — so the 17 GB ASR-HF checkpoint OOM'd a 16 GB GPU
    (crashed at 15.14/15.99 GiB). ASR now follows the same device contract
    as the TTS loader (DF-003): the model is built entirely on CPU, the
    patcher owns the single host-to-device transfer after arbitration, and
    the tree is converted to ComfyUI's lowvram streaming protocol
    (`convert_tree_for_streaming`, incl. the native RMSNorm classes) so an
    oversized model partially offloads instead of dying — no OOM.
*   **Official models are no longer re-downloaded when they already exist
    under a secondary models root.** `_resolve_official_model_dir` only
    checked the first registered `tts` folder, so a checkpoint placed
    manually under `extra_model_paths.yaml` root (e.g.
    `models/tts/VibeVoice/VibeVoice-ASR-HF`) triggered a fresh
    download into the primary root. All registered tts roots are now
    searched in order and the first existing candidate directory wins;
    incomplete (partial) downloads resume in place instead of being
    orphaned. This applies to TTS official models too.
*   **ASR model loading crashed with `Unknown dtype: sdpa`.** The shared
    patcher invoked the handler's `load_model` positionally:
    `load_model(target_device, attention_mode)`. The TTS handler signature
    is `(device, attention_mode)` so it always worked there, but the ASR
    handler is `(device, dtype_str, attention_mode)` — the positional call
    landed the attention mode in the dtype slot. The call is now
    keyword-based (`attention_mode=`), which is signature-agnostic for all
    handlers.
*   Locally discovered directories whose name contains "ASR" are now
    classified into the ASR family (previously they defaulted to TTS and
    never appeared in the ASR node's dropdown).

### 📦 Behind the scenes
*   The vendored ASR path (`src/vibevoice`) remains for externally-loaded
    and streaming checkpoints: absolute imports fixed to package-relative,
    transformers 5.x `prepare_inputs_for_generation` loop fixes, the
    chunked streaming protocol (`streaming_generate` port) for
    `VibeVoice-ASR-Streaming-*` directories, and checkpoint-shipped
    tokenizer files win over the base Qwen tokenizer (mismatch logged).
    `apply_chat_template(tokenize=True)` passes `return_dict=False`
    (transformers ≥ 4.44 returns a `BatchEncoding` there, which broke the
    prompt assembly).

</details>

<details>
<summary><strong>v2.8.1 - Flat-Block GGUF Recovery (quantui-rs files load again)</strong></summary>

### 🐛 Bug Fixes
*   **GGUF files written by quantui-rs (e.g. a whole-checkpoint
    `VibeVoice-1.5B-q8_0.gguf` or `VibeVoice-7B-q8_0.gguf` converted with
    conv layers included) no
    longer fail to load** with `ValueError: ... could not determine the
    VibeVoice architecture` (the underlying error was `Quantized tensor
    row size (7) is not a multiple of Q8_0 block size (32)`). Root cause:
    quantui-rs stores tensors whose last dimension is below the quant
    block size (102 small conv kernels in the tokenizers) as a *flat
    pool of whole blocks*, which the stock GGUF reader rejects at open
    time — so the file was never even fingerprinted. A new tolerant
    reader opens such files (logged once as a warning), keeping header
    dims intact.
*   **Mixed naming conventions no longer fail key mapping.** The 7B
    quantui-rs export keeps HF names for everything but the lm_head,
    which it renames llama.cpp-style to `output.weight` — the census is
    1204 HF keys + 1 llamacpp key, and the old mapper hard-rejected such
    'mixed' files. A mixed census now resolves by majority vote: the
    dominant convention is kept and minority keys are mapped through the
    other convention's alias table (an exact 50/50 tie or an unknown
    minority key still fails loudly).
*   **Quantized non-Linear weights (embeddings, conv heads) now load via
    dequant-at-load** instead of raising `QuantTargetMismatch`: quantui-rs
    quantizes `embed_tokens` and the conv heads too, and those can't be
    quant-resident `GGUFLinear` targets. They are materialized once at
    load (fp32) after shape validation — the Linear weights in the same
    file remain quant-resident.
*   **A quantized `lm_head` under a tied-embeddings config is now rejected
    loudly** at install time; previously `tie_weights()` would silently
    overwrite it with the embedding weight.
*   Values were verified bit-close against the original safetensors
    checkpoint (embedding max err 0.0027 on |w|max 0.68 = normal Q8_0
    loss; head conv max err 0.00011).
*   Fully spec-conformant GGUF files are unaffected: the stock reader path
    runs first and the recovery only engages on the block-size error.

</details>

<details>
<summary><strong>v2.8.0 - FP8-Resident Loading + Streaming Safetensors (RAM-spike fix)</strong></summary>

### 🚀 Performance / Memory
*   **fp8 checkpoints (e.g. `VibeVoice-7B-fp8_e4m3.safetensors`) no longer
    spike RAM or flood `[WARNING] Pin error.`** Previously the file was
    dequantized to bf16 *at load time* (full in-memory dict + fp32
    transients, ~25 GB peak for the 7B file), producing a ~16 GB model that
    no longer fit VRAM — ComfyUI then partially offloaded it, and core's
    best-effort pinned-memory registration failed under the RAM pressure.
*   **fp8 weights now stay resident:** fp8 storage + per-tensor fp32 scales
    are installed untouched (VRAM residency ≈ file size, ~9 GB for the 7B
    file, so it fully fits a 12 GB card) and are dequantized per matrix
    multiply via comfy-kitchen's `dequantize_per_tensor_fp8`
    (triton/cuda/eager backends; bit-exact vs float math).
*   **Quantized safetensors now stream per-tensor:** the full-file state
    dict is never materialized, so peak host RAM is bounded by the model
    plus one tensor in flight instead of ~2× the dequantized checkpoint.
    Dense bf16 files and `.bin`/`.pt` keep the existing batch loader.

### 🛡️ Safety
*   fp8 files with per-row scales — or boxes without a working comfy-kitchen
    fp8 backend — transparently fall back to the previous dequant-at-load
    behavior.
*   A quantized checkpoint whose `lm_head` is quantized while the config
    ties word embeddings is rejected with a clear error (tied heads cannot
    be quant-resident).
*   Mid-stream shape / storage-dtype disagreements fail fast with the same
    actionable config-mismatch message as the dense path.

</details>

<details>
<summary><strong>v2.7.1 - Config dtype access fix</strong></summary>

### 🐛 Bug Fixes
*   Version-safe `torch_dtype` config access — silences the transformers
    deprecation warning on newer versions.

</details>

<details>
<summary><strong>v2.7.0 - Auto-detect config, drop VibeVoice-Large</strong></summary>

### ✨ New Features
*   `config_name` gains **Auto-detect** (default): the architecture family is
    resolved from the weight file's embedding fingerprint before any heavy
    load; explicit selections that contradict the weights are auto-corrected
    with a warning.
*   The legacy `VibeVoice-Large` option is removed.

</details>

<details>
<summary><strong>v2.6.0 - Native Lowvram Streaming (oversized models work)</strong></summary>

### 🚀 Performance / Memory
*   **Models larger than your VRAM now load and generate correctly** instead
    of crashing with `Input type (CUDABFloat16Type) and weight type
    (CPUBFloat16Type)` (VibeVoice-7B). The tree is converted at load time to
    comfy-native streaming modules (`comfy_cast_weights` +
    `cast_bias_weight(offloadable=True)`), so ComfyUI's lowvram arbiter can
    partially load and partially offload it like any native model — hot
    layers stay in VRAM, cold layers stream from CPU on demand.
*   Previously, foreign transformers modules were silently stranded on CPU by
    core's `partially_unload` and had no way back; the temporary
    force-full-placement workaround (which blocked legitimate offloading) is
    removed in favor of the native protocol.

### 🛡️ Safety
*   Quant residents (GGUF raw blocks / ConvRot INT8) participate in streaming
    with their storage dtype pinned (`weight_comfy_model_dtype`) and a
    dtype-preserving pull path — raw bytes are moved, never recast.
*   Vendored norm classes (RMSNorm/ConvRMSNorm/LayerNorm variants, Qwen2
    RMSNorm) gained streaming forwards with parity tests against the
    originals.

</details>

<details>
<summary><strong>v2.5.0 - Quant-Resident Runtime: GGUF Raw-Block Residency + ConvRot INT8</strong></summary>

### 🚀 Performance / Memory
*   **GGUF loading no longer spikes RAM (~2× full-float size eliminated).**
    Quantized weights now stay **raw-block resident end-to-end**: the loader
    copies exact on-disk block bytes into `uint8` parameters (zero float
    materialization), transfers raw bytes to VRAM, and dequantizes each block
    to the activation dtype only during the matrix multiply. A ~4 GB Q8_0
    VibeVoice GGUF that previously spiked **~10 GB of RAM** and then occupied
    full-float VRAM now peaks near its file size and resides in VRAM at roughly
    the file's size.
*   **Dense tensors load at their native dtype** (F32/F16/BF16 zero-copy views)
    instead of a uniform fp32 expansion — the 456 MB BF16 embedding in the
    common 1.5B GGUF no longer doubles during load.
*   **Size accounting is truthful**: ComfyUI's VRAM arbitration and lowvram
    partial loading now see actual residency bytes (raw blocks count as one
    byte/weight).

### ✨ New Features
*   **ConvRot INT8 checkpoints are supported.** Safetensors models carrying
    `*.comfy_quant` metadata (`int8_tensorwise`, `convrot=true`) execute through
    comfy-kitchen's INT8 ConvRot kernels (cuda/triton when available, eager
    otherwise); int8 weights + fp32 scales stay resident.
*   **Supported GGML types:** Q8_0 / Q4_K / Q5_K / Q6_K become quant-resident
    linears; F32 / F16 / BF16 pass through natively. Other types fail fast with
    an actionable error naming the tensor and supported list.
*   **Quantized safetensors checkpoints execute end-to-end.** Beyond rotated
    ConvRot INT8 layers, plain rowwise INT8 layers (mixed checkpoints ship
    both) and rowwise FP8 e4m3/e5m2 layers are dequantized to their declared
    dtype at load time using their per-row/per-tensor scales, and unrotated
    `int8_blockwise` checkpoints (per-gs×gs-block scale grids) dequantize the
    same way. Unsupported `.comfy_quant` formats now hard-fail immediately
    instead of silently misloading integer weights as floats, a dense-load
    gate rejects any checkpoint carrying unplanned quantized storages with an
    actionable message, and ROTATED quantized non-Linear layers (e.g. an
    int8-rotated embedding) fail with explicit re-export guidance — they are
    mathematically unexecutable without the converter's private rotation.
*   Both llama.cpp-style and HF pass-through GGUF key naming are mapped;
    unknown keys hard-fail listing the offenders. `tools/probe_gguf.py`
    inventories any file (types, scheme, residency coverage).

### 🛡️ Safety
*   Dtype casting never touches quant-resident storage (raw uint8/int8 weights,
    fp32 scales), so requesting a different model dtype can no longer corrupt
    quantized weights. Exclusivity is validated up-front (GGUF ⊕ ConvRot ⊕
    bnb-4bit; SageAttention over K-quants warns only).

</details>

<details>
<summary><strong>v2.4.0 - Unload Previous Model on Change (Memory Fix)</strong></summary>

### 🐛 Bug Fixes
*   **Switching models no longer keeps the old model's memory filled while the
    new one loads.** Selecting a different model in *Load VibeVoice Model*
    (or changing the TTS/ASR dropdown model) now **fully releases the previous
    model — RAM, VRAM, and ComfyUI's loaded-model registry — before any byte of
    the new model is read**, so peak memory during a swap stays at ~1× model
    size instead of ~2×. Eviction unregisters the old patcher from
    `model_management.current_loaded_models` (with finalizer detach), nulls all
    references, and runs an explicit GC + cache-empty pass.
*   **External cache keys are now file-identity-aware.** Two different weight
    files that share the same config name (e.g. a BF16 and a GGUF build of
    VibeVoice-1.5B) previously collided in the cache: the freshly loaded model
    was silently ignored and inference ran on the old weights. Cache identity
    now includes the weight file name, modification time, size, resolved
    attention mode, 4-bit flag, and dtype — switching `quantize_llm_4bit`,
    `dtype`, or the weight file always runs the newly selected build.

### ⚠️ Behavior Changes
*   **Single active model per family** (TTS / ASR), matching common TTS node
    semantics: loading a second VibeVoice model evicts the first. Workflows no
    longer keep two VibeVoice models resident simultaneously.
*   If loading a new model fails after a switch, the previous model has already
    been released; re-select it to load it again.

</details>

<details open>
<summary><strong>v2.3.2 - Fix Gibberish Output (RoPE inv_freq Regression)</strong></summary>

### 🐛 Bug Fixes
*   **Short (~2 s) gibberish audio after the v2.3.0 fast-loading change.** The v2.3.0 meta-init
    path zero-materialized the RoPE `inv_freq` / `original_inv_freq` buffers. These buffers are
    **computed from the config inside `Qwen2RotaryEmbedding.__init__`** and are non-persistent —
    they never appear in the checkpoint. Under `torch.device("meta")` instantiation they became
    meta tensors and the generic zero-materialization destroyed them. With `inv_freq == 0`, RoPE
    yields `cos(0)=1` / `sin(0)=0`, i.e. **no positional encoding**: the language model cannot
    order tokens, so it emits gibberish syllables and hits the speech-end token almost
    immediately.
*   **Diagnosis:** a deterministic eager-vs-meta diff on the real VibeVoice-1.5B checkpoint showed
    these two rotary buffers were the **only** tensors that differed (all 1204 checkpoint params
    matched exactly).
*   **Fix:** `_apply_state_dict` now calls `_recompute_rope_buffers()`, which re-runs each rotary
    module's own config-based computation on CPU (mirroring the eager `__init__`) and
    re-registers `inv_freq` / `original_inv_freq`. Covered by
    `tests/test_assign_loading.py::TestRopeRecomputeRegression`.

</details>

<details open>
<summary><strong>v2.3.1 - Fix Silent Output (Sentinel Buffer Regression)</strong></summary>

### 🐛 Bug Fixes
*   **Silent / empty audio output after the v2.3.0 fast-loading change.** The v2.3.0 hotfix
    (commit `6b99be9`) materialized *all* meta buffers with zeros after `assign` loading. This
    incorrectly zeroed the `speech_scaling_factor` / `speech_bias_factor` sentinel buffers, which
    the model registers as `float('nan')` and computes at inference time. The diffusion-inversion
    gate (`modeling_vibevoice.py:623`, `if not torch.isnan(sf) and not torch.isnan(bf)`) then
    evaluated `True` and applied `speech / 0 - 0`, producing a silent 2-second file. Both default
    and GGUF models were affected.
*   **Fix:** `_apply_state_dict` now restores known sentinel buffers to their intended initial
    values (`speech_scaling_factor` / `speech_bias_factor` → `nan`, `fix_std` → config value)
    instead of zeroing them. Ordinary meta buffers (e.g. `position_ids`) are still zeroed.
    Covered by `tests/test_assign_loading.py::TestSentinelBufferRegression`.

</details>

<details open>
<summary><strong>v2.3.0 - Fast Loading & ComfyUI-Conformant Model Management</strong></summary>

### ⚡ Performance
*   **Model loading no longer fills RAM before moving to VRAM.** Models are now
    instantiated on the `meta` device (zero RAM) and weights are assigned directly from
    the checkpoint (`load_state_dict(assign=True)`), eliminating the old
    "allocate empty model on CPU → copy weights → cast dtype → bulk move to GPU" chain.
    Loading a merged safetensors checkpoint now goes straight from disk to the target
    device with a single managed transfer.
*   **Redundant CPU passes removed.** The unconditional `model.to(dtype=...)` full-model
    cast is replaced by a conditional per-parameter cast that skips entirely when the
    model is already in the target dtype. The checkpoint `state_dict` is released
    immediately after assignment instead of lingering in RAM.

### 🔧 Changes
*   `modules/loader.py`: meta-context instantiation (`_instantiate_model`, `use_meta`
    escape hatch), assign-based weight loading (`_apply_state_dict` with `tie_weights()`
    re-tie + meta-straggler materialization), conditional dtype cast, `del state_dict`,
    and post-load size refinement (`_refine_size`).
*   `modules/patcher.py`: **non-destructive offload contract** — the default
    `unpatch_model` (ComfyUI-initiated) now keeps the model in CPU RAM so the next
    `patch_model` is a pure host-to-device transfer instead of a disk reload;
    `destroy=True` keeps the explicit full-free path and `warm=True` keeps the warm
    re-attach path. The bulk `handler.model.to()` pre-move was removed from
    `patch_model`; the single H2D transfer is owned by ComfyUI's `ModelPatcher.load()`.
    `is_loaded` is now device-aware (a CPU-offloaded model is not "loaded for inference").
*   `modules/external_loader.py` / `modules/asr_loader.py`: same meta-init, conditional
    cast, state-dict release, and size refinement applied to the external and ASR paths.
*   `modules/generation.py` / `modules/asr_generation.py`: user-requested force offload
    routes through `destroy=True` (cold) while retaining the warm path.

### 🧪 Tests
*   `tests/test_meta_init_feasibility.py`, `tests/test_assign_loading.py`,
    `tests/test_offload_contract.py`: new suites covering meta-init for all three model
    classes, assign-based loading + re-tying, and the routine/destroy/warm offload
    contract. Full suite: 770 passed / 5 pre-existing failures / 4 skipped.

</details>

<details>
<summary><strong>v2.2.3 - Guard Empty Generation + Low-Bit Quant Warnings</strong></summary>

### 🐛 Fixes
*   **`generate_audio()` no longer crashes with `AttributeError: 'NoneType' object has no
    attribute 'ndim'`** when the model produces no speech outputs. Corrupt or
    over-quantized weights (e.g. a naive int8 cast without dequantization scales) make the
    autoregressive loop emit no `speech_diffusion_id` token, so `speech_outputs[0]` is
    `None`. The non-streaming path now raises a clear, actionable `RuntimeError` pointing
    at the checkpoint quality (the streaming path already had this guard).

### 🔧 Changes
*   `modules/generation.py`: `generate_audio()` guards `None`/empty `speech_outputs` and
    raises a descriptive `RuntimeError` recommending a higher-quality checkpoint.
*   `modules/external_loader.py`: new defensive load-time check
    `warn_if_lowbit_quantization()` wired into both `load_external_vibevoice_model()` and
    `load_external_vibevoice_asr_model()`. It parses only the safetensors header (no tensor
    data) to flag raw-integer tensors with no scale/zero-point metadata (naive int cast),
    and inspects the GGUF tensor table to flag sub-4-bit I-quant checkpoints that commonly
    degrade TTS quality (garbled syllables / reference-audio echo).

### 🧪 Tests
*   `tests/test_generation.py`: `TestGenerateAudioNoneSpeechOutputs` (5 tests) — `None`,
    empty, and `[None]` speech outputs raise a clear `RuntimeError`; valid tensors still
    pass through.
*   `tests/test_external_loader.py`: `TestInspectSafetensorsQuantization` +
    `TestWarnIfLowbitQuantization` (10 tests) — naive int8 cast detected, proper quant with
    scales not flagged, GGUF sub-4-bit warned, high-bit/full-precision silent.

</details>

<details>
<summary><strong>v2.2.2 - Fix external_model Validation Bypass</strong></summary>

### 🐛 Fixes
*   **Connecting a `Load VibeVoice Model` node to the `external_model` input no longer
    fails prompt validation.** During prompt validation ComfyUI resolves *linked* inputs
    to `None` (no execution cache exists yet — see `execution.get_input_data` /
    `mark_missing`), so the previous `kwargs.get("external_model") is not None` check
    never triggered. The node then fell through and rejected the stale `model_name`
    widget value (e.g. a streaming model left in the TTS node dropdown). All three nodes
    (TTS, Realtime TTS, ASR) now detect a *connected* external model by its presence in
    `kwargs` (`"external_model" in kwargs`), which is always true for a linked input.

### 🔧 Changes
*   `nodes/tts_node.py`, `nodes/realtime_node.py`, `nodes/asr_node.py`:
    `validate_inputs()` bypass now keys on input presence instead of a non-`None` value.

### 🧪 Tests
*   Regression tests in `tests/test_node_schema.py`, `tests/test_realtime_node.py`, and
    `tests/test_asr_node.py` simulate the linked-input scenario (`external_model=None`)
    and confirm validation passes while the unconnected path still rejects wrong types.

</details>

<details>
<summary><strong>v2.2.1 - GGUF Support in Load VibeVoice Model</strong></summary>

### 🐛 Fixes
*   **`.gguf` files now appear in the `Load VibeVoice Model` dropdown.** ComfyUI's
    `get_filename_list("diffusion_models")` filters by `supported_pt_extensions`, which
    excludes `.gguf`. The node now merges that list with `.gguf` files scanned from the
    `diffusion_models` folders and the ComfyUI-GGUF `unet_gguf` folder (when registered).
*   **`.gguf` weights now load correctly.** ComfyUI's `load_torch_file` routes `.gguf` to
    `torch.load` (which fails). The external loader now parses GGUF containers directly via
    the `gguf` Python package (`GGUFReader` + `dequantize`), producing a dequantized CPU
    state dict for both the TTS/streaming and ASR branches.

### 🔧 Changes
*   `modules/external_loader.py`: new `_load_gguf_state_dict()` + `_load_weight_state_dict()`
    dispatcher; both `load_external_vibevoice_model()` and `load_external_vibevoice_asr_model()`
    route through it.
*   `nodes/external_loader_node.py`: new `list_external_model_files()` (dropdown) and
    `resolve_weight_path()` (diffusion_models → unet_gguf fallback).

### 🧪 Tests
*   New `tests/test_gguf_loading.py` (19 tests): real GGUF round-trip via the `gguf` package,
    dispatch routing, dropdown listing, and path-resolution fallback ordering.

</details>

<details>
<summary><strong>v2.2.0 - External Model Input (Load Your Own Weights)</strong></summary>

### ✨ Highlights
*   **New `VibeVoiceLoadExternalModel` node:** load a VibeVoice checkpoint from your own
    `.safetensors`/`.pt` file in `models/diffusion_models` (or `models/unet`) instead of the
    built-in HuggingFace download path. ComfyUI's stock `Load Diffusion Model` cannot parse
    VibeVoice weights, so this node provides a dedicated `VIBEVOICE_MODEL` output type.
*   **Optional `external_model` input on all three nodes:** TTS, Realtime TTS, and ASR now accept
    an externally loaded model bundle. When wired, the node skips its own loader entirely and uses
    the provided model/processor.
*   **Sidecar config binding:** the loader resolves architecture config via a sidecar
    `<weight>.config.json` (or a `config.json` next to the file), falling back to the packaged
    default for the selected `config_name`. Optional `<weight>.preprocessor.json` and a local
    `tokenizer.json` are honored as well.
*   **Type guards:** each node rejects mismatched model kinds with a clear error (e.g. a Realtime
    model into the TTS node, an ASR model into TTS, a non-streaming model into Realtime).

### 🔧 Changes
*   New `modules/custom_types.py`: `VibeVoiceModel = io.Custom("VIBEVOICE_MODEL")`.
*   New `modules/external_loader.py`: sidecar resolution + `load_external_vibevoice_model()`
    (TTS/streaming) and `load_external_vibevoice_asr_model()` (ASR) — CPU-first load, in-memory
    state-dict injection, dtype cast, optional 4-bit quantization (TTS only) and SageAttention.
*   New `nodes/external_loader_node.py`: the loader node (registered in `vibevoice_nodes.py`).
*   `modules/generation.py`: `ExternalVibeVoiceModelHandler` + `load_vibevoice_from_external()`.
*   `modules/asr_generation.py`: `ExternalVibeVoiceASRModelHandler` + `load_asr_from_external()`.
*   `nodes/tts_node.py`, `nodes/realtime_node.py`, `nodes/asr_node.py`: optional `external_model`
    input, early validation pass-through, and execute-branch routing with kind guards.

### 🧪 Tests
*   New `tests/test_custom_types.py` (5), `tests/test_external_loader.py` (39),
    `tests/test_external_loader_node.py` (13).
*   Extended `tests/test_node_schema.py`, `tests/test_generation.py` (+15),
    `tests/test_asr_generation.py` (+16), `tests/test_realtime_node.py` (+7),
    `tests/test_asr_node.py`, `tests/test_patcher_behavioral.py`, `tests/test_integration.py`,
    `tests/test_docs_consistency.py` (+3), `tests/test_workflow.py` (+5), `tests/test_imports.py`.

</details>

<details>
<summary><strong>v2.1.1 - Standard ComfyUI Progress Bar During Inference</strong></summary>

### ✨ Highlights
*   **Live progress bar:** All three nodes (TTS, Realtime TTS, ASR) now drive the standard
    ComfyUI frontend progress bar during inference. Previously the bar sat at 0% for the whole
    generation and jumped to 100% only at the end.
*   **Responsive cancel:** the progress hook checks ComfyUI's interrupt flag on every loop step,
    so pressing cancel stops generation promptly instead of waiting for the current blocking call.
*   **Guaranteed 100%:** a final progress event is always emitted, even when generation stops
    early (EOS) or raises.

### 🔧 Changes
*   Vendored `generate()` (non-streaming + streaming) gained an optional, framework-agnostic
    `progress_callback(current, total)` hook fired once per AR loop step (vendored code stays
    `comfy`-free; `None` = disabled, fully backward compatible).
*   `modules/generation.py`: `generate_audio()` / `generate_streaming_audio()` wrap the hook with
    `comfy.utils.ProgressBar` (throttled WebSocket updates; the bar total self-corrects via
    `update_absolute(value, total=...)` once the loop reports its real budget).
*   `modules/asr_generation.py`: ASR reports per-token progress through an HF `BaseStreamer`
    (greedy/sampling only; beam search falls back to a single 0→100% bar).

### 🧪 Tests
*   New `tests/test_generate_progress_callback.py` (5 tests) and
    `tests/test_streaming_progress_callback.py` (4 tests): drive the real vendored loops with
    scripted mocks and lock the callback contract (monotonic, bounded, call counts, interrupt
    propagation, output determinism).
*   New `tests/test_streaming_progress.py` (4 tests); extended `tests/test_generation.py` (+5),
    `tests/test_asr_generation.py` (+6), `tests/test_integration.py` (+3 node-level tests).

</details>

<details>
<summary><strong>v2.1.0 - torchaudio-Primary Audio Backend (librosa now optional)</strong></summary>

### ✨ Highlights
*   **torchaudio is now the primary audio library.** Resampling uses `torchaudio.functional.resample`
    (Kaiser-windowed sinc — the same family ComfyUI core uses), matching the ComfyUI-core idiom.
*   **`librosa` is no longer a hard dependency.** It was declared but never actually imported
    (a phantom dependency). It is now an *optional* last-resort fallback; the node never crashes
    when librosa (or scipy) is missing or broken (e.g. the empty librosa namespace stub shipped in
    some embedded Pythons).
*   **ComfyUI built-ins adopted:** file decoding now uses PyAV (`av`) — ComfyUI's own audio decoder —
    with `soundfile`/`torchaudio`/`librosa` as guarded fallbacks. This also fixes `.m4a`/`.ogg`
    reference-audio loading, which `soundfile` (libsndfile) cannot decode.
*   **Dependency surface minimized:** `librosa` and `scipy` moved to the optional
    `[audio-extra]` group; `torchaudio` + `soundfile` remain the working default.

### 🔧 Changes
*   New `modules/audio_backend.py`: single dependency-resilient backend for resample / load / save
    with import-time capability detection (`_HAS_*` flags) and graceful fallback ordering.
*   `modules/audio_utils.py`: `resample_audio()` delegates to the backend; `preprocess_comfy_audio()`
    now resamples in tensor space (no numpy round-trip).
*   Vendored processors (`_load_audio_from_path`, `save_audio`, ASR file loading) route through the
    backend; no hard `ffmpeg`/`soundfile` requirement for the common wav/flac path.

### 🧪 Tests
*   New `tests/test_audio_backend.py` (41 tests): import resilience with any optional lib blocked,
    resample correctness/priority/fallbacks, numpy↔tensor parity, file I/O roundtrips, `f32_pcm`.
*   New `tests/test_processor_io_backend.py` (17 tests): real vendored processor I/O via the backend.
*   Extended `tests/test_audio_utils.py`, `tests/test_pyproject.py`, `tests/test_imports.py`.

</details>

<details>
<summary><strong>v2.0.2 - Negative-Branch RoPE Position Fix (SDPA Shape Crash)</strong></summary>

### 🐛 Fixes
*   **BUG-011 — Negative-branch RoPE `position_ids` desync:** Fixed a `RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [12, 3] but got: [12, 2]` crash during TTS generation. In the non-streaming `generate()` CFG loop, the negative (unconditional) forward fed a **single-token** `inputs_embeds` `(B,1,H)` together with a **full-length** `position_ids` `(B, step+1)`. In transformers 5.x the explicit `position_ids` drive RoPE directly, so q/k silently broadcast against full-length cos/sin and expanded to seq-len `step+1` while v (never rotated) stayed length 1 — the KV cache accumulated `step+1` keys but only 1 value per step, and SDPA's `attn @ value` crashed at AR step 1. The negative forward now passes **current-only** `position_ids` (`neg_position_ids[:, -1:]`), matching its single input token; the attention mask stays full-length.

### 🧪 Tests
*   New `tests/test_generate_neg_position_ids.py` (4 tests): drives the real `generate()` through several AR steps with a recording inner LM and locks the invariant that `position_ids` length always equals `inputs_embeds` sequence length (red/green verified against the buggy code).

</details>

<details>
<summary><strong>v2.0.1 - CPU-First Model Loading (VRAM Round-Trip Fix)</strong></summary>

### 🐛 Fixes
*   **Load Device Flow — DF-001..DF-006:** Fixed a GPU→RAM→GPU round-trip during model loading. Previously the checkpoint state dict was loaded directly onto CUDA (a full-model VRAM spike outside ComfyUI's arbitration), copied back to CPU-resident parameters, then moved to CUDA again. Now the loader builds the model **entirely on CPU** (state dict, `load_state_dict`, dtype cast, and 4-bit quantization all on CPU), and `VibeVoicePatcher.patch_model` owns the **single** host-to-device transfer after ComfyUI's `load_models_gpu` VRAM arbitration. Peak VRAM during load drops from ≈2× model size to ≈1× model size.
*   **Dtype Threading — DF-004 / AUD-008:** The user-selected dtype is now threaded from the node through the handler into the loader and applied on CPU before the transfer; the patcher's dtype cast is now a mismatch-only guard (no redundant GPU cast).
*   **Handler No-Move — DF-003:** `VibeVoiceModelHandler.load_model` no longer moves the model; device placement is owned solely by the patcher.

### 🧪 Tests
*   New `tests/test_load_device_flow.py` (36 tests): device-ledger doubles asserting CPU-only loading, single H2D transfer, no round-trip, dtype threading, cast guard, VRAM arbitration, and a `[GPU-OPTIONAL]` peak-VRAM measurement.

</details>

<details>
<summary><strong>v2.0.0 - V3 Extension &amp; VRAM Parity</strong></summary>

### ✨ Highlights
*   **V3 Extension API:** Migrated the custom-node entrypoint to the ComfyUI V3 `ComfyExtension` / `io.ComfyNode` schema — type-filtered model dropdowns and declarative inputs/outputs.
*   **VRAM Parity (ASR) — CRIT-001:** The ASR path now runs under the same `VibeVoicePatcher` / `model_management.load_model_gpu` orchestration as TTS, clearing the dedicated ASR cache on unload.
*   **Warm Re-attach — NTH-004:** `force_offload` can retain model tensors on the intermediate device for a fast re-attach on the next run instead of reloading from disk.
*   **Streaming TTS Node — NTH-001:** `VibeVoice-Realtime-0.5B` is now reachable through a dedicated `VibeVoice Realtime TTS` node that shares the patcher / attention machinery.
*   **Maintainability — IMP-004:** TTS/ASR download, discovery, and sharded-load logic is now shared via `BaseVibeVoiceLoader`.
*   **Device &amp; Attention Honesty — IMP-003 / IMP-001:** MPS/XPU/NPU device selection is honored when available; `flash_attention_2` is only offered when `flash-attn` + CUDA are present.
*   **Docs Consistency — CRIT-003:** README zero-shot wording now matches `generate_audio` (at least one reference voice is required).

</details>

<details>
<summary><strong>v1.5.0 - Stability and Prompting</strong></summary>

### ✨ New Features & Improvements
*   **Total Generation Stability:** Fixed the bug where a speaker's voice could unintentionally change or blend with another reference voice mid-sentence.
*   **Improved Voice Cloning Fidelity** 
*   **Consistent Speaker Tagging:** The node now intelligently handles multiple script formats (`[1]`, `[1]:`, and `Speaker 1:`) to produce identical, high-quality results, removing all previous inconsistencies.
*   **Hybrid Voice Cloning:** Mix and match cloned speakers in the same script — at least one reference audio is required; speakers without their own reference are cloned from the provided reference(s).
</details>

<details>
<summary><strong>v1.3.0 - SageAttention & Quantization Overhaul</strong></summary>

*   **SageAttention Support:** Full integration with the `sageattention` library for a high-performance, mixed-precision attention option.
*   **Robust 4-Bit LLM Quantization:** The "Quantize LLM (4-bit)" option is now highly stable and delivers significant VRAM savings.
*   **Smart Configuration & Fallbacks:** The node now automatically handles incompatible settings (e.g., 4-bit with `flash_attention_2`) by gracefully falling back to a stable alternative (`sdpa`) and notifying the user.
</details>

<details>
<summary><strong>v1.2.0 - Compatibility Update</strong></summary>

*   **Transformers Library:** Includes automatic detection and compatibility for both older and newer versions of the Transformers library (pre- and post-4.56).
*   **Bug Fixes:** Resolved issues with `Force Offload` and multi-speaker generation on newer Transformers versions.
</details>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Tips from the Original Authors

*   **Punctuation:** For Chinese text, using English punctuation (commas and periods) can improve stability.
*   **Model Choice:** The 7B model variant (`VibeVoice-Large`) is generally more stable.
*   **Spontaneous Sounds/Music:** The model may spontaneously generate background music, especially if the reference audio contains it or if the text includes introductory phrases like "Welcome to...". This is an emergent capability and cannot be directly controlled.
*   **Singing:** The model was not trained on singing data, but it may attempt to sing as an emergent behavior. Results may vary.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

This project is distributed under the MIT License. See `LICENSE.txt` for more information. The VibeVoice model and its components are subject to the licenses provided by Microsoft. Please use responsibly.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

*   **Microsoft** for creating and open-sourcing the [VibeVoice](https://github.com/microsoft/VibeVoice) project.
*   **The ComfyUI team** for their incredible and extensible platform.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=wildminder/ComfyUI-VibeVoice&type=Timeline)](https://www.star-history.com/#wildminder/ComfyUI-VibeVoice&Timeline)


<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/wildminder/ComfyUI-VibeVoice.svg?style=for-the-badge
[contributors-url]: https://github.com/wildminder/ComfyUI-VibeVoice/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/wildminder/ComfyUI-VibeVoice.svg?style=for-the-badge
[forks-url]: https://github.com/wildminder/ComfyUI-VibeVoice/network/members
[stars-shield]: https://img.shields.io/github/stars/wildminder/ComfyUI-VibeVoice.svg?style=for-the-badge
[stars-url]: https://github.com/wildminder/ComfyUI-VibeVoice/stargazers
[issues-shield]: https://img.shields.io/github/issues/wildminder/ComfyUI-VibeVoice.svg?style=for-the-badge
[issues-url]: https://github.com/wildminder/ComfyUI-VibeVoice/issues
