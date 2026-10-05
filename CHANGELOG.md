<a id="changelog-top"></a>

# Changelog

Release history for ComfyUI-VibeVoice. Newest first.

<p align="right">(<a href="README.md">back to README</a>)</p>

---

<details>
<summary><strong>Unreleased - fix the ~2x host-RAM spike on an external load</strong></summary>

### Changed
*   **The dense external single-file route now selects ComfyUI's DynamicVRAM
    patcher.** A plain-BF16 external checkpoint was landing *entirely* in host RAM,
    because the pack subclassed the legacy `comfy.model_patcher.ModelPatcher` rather
    than `CoreModelPatcher` (which core rebinds to `ModelPatcherDynamic` once aimdo
    initialises). The observed cost was **1.26×** file size on the 5.41 GB
    `VibeVoice-1.5B` and **1.74×** on the 16.66 GB ASR checkpoint. The route now
    constructs `ModelPatcherDynamic` and loads it via `load_models_gpu()`.

    **The mechanism is wired; the outcome is NOT achieved and NOT measured.** The
    weights are **not** demand-paged. `ModelPatcherDynamic` gates paging on a
    per-module `comfy_cast_weights` attribute (`comfy/model_patcher.py:1967`,
    allocating at `:1993`), and this route suppresses
    `convert_tree_for_streaming` (`modules/external_loader.py:1904`, `:2239`) —
    which is the only thing in this pack that sets that attribute on these
    transformers modules. Every module therefore takes the eager branch at
    `:2000-2009`: host tensors are retained by reference in `ModelPatcher.backup`
    (so the 16.66 GB stays in host RAM) and a full device copy is made on top
    (so VRAM is no longer paged either). This trips the port plan's own stop
    condition and needs re-planning. Until then, treat the dense route's
    host-RAM behaviour as unchanged from the legacy patcher.

    **Scope is dense only.** The entire behavioural decision is one function,
    `select_patcher_class` in `modules/patcher.py`. Four families deliberately did
    **not** move and keep the legacy patcher byte-identical:
    *   **GGUF** (`gguf_block`) — installs quant residents that already stream natively.
    *   **ConvRot INT8** (`convrot_int8`) — same.
    *   **fp8-resident** (`fp8_resident`) — same.
    *   The **standard-directory dropdown loaders** (no external bundle, so
        `weight_family` is `None`).

    A dynamic patcher would gain nothing for the first three — they are quant
    residents, not full float state dicts — while changing their offload order.

*   **Silent fallback to legacy.** If ComfyUI's DynamicVRAM is unavailable (aimdo did
    not initialise, or the load device is not CUDA), the selector returns the legacy
    patcher with no error, no warning and no user-visible change. Machines without
    DynamicVRAM keep exactly the previous, known-good behaviour.

### Fixed
*   Re-running the `Load VibeVoice Model` node with the **same** model no longer builds a
    second copy of the weights. The node computed an unload-before-load key, skipped
    eviction for an identical re-run (correctly — the live patcher is reused), and then
    rebuilt the model anyway. The outgoing weights stayed reachable from ComfyUI's node
    output cache *and* from the live patcher's handler, so the process held two full
    models at once. An identical re-execution now reuses the resident model.
*   The returned bundle is now the same object the reused patcher wraps. Previously a
    rebuilt bundle was discarded in favour of the already-loaded weights, so the rebuild
    was wasted work as well as wasted RAM.

### Diagnostics
*   README: the "Host RAM during an external load" note no longer claims the load path
    was exonerated, and no longer claims the legacy-patcher opt-out is permanent. It
    now states the dense-only scope, the four families left on the legacy patcher, and
    the silent fallback.
*   `.dev/scratch/probe_external_asr_ram.py` gained a `PATCHER SELECTION` block
    reporting, per load phase, the patcher class actually constructed,
    `patcher.is_dynamic()`, and `patcher.loaded_size()` split into
    `model_loaded_weight_memory` and the demand-paged remainder. Requires
    `--drive-consumer` (only the consumer path builds a patcher). All pre-existing
    counters and the `TIMELINE` / `PEAK` / `PHASE COUNTS` blocks are unchanged, so
    earlier and later pastes stay comparable.

### Not yet confirmed by live measurement
*   **The RAM fix is NOT working, and NOT confirmed.** No real checkpoint was loaded
    to produce this change; the evidence behind it is a green unit-test suite, which
    is a non-regression signal only. No host-RAM number has been observed since the
    port. Worse, reading core's source says the port cannot have worked as written —
    see the "mechanism is wired" note above. Running the probe is still worth doing,
    but it will measure the eager path, not paging: expect host RAM unchanged.
    Run it on both the 5.41 GB and the 16.66 GB checkpoints and paste back the
    `PEAK` / `TIMELINE` / `PHASE COUNTS` / `PATCHER SELECTION` blocks either way.

</details>

---

<details>
<summary><strong>v2.12.1 - load the single-file realtime checkpoint externally</strong></summary>

### Loading
*   A single-file `VibeVoice-Realtime-0.5B` safetensors export now loads through
    `Load VibeVoice Model` with no sidecar. The architecture config ships with the
    node (a verbatim copy of the published `config.json`), so the realtime option
    in `config_name` resolves instead of raising "no packaged default".
*   `Auto-detect` recognizes the realtime family from its embedding fingerprint
    (hidden 896, vocab 151936), which it shares with no other family. Detection
    stays exact-match-only: an unmatched shape is still not guessed.
*   `config_name` accepts the lowercase spellings `vibevoice-realtime` and
    `vibevoice_realtime` in saved workflows.

### Fixed
*   The "intentionally absent" rule for the acoustic-tokenizer encoder matched a
    key namespace no released checkpoint uses, so every realtime load warned
    about 276 keys it should have known were absent. Both the rooted and the
    bare spelling are matched now, and the rule still refuses to silence a
    partially present encoder.
*   Both "no architecture config" errors name the actual remedy: an option
    without a packaged default is named as such, and the auto-detect error no
    longer recommends an option that would fail the same way.

### Diagnostics
*   README: the sidecar table's fallback column is now the real packaged-default
    set instead of "`1.5B` / `7B` only", auto-detect is documented as
    recognizing all four families, and a note records that realtime exports are
    decoder-only (276 encoder keys intentionally absent). A test now keeps that
    table in step with the loader's.

</details>

---

<details>
<summary><strong>v2.12.0 - external VibeVoice-ASR (native) loading</strong></summary>

### Loading
*   The `Load VibeVoice Model` node, `config_name "VibeVoice-ASR"`, loads a single-file
    `VibeVoice-ASR-HF` checkpoint end to end.
*   Native checkpoints (`model_type: "vibevoice_asr"`) are built with the transformers
    classes; the vendored ASR tree cannot consume their `language_model.model.*` keys.
*   The architecture config, tokenizer config, processor config and chat template ship with
    the node, so nothing has to be written into the model's own folder.
*   Auto-detect recognizes an ASR checkpoint by the module names only it carries, instead of
    guessing from the embedding shape it shares with VibeVoice-7B.

### Diagnostics
*   The post-load dtype cast logs the SOURCE dtype, the target and how many
    parameters were converted, e.g. `Model cast torch.float32 -> torch.bfloat16
    (812 mismatched params)`. A cast allocates a second copy of every
    mismatched parameter before the old one is freed, so the line is the
    fastest way to tell a cast-heavy load from a plain one. A checkpoint that
    already matches stays silent and costs nothing.

### Documentation
*   The external-model section now states the measured host-RAM cost of a load
    (~1x the checkpoint, ~1.2x working set) instead of leaving the impression
    that ComfyUI core's near-zero figure applies here. It explains that the
    legacy `ModelPatcher` opt-out is why, points at the peak-RAM probe, and asks
    for the load timeline back when reporting a large-RAM load.

### Tests
*   The external ASR loader is exercised end to end on a tiny native checkpoint: config,
    processor, weight binding, and the rotary buffers a meta-device load must materialize.
*   That load is also asserted to have been converted for streaming. The conversion is
    wrapped in a defensive `try`/`except` that only warns, so a dropped call would
    otherwise leave the full-size checkpoint unstreamable behind a green suite.
*   The packaged tokenizer is diffed against the published `VibeVoice-ASR-HF` tokenizer
    (vocab size, symmetric difference, token→id mapping, added tokens) on hosts that have
    it; the digest pin still guards hosts that do not.
*   The full-size checkpoint is never loaded by the test suite.

### Docs
*   **Documentation correction, no behaviour change and no user-visible fix.** Three
    docstrings claimed that `safe_open` / `comfy.utils.load_torch_file` hand back
    zero-copy mmap views into the checkpoint file. Measured on this stack, they do not:
    `get_tensor` returns an owned, deserialised copy, and holding a file's tensors costs
    the file size in private commit. Corrected in `iter_safetensors_tensors`,
    `_load_state_dict_into_model_from_memory` and `_stream_apply_dense`.
*   The safety rationale is kept and narrowed, not deleted. The per-tensor `clone()`
    before assign is still the fix for the ghost-RAM / `[WARNING] Pin error.` flood, and
    it is still required for `.bin`/`.pt` when `MMAP_TORCH_FILES` is set and for the
    aimdo `load_safetensors` path. Under the current configuration
    (`MMAP_TORCH_FILES=False`, `aimdo_enabled=False`) it is a redundant copy that costs
    one tensor of transient memory; the code is unchanged.
*   The RAM plan's stop condition fired on its own evidence and the streaming re-route is
    not shipped; the reason is recorded in
    `.dev/docs/plans/2026-09-28-external-dense-load-ram-spike.md` (S1 RESULT).

</details>

<details>
<summary><strong>v2.11.0 - loading, numerics and release hygiene</strong></summary>

### Loading
*   GGUF weights install per tensor instead of buffering the whole checkpoint.
*   Dequant-at-load targets the destination parameter dtype, removing fp32 scratch.
*   A released model bundle is rebuilt from its recorded source instead of failing.
*   Host memory is released after install.

### Numerics
*   Q8_0 dequant rounds once, in fp32; the activation-dtype path is reverted.
*   Redundant weight copies removed from the dequant kernel (bitwise-identical).
*   Precision gates are bitwise, not tolerance-based.

### Docs
*   README condensed; changelog moved to `CHANGELOG.md`.
*   Third-party project references removed from source comments.

</details>

### • Tests
*   The `transformers` range guard keeps its teeth against a self-contained measured-version
    set: removing the `<5.4` cap, or claiming a version nobody measured, still fails.
*   The realtime GPU-audit fixtures skip instead of erroring when the local diagnostic
    script is not present.


</details>

<details>
<summary><strong>v2.9.0 - One canonical TTS node with correct realtime-model support</strong></summary>

### ◆ New Features
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

### • Changes
*   **Deprecated forwarding shim.** `nodes/realtime_node.py` keeps the `VibeVoiceRealtime`
    node ID and its exact legacy input order, but holds no loading or generation logic:
    validation and execution delegate to `VibeVoiceTTSNode` and one process-level
    deprecation warning is logged. Removed in the next release.
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

### • Tests
*   `tests/test_unified_tts_node.py` covers schema/append-only ordering, preset discovery
    fallback, the full validation truth table, mutually exclusive routing, external
    realtime/ASR handling, the loaded-pair safety net, independent steps/length, ignored-input
    warnings, and the shared output/offload/cancel paths.
*   `tests/test_realtime_node.py` covers the shim: legacy widget order, the one-time
    deprecation warning, `stream` stripping, delegation-only source assertions, unique
    node IDs, and a `tests/fixtures/legacy_realtime_workflow.json` compatibility fixture.
*   Both serialization `xfail`s are **resolved**: real `BaseModelOutputWithPast` /
    `DynamicCache` round trips and the real `en-Carter_man.pt` prompt now load on
    transformers 5.3 with safe loading retained.
*   `tests/test_integration.py` realtime progress/cancellation coverage retargeted to the
    canonical node, and new `tests/test_e2e_smoke_contract.py` pins the smoke script's CLI
    and realtime-branch wiring.


</details>

<details>
<summary><strong>v2.8.2 - Native VibeVoice-ASR-HF: correct, working transcription</strong></summary>

### ◆ New Features
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

### • Bug Fixes
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

### • Behind the scenes
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

### • Bug Fixes
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

### » Performance / Memory
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

### • Safety
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

### • Bug Fixes
*   Version-safe `torch_dtype` config access — silences the transformers
    deprecation warning on newer versions.


</details>

<details>
<summary><strong>v2.7.0 - Auto-detect config, drop VibeVoice-Large</strong></summary>

### ◆ New Features
*   `config_name` gains **Auto-detect** (default): the architecture family is
    resolved from the weight file's embedding fingerprint before any heavy
    load; explicit selections that contradict the weights are auto-corrected
    with a warning.
*   The legacy `VibeVoice-Large` option is removed.


</details>

<details>
<summary><strong>v2.6.0 - Native Lowvram Streaming (oversized models work)</strong></summary>

### » Performance / Memory
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

### • Safety
*   Quant residents (GGUF raw blocks / ConvRot INT8) participate in streaming
    with their storage dtype pinned (`weight_comfy_model_dtype`) and a
    dtype-preserving pull path — raw bytes are moved, never recast.
*   Vendored norm classes (RMSNorm/ConvRMSNorm/LayerNorm variants, Qwen2
    RMSNorm) gained streaming forwards with parity tests against the
    originals.


</details>

<details>
<summary><strong>v2.5.0 - Quant-Resident Runtime: GGUF Raw-Block Residency + ConvRot INT8</strong></summary>

### » Performance / Memory
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

### ◆ New Features
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

### • Safety
*   Dtype casting never touches quant-resident storage (raw uint8/int8 weights,
    fp32 scales), so requesting a different model dtype can no longer corrupt
    quantized weights. Exclusivity is validated up-front (GGUF ⊕ ConvRot ⊕
    bnb-4bit; SageAttention over K-quants warns only).


</details>

<details>
<summary><strong>v2.4.0 - Unload Previous Model on Change (Memory Fix)</strong></summary>

### • Bug Fixes
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

### ! Behavior Changes
*   **Single active model per family** (TTS / ASR), matching common TTS node
    semantics: loading a second VibeVoice model evicts the first. Workflows no
    longer keep two VibeVoice models resident simultaneously.
*   If loading a new model fails after a switch, the previous model has already
    been released; re-select it to load it again.


</details>

<details open>
<summary><strong>v2.3.2 - Fix Gibberish Output (RoPE inv_freq Regression)</strong></summary>

### • Bug Fixes
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

### • Bug Fixes
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

### » Performance
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

### • Changes
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

### • Tests
*   `tests/test_meta_init_feasibility.py`, `tests/test_assign_loading.py`,
    `tests/test_offload_contract.py`: new suites covering meta-init for all three model
    classes, assign-based loading + re-tying, and the routine/destroy/warm offload
    contract. Full suite: 770 passed / 5 pre-existing failures / 4 skipped.


</details>

<details>
<summary><strong>v2.2.3 - Guard Empty Generation + Low-Bit Quant Warnings</strong></summary>

### • Fixes
*   **`generate_audio()` no longer crashes with `AttributeError: 'NoneType' object has no
    attribute 'ndim'`** when the model produces no speech outputs. Corrupt or
    over-quantized weights (e.g. a naive int8 cast without dequantization scales) make the
    autoregressive loop emit no `speech_diffusion_id` token, so `speech_outputs[0]` is
    `None`. The non-streaming path now raises a clear, actionable `RuntimeError` pointing
    at the checkpoint quality (the streaming path already had this guard).

### • Changes
*   `modules/generation.py`: `generate_audio()` guards `None`/empty `speech_outputs` and
    raises a descriptive `RuntimeError` recommending a higher-quality checkpoint.
*   `modules/external_loader.py`: new defensive load-time check
    `warn_if_lowbit_quantization()` wired into both `load_external_vibevoice_model()` and
    `load_external_vibevoice_asr_model()`. It parses only the safetensors header (no tensor
    data) to flag raw-integer tensors with no scale/zero-point metadata (naive int cast),
    and inspects the GGUF tensor table to flag sub-4-bit I-quant checkpoints that commonly
    degrade TTS quality (garbled syllables / reference-audio echo).

### • Tests
*   `tests/test_generation.py`: `TestGenerateAudioNoneSpeechOutputs` (5 tests) — `None`,
    empty, and `[None]` speech outputs raise a clear `RuntimeError`; valid tensors still
    pass through.
*   `tests/test_external_loader.py`: `TestInspectSafetensorsQuantization` +
    `TestWarnIfLowbitQuantization` (10 tests) — naive int8 cast detected, proper quant with
    scales not flagged, GGUF sub-4-bit warned, high-bit/full-precision silent.


</details>

<details>
<summary><strong>v2.2.2 - Fix external_model Validation Bypass</strong></summary>

### • Fixes
*   **Connecting a `Load VibeVoice Model` node to the `external_model` input no longer
    fails prompt validation.** During prompt validation ComfyUI resolves *linked* inputs
    to `None` (no execution cache exists yet — see `execution.get_input_data` /
    `mark_missing`), so the previous `kwargs.get("external_model") is not None` check
    never triggered. The node then fell through and rejected the stale `model_name`
    widget value (e.g. a streaming model left in the TTS node dropdown). All three nodes
    (TTS, Realtime TTS, ASR) now detect a *connected* external model by its presence in
    `kwargs` (`"external_model" in kwargs`), which is always true for a linked input.

### • Changes
*   `nodes/tts_node.py`, `nodes/realtime_node.py`, `nodes/asr_node.py`:
    `validate_inputs()` bypass now keys on input presence instead of a non-`None` value.

### • Tests
*   Regression tests in `tests/test_node_schema.py`, `tests/test_realtime_node.py`, and
    `tests/test_asr_node.py` simulate the linked-input scenario (`external_model=None`)
    and confirm validation passes while the unconnected path still rejects wrong types.


</details>

<details>
<summary><strong>v2.2.1 - GGUF Support in Load VibeVoice Model</strong></summary>

### • Fixes
*   **`.gguf` files now appear in the `Load VibeVoice Model` dropdown.** ComfyUI's
    `get_filename_list("diffusion_models")` filters by `supported_pt_extensions`, which
    excludes `.gguf`. The node now merges that list with `.gguf` files scanned from the
    `diffusion_models` folders and the ComfyUI-GGUF `unet_gguf` folder (when registered).
*   **`.gguf` weights now load correctly.** ComfyUI's `load_torch_file` routes `.gguf` to
    `torch.load` (which fails). The external loader now parses GGUF containers directly via
    the `gguf` Python package (`GGUFReader` + `dequantize`), producing a dequantized CPU
    state dict for both the TTS/streaming and ASR branches.

### • Changes
*   `modules/external_loader.py`: new `_load_gguf_state_dict()` + `_load_weight_state_dict()`
    dispatcher; both `load_external_vibevoice_model()` and `load_external_vibevoice_asr_model()`
    route through it.
*   `nodes/external_loader_node.py`: new `list_external_model_files()` (dropdown) and
    `resolve_weight_path()` (diffusion_models → unet_gguf fallback).

### • Tests
*   New `tests/test_gguf_loading.py` (19 tests): real GGUF round-trip via the `gguf` package,
    dispatch routing, dropdown listing, and path-resolution fallback ordering.


</details>

<details>
<summary><strong>v2.2.0 - External Model Input (Load Your Own Weights)</strong></summary>

### ◆ Highlights
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

### • Changes
*   New `modules/custom_types.py`: `VibeVoiceModel = io.Custom("VIBEVOICE_MODEL")`.
*   New `modules/external_loader.py`: sidecar resolution + `load_external_vibevoice_model()`
    (TTS/streaming) and `load_external_vibevoice_asr_model()` (ASR) — CPU-first load, in-memory
    state-dict injection, dtype cast, optional 4-bit quantization (TTS only) and SageAttention.
*   New `nodes/external_loader_node.py`: the loader node (registered in `vibevoice_nodes.py`).
*   `modules/generation.py`: `ExternalVibeVoiceModelHandler` + `load_vibevoice_from_external()`.
*   `modules/asr_generation.py`: `ExternalVibeVoiceASRModelHandler` + `load_asr_from_external()`.
*   `nodes/tts_node.py`, `nodes/realtime_node.py`, `nodes/asr_node.py`: optional `external_model`
    input, early validation pass-through, and execute-branch routing with kind guards.

### • Tests
*   New `tests/test_custom_types.py` (5), `tests/test_external_loader.py` (39),
    `tests/test_external_loader_node.py` (13).
*   Extended `tests/test_node_schema.py`, `tests/test_generation.py` (+15),
    `tests/test_asr_generation.py` (+16), `tests/test_realtime_node.py` (+7),
    `tests/test_asr_node.py`, `tests/test_patcher_behavioral.py`, `tests/test_integration.py`,
    `tests/test_docs_consistency.py` (+3), `tests/test_workflow.py` (+5), `tests/test_imports.py`.


</details>

<details>
<summary><strong>v2.1.1 - Standard ComfyUI Progress Bar During Inference</strong></summary>

### ◆ Highlights
*   **Live progress bar:** All three nodes (TTS, Realtime TTS, ASR) now drive the standard
    ComfyUI frontend progress bar during inference. Previously the bar sat at 0% for the whole
    generation and jumped to 100% only at the end.
*   **Responsive cancel:** the progress hook checks ComfyUI's interrupt flag on every loop step,
    so pressing cancel stops generation promptly instead of waiting for the current blocking call.
*   **Guaranteed 100%:** a final progress event is always emitted, even when generation stops
    early (EOS) or raises.

### • Changes
*   Vendored `generate()` (non-streaming + streaming) gained an optional, framework-agnostic
    `progress_callback(current, total)` hook fired once per AR loop step (vendored code stays
    `comfy`-free; `None` = disabled, fully backward compatible).
*   `modules/generation.py`: `generate_audio()` / `generate_streaming_audio()` wrap the hook with
    `comfy.utils.ProgressBar` (throttled WebSocket updates; the bar total self-corrects via
    `update_absolute(value, total=...)` once the loop reports its real budget).
*   `modules/asr_generation.py`: ASR reports per-token progress through an HF `BaseStreamer`
    (greedy/sampling only; beam search falls back to a single 0→100% bar).

### • Tests
*   New `tests/test_generate_progress_callback.py` (5 tests) and
    `tests/test_streaming_progress_callback.py` (4 tests): drive the real vendored loops with
    scripted mocks and lock the callback contract (monotonic, bounded, call counts, interrupt
    propagation, output determinism).
*   New `tests/test_streaming_progress.py` (4 tests); extended `tests/test_generation.py` (+5),
    `tests/test_asr_generation.py` (+6), `tests/test_integration.py` (+3 node-level tests).


</details>

<details>
<summary><strong>v2.1.0 - torchaudio-Primary Audio Backend (librosa now optional)</strong></summary>

### ◆ Highlights
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

### • Changes
*   New `modules/audio_backend.py`: single dependency-resilient backend for resample / load / save
    with import-time capability detection (`_HAS_*` flags) and graceful fallback ordering.
*   `modules/audio_utils.py`: `resample_audio()` delegates to the backend; `preprocess_comfy_audio()`
    now resamples in tensor space (no numpy round-trip).
*   Vendored processors (`_load_audio_from_path`, `save_audio`, ASR file loading) route through the
    backend; no hard `ffmpeg`/`soundfile` requirement for the common wav/flac path.

### • Tests
*   New `tests/test_audio_backend.py` (41 tests): import resilience with any optional lib blocked,
    resample correctness/priority/fallbacks, numpy↔tensor parity, file I/O roundtrips, `f32_pcm`.
*   New `tests/test_processor_io_backend.py` (17 tests): real vendored processor I/O via the backend.
*   Extended `tests/test_audio_utils.py`, `tests/test_pyproject.py`, `tests/test_imports.py`.


</details>

<details>
<summary><strong>v2.0.2 - Negative-Branch RoPE Position Fix (SDPA Shape Crash)</strong></summary>

### • Fixes
*   **BUG-011 — Negative-branch RoPE `position_ids` desync:** Fixed a `RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [12, 3] but got: [12, 2]` crash during TTS generation. In the non-streaming `generate()` CFG loop, the negative (unconditional) forward fed a **single-token** `inputs_embeds` `(B,1,H)` together with a **full-length** `position_ids` `(B, step+1)`. In transformers 5.x the explicit `position_ids` drive RoPE directly, so q/k silently broadcast against full-length cos/sin and expanded to seq-len `step+1` while v (never rotated) stayed length 1 — the KV cache accumulated `step+1` keys but only 1 value per step, and SDPA's `attn @ value` crashed at AR step 1. The negative forward now passes **current-only** `position_ids` (`neg_position_ids[:, -1:]`), matching its single input token; the attention mask stays full-length.

### • Tests
*   New `tests/test_generate_neg_position_ids.py` (4 tests): drives the real `generate()` through several AR steps with a recording inner LM and locks the invariant that `position_ids` length always equals `inputs_embeds` sequence length (red/green verified against the buggy code).


</details>

<details>
<summary><strong>v2.0.1 - CPU-First Model Loading (VRAM Round-Trip Fix)</strong></summary>

### • Fixes
*   **Load Device Flow — DF-001..DF-006:** Fixed a GPU→RAM→GPU round-trip during model loading. Previously the checkpoint state dict was loaded directly onto CUDA (a full-model VRAM spike outside ComfyUI's arbitration), copied back to CPU-resident parameters, then moved to CUDA again. Now the loader builds the model **entirely on CPU** (state dict, `load_state_dict`, dtype cast, and 4-bit quantization all on CPU), and `VibeVoicePatcher.patch_model` owns the **single** host-to-device transfer after ComfyUI's `load_models_gpu` VRAM arbitration. Peak VRAM during load drops from ≈2× model size to ≈1× model size.
*   **Dtype Threading — DF-004 / AUD-008:** The user-selected dtype is now threaded from the node through the handler into the loader and applied on CPU before the transfer; the patcher's dtype cast is now a mismatch-only guard (no redundant GPU cast).
*   **Handler No-Move — DF-003:** `VibeVoiceModelHandler.load_model` no longer moves the model; device placement is owned solely by the patcher.

### • Tests
*   New `tests/test_load_device_flow.py` (36 tests): device-ledger doubles asserting CPU-only loading, single H2D transfer, no round-trip, dtype threading, cast guard, VRAM arbitration, and a `[GPU-OPTIONAL]` peak-VRAM measurement.


</details>

<details>
<summary><strong>v2.0.0 - V3 Extension &amp; VRAM Parity</strong></summary>

### ◆ Highlights
*   **V3 Extension API:** Migrated the custom-node entrypoint to the ComfyUI V3 `ComfyExtension` / `io.ComfyNode` schema — type-filtered model dropdowns and declarative inputs/outputs.
*   **VRAM Parity (ASR) — CRIT-001:** The ASR path now runs under the same `VibeVoicePatcher` / `model_management.load_model_gpu` orchestration as TTS, clearing the dedicated ASR cache on unload.
*   **Warm Re-attach — NTH-004:** `force_offload` can retain model tensors on the intermediate device for a fast re-attach on the next run instead of reloading from disk.
*   **Streaming TTS Node — NTH-001:** `VibeVoice-Realtime-0.5B` became reachable from the patcher / attention machinery (it now runs on the single `VibeVoice TTS` node).
*   **Maintainability — IMP-004:** TTS/ASR download, discovery, and sharded-load logic is now shared via `BaseVibeVoiceLoader`.
*   **Device &amp; Attention Honesty — IMP-003 / IMP-001:** MPS/XPU/NPU device selection is honored when available; `flash_attention_2` is only offered when `flash-attn` + CUDA are present.
*   **Docs Consistency — CRIT-003:** README zero-shot wording now matches `generate_audio` (at least one reference voice is required).


</details>

<details>
<summary><strong>v1.5.0 - Stability and Prompting</strong></summary>

### ◆ New Features & Improvements
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
