"""VibeVoice External Model Loader Node (V3 Schema).

Loads a VibeVoice model from a standalone weight file (safetensors / .bin /
.gguf) placed in ComfyUI's ``diffusion_models`` folder, binding the required
config / tokenizer / preprocessor JSONs via sidecar files or packaged defaults.

This node bypasses ComfyUI's ``model_detection.detect_unet_config()`` (which
does not recognize VibeVoice's state-dict keys) and outputs a
``VIBEVOICE_MODEL`` custom type consumed by the TTS / Realtime / ASR nodes via
their optional ``external_model`` input.
"""

import logging

import folder_paths
from comfy_api.latest import io

from ..modules.custom_types import VibeVoiceModel
from ..modules.external_loader import (
    load_external_vibevoice_model,
    AUTO_CONFIG_NAME,
    EXTERNAL_CONFIG_OPTIONS,
    is_asr_config_name,
    normalize_config_name,
    reconciled_config_name,
    remember_reconciled_config,
    resolve_auto_config_name,
)
from ..modules.attention_utils import (
    get_available_attention_modes,
    resolve_attention_mode,
    check_dtype_attention_compatible,
)
from ..modules.dtype_utils import get_dtype_options, DTYPE_AUTO
from ..modules.model_registry import (
    FAMILY_ASR,
    FAMILY_TTS,
    evict_if_changed,
    get_live_bundle,
    identity_for_external,
)
from ..modules.utils import VIBEVOICE_ASR_PATCHER_CACHE, VIBEVOICE_PATCHER_CACHE

logger = logging.getLogger(__name__)


def list_external_model_files() -> list:
    """Build the selectable weight-file list, including ``.gguf`` files.

    ComfyUI's ``get_filename_list("diffusion_models")`` filters by
    ``supported_pt_extensions``, which excludes ``.gguf``. This helper merges
    that list with ``.gguf`` files found in the same folders, plus the
    ``unet_gguf`` folder if it is registered (e.g. by ComfyUI-GGUF).

    Returns:
        Sorted, de-duplicated list of weight-file names (relative paths).
    """
    files = set()

    # Standard diffusion_models list (safetensors / bin / pt / ...).
    try:
        files.update(folder_paths.get_filename_list("diffusion_models"))
    except Exception:
        pass

    # .gguf files are excluded from supported_pt_extensions, so scan the
    # diffusion_models folders directly for them.
    try:
        for folder in folder_paths.get_folder_paths("diffusion_models"):
            found, _ = folder_paths.recursive_search(folder)
            for name in found:
                if name.lower().endswith(".gguf"):
                    files.add(name)
    except Exception:
        pass

    # ComfyUI-GGUF registers a dedicated "unet_gguf" folder; include it if present.
    try:
        files.update(folder_paths.get_filename_list("unet_gguf"))
    except Exception:
        pass

    return sorted(files)


def resolve_weight_path(model_file: str) -> str:
    """Resolve a selected weight-file name to an absolute path.

    Tries the ``diffusion_models`` folder first, then the ``unet_gguf`` folder
    (registered by ComfyUI-GGUF). Raises ``FileNotFoundError`` if the file is
    not found in either.

    Args:
        model_file: Relative weight-file name from the dropdown.

    Returns:
        Absolute path to the weight file.
    """
    for folder_name in ("diffusion_models", "unet_gguf"):
        try:
            return folder_paths.get_full_path_or_raise(folder_name, model_file)
        except Exception:
            continue
    raise FileNotFoundError(
        f"Weight file '{model_file}' not found in diffusion_models or unet_gguf folders."
    )


class VibeVoiceExternalLoaderNode(io.ComfyNode):
    """Load a VibeVoice model from an external weight file.

    The weight file is selected from ComfyUI's ``diffusion_models`` folder.
    The architecture config / tokenizer / preprocessor JSONs are bound via
    sidecar files next to the weight file, or fall back to packaged defaults
    selected by the ``config_name`` dropdown.
    """

    CATEGORY = "WMNodes/sound/tts"

    @classmethod
    def define_schema(cls) -> io.Schema:
        model_files = list_external_model_files()
        if not model_files:
            model_files = ["No model files found in diffusion_models"]

        config_options = list(EXTERNAL_CONFIG_OPTIONS)
        attention_modes = get_available_attention_modes()
        dtype_options = get_dtype_options()

        return io.Schema(
            node_id="VibeVoiceLoadExternalModel",
            display_name="Load VibeVoice Model",
            category=cls.CATEGORY,
            description=(
                "Load a VibeVoice model from an external weight file "
                "(safetensors/bin/gguf) in the diffusion_models folder. "
                "Bind sidecar config/tokenizer JSONs or use packaged defaults."
            ),
            inputs=[
                io.Combo.Input(
                    "model_file",
                    options=model_files,
                    default=model_files[0],
                    tooltip=(
                        "Weight file in the diffusion_models folder. Place the "
                        "VibeVoice .safetensors (or .bin/.gguf) file there."
                    ),
                ),
                io.Combo.Input(
                    "config_name",
                    options=config_options,
                    default=AUTO_CONFIG_NAME,
                    tooltip=(
                        "Architecture config to use. Auto-detect reads the "
                        "weight file's embedding fingerprint and selects the "
                        "matching family; an explicit selection that "
                        "contradicts the weights is auto-corrected with a "
                        "warning. A sidecar config next to the weight file "
                        "takes priority over the packaged default fallback."
                    ),
                ),
                io.Combo.Input(
                    "attention_mode",
                    options=attention_modes,
                    default="sdpa",
                    tooltip="Attention implementation: Eager (safest), SDPA (balanced), Flash Attention 2 (fastest), Sage (quantized)",
                ),
                io.Boolean.Input(
                    "quantize_llm_4bit",
                    default=False,
                    label_on="Q4 (LLM only)",
                    label_off="Full precision",
                    tooltip="Quantize the Qwen2.5 LLM to 4-bit NF4 via bitsandbytes.",
                ),
                io.Combo.Input(
                    "dtype",
                    options=dtype_options,
                    default=DTYPE_AUTO,
                    tooltip="Data type for model precision. 'auto' selects optimal type for device.",
                ),
            ],
            outputs=[
                VibeVoiceModel.Output(display_name="VibeVoice Model"),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        """Accept current options AND removed legacy aliases.

        Overriding this with ``**kwargs`` makes ComfyUI core skip its own
        combo-membership check (execution.py defers to the node when the
        validate function has var-keywords), so saved workflows carrying the
        removed ``VibeVoice-Large`` value still queue; ``execute()`` then
        normalizes the alias onto its canonical option.
        """
        cfg = kwargs.get("config_name")
        if cfg not in EXTERNAL_CONFIG_OPTIONS and \
                normalize_config_name(cfg) not in EXTERNAL_CONFIG_OPTIONS:
            return (
                f"config_name '{cfg}' is not valid. "
                f"Choose one of: {', '.join(EXTERNAL_CONFIG_OPTIONS)}"
            )

        # This node is the single owner of the dtype/attention pair for every
        # model it builds (TTS and ASR alike), so the sage fp32/bf16
        # cross-check belongs here as well as on the generation nodes. The
        # ASR branch additionally drops sage entirely (see
        # modules/attention_utils.ASR_EXCLUDED_ATTENTION_MODES).
        message = check_dtype_attention_compatible(
            kwargs.get("dtype"),
            kwargs.get("attention_mode"),
            bool(kwargs.get("quantize_llm_4bit")),
        )
        if message is not None:
            return message
        return True

    @classmethod
    def execute(
        cls,
        model_file: str,
        config_name: str,
        attention_mode: str,
        quantize_llm_4bit: bool,
        dtype: str,
    ) -> io.NodeOutput:
        # Normalize removed legacy aliases (e.g. "VibeVoice-Large" -> 7B)
        # BEFORE the identity is computed, so legacy and modern workflows
        # produce the same cache/eviction key for the same effective model.
        config_name = normalize_config_name(config_name)

        weight_path = resolve_weight_path(model_file)

        # Auto-detect resolution (plan 2026-08-27, D6): resolve the effective
        # family BEFORE the identity is computed. The consumer derives its
        # patcher-cache key from the bundle's recorded model_name, so the
        # node's request key must carry the same resolved name — otherwise the
        # unload-before-load gate and patcher cache would churn on every run
        # (same principle as alias normalization above). Inconclusive
        # detection fails fast here with an actionable error.
        if config_name == AUTO_CONFIG_NAME:
            config_name = resolve_auto_config_name(weight_path)

        # The name the USER asked for, captured before anything downstream can
        # rewrite it. The loader may still reconcile it against the weights'
        # architecture fingerprint (reconcile_config), and the memo below is
        # keyed by this requested name.
        requested_config_name = config_name

        # Unload-before-load gate (plan 2026-08-20, C1/RC-1/RC-5): compute the
        # request identity EXACTLY as the external loader will record it on the
        # bundle (same resolution rules), and fully release any different
        # active model BEFORE the new one is built on CPU. This is what keeps
        # peak RAM at ~1x model size during a model swap.
        is_asr = is_asr_config_name(config_name)
        family = FAMILY_ASR if is_asr else FAMILY_TTS
        use_llm_4bit = False if is_asr else bool(quantize_llm_4bit)
        resolved_attn = resolve_attention_mode(attention_mode, use_llm_4bit)
        # The name the consumer will key the built patcher off is the bundle's
        # POST-reconciliation model_name. The gate cannot run after the load
        # (peak-RAM), so on a memo miss the requested name is used — byte-
        # identical to the pre-memo behaviour, i.e. the first run of a
        # reconciling model still evicts, and every later run is a strict
        # no-op. Family/attention resolution above is unaffected: the ASR
        # branch never reconciles, and reconciliation only ever substitutes a
        # TTS family detected from the same fingerprint.
        config_name = reconciled_config_name(
            weight_path, requested_config_name
        )
        # The request key MUST live in the same namespace as the consumer's
        # cache key ("asr_external" vs "external") — a prefix mismatch would
        # make every identical re-run of an ASR model spuriously evict and
        # rebuild the live patcher.
        request_key = identity_for_external(
            weight_path,
            config_name,
            resolved_attn,
            use_llm_4bit=use_llm_4bit,
            dtype_str=dtype,
            prefix="asr_external" if is_asr else "external",
        )
        evict_if_changed(
            family,
            request_key,
            (VIBEVOICE_ASR_PATCHER_CACHE if is_asr else VIBEVOICE_PATCHER_CACHE,),
        )
        # The gate above is a strict no-op for a SAME-key re-execution (that is
        # what lets a cached re-run reuse its live patcher), but without this
        # short-circuit the node still REBUILDS the model on every run. The
        # outgoing weights remain reachable from two places — ComfyUI's node
        # output cache and the live patcher's handler — so the rebuild overlaps
        # a second full copy: the ~2x host-RAM spike on a 16.6 GB checkpoint,
        # invisible in the load path itself, which measures ~1.0x file
        # (plan §11.8).
        #
        # Reusing the resident model is not merely a RAM optimisation: the
        # consumer's patcher is keyed by this same identity and is NOT rebuilt
        # on a hit, so a rebuilt bundle would be discarded in favour of the
        # already-loaded weights. Short-circuiting keeps exactly one model, and
        # keeps the node's output consistent with the patcher the consumer
        # will actually use.
        cached_bundle = get_live_bundle(request_key)
        if cached_bundle is not None:
            logger.debug(
                f"Reusing the resident model for {request_key!r} "
                f"(identical re-execution; skipping the rebuild)"
            )
            return io.NodeOutput(cached_bundle)

        model_bundle = load_external_vibevoice_model(
            weight_path=weight_path,
            config_name=config_name,
            attention_mode=attention_mode,
            use_llm_4bit=quantize_llm_4bit,
            dtype_str=dtype,
        )

        # Remember what this load actually built, so the NEXT run computes the
        # same request key the consumer will compute. Guarded: a hand-built or
        # mocked bundle without 'model_name' must never break execute().
        try:
            remember_reconciled_config(
                weight_path, requested_config_name, model_bundle["model_name"]
            )
        except Exception as e:
            logger.debug(f"Could not memoize reconciled config name: {e}")

        return io.NodeOutput(model_bundle)
