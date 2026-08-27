"""Phase 1 gate: meta-init feasibility for the vendored VibeVoice model classes.

These tests instantiate each model class under ``torch.device("meta")`` and
assert that every parameter lands on the meta device (zero allocation, zero
RNG) while the numpy-based ``DPMSolverMultistepScheduler`` keeps real CPU
tensors. A GO result here is the precondition for Phase 3 (meta-context
instantiation in the loader), which eliminates the random-init CPU pass (RC-1).

The vendored package is mocked by ``conftest.py``; we load the REAL modules
here by stubbing ``diffusers`` (incompatible install) and registering the
package hierarchy manually — the same technique used by
``tests/test_model_forward.py``.
"""

import os
import sys
import types
import importlib.util

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# diffusers stub (dpm_solver imports it; the real install is incompatible).
# ---------------------------------------------------------------------------
def _stub_diffusers():
    if "diffusers" in sys.modules and isinstance(sys.modules["diffusers"], types.ModuleType):
        # A real (or previously-stubbed) module is present; only stub if it is
        # not already providing the names we need.
        existing = sys.modules["diffusers"]
        if hasattr(existing, "configuration_utils") and hasattr(existing, "schedulers"):
            return
    d = types.ModuleType("diffusers")
    cu = types.ModuleType("diffusers.configuration_utils")

    class _ConfigMixin:
        pass

    def _register_to_config(*a, **k):
        if a and callable(a[0]):
            return a[0]

        def _wrap(fn):
            return fn

        return _wrap

    cu.ConfigMixin = _ConfigMixin
    cu.register_to_config = _register_to_config
    du = types.ModuleType("diffusers.utils")
    du.deprecate = lambda *a, **k: None
    tu = types.ModuleType("diffusers.utils.torch_utils")
    tu.randn_tensor = lambda *a, **k: None
    su = types.ModuleType("diffusers.schedulers.scheduling_utils")

    class _KarrasDiffusionSchedulers:
        # Iterated at class-body time in dpm_solver: `[e.name for e in ...]`
        def __iter__(self):
            return iter([])

    class _SchedulerMixin(_ConfigMixin):
        # Real diffusers: SchedulerMixin subclasses ConfigMixin (valid MRO).
        pass

    su.KarrasDiffusionSchedulers = _KarrasDiffusionSchedulers()
    su.SchedulerMixin = _SchedulerMixin
    su.SchedulerOutput = object
    d.configuration_utils = cu
    d.utils = du
    d.schedulers = su
    sys.modules["diffusers"] = d
    sys.modules["diffusers.configuration_utils"] = cu
    sys.modules["diffusers.utils"] = du
    sys.modules["diffusers.utils.torch_utils"] = tu
    sys.modules["diffusers.schedulers"] = su
    sys.modules["diffusers.schedulers.scheduling_utils"] = su


# ---------------------------------------------------------------------------
# Real-module loading (bypasses the conftest mocks).
# ---------------------------------------------------------------------------
def _mk_pkg(name, path):
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    sys.modules[name] = mod
    return mod


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relpath))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_real_modules():
    """Load the real vendored modules once and return a namespace dict."""
    if hasattr(_load_real_modules, "_cache"):
        return _load_real_modules._cache

    _stub_diffusers()

    # Remove conftest-installed mocks ONLY for the subtrees we are about to
    # load with real code (modular + schedule). The src.vibevoice.configs and
    # src.vibevoice.processor mocks are deliberately left in place — other
    # tests (test_package_structure.py) assert they remain importable, and
    # popping them here polluted sys.modules for later tests.
    for mod in list(sys.modules):
        if (
            mod == "src.vibevoice"
            or mod.startswith("src.vibevoice.modular")
            or mod.startswith("src.vibevoice.schedule")
        ):
            sys.modules.pop(mod, None)

    src_vv = os.path.join(ROOT, "src", "vibevoice")
    _mk_pkg("src", os.path.join(ROOT, "src"))
    _mk_pkg("src.vibevoice", src_vv)
    _mk_pkg("src.vibevoice.modular", os.path.join(src_vv, "modular"))
    _mk_pkg("src.vibevoice.schedule", os.path.join(src_vv, "schedule"))

    ns = {}
    ns["cfg"] = _load(
        "src.vibevoice.modular.configuration_vibevoice",
        "src/vibevoice/modular/configuration_vibevoice.py")
    ns["cfg_stream"] = _load(
        "src.vibevoice.modular.configuration_vibevoice_streaming",
        "src/vibevoice/modular/configuration_vibevoice_streaming.py")
    _load("src.vibevoice.modular.modular_vibevoice_tokenizer",
          "src/vibevoice/modular/modular_vibevoice_tokenizer.py")
    _load("src.vibevoice.modular.modular_vibevoice_diffusion_head",
          "src/vibevoice/modular/modular_vibevoice_diffusion_head.py")
    _load("src.vibevoice.schedule.dpm_solver",
          "src/vibevoice/schedule/dpm_solver.py")
    ns["modeling"] = _load(
        "src.vibevoice.modular.modeling_vibevoice",
        "src/vibevoice/modular/modeling_vibevoice.py")
    _load("src.vibevoice.modular.modeling_vibevoice_streaming",
          "src/vibevoice/modular/modeling_vibevoice_streaming.py")
    ns["modeling_stream_infer"] = _load(
        "src.vibevoice.modular.modeling_vibevoice_streaming_inference",
        "src/vibevoice/modular/modeling_vibevoice_streaming_inference.py")
    ns["modeling_asr"] = _load(
        "src.vibevoice.modular.modeling_vibevoice_asr",
        "src/vibevoice/modular/modeling_vibevoice_asr.py")
    ns["configs_dir"] = os.path.join(src_vv, "configs")

    _load_real_modules._cache = ns
    return ns


def _meta_stats(model):
    """Return (n_params, n_meta, scheduler_real_or_None)."""
    params = list(model.parameters())
    n_meta = sum(1 for p in params if p.is_meta)
    sched_ok = None
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "noise_scheduler"):
        sig = getattr(inner.noise_scheduler, "sigmas", None)
        if sig is not None and isinstance(sig, torch.Tensor):
            sched_ok = (not sig.is_meta)
    return len(params), n_meta, sched_ok


# ---------------------------------------------------------------------------
# Fixtures / configs
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real():
    return _load_real_modules()


@pytest.fixture(scope="module")
def std_config(real):
    cfg_path = os.path.join(real["configs_dir"], "default_VibeVoice-1.5B_config.json")
    config = real["cfg"].VibeVoiceConfig.from_pretrained(cfg_path)
    real["cfg"].set_config_dtype(config, torch.bfloat16)
    return config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestMetaInit:
    def test_standard_model_meta_init(self, real, std_config):
        with torch.device("meta"):
            model = real["modeling"].VibeVoiceForConditionalGeneration(std_config)
        n_params, n_meta, sched_ok = _meta_stats(model)
        assert n_params > 0
        assert n_meta == n_params, "all parameters must be on meta device"
        # numpy scheduler path keeps real CPU tensors
        assert sched_ok is True, "noise_scheduler.sigmas must be a real tensor"

    def test_streaming_model_meta_init(self, real, std_config):
        stream_config = real["cfg_stream"].VibeVoiceStreamingConfig(
            acoustic_tokenizer_config=std_config.acoustic_tokenizer_config,
            decoder_config=std_config.decoder_config,
            diffusion_head_config=std_config.diffusion_head_config,
            tts_backbone_num_hidden_layers=20,
        )
        real["cfg"].set_config_dtype(stream_config, torch.bfloat16)
        with torch.device("meta"):
            model = real["modeling_stream_infer"].VibeVoiceStreamingForConditionalGenerationInference(
                stream_config)
        n_params, n_meta, sched_ok = _meta_stats(model)
        assert n_params > 0
        assert n_meta == n_params, "all parameters must be on meta device"
        assert sched_ok is True, "noise_scheduler.sigmas must be a real tensor"

    def test_asr_model_meta_init(self, real, std_config):
        asr_config = real["cfg"].VibeVoiceASRConfig(
            acoustic_tokenizer_config=std_config.acoustic_tokenizer_config,
            semantic_tokenizer_config=std_config.semantic_tokenizer_config,
            decoder_config=std_config.decoder_config,
        )
        real["cfg"].set_config_dtype(asr_config, torch.bfloat16)
        with torch.device("meta"):
            model = real["modeling_asr"].VibeVoiceASRForConditionalGeneration(asr_config)
        n_params, n_meta, sched_ok = _meta_stats(model)
        assert n_params > 0
        assert n_meta == n_params, "all parameters must be on meta device"
        # ASR has no noise_scheduler; sched_ok is None — that is acceptable.
        assert sched_ok in (None, True)

    def test_meta_init_allocates_no_real_storage(self, real, std_config):
        """Meta init must allocate zero real memory (RC-1 elimination proof).

        Every parameter produced under the meta context must have its storage
        on the meta device — meta storage carries a logical size but consumes
        no RAM, so no real memory was used for weights.
        """
        with torch.device("meta"):
            model = real["modeling"].VibeVoiceForConditionalGeneration(std_config)
        for name, p in model.named_parameters():
            assert p.is_meta, f"{name} is not on meta device"
            assert p.untyped_storage().device.type == "meta", (
                f"{name} has real storage on {p.untyped_storage().device}")


# ---------------------------------------------------------------------------
# Phase 3: VibeVoiceLoader._instantiate_model meta-context behavior
# ---------------------------------------------------------------------------
class _TinyCtorModel(torch.nn.Module):
    """Minimal stand-in for the vendored model classes (config-ctor shape)."""

    def __init__(self, config):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)


class TestInstantiateModelMeta:
    """Contract: _instantiate_model builds under a meta context by default."""

    def _instantiate(self, use_meta=None, streaming=False):
        from unittest.mock import patch, MagicMock
        from ComfyUI_VibeVoice.modules.loader import VibeVoiceLoader

        target = ("ComfyUI_VibeVoice.modules.loader."
                  "VibeVoiceStreamingForConditionalGenerationInference"
                  if streaming else
                  "ComfyUI_VibeVoice.modules.loader.VibeVoiceForConditionalGeneration")
        kwargs = {} if use_meta is None else {"use_meta": use_meta}
        with patch(target, _TinyCtorModel):
            return VibeVoiceLoader._instantiate_model(
                config=MagicMock(),
                is_streaming=streaming,
                attn_implementation="eager",
                final_load_dtype=torch.bfloat16,
                **kwargs,
            )

    def test_instantiate_model_defaults_to_meta(self):
        model = self._instantiate()
        params = list(model.parameters())
        assert params, "tiny model must have parameters"
        assert all(p.is_meta for p in params), "default construction must be meta"

    def test_instantiate_model_use_meta_false_eager(self):
        model = self._instantiate(use_meta=False)
        params = list(model.parameters())
        assert params
        assert all(not p.is_meta for p in params), "use_meta=False must be eager"
        assert all(p.device.type == "cpu" for p in params)

    def test_no_random_init_on_real_storage_fast_path(self):
        """RC-1 proof: any initializer invoked during meta construction must
        only ever touch meta tensors (no real RNG fill of real storage)."""
        import torch.nn.init as init

        seen = []
        orig = {k: getattr(init, k) for k in ("normal_", "kaiming_uniform_", "uniform_", "constant_")}

        def _spy(name):
            def _wrapped(tensor, *a, **k):
                seen.append((name, tensor.is_meta))
                return orig[name](tensor, *a, **k)
            return _wrapped

        for k in orig:
            setattr(init, k, _spy(k))
        try:
            self._instantiate(use_meta=True)
        finally:
            for k, fn in orig.items():
                setattr(init, k, fn)

        # Every initializer call (if any) must have operated on a meta tensor.
        assert all(is_meta for _, is_meta in seen), (
            "an initializer touched a REAL tensor during meta construction")


# ---------------------------------------------------------------------------
# transformers v5 torch_dtype deprecation compat (vendored helpers)
# ---------------------------------------------------------------------------
class TestConfigDtypeCompat:
    """The vendored set/get_config_dtype helpers must round-trip the dtype on
    a REAL config without triggering the transformers v5 deprecation warning,
    and model construction must read it back correctly."""

    def test_set_get_roundtrip_real_config(self, real):
        cfg_path = os.path.join(real["configs_dir"], "default_VibeVoice-1.5B_config.json")
        config = real["cfg"].VibeVoiceConfig.from_pretrained(cfg_path)
        real["cfg"].set_config_dtype(config, torch.bfloat16)
        assert real["cfg"].get_config_dtype(config) == torch.bfloat16

    def test_get_config_dtype_from_packaged_config(self, real):
        """Packaged config.json carries "torch_dtype": "bfloat16"; transformers
        v5 converts it to the canonical ``dtype`` at load (no warning), and the
        helper must resolve it."""
        cfg_path = os.path.join(real["configs_dir"], "default_VibeVoice-1.5B_config.json")
        config = real["cfg"].VibeVoiceConfig.from_pretrained(cfg_path)
        assert real["cfg"].get_config_dtype(config) == torch.bfloat16

    def test_model_reads_dtype_via_helper(self, real, std_config):
        """Model construction must resolve the dtype through get_config_dtype
        (never via the deprecated config.torch_dtype attribute). Under meta
        init the .to(dtype) casts are skipped, so we spy on the helper call
        instead of inspecting parameter dtypes."""
        modeling = real["modeling"]
        real_fn = modeling.get_config_dtype
        calls = []

        def _spy(config):
            calls.append(config)
            return real_fn(config)

        modeling.get_config_dtype = _spy
        try:
            with torch.device("meta"):
                modeling.VibeVoiceForConditionalGeneration(std_config)
        finally:
            modeling.get_config_dtype = real_fn

        assert calls, "model construction did not call get_config_dtype"
        assert real_fn(std_config) == torch.bfloat16

    def test_convert_dtype_to_string_handles_both_keys(self, real):
        fn = real["cfg"]._convert_dtype_to_string
        d = fn({"torch_dtype": torch.bfloat16, "dtype": torch.float16})
        assert d["torch_dtype"] == "bfloat16"
        assert d["dtype"] == "float16"
