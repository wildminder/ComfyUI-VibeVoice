"""Regression test for BUG-006 scheduler IndexError.

Root cause: `VibeVoiceForConditionalGeneration._sample_one_latent` runs once per
AR step, but the vendored DPMSolverMultistepScheduler only resets its internal
`_step_index` inside `set_timesteps()`. `generate()` called `set_timesteps()`
once before the AR loop, so from the 2nd latent the final-step `lower_order_final`
guard never triggered and the 2nd-order update read `sigmas[step_index + 1]`,
raising `IndexError: index N is out of bounds for dimension 0 with size N`.

The fix adds `self.model.noise_scheduler.set_timesteps(num_steps)` at the START of
every `_sample_one_latent` call (mirroring the streaming `sample_speech_tokens`).

This test drives the REAL `_sample_one_latent` with a REAL scheduler and asserts
that several consecutive latents sample without an IndexError and produce finite,
correctly-shaped output. It also locks in the bug: driving the scheduler with the
OLD once-only `set_timesteps` pattern must raise IndexError.
"""

import os
import sys
import types
import functools
import inspect
import importlib.util

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Faithful diffusers stub (the validation env has the diffusers<->huggingface_hub
# clash; we stub just enough and load the vendored scheduler standalone).
# ---------------------------------------------------------------------------
def _stub_diffusers_faithful():
    if "diffusers" in sys.modules and not isinstance(sys.modules["diffusers"], types.ModuleType):
        return
    if isinstance(sys.modules.get("diffusers", None), types.ModuleType) and getattr(
        sys.modules["diffusers"], "__vibevoice_stub__", False
    ):
        return

    d = types.ModuleType("diffusers")
    d.__vibevoice_stub__ = True

    cu = types.ModuleType("diffusers.configuration_utils")

    class _ConfigHolder:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _ConfigMixin:
        def __init__(self, **kwargs):
            self.config = _ConfigHolder()

    def _register_to_config(func=None, **config_kwargs):
        if func is None:
            return functools.partial(_register_to_config, **config_kwargs)
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            if not hasattr(self, "config"):
                self.config = _ConfigHolder()
            for name, val in bound.arguments.items():
                if name == "self":
                    continue
                setattr(self.config, name, val)

        return wrapper

    cu.ConfigMixin = _ConfigMixin
    cu.register_to_config = _register_to_config

    du = types.ModuleType("diffusers.utils")
    du.deprecate = lambda *a, **k: None
    tu = types.ModuleType("diffusers.utils.torch_utils")
    tu.randn_tensor = lambda *a, **k: None
    su = types.ModuleType("diffusers.schedulers.scheduling_utils")
    su.KarrasDiffusionSchedulers = []  # iterable for `_compatibles = [e.name for e in ...]`

    class _SchedulerMixin:
        pass

    su.SchedulerMixin = _SchedulerMixin

    class _SchedulerOutput:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    su.SchedulerOutput = _SchedulerOutput

    d.configuration_utils = cu
    d.utils = du
    d.schedulers = su
    d.schedulers.scheduling_utils = su
    # Direct assignment (NOT setdefault): this must override any diffusers stub
    # another test module may have already installed under the same sys.modules key.
    sys.modules["diffusers"] = d
    sys.modules["diffusers.configuration_utils"] = cu
    sys.modules["diffusers.utils"] = du
    sys.modules["diffusers.utils.torch_utils"] = tu
    sys.modules["diffusers.schedulers"] = su
    sys.modules["diffusers.schedulers.scheduling_utils"] = su


def _load_real_modeling_module():
    _stub_diffusers_faithful()
    for mod in (
        "src.vibevoice",
        "src.vibevoice.modular",
        "src.vibevoice.modular.modeling_vibevoice",
    ):
        sys.modules.pop(mod, None)

    root = os.path.join(os.getcwd(), "src", "vibevoice")
    mod_pkg = types.ModuleType("src.vibevoice.modular")
    mod_pkg.__path__ = [os.path.join(root, "modular")]
    sys.modules.setdefault("src.vibevoice", types.ModuleType("src.vibevoice"))
    sys.modules["src.vibevoice"].__path__ = [root]
    sys.modules["src.vibevoice.modular"] = mod_pkg

    cfg_mod = types.ModuleType("src.vibevoice.modular.configuration_vibevoice")

    class _FakeConfig:
        __name__ = "VibeVoiceConfig"
        model_type = "vibevoice"

    cfg_mod.VibeVoiceConfig = _FakeConfig
    # modeling_vibevoice imports the version-safe dtype helper from this
    # module; provide the real implementation (never mocked).
    from ComfyUI_VibeVoice.modules.dtype_utils import get_config_dtype, set_config_dtype
    cfg_mod.get_config_dtype = get_config_dtype
    cfg_mod.set_config_dtype = set_config_dtype
    sys.modules["src.vibevoice.modular.configuration_vibevoice"] = cfg_mod

    _HERE = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(
        os.path.join(_HERE, "..", "src", "vibevoice", "modular", "modeling_vibevoice.py")
    )
    spec = importlib.util.spec_from_file_location(
        "src.vibevoice.modular.modeling_vibevoice", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["src.vibevoice.modular.modeling_vibevoice"] = module
    spec.loader.exec_module(module)
    return module


_modeling = _load_real_modeling_module()
VibeVoiceForConditionalGeneration = _modeling.VibeVoiceForConditionalGeneration


def _make_real_scheduler(num_train_timesteps=1000, num_inference_steps=10):
    # Import the vendored scheduler standalone (so we never pull the broken
    # vibevoice package __init__). Returns a REAL DPMSolverMultistepScheduler.
    _stub_diffusers_faithful()
    sched_spec = importlib.util.spec_from_file_location(
        "dpm_solver_standalone_sched",
        os.path.join(os.getcwd(), "src", "vibevoice", "schedule", "dpm_solver.py"),
    )
    sched_mod = importlib.util.module_from_spec(sched_spec)
    sched_spec.loader.exec_module(sched_mod)
    Scheduler = sched_mod.DPMSolverMultistepScheduler
    s = Scheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="scaled_linear",
        prediction_type="v_prediction",
    )
    s.set_timesteps(num_inference_steps)
    return s


def _make_fake_self(num_steps=10, vae_dim=8, batch=1):
    """Build a fake `self` exposing the attributes _sample_one_latent touches,
    wired to a REAL DPMSolverMultistepScheduler."""
    scheduler = _make_real_scheduler(num_inference_steps=num_steps)

    class _PredHead:
        device = torch.device("cpu")

        def __call__(self, noisy, timesteps, condition=None):
            # noisy: (2B, vae_dim); return matching noise estimate.
            return torch.randn(noisy.shape[0], vae_dim)

    class _AcousticCfg:
        vae_dim = None

    _acfg = _AcousticCfg()
    _acfg.vae_dim = vae_dim

    class _ModelCfg:
        acoustic_tokenizer_config = _acfg

    class _Model:
        model = None

    # Real scheduler + fake prediction head + NaN scaling factors (skip inverse scaling).
    inner = types.SimpleNamespace(
        noise_scheduler=scheduler,
        prediction_head=_PredHead(),
        speech_scaling_factor=torch.tensor(float("nan")),
        speech_bias_factor=torch.tensor(float("nan")),
    )
    fake = types.SimpleNamespace(
        model=inner,
        config=_ModelCfg(),
    )
    return fake


def test_sample_one_latent_multiple_ar_steps_no_indexerror():
    """Driving _sample_one_latent repeatedly (one call per AR step) must not
    raise IndexError and must return finite (B, vae_dim) latents."""
    real_method = VibeVoiceForConditionalGeneration._sample_one_latent
    num_steps = 10
    vae_dim = 8
    fake = _make_fake_self(num_steps=num_steps, vae_dim=vae_dim, batch=1)

    bound = types.MethodType(real_method, fake)
    condition = torch.randn(1, vae_dim)
    neg_condition = torch.randn(1, vae_dim)

    last = None
    for _ in range(5):  # 5 AR steps -> 5 latent samples
        last = bound(condition, neg_condition, cfg_scale=3.0, num_steps=num_steps)
        assert last.shape == (1, vae_dim)
        assert torch.isfinite(last).all(), "latent contains non-finite values"

    assert last is not None


def test_scheduler_once_only_pattern_raises_indexerror():
    """Lock in the bug: the OLD usage (set_timesteps once, then multiple full
    step-loops) overruns `sigmas` and raises IndexError. The fix in
    _sample_one_latent resets per latent so this never happens there."""
    scheduler = _make_real_scheduler(num_inference_steps=10)
    n_steps = len(scheduler.timesteps)

    def run_step(s):
        t = s.timesteps[s.step_index if s.step_index is not None else 0]
        return s.step(torch.randn(2, 8), t, torch.randn(2, 8)).prev_sample

    with pytest.raises(IndexError):
        for _latent in range(3):
            for _ in range(n_steps):
                run_step(scheduler)
