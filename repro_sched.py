import sys, types, functools, inspect, importlib.util

# --- Minimal diffusers stub (validates env has the diffusers<->huggingface_hub clash) ---
# dpm_solver.py imports ONLY from diffusers + stdlib, so we stub just enough and load the
# vendored file standalone (bypassing the full vibevoice package __init__ chain).
d = types.ModuleType("diffusers")

cu = types.ModuleType("diffusers.configuration_utils")

class _ConfigHolder:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class _ConfigMixin:
    def __init__(self, **kwargs):
        self.config = _ConfigHolder()

def _register_to_config(func=None, **config_kwargs):
    """Identity-ish decorator that mirrors diffusers: stores __init__ kwargs into self.config."""
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
sys.modules.setdefault("diffusers", d)
sys.modules.setdefault("diffusers.configuration_utils", cu)
sys.modules.setdefault("diffusers.utils", du)
sys.modules.setdefault("diffusers.utils.torch_utils", tu)
sys.modules.setdefault("diffusers.schedulers", su)
sys.modules.setdefault("diffusers.schedulers.scheduling_utils", su)

# Load the vendored scheduler file directly.
spec = importlib.util.spec_from_file_location(
    "dpm_solver_standalone", "src/vibevoice/schedule/dpm_solver.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
DPMSolverMultistepScheduler = mod.DPMSolverMultistepScheduler

import torch

def make_sched():
    return DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="v_prediction",
    )

def run_step(s):
    t = s.timesteps[s.step_index if s.step_index is not None else 0]
    return s.step(torch.randn(2, 8), t, torch.randn(2, 8)).prev_sample

# Simulate generate(): set_timesteps ONCE, sample N latents (loops), no re-set per latent.
# This is what _sample_one_latent currently does (set_timesteps called once in generate()).
sched = make_sched()
sched.set_timesteps(10)
print("len(timesteps)=", len(sched.timesteps), "len(sigmas)=", len(sched.sigmas))
try:
    for latent in range(4):
        for _ in range(len(sched.timesteps)):
            run_step(sched)
    print("NO CRASH (unexpected for buggy path)")
except IndexError as e:
    print("CRASH (reproduced user bug):", e)

# Now simulate the FIX: set_timesteps at start of EACH latent
# (mirroring streaming sample_speech_tokens line 891).
sched2 = make_sched()
try:
    for latent in range(4):
        sched2.set_timesteps(10)
        for _ in range(len(sched2.timesteps)):
            run_step(sched2)
    print("FIX OK: no crash with per-latent set_timesteps")
except IndexError as e:
    print("FIX STILL CRASHES:", e)
