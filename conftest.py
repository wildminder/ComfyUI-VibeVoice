"""Pytest configuration for ComfyUI-VibeVoice tests.

Sets up the ComfyUI path, mocks the network server, and registers the
custom node package under a stable alias so relative imports work.
"""

import sys
import os
import importlib.util
from unittest.mock import MagicMock, patch
import pytest

# torch is required by the behavioral patcher stub (_TinyHandler) below.
import torch  # noqa: E402

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ====================================================================
# 1. PATH CONFIGURATION
# ====================================================================
# ComfyUI root — adjust if your checkout lives elsewhere.
COMFYUI_ROOT = os.environ.get("COMFYUI_ROOT", r"C:\_Dev\ComfyUI_dev\ComfyUI")

if COMFYUI_ROOT not in sys.path:
    sys.path.insert(0, COMFYUI_ROOT)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ====================================================================
# 2. MOCK ONLY THE NETWORK SERVER
# ====================================================================
# We let comfy and folder_paths load normally from COMFYUI_ROOT.
# We only mock 'server' and 'aiohttp' to prevent network activity.
_mock_server = MagicMock()
_mock_prompt_server_instance = MagicMock()
_mock_server.PromptServer.instance = _mock_prompt_server_instance
sys.modules.setdefault("server", _mock_server)
sys.modules.setdefault("aiohttp", MagicMock())
sys.modules.setdefault("aiohttp.web", MagicMock())

# ====================================================================
# 2b. MOCK HEAVY VENDORED DEPENDENCIES
# ====================================================================
# The vendored VibeVoice source imports diffusers, which may be broken
# in some environments (e.g. huggingface_hub version mismatch). We mock
# the heavy vendored modules so tests can import modules.loader etc.
# without actually loading the full model stack.
# Tests that need real model behavior already mock at the function level.
# Mock the entire src.vibevoice package to prevent loading heavy dependencies
# This must be done BEFORE any imports that might trigger the vendored code
_src_vibevoice_mock = MagicMock()
_src_vibevoice_mock.__path__ = []  # Make it a package
sys.modules["src.vibevoice"] = _src_vibevoice_mock
sys.modules["src.vibevoice.modular"] = MagicMock()
sys.modules["src.vibevoice.processor"] = MagicMock()
sys.modules["src.vibevoice.schedule"] = MagicMock()
sys.modules["src.vibevoice.configs"] = MagicMock()

# Also mock the aliased versions
sys.modules["ComfyUI_VibeVoice.src.vibevoice"] = _src_vibevoice_mock
sys.modules["ComfyUI_VibeVoice.src.vibevoice.modular"] = MagicMock()
sys.modules["ComfyUI_VibeVoice.src.vibevoice.processor"] = MagicMock()
sys.modules["ComfyUI_VibeVoice.src.vibevoice.schedule"] = MagicMock()
sys.modules["ComfyUI_VibeVoice.src.vibevoice.configs"] = MagicMock()

# Mock specific submodules that are imported directly
_vibevoice_mock_modules = [
    "src.vibevoice.modular.configuration_vibevoice",
    "src.vibevoice.modular.configuration_vibevoice_streaming",
    "src.vibevoice.modular.modeling_vibevoice",
    "src.vibevoice.modular.modeling_vibevoice_asr",
    "src.vibevoice.modular.modeling_vibevoice_streaming",
    "src.vibevoice.modular.modeling_vibevoice_streaming_inference",
    "src.vibevoice.modular.modular_vibevoice_diffusion_head",
    "src.vibevoice.modular.modular_vibevoice_text_tokenizer",
    "src.vibevoice.modular.modular_vibevoice_tokenizer",
    "src.vibevoice.modular.sage_attention_patch",
    "src.vibevoice.modular.streamer",
    "src.vibevoice.processor.audio_utils",
    "src.vibevoice.processor.vibevoice_asr_processor",
    "src.vibevoice.processor.vibevoice_processor",
    "src.vibevoice.processor.vibevoice_streaming_processor",
    "src.vibevoice.processor.vibevoice_tokenizer_processor",
    "src.vibevoice.schedule.dpm_solver",
    "src.vibevoice.schedule.timestep_sampler",
    # Also mock the aliased versions
    "ComfyUI_VibeVoice.src.vibevoice.modular.configuration_vibevoice",
    "ComfyUI_VibeVoice.src.vibevoice.modular.configuration_vibevoice_streaming",
    "ComfyUI_VibeVoice.src.vibevoice.modular.modeling_vibevoice",
    "ComfyUI_VibeVoice.src.vibevoice.modular.modeling_vibevoice_asr",
    "ComfyUI_VibeVoice.src.vibevoice.modular.modeling_vibevoice_streaming",
    "ComfyUI_VibeVoice.src.vibevoice.modular.modeling_vibevoice_streaming_inference",
    "ComfyUI_VibeVoice.src.vibevoice.modular.modular_vibevoice_diffusion_head",
    "ComfyUI_VibeVoice.src.vibevoice.modular.modular_vibevoice_text_tokenizer",
    "ComfyUI_VibeVoice.src.vibevoice.modular.modular_vibevoice_tokenizer",
    "ComfyUI_VibeVoice.src.vibevoice.modular.sage_attention_patch",
    "ComfyUI_VibeVoice.src.vibevoice.modular.streamer",
    "ComfyUI_VibeVoice.src.vibevoice.processor.audio_utils",
    "ComfyUI_VibeVoice.src.vibevoice.processor.vibevoice_asr_processor",
    "ComfyUI_VibeVoice.src.vibevoice.processor.vibevoice_processor",
    "ComfyUI_VibeVoice.src.vibevoice.processor.vibevoice_streaming_processor",
    "ComfyUI_VibeVoice.src.vibevoice.processor.vibevoice_tokenizer_processor",
    "ComfyUI_VibeVoice.src.vibevoice.schedule.dpm_solver",
    "ComfyUI_VibeVoice.src.vibevoice.schedule.timestep_sampler",
]
for _mod_name in _vibevoice_mock_modules:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

# ====================================================================
# 3. NATIVE ALIAS FOR THE CUSTOM NODE PACKAGE
# ====================================================================
# ComfyUI loads custom nodes by directory name. When the directory is
# "ComfyUI-VibeVoice" the import name becomes "ComfyUI-VibeVoice" which
# is not a valid Python identifier. We register it under a stable alias.
_PKG_ALIAS = "ComfyUI_VibeVoice"
if _PKG_ALIAS not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        _PKG_ALIAS,
        os.path.join(ROOT_DIR, "__init__.py"),
        submodule_search_locations=[ROOT_DIR],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[_PKG_ALIAS] = pkg
    # The __init__.py has a pytest guard that exits early, so this is safe.
    try:
        spec.loader.exec_module(pkg)
    except Exception:
        # If __init__.py fails (e.g. missing deps during collection),
        # the alias is still registered for submodule imports.
        pass

# Register submodules under the package alias so relative imports work
for _submod in ("modules", "nodes", "src"):
    _submod_path = os.path.join(ROOT_DIR, _submod)
    if os.path.isdir(_submod_path):
        _submod_alias = f"{_PKG_ALIAS}.{_submod}"
        if _submod_alias not in sys.modules:
            _submod_spec = importlib.util.spec_from_file_location(
                _submod_alias,
                os.path.join(_submod_path, "__init__.py"),
                submodule_search_locations=[_submod_path],
            )
            _submod_pkg = importlib.util.module_from_spec(_submod_spec)
            sys.modules[_submod_alias] = _submod_pkg
            try:
                _submod_spec.loader.exec_module(_submod_pkg)
            except Exception:
                pass


# ====================================================================
# 4. GLOBAL FIXTURES
# ====================================================================
@pytest.fixture
def mock_prompt_server():
    """Provide a shared PromptServer mock with reset send_sync."""
    _mock_server.PromptServer.instance = _mock_prompt_server_instance
    _mock_prompt_server_instance.send_sync.reset_mock()
    return _mock_prompt_server_instance


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary model directory mimicking models/tts/VibeVoice."""
    model_dir = tmp_path / "models" / "tts" / "VibeVoice"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


@pytest.fixture(autouse=True)
def comfyui_env():
    """Indicate that we are running in a ComfyUI test environment."""
    return True


# ====================================================================
# 5. BEHAVIORAL PATCHER STUB (NTH-003)
# ====================================================================
# A tiny torch.nn.Module stand-in for the real VibeVoice handler so the
# patcher's device transitions and cache keying can be exercised in CI
# without loading gigabytes of weights. Shared with CRIT-001 / NTH-004 tests.
class _TinyHandler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # AUD-010: mirror the real VibeVoiceModelHandler — the heavy model is
        # lazily created by load_model(), not pre-set in __init__. This lets the
        # patcher's lazy-load branch (``if self.model.model is None``) be
        # exercised by the behavioral tests.
        self.model = None
        self.processor = object()
        self.model_pack_name = "tiny"
        self.cache_key = "tiny"
        self.size = 1024

    def load_model(self, device, attention_mode: str = "sdpa"):
        # Mirror the real handler: lazy instantiation + move onto target device.
        if self.model is None:
            self.model = torch.nn.Linear(8, 8)
        self.model.to(device)


@pytest.fixture
def tiny_handler():
    """A fresh tiny stub handler (no patcher)."""
    return _TinyHandler()


@pytest.fixture
def tiny_patcher(tiny_handler):
    """A VibeVoicePatcher wrapping the tiny stub handler, ModelPatcher.__init__ mocked."""
    from ComfyUI_VibeVoice.modules.patcher import VibeVoicePatcher

    with patch("comfy.model_patcher.ModelPatcher.__init__"):
        patcher = VibeVoicePatcher(
            tiny_handler,
            attention_mode="sdpa",
            load_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            size=1024,
        )
    # Attributes ModelPatcher.__init__ would normally set.
    patcher.load_device = torch.device("cpu")
    patcher.offload_device = torch.device("cpu")
    patcher.model = tiny_handler
    patcher.pinned = set()  # ModelPatcher.__del__ → unpin_all_weights() needs this
    return patcher


# ====================================================================
# 6. SYNTHETIC GGUF FIXTURES (quant-resident runtime)
# ====================================================================
# Spec-driven builders producing real .gguf files via gguf.GGUFWriter so the
# loader/planner/modules are exercised against genuine container bytes.
# K-quants cannot be produced by gguf-py (dequantize-only), so Q4_K/Q5_K/Q6_K
# blocks are handcrafted with controlled fp16 scales — valid blocks whose
# dequantized values are finite and bounded (usable by forward-parity tests).

_GGUF_BLOCK_SHAPES = {
    "Q8_0": (32, 34),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q6_K": (256, 210),
}


def craft_kquant_blocks(qtype_name: str, n_elements: int, seed: int = 0):
    """Handcraft one flat uint8 array of raw K-quant/Q8_0 blocks.

    Scales are small sane fp16 values and quanta bounded, keeping
    dequantized magnitudes well inside float range.
    """
    import numpy as np

    block_size, type_size = _GGUF_BLOCK_SHAPES[qtype_name]
    n_blocks = n_elements // block_size
    rng = np.random.default_rng(seed)
    b = np.zeros((n_blocks, type_size), dtype=np.uint8)

    def _fp16_bytes(values):
        return np.asarray(values, dtype=np.float16).view(np.uint8).reshape(-1)

    d = 0.03 + np.abs(rng.standard_normal(n_blocks)) * 0.02
    if qtype_name == "Q8_0":
        qs = rng.integers(0, 256, size=(n_blocks, 32), dtype=np.uint8)
        b[:, 0:2] = _fp16_bytes(d).reshape(n_blocks, 2)
        b[:, 2:] = qs
    elif qtype_name in ("Q4_K", "Q5_K"):
        dmin = 0.0005 + np.abs(rng.standard_normal(n_blocks)) * 0.001
        b[:, 0:2] = _fp16_bytes(d).reshape(n_blocks, 2)
        b[:, 2:4] = _fp16_bytes(dmin).reshape(n_blocks, 2)
        b[:, 4:16] = rng.integers(0, 64, size=(n_blocks, 12), dtype=np.uint8)
        if qtype_name == "Q4_K":
            b[:, 16:] = rng.integers(0, 256, size=(n_blocks, 128), dtype=np.uint8)
        else:
            b[:, 16:48] = rng.integers(0, 256, size=(n_blocks, 32), dtype=np.uint8)
            b[:, 48:] = rng.integers(0, 256, size=(n_blocks, 128), dtype=np.uint8)
    elif qtype_name == "Q6_K":
        # scales int8 near zero; keep |scale| small so products stay finite
        b[:, 0:128] = rng.integers(0, 256, size=(n_blocks, 128), dtype=np.uint8)
        b[:, 128:192] = rng.integers(0, 256, size=(n_blocks, 64), dtype=np.uint8)
        b[:, 192:208] = rng.integers(60, 68, size=(n_blocks, 16)).astype(np.uint8)
        b[:, 208:210] = _fp16_bytes(
            0.01 + np.abs(rng.standard_normal(n_blocks)) * 0.01
        ).reshape(n_blocks, 2)
    else:
        raise ValueError(f"craft_kquant_blocks: unsupported type {qtype_name}")
    return b


def write_synthetic_gguf(path, spec, seed: int = 0):
    """Write a .gguf file from ``spec`` and return ``path``.

    Args:
        path: Target file path (str or pathlib.Path).
        spec: Iterable of ``(name, qtype_str_or_float, shape)`` tuples where
            shape is the TORCH-logical shape (out, in) for weights.
            Float types: 'F32' / 'F16' / 'BF16'. Quant types: 'Q8_0' /
            'Q4_K' / 'Q5_K' / 'Q6_K'.
        seed: Deterministic content seed.

    Returns:
        The written path (as given).
    """
    import numpy as np
    import gguf as _gguf
    from gguf.constants import GGMLQuantizationType as T

    writer = _gguf.GGUFWriter(str(path), "vibevoice")
    try:
        for i, (name, kind, shape) in enumerate(spec):
            n_elem = int(np.prod(shape))
            if kind == "F32":
                rng = np.random.default_rng(seed * 1000 + i)
                writer.add_tensor(name, (rng.standard_normal(shape) * 0.05).astype(np.float32),
                                  raw_dtype=T.F32)
            elif kind == "F16":
                rng = np.random.default_rng(seed * 1000 + i)
                writer.add_tensor(name, (rng.standard_normal(shape) * 0.05).astype(np.float16),
                                  raw_dtype=T.F16)
            elif kind == "BF16":
                rng = np.random.default_rng(seed * 1000 + i)
                f32 = (rng.standard_normal(shape) * 0.05).astype(np.float32)
                u16 = (f32.view(np.uint32) >> 16).astype(np.uint16)
                rows = shape[0]
                writer.add_tensor(name, u16.view(np.uint8).reshape(rows, -1),
                                  raw_dtype=T.BF16)
            elif kind in _GGUF_BLOCK_SHAPES:
                block_size, type_size = _GGUF_BLOCK_SHAPES[kind]
                if n_elem % block_size != 0 or shape[-1] % block_size != 0:
                    raise ValueError(
                        f"{kind} requires last dim multiple of {block_size}: {shape}"
                        )
                bytes_flat = craft_kquant_blocks(kind, n_elem, seed=seed * 1000 + i)
                rows = shape[0]
                bytes_per_row = shape[-1] // block_size * type_size
                writer.add_tensor(name, bytes_flat.reshape(rows, bytes_per_row),
                                  raw_dtype=getattr(T, kind))
            else:
                raise ValueError(f"write_synthetic_gguf: unknown kind {kind!r}")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    return path


@pytest.fixture
def make_gguf_file(tmp_path):
    """Factory fixture: write_synthetic_gguf(tmp_path/'<tag>.gguf', spec)."""
    def _make(spec, tag="model", seed=0):
        return write_synthetic_gguf(tmp_path / f"{tag}.gguf", spec, seed=seed)
    return _make


def build_stub_vv(n_layers: int = 1, hidden: int = 64, ffn: int = 128,
                  vocab: int = 96, seed: int = 0):
    """Build a tiny real-parameter module tree mirroring VibeVoice topology.

    Module paths match the HF pass-through naming seen in real checkpoints:
        model.language_model.layers.N.{self_attn.{q,k,v,o}_proj,
                                        mlp.{gate,up,down}_proj,
                                        input_layernorm, post_attention_layernorm}
        model.language_model.embed_tokens / .norm
        lm_head (untied here for simplicity)
        model.prediction_head.cond_proj / final_layer.linear

    Returns an eager (non-meta) nn.Module in eval mode.
    """
    import torch.nn as nn

    class _Lin(nn.Linear):
        def __init__(self, i, o):
            super().__init__(i, o, bias=False)

    class _SelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = _Lin(hidden, hidden)
            self.k_proj = _Lin(hidden, hidden // 4)
            self.v_proj = _Lin(hidden, hidden // 4)
            self.o_proj = _Lin(hidden, hidden)

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = _Lin(hidden, ffn)
            self.up_proj = _Lin(hidden, ffn)
            self.down_proj = _Lin(ffn, hidden)

    class _Norm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden))

        def forward(self, x):
            # RMSNorm-shaped residual-scale (matches real model semantics
            # closely enough for pipeline tests).
            return x * self.weight

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _SelfAttn()
            self.mlp = _MLP()
            self.input_layernorm = _Norm()
            self.post_attention_layernorm = _Norm()

    class _LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab, hidden)
            self.layers = nn.ModuleList([_Layer() for _ in range(n_layers)])
            self.norm = _Norm()

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = _LM()
            self.prediction_head = nn.Module()
            self.prediction_head.cond_proj = _Lin(hidden, hidden)
            self.prediction_head.final_layer = nn.Module()
            self.prediction_head.final_layer.linear = _Lin(hidden, hidden // 4)
            self.speech_scaling_factor = nn.Parameter(torch.tensor(float("nan")))
            self.speech_bias_factor = nn.Parameter(torch.tensor(float("nan")))

    class _VV(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Model()
            self.lm_head = _Lin(hidden, vocab)

    torch.manual_seed(seed)
    m = _VV().eval()

    # Give every linear distinct float32 weights so parity checks are strict.
    g = torch.Generator().manual_seed(seed)
    for name, p in m.named_parameters():
        if p.dtype.is_floating_point:
            p.data = torch.randn(p.shape, generator=g) * 0.02
    m.model.speech_scaling_factor.data.fill_(float("nan"))
    m.model.speech_bias_factor.data.fill_(float("nan"))
    return m


def stub_vv_gguf_spec(n_layers: int = 1, hidden: int = 64, ffn: int = 128,
                      vocab: int = 96, qtype: str = "Q8_0"):
    """GGUF writer spec for :func:`build_stub_vv` linears (HF naming).

    Quantizable linears use ``qtype``; everything else stays float
    (F32 norms / BF16 embeddings + remaining linears), mirroring the real
    vibevoice-1.5b-q8_0.gguf layout.
    """
    spec = [
        ("model.language_model.embed_tokens.weight", "BF16", (vocab, hidden)),
        ("model.language_model.norm.weight", "F32", (hidden,)),
    ]
    for i in range(n_layers):
        p = f"model.language_model.layers.{i}"
        spec += [
            (f"{p}.input_layernorm.weight", "F32", (hidden,)),
            (f"{p}.post_attention_layernorm.weight", "F32", (hidden,)),
            (f"{p}.self_attn.q_proj.weight", qtype, (hidden, hidden)),
            (f"{p}.self_attn.k_proj.weight", qtype, (hidden // 4, hidden)),
            (f"{p}.self_attn.v_proj.weight", qtype, (hidden // 4, hidden)),
            (f"{p}.self_attn.o_proj.weight", qtype, (hidden, hidden)),
            (f"{p}.mlp.gate_proj.weight", qtype, (ffn, hidden)),
            (f"{p}.mlp.up_proj.weight", qtype, (ffn, hidden)),
            (f"{p}.mlp.down_proj.weight", qtype, (hidden, ffn)),
        ]
    spec += [
        ("model.prediction_head.cond_proj.weight", "BF16", (hidden, hidden)),
        ("model.prediction_head.final_layer.linear.weight", "F32", (hidden // 4, hidden)),
        ("lm_head.weight", "BF16", (vocab, hidden)),
    ]
    return spec


@pytest.fixture
def make_stub_vv():
    return build_stub_vv
