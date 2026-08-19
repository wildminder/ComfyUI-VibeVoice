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
