"""Device-flow tests for the VibeVoice load path.

Plan: docs/plans/2026-08-15-vibevoice-load-device-flow-fix.md

Phases covered here:
- P0.1: device-ledger test double (records .to()/load_state_dict traffic
  without moving real tensors — no CUDA hardware required).
- P0.2: reproduction of the CURRENT defective flow (DF-001/DF-002):
  state dict loaded straight to CUDA, then GPU->CPU copy via
  load_state_dict, then CPU->GPU move. These tests document the defect
  and pass against the pre-fix code; they are replaced by the inverted
  contract tests in Phase 1.

Determinism: no network, no real GPU. Device traffic is asserted via the
ledger double; ``torch.device("cuda")`` objects are used freely (they can
be constructed without CUDA hardware).
"""

import contextlib

import torch
import pytest
from unittest.mock import patch, MagicMock

from ComfyUI_VibeVoice.modules.loader import (
    VibeVoiceLoader,
    VibeVoiceModelHandler,
    LOADED_MODELS_CACHE,
)

CUDA = torch.device("cuda")
CPU = torch.device("cpu")


# ====================================================================
# Device-ledger test double
# ====================================================================
def _parse_to_call(args, kwargs):
    """Parse ``nn.Module.to(*args, **kwargs)`` into ``(device, dtype)``.

    Supports the call forms used by the production code:
    ``to(device)``, ``to(dtype)``, ``to(device, dtype)``,
    ``to(device=..., dtype=...)``. No CUDA hardware needed.
    """
    device = kwargs.get("device", None)
    dtype = kwargs.get("dtype", None)
    for a in args:
        if isinstance(a, torch.device):
            device = a
        elif isinstance(a, torch.dtype):
            dtype = a
        elif isinstance(a, str):
            device = torch.device(a)
        elif isinstance(a, (tuple, list)) and len(a) == 2:
            device, dtype = a
    if device is not None and not isinstance(device, torch.device):
        device = torch.device(device)
    return device, dtype


class _LedgerModel(torch.nn.Module):
    """Fake model that records device/dtype traffic instead of moving tensors.

    Every ``.to()`` / ``load_state_dict()`` / ``eval()`` call is appended to
    the shared ``ledger`` list as a tuple:
    - ``("to", device_or_None, dtype_or_None)``
    - ``("load_state_dict", strict, assign)``
    - ``("eval", None, None)``
    """

    def __init__(self, ledger):
        super().__init__()
        self._ledger = ledger
        self._device = CPU
        self._dtype = torch.float32
        # Real parameter so the streaming assign (_stream_apply_dense) has a
        # target for the ("w", ...) tensors the tests stream in.
        self.w = torch.nn.Parameter(torch.zeros(1))

    def to(self, *args, **kwargs):
        device, dtype = _parse_to_call(args, kwargs)
        self._ledger.append(("to", device, dtype))
        if device is not None:
            self._device = device
        if dtype is not None:
            self._dtype = dtype
        return self

    def load_state_dict(self, state_dict, strict=True, assign=False, **kwargs):
        self._ledger.append(("load_state_dict", strict, assign))
        return ([], [])

    def eval(self):
        self._ledger.append(("eval", None, None))
        return self

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype


@pytest.fixture
def ledger():
    """Fresh device ledger for one test."""
    return []


@pytest.fixture
def ledger_model(ledger):
    """A fresh ledger-backed fake model (starts on CPU, fp32)."""
    return _LedgerModel(ledger)


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Isolate LOADED_MODELS_CACHE between tests."""
    LOADED_MODELS_CACHE.clear()
    yield
    LOADED_MODELS_CACHE.clear()


# ====================================================================
# Loader seam patching helper
# ====================================================================
@contextlib.contextmanager
def _patched_loader(ledger_model, registry=None):
    """Patch every loader seam EXCEPT ``_load_state_dict_into_model``.

    The state-dict loading path is the unit under test, so it stays real;
    its own seams (``_resolve_checkpoint_path``, the streaming tensor
    iterators ``iter_checkpoint_tensors`` / ``iter_sharded_tensors``) are
    patched by the individual tests.
    """

    class _FakeStreamingCfg:  # isinstance() target must be a real class
        pass

    registry = registry or {
        "TestModel": {"type": "official", "repo_id": "test/repo", "path": "x"}
    }
    with patch("ComfyUI_VibeVoice.modules.loader.AVAILABLE_VIBEVOICE_MODELS", registry), \
         patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceStreamingConfig", _FakeStreamingCfg), \
         patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._resolve_model_paths",
               return_value=("mp", "cp", "pp", "td")), \
         patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._load_config",
               return_value=MagicMock(spec=[])), \
         patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._load_tokenizer",
               return_value=MagicMock()), \
         patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._load_processor",
               return_value=MagicMock()), \
         patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._instantiate_model",
               return_value=ledger_model):
        yield


# ====================================================================
# P0.1 — Ledger harness self-tests
# ====================================================================
class TestP01LedgerHarness:
    """The device-ledger double must faithfully record traffic."""

    def test_ledger_starts_empty(self, ledger, ledger_model):
        assert ledger == []
        assert ledger_model.device == CPU

    def test_records_positional_device(self, ledger, ledger_model):
        ledger_model.to(CUDA)
        assert ledger == [("to", CUDA, None)]
        assert ledger_model.device == CUDA

    def test_records_string_device(self, ledger, ledger_model):
        ledger_model.to("cuda")
        assert ledger[0][0] == "to"
        assert ledger[0][1] == CUDA

    def test_records_dtype_only(self, ledger, ledger_model):
        ledger_model.to(dtype=torch.bfloat16)
        assert ledger == [("to", None, torch.bfloat16)]
        assert ledger_model.device == CPU  # device unchanged
        assert ledger_model.dtype == torch.bfloat16

    def test_records_device_and_dtype_kwargs(self, ledger, ledger_model):
        ledger_model.to(device=CUDA, dtype=torch.float16)
        assert ledger == [("to", CUDA, torch.float16)]
        assert ledger_model.device == CUDA
        assert ledger_model.dtype == torch.float16

    def test_records_device_dtype_pair(self, ledger, ledger_model):
        ledger_model.to((CUDA, torch.bfloat16))
        assert ledger == [("to", CUDA, torch.bfloat16)]

    def test_records_load_state_dict(self, ledger, ledger_model):
        result = ledger_model.load_state_dict({"w": torch.zeros(1)}, strict=False)
        assert result == ([], [])
        assert ledger == [("load_state_dict", False, False)]

    def test_records_eval(self, ledger, ledger_model):
        assert ledger_model.eval() is ledger_model
        assert ledger == [("eval", None, None)]

    def test_to_returns_self_for_chaining(self, ledger, ledger_model):
        assert ledger_model.to(CUDA).to(dtype=torch.float16) is ledger_model
        assert len(ledger) == 2


# ====================================================================
# P1.1/P1.2 — State dict is ALWAYS loaded onto CPU (DF-001/DF-002 fix)
# ====================================================================
class TestP1StateDictLoadedToCpu:
    """Contract: checkpoints are read to CPU regardless of target device."""

    @pytest.mark.parametrize("model_type", ["official", "local_dir", "standalone"])
    def test_state_dict_loaded_to_cpu_single_file(self, ledger, ledger_model, model_type):
        registry = {"TestModel": {"type": model_type, "repo_id": "test/repo", "path": "x"}}
        with _patched_loader(ledger_model, registry=registry), \
             patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._resolve_checkpoint_path",
                   return_value=("fake.safetensors", False)), \
             patch("ComfyUI_VibeVoice.modules.base_loader.BaseVibeVoiceLoader.iter_checkpoint_tensors",
                   return_value=iter([("w", torch.zeros(1))])) as mock_iter:
            VibeVoiceLoader.load_model(
                "TestModel", CUDA, attention_mode="sdpa", dtype_str="bf16"
            )

        mock_iter.assert_called_once_with("fake.safetensors")
        # Streaming iterator contract: tensors always arrive on CPU (the
        # iterator has no device parameter); the patcher owns the single
        # H2D transfer after ComfyUI's VRAM arbitration.
        assert ledger_model.w.device.type == "cpu"

    def test_state_dict_loaded_to_cpu_sharded(self, ledger, ledger_model):
        """Sharded path: the streaming shard iterator feeds the assign loop."""
        captured = {}

        def fake_iter(model_dir):
            captured["model_dir"] = model_dir
            yield "w", torch.zeros(1)

        with _patched_loader(ledger_model), \
             patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._resolve_checkpoint_path",
                   return_value=("model.safetensors.index.json", True)), \
             patch("ComfyUI_VibeVoice.modules.base_loader.BaseVibeVoiceLoader.iter_sharded_tensors",
                   side_effect=fake_iter) as mock_iter:
            VibeVoiceLoader.load_model(
                "TestModel", CUDA, attention_mode="sdpa", dtype_str="bf16"
            )

        mock_iter.assert_called_once()
        assert captured["model_dir"] == "mp"
        # Shards stream onto CPU; the merged-dict device arg is gone.
        assert ledger_model.w.device.type == "cpu"

    def test_load_state_dict_into_model_ignores_device_arg(self, ledger, ledger_model):
        """Direct unit test of _load_state_dict_into_model: device arg is reserved."""
        with patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._resolve_checkpoint_path",
                   return_value=("fake.safetensors", False)), \
             patch("ComfyUI_VibeVoice.modules.base_loader.BaseVibeVoiceLoader.iter_checkpoint_tensors",
                   return_value=iter([("w", torch.zeros(1))])):
            VibeVoiceLoader._load_state_dict_into_model(
                model=ledger_model,
                model_path="mp",
                model_type="official",
                model_info={"type": "official"},
                device=CUDA,  # must be ignored for placement
            )
        # Streaming iterator yields CPU tensors unconditionally.
        assert ledger_model.w.device.type == "cpu"


# ====================================================================
# P1.3 — Loader returns a CPU model in the final dtype (DF-002/DF-003 fix)
# ====================================================================
class TestP1LoaderReturnsCpuModel:
    """Contract: the loader performs ZERO GPU operations."""

    def test_loader_returns_cpu_model(self, ledger, ledger_model):
        with _patched_loader(ledger_model), \
             patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._resolve_checkpoint_path",
                   return_value=("fake.safetensors", False)), \
             patch("ComfyUI_VibeVoice.modules.base_loader.BaseVibeVoiceLoader.iter_checkpoint_tensors",
                   return_value=iter([("w", torch.zeros(1))])):
            model, _ = VibeVoiceLoader.load_model(
                "TestModel", CUDA, attention_mode="sdpa", dtype_str="bf16"
            )

        assert model is ledger_model
        # No CUDA traffic anywhere in the loader.
        cuda_moves = [
            e for e in ledger
            if e[0] == "to" and e[1] is not None and e[1].type == "cuda"
        ]
        assert cuda_moves == [], f"loader must not touch CUDA, ledger={ledger}"
        assert ledger_model.device == CPU

    def test_loader_applies_dtype_on_cpu(self, ledger, ledger_model):
        """Plan 2026-08-18 D4/RC-3: final dtype applied via the conditional
        cast helper (not an unconditional ``.to()``), and never on CUDA."""
        cast_calls = []

        def fake_cast(model, dtype):
            cast_calls.append(dtype)
            model._dtype = dtype  # simulate the in-place cast on the double

        with _patched_loader(ledger_model), \
             patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._resolve_checkpoint_path",
                   return_value=("fake.safetensors", False)), \
             patch("ComfyUI_VibeVoice.modules.base_loader.BaseVibeVoiceLoader.iter_checkpoint_tensors",
                   return_value=iter([("w", torch.zeros(1))])), \
             patch("ComfyUI_VibeVoice.modules.loader.cast_model_to_dtype_if_needed",
                   side_effect=fake_cast):
            VibeVoiceLoader.load_model(
                "TestModel", CUDA, attention_mode="sdpa", dtype_str="fp16"
            )

        assert cast_calls == [torch.float16], (
            "loader must apply the final dtype via the conditional cast helper")
        # The dtype cast must not be combined with a CUDA device move.
        cuda_moves = [
            e for e in ledger
            if e[0] == "to" and e[1] is not None and e[1].type == "cuda"
        ]
        assert cuda_moves == [], f"dtype cast must happen on CPU, got {ledger}"
        assert ledger_model.dtype == torch.float16

    def test_4bit_quantization_runs_on_cpu(self, ledger, ledger_model):
        """bnb replacement must run while the model is still CPU-resident."""
        try:
            import transformers.integrations.bitsandbytes  # noqa: F401
            _bnb_target = "transformers.integrations.bitsandbytes.replace_with_bnb_linear"
        except ImportError:
            _bnb_target = "transformers.utils.bitsandbytes.replace_with_bnb_linear"

        def fake_replace(model, quantization_config=None, modules_to_not_convert=None):
            # At quantization time the model must still be on CPU.
            assert model.device == CPU, "4-bit quantization must run on CPU"
            assert not any(
                e[0] == "to" and e[1] is not None and e[1].type == "cuda"
                for e in ledger
            ), "no CUDA traffic before quantization"

        with _patched_loader(ledger_model), \
             patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._resolve_checkpoint_path",
                   return_value=("fake.safetensors", False)), \
             patch("ComfyUI_VibeVoice.modules.base_loader.BaseVibeVoiceLoader.iter_checkpoint_tensors",
                   return_value=iter([("w", torch.zeros(1))])), \
             patch("ComfyUI_VibeVoice.modules.loader.BitsAndBytesConfig"), \
             patch(_bnb_target, side_effect=fake_replace) as mock_replace:
            VibeVoiceLoader.load_model(
                "TestModel", CUDA, attention_mode="sdpa",
                use_llm_4bit=True, dtype_str="bf16",
            )
        mock_replace.assert_called_once()


# ====================================================================
# P1.4 — Handler performs NO device move (DF-003, part 1)
# ====================================================================
class TestP1HandlerNoDeviceMove:
    """Contract: VibeVoiceModelHandler.load_model never moves the model."""

    def test_handler_load_model_does_not_move_device(self, ledger, ledger_model):
        handler = VibeVoiceModelHandler("VibeVoice-1.5B", attention_mode="sdpa")
        with patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader.load_model",
                   return_value=(ledger_model, MagicMock())):
            handler.load_model(CUDA, attention_mode="sdpa")

        assert handler.model is ledger_model
        assert ledger == [], f"handler must not move the model, ledger={ledger}"
        assert ledger_model.device == CPU


# ====================================================================
# P1.5 — Patcher owns the SINGLE host-to-device transfer (DF-003, part 2)
# ====================================================================
class _LedgerHandler(torch.nn.Module):
    """Handler double mirroring the FIXED VibeVoiceModelHandler.

    ``load_model`` simulates the CPU-only loader: it (optionally) records a
    ``load_state_dict`` on the ledger model and assigns it — with NO device
    move, exactly like the production handler after the DF-003 fix.
    """

    def __init__(self, ledger, model=None, simulate_loader=True):
        super().__init__()
        self._ledger = ledger
        self._pending_model = model
        self._simulate_loader = simulate_loader
        self.model = None
        self.processor = object()
        self.model_pack_name = "ledger"
        self.cache_key = "ledger_test"
        self.size = 1024
        self.load_model_calls = []

    def load_model(self, device, attention_mode="sdpa"):
        self.load_model_calls.append(device)
        if self._pending_model is not None:
            if self._simulate_loader:
                self._pending_model.load_state_dict({"w": torch.zeros(1)}, strict=False)
            self.model = self._pending_model


def _make_ledger_patcher(handler, target_dtype=None):
    """Build a VibeVoicePatcher with ModelPatcher.__init__/patch_model mocked."""
    from ComfyUI_VibeVoice.modules.patcher import VibeVoicePatcher

    with patch("comfy.model_patcher.ModelPatcher.__init__"):
        patcher = VibeVoicePatcher(
            handler,
            attention_mode="sdpa",
            dtype=target_dtype,
            load_device=CUDA,
            offload_device=CPU,
            size=1024,
        )
    patcher.load_device = CUDA
    patcher.offload_device = CPU
    patcher.model = handler
    patcher.size = getattr(handler, "size", 1024)
    # Attributes ModelPatcher.__init__ would normally set (avoids __del__ noise).
    patcher.pinned = {}
    patcher.is_injected = False
    patcher.model_options = {"transformer_options": {}}
    patcher.callbacks = {}
    return patcher


class TestP1PatcherSingleH2D:
    """Contract (plan 2026-08-18, D6/RC-5): the patcher performs NO device
    move of its own. The single host-to-device transfer is owned by
    ``super().patch_model()`` → ``ModelPatcher.load()``. Because super is
    mocked in these tests, the ledger must record ZERO cuda moves from the
    patcher's own code; the transfer is asserted via the super() call args.
    """

    def test_patcher_single_h2d_transfer_cold_load(self, ledger, ledger_model):
        handler = _LedgerHandler(ledger, model=ledger_model)
        patcher = _make_ledger_patcher(handler)

        with patch("comfy.model_patcher.ModelPatcher.patch_model") as mock_super:
            patcher.patch_model(device_to=CUDA)

        mock_super.assert_called_once()
        # Loader ran once (cold load), model assigned on CPU.
        assert len(handler.load_model_calls) == 1
        kinds = [e[0] for e in ledger]
        assert "load_state_dict" in kinds
        # D6: the patcher itself performs NO cuda move — the H2D is owned by
        # super().patch_model() → load() (mocked here).
        cuda_moves = [
            e for e in ledger
            if e[0] == "to" and e[1] is not None and e[1].type == "cuda"
        ]
        assert len(cuda_moves) == 0, f"patcher must not bulk-move, ledger={ledger}"
        # The transfer target is delegated to super with the right device.
        assert mock_super.call_args.kwargs["device_to"] == CUDA

    def test_patcher_warm_reattach_single_h2d(self, ledger, ledger_model):
        handler = _LedgerHandler(ledger, model=None, simulate_loader=False)
        handler.model = ledger_model  # already loaded (warm state)
        patcher = _make_ledger_patcher(handler)
        patcher._warm_offloaded = True  # warm-offloaded state

        with patch("comfy.model_patcher.ModelPatcher.patch_model") as mock_super:
            patcher.patch_model(device_to=CUDA)

        # No re-load from disk.
        assert handler.load_model_calls == []
        # D6: no cuda move from the patcher's own code (delegated to super).
        cuda_moves = [
            e for e in ledger
            if e[0] == "to" and e[1] is not None and e[1].type == "cuda"
        ]
        assert len(cuda_moves) == 0, f"ledger={ledger}"
        assert patcher._warm_offloaded is False
        assert mock_super.call_args.kwargs["device_to"] == CUDA

    def test_no_gpu_cpu_gpu_round_trip(self, ledger, ledger_model):
        """Full cycle: cold load -> warm offload -> re-attach.

        D6: within the load cycle the patcher records NO device move at all
        (the H2D is delegated to super). The warm-offload CPU move belongs to
        the offload cycle, not the load, so no GPU->CPU->GPU round-trip can
        appear in the load cycle.
        """
        handler = _LedgerHandler(ledger, model=ledger_model)
        patcher = _make_ledger_patcher(handler)

        with patch("comfy.model_patcher.ModelPatcher.patch_model"), \
             patch("comfy.model_patcher.ModelPatcher.unpatch_model"):
            # Cycle 1: cold load
            patcher.patch_model(device_to=CUDA)
            load_cycle_1 = list(ledger)
            # Warm offload (tensors retained on intermediate/CPU device)
            patcher.unpatch_model(device_to=CPU, unpatch_weights=True, warm=True)

        devices = [
            e[1].type for e in load_cycle_1
            if e[0] == "to" and e[1] is not None
        ]
        # D6: the load cycle contains NO patcher-owned device move.
        assert "cuda" not in devices, (
            f"patcher must not bulk-move during load: {load_cycle_1}"
        )
        # The warm offload itself moved the model off-GPU exactly once.
        offload_moves = [
            e for e in ledger[len(load_cycle_1):]
            if e[0] == "to" and e[1] is not None and e[1].type == "cpu"
        ]
        assert len(offload_moves) == 1

    def test_patcher_uses_load_device_when_device_to_none(self, ledger, ledger_model):
        handler = _LedgerHandler(ledger, model=None, simulate_loader=False)
        handler.model = ledger_model
        patcher = _make_ledger_patcher(handler)

        with patch("comfy.model_patcher.ModelPatcher.patch_model") as mock_super:
            patcher.patch_model(device_to=None)

        # D6: no cuda move from the patcher's own code.
        cuda_moves = [
            e for e in ledger
            if e[0] == "to" and e[1] is not None and e[1].type == "cuda"
        ]
        assert len(cuda_moves) == 0
        # device_to=None falls back to load_device and is delegated to super.
        assert mock_super.call_args.kwargs["device_to"] == CUDA


# ====================================================================
# P2.1 — User dtype is threaded into the handler (DF-004 / AUD-008)
# ====================================================================
class TestP2DtypeThreading:
    """Contract: node dtype -> handler.dtype_str -> loader dtype_str."""

    def test_handler_init_stores_dtype_str(self):
        handler = VibeVoiceModelHandler("VibeVoice-1.5B", dtype_str="fp16")
        assert handler.dtype_str == "fp16"

    def test_handler_init_defaults_dtype_str_auto(self):
        handler = VibeVoiceModelHandler("VibeVoice-1.5B")
        assert handler.dtype_str == "auto"

    def test_generation_passes_dtype_str_to_handler(self):
        """load_vibevoice_model must forward the node's dtype to the handler."""
        from ComfyUI_VibeVoice.modules import generation as gen

        captured = {}

        class _HandlerSpy:
            def __init__(self, model_name, attention_mode="eager",
                         use_llm_4bit=False, dtype_str="auto"):
                captured["dtype_str"] = dtype_str
                captured["model_name"] = model_name
                self.model = MagicMock()
                self.processor = MagicMock()
                self.size = 1024
                self.cache_key = "spy"

        with patch.object(gen, "VIBEVOICE_PATCHER_CACHE", {}), \
             patch.object(gen, "VibeVoiceModelHandler", _HandlerSpy), \
             patch.object(gen, "VibeVoicePatcher") as mock_patcher_cls, \
             patch.object(gen.model_management, "load_model_gpu"):
            mock_patcher = MagicMock()
            mock_patcher.model.model = MagicMock()
            mock_patcher.model.processor = MagicMock()
            mock_patcher_cls.return_value = mock_patcher

            gen.load_vibevoice_model(
                "VibeVoice-1.5B", device="cuda", dtype="fp16",
                attention_mode="sdpa",
            )

        assert captured["dtype_str"] == "fp16"
        # The patcher still receives the resolved torch dtype (unchanged).
        assert mock_patcher_cls.call_args.kwargs["dtype"] == torch.float16

    def test_handler_forwards_dtype_str_to_loader(self):
        """P2.2: handler.load_model passes dtype_str into VibeVoiceLoader."""
        handler = VibeVoiceModelHandler("VibeVoice-1.5B", dtype_str="bf16")
        with patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader.load_model",
                   return_value=(MagicMock(), MagicMock())) as mock_load:
            handler.load_model(CUDA, attention_mode="sdpa")

        mock_load.assert_called_once()
        assert mock_load.call_args.kwargs.get("dtype_str") == "bf16"


# ====================================================================
# P2.3 — Patcher dtype cast is a mismatch-only guard (DF-004)
# ====================================================================
class TestP2PatcherCastGuard:
    """Contract: cast only when the model dtype differs from target."""

    def test_patcher_skips_cast_when_dtype_matches(self, ledger, ledger_model):
        ledger_model._dtype = torch.float16
        # The cast guard walks REAL parameter dtypes (representative_dtype),
        # so the ledger's parameter must match the target dtype too.
        ledger_model.w = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16))
        handler = _LedgerHandler(ledger, model=None, simulate_loader=False)
        handler.model = ledger_model
        patcher = _make_ledger_patcher(handler, target_dtype=torch.float16)

        with patch("comfy.model_patcher.ModelPatcher.patch_model"), \
             patch("ComfyUI_VibeVoice.modules.patcher.cast_model_to_dtype") as mock_cast:
            patcher.patch_model(device_to=CUDA)

        mock_cast.assert_not_called()

    def test_patcher_casts_when_dtype_differs(self, ledger, ledger_model):
        ledger_model._dtype = torch.float32
        handler = _LedgerHandler(ledger, model=None, simulate_loader=False)
        handler.model = ledger_model
        patcher = _make_ledger_patcher(handler, target_dtype=torch.float16)

        with patch("comfy.model_patcher.ModelPatcher.patch_model"), \
             patch("ComfyUI_VibeVoice.modules.patcher.cast_model_to_dtype") as mock_cast:
            patcher.patch_model(device_to=CUDA)

        mock_cast.assert_called_once()
        assert mock_cast.call_args.args[1] == torch.float16

    def test_patcher_no_cast_when_target_dtype_none(self, ledger, ledger_model):
        handler = _LedgerHandler(ledger, model=None, simulate_loader=False)
        handler.model = ledger_model
        patcher = _make_ledger_patcher(handler, target_dtype=None)

        with patch("comfy.model_patcher.ModelPatcher.patch_model"), \
             patch("ComfyUI_VibeVoice.modules.patcher.cast_model_to_dtype") as mock_cast:
            patcher.patch_model(device_to=CUDA)

        mock_cast.assert_not_called()


# ====================================================================
# P3 — assign=True decision lock (DF-005: DEFERRED)
# ====================================================================
class _TiedTinyModel(torch.nn.Module):
    """Minimal model with a tied lm_head (mirrors VibeVoice-1.5B).

    VibeVoice-1.5B config has ``tie_word_embeddings: true`` and
    ``_tied_weights_keys = ["lm_head.weight"]``; the checkpoint does NOT
    contain ``lm_head.weight``.
    """

    def __init__(self, vocab=4, dim=4):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        self.lm_head = torch.nn.Linear(dim, vocab, bias=False)
        self.lm_head.weight = self.embed.weight  # tie

    @property
    def dtype(self):
        return self.embed.weight.dtype


class TestP3AssignWithRetie:
    """DF-005 un-deferred (plan 2026-08-18, D2): assign=True + re-tie.

    With ``tie_word_embeddings=true`` the checkpoint omits ``lm_head.weight``.
    ``assign=True`` replaces the embedding's parameter object and breaks the
    tie — so the loader re-invokes ``tie_weights()`` after loading (the same
    mitigation transformers' ``from_pretrained`` applies). This eliminates the
    copy pass and halves peak host RAM (RC-2).
    """

    def test_copy_semantics_preserves_tying(self):
        """Documentation: copy semantics keeps tying (old behavior)."""
        model = _TiedTinyModel()
        sd = {"embed.weight": torch.ones(4, 4)}
        model.load_state_dict(sd, strict=False)  # copy semantics
        assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()
        assert torch.equal(model.lm_head.weight, torch.ones(4, 4))

    def test_assign_breaks_tying_without_retie(self):
        """Documentation: the risk that assign=True alone would cause."""
        model = _TiedTinyModel()
        sd = {"embed.weight": torch.ones(4, 4)}
        model.load_state_dict(sd, strict=False, assign=True)
        # assign=True replaced embed.weight with the sd tensor object;
        # lm_head.weight still references the OLD random parameter.
        assert model.lm_head.weight.data_ptr() != model.embed.weight.data_ptr()
        assert not torch.equal(model.lm_head.weight, torch.ones(4, 4))

    def test_loader_uses_assign_semantics(self, ledger):
        """The production loader MUST assign per-tensor (D2, streaming form).

        Plan 2026-08-28: the batch ``load_state_dict(assign=True)`` call was
        replaced by ``_stream_apply_dense`` (per-tensor ``set_attr_param`` of
        a private clone). The contract is unchanged: parameter objects are
        REPLACED with the checkpoint values, never copied into.
        """
        ledger_model = _LedgerModel(ledger)
        with _patched_loader(ledger_model), \
             patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._resolve_checkpoint_path",
                   return_value=("fake.safetensors", False)), \
             patch("ComfyUI_VibeVoice.modules.base_loader.BaseVibeVoiceLoader.iter_checkpoint_tensors",
                   return_value=iter([("w", torch.full((1,), 3.0))])):
            VibeVoiceLoader.load_model(
                "TestModel", CUDA, attention_mode="sdpa", dtype_str="bf16"
            )

        assert not any(e[0] == "load_state_dict" for e in ledger), (
            "streaming assign must not route through load_state_dict"
        )
        # The parameter object now holds the checkpoint value (assign).
        assert float(ledger_model.w.item()) == 3.0

    def test_loader_reties_after_assign(self, ledger):
        """The loader must re-invoke tie_weights() after assign load (D2)."""
        ledger_model = _LedgerModel(ledger)
        ledger_model.tie_weights = MagicMock()
        # Config gate: tied
        ledger_model.config = MagicMock()
        ledger_model.config.decoder_config.tie_word_embeddings = True
        ledger_model.config.tie_word_embeddings = False

        with _patched_loader(ledger_model), \
             patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._resolve_checkpoint_path",
                   return_value=("fake.safetensors", False)), \
             patch("ComfyUI_VibeVoice.modules.base_loader.BaseVibeVoiceLoader.iter_checkpoint_tensors",
                   # bf16 = target dtype, so the conditional cast takes its
                   # fast path and the only tie comes from the assign fixups.
                   return_value=iter([("w", torch.zeros(1, dtype=torch.bfloat16))])):
            VibeVoiceLoader.load_model(
                "TestModel", CUDA, attention_mode="sdpa", dtype_str="bf16"
            )

        ledger_model.tie_weights.assert_called_once()
        # tie_weights runs in the post-assign fixups, which execute only
        # after the assign loop has delivered the weights.
        assert hasattr(ledger_model, "w")


# ====================================================================
# P4 — VRAM arbitration correctness (DF-006)
# ====================================================================
class TestP4VramArbitration:
    """Contract: ComfyUI can arbitrate VRAM BEFORE any GPU traffic.

    ``load_models_gpu`` computes ``total_memory_required`` from
    ``patcher.model_size()`` (the handler's static ``size`` estimate) and
    calls ``free_memory()`` BEFORE ``model_load`` → ``patch_model``. With
    the loader now CPU-only, no untracked VRAM spike exists inside the
    lazy load.
    """

    def test_model_size_reported_before_load(self, ledger):
        """model_size() > 0 prior to any patch_model call (static estimate)."""
        handler = _LedgerHandler(ledger, model=None, simulate_loader=False)
        handler.size = int(3.0 * (1024**3))
        patcher = _make_ledger_patcher(handler)

        # No load has happened yet — the estimate must still be available
        # so load_models_gpu can arbitrate VRAM up front.
        assert handler.model is None
        assert patcher.model_size() == handler.size
        assert patcher.model_size() > 0

    def test_loaded_size_tracking_after_load(self, ledger):
        """After a full load, ModelPatcher.load tracks loaded weight memory.

        Reuses the AUD-002 weight-visibility pattern: once the handler's
        inner model is assigned, its parameters are visible to
        ``_load_list()`` and ``ModelPatcher.load`` can set
        ``model_loaded_weight_memory``.
        """
        import comfy.model_patcher

        class _RealInner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)

        class _RealHandler(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = None
                self.processor = object()
                self.model_pack_name = "p4"
                self.cache_key = "p4"
                self.size = 1024

            def load_model(self, device, attention_mode="sdpa"):
                # FIXED contract: build on CPU, NO device move.
                if self.model is None:
                    self.model = _RealInner()

        handler = _RealHandler()
        # REAL ModelPatcher.__init__ (CPU devices) so patcher.load() works.
        patcher = VibeVoicePatcherForP4(handler)

        # Simulate the ComfyUI sequence: lazy load (CPU-only) then load().
        handler.load_model(CPU)
        assert patcher.loaded_size() == 0

        # ModelPatcher.load moves weights and sets model_loaded_weight_memory.
        patcher.load(device_to=CPU, lowvram_model_memory=0,
                     force_patch_weights=False, full_load=True)
        assert handler.model_loaded_weight_memory > 0
        assert patcher.loaded_size() == handler.model_loaded_weight_memory

    def test_no_untracked_vram_during_lazy_load(self, ledger, ledger_model):
        """The lazy load itself must perform zero CUDA operations.

        DF-006: the step-2 VRAM spike existed because the lazy load touched
        CUDA. With the fix, the entire handler.load_model → loader path is
        CPU-only, so ComfyUI's free_memory() arbitration (which runs before
        patch_model) sees all GPU traffic.
        """
        handler = VibeVoiceModelHandler("VibeVoice-1.5B", attention_mode="sdpa")
        registry = {"VibeVoice-1.5B": {"type": "official", "repo_id": "test/repo", "path": "x"}}
        with _patched_loader(ledger_model, registry=registry), \
             patch("ComfyUI_VibeVoice.modules.loader.VibeVoiceLoader._resolve_checkpoint_path",
                   return_value=("fake.safetensors", False)), \
             patch("ComfyUI_VibeVoice.modules.base_loader.BaseVibeVoiceLoader.iter_checkpoint_tensors",
                   return_value=iter([("w", torch.zeros(1))])):
            handler.load_model(CUDA, attention_mode="sdpa")

        assert ledger_model.w.device.type == "cpu"
        assert ledger_model.device == CPU
        assert not any(
            e[0] == "to" and e[1] is not None and e[1].type == "cuda"
            for e in ledger
        ), f"lazy load touched CUDA: {ledger}"


def VibeVoicePatcherForP4(handler):
    """Build a patcher with a REAL ModelPatcher.__init__ (CPU devices)."""
    from ComfyUI_VibeVoice.modules.patcher import VibeVoicePatcher
    return VibeVoicePatcher(
        handler,
        attention_mode="sdpa",
        load_device=CPU,
        offload_device=CPU,
        size=1024,
    )


# ====================================================================
# P4.2 — [GPU-OPTIONAL] real peak-VRAM measurement
# ====================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestP42GpuPeakVram:
    """[GPU-OPTIONAL] Verify peak VRAM ≈ 1× model size during patch_model.

    Runs the full patcher flow with a real (small) model on CUDA and
    asserts the peak allocation is bounded by ~1× the model weight size —
    NOT ~2× (which the pre-fix disk→VRAM→RAM→VRAM round-trip caused).
    """

    def test_vram_peak_single_model_size(self, ledger):
        class _RealInner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # ~4 MB of weights
                self.linear = torch.nn.Linear(1024, 1024)

        class _RealHandler(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = None
                self.processor = object()
                self.model_pack_name = "gpu"
                self.cache_key = "gpu"
                self.size = 1024 * 1024 * 4

            def load_model(self, device, attention_mode="sdpa"):
                if self.model is None:
                    self.model = _RealInner()  # CPU-only, per the fix

        handler = _RealHandler()
        patcher = _make_ledger_patcher(handler)
        patcher.load_device = CUDA
        patcher.pinned = set()

        # ``max_memory_allocated`` is a *process-wide* peak, and
        # ``reset_peak_memory_stats`` only resets the counter to whatever is
        # already resident — it does not free anything. A real checkpoint left
        # on the device by an earlier test (the opt-in RUN_VIBEVOICE_E2E suite
        # keeps one) therefore lands in the peak and this 4 MB model is measured
        # as 2.7 GB. Subtract the resident baseline so the number is this
        # model's own peak, which is what the assertion is about.
        torch.cuda.synchronize()
        resident_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with patch("comfy.model_patcher.ModelPatcher.patch_model"):
            patcher.patch_model(device_to=CUDA)
        torch.cuda.synchronize()

        model_bytes = sum(
            p.numel() * p.element_size() for p in handler.model.parameters()
        )
        peak = torch.cuda.max_memory_allocated() - resident_before
        # Peak must be bounded by ~1.5× model size (single transfer + slack),
        # never ~2× (state-dict + model simultaneously on GPU).
        assert peak <= int(model_bytes * 1.5), (
            f"peak VRAM {peak} exceeds 1.5x model size {model_bytes} "
            f"(resident baseline {resident_before} subtracted)"
        )
        # Cleanup
        handler.model.to(CPU)
        torch.cuda.empty_cache()
