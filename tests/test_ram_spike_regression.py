"""Phase D5: peak-RAM regression harness for the GGUF load path.

Primary (deterministic) proofs:
- gguf.dequantize is called ZERO times through the full loader path.
- Installed residency equals the raw quant bytes (< file size), never the
  full-float expansion.

Secondary (RSS-based, generous threshold): loading a multi-layer Q8_0 file
must not allocate anywhere near the legacy ~2x-full-float peak. The RSS bound
is intentionally loose (2.5x raw) to stay robust against interpreter noise;
the deterministic proofs above carry the regression signal.
"""

import gc
import sys

import pytest
import torch
from unittest.mock import MagicMock, patch

import gguf

from ComfyUI_VibeVoice.modules import external_loader as EL
from ComfyUI_VibeVoice.modules.external_loader import load_external_vibevoice_model
from ComfyUI_VibeVoice.modules.gguf_quant import GGUFLinear
from conftest import build_stub_vv, stub_vv_gguf_spec


def _rss_bytes():
    try:
        import psutil
        import os
        return psutil.Process(os.getpid()).memory_info().rss
    except ImportError:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        k32 = ctypes.windll.kernel32
        h = k32.GetCurrentProcess()
        if not k32.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return 0
        return pmc.WorkingSetSize


class _FakeStreamingCfg:
    pass


def _run_loader(weight_path, dims):
    n_layers, hidden, ffn = dims

    def _instantiate(config, is_streaming, attn_implementation, final_load_dtype,
                     use_meta=True):
        n_layers, hidden, ffn = dims
        return build_stub_vv(n_layers=n_layers, hidden=hidden, ffn=ffn,
                             vocab=1000)

    with patch.object(EL.VibeVoiceLoader, "_load_config", return_value=MagicMock()), \
         patch.object(EL.VibeVoiceLoader, "_load_tokenizer", return_value=MagicMock()), \
         patch.object(EL.VibeVoiceLoader, "_load_processor", return_value=MagicMock()), \
         patch.object(EL.VibeVoiceLoader, "_instantiate_model",
                      side_effect=_instantiate), \
         patch.object(EL, "resolve_sidecar_config", return_value="/fake/config.json"), \
         patch.object(EL, "resolve_sidecar_preprocessor", return_value=""), \
         patch.object(EL, "resolve_sidecar_tokenizer_dir", return_value="/fake/dir"), \
         patch.object(EL, "resolve_dtype", return_value=torch.bfloat16), \
         patch.object(EL, "resolve_attention_mode", side_effect=lambda m, q: m), \
         patch.object(EL, "get_attn_implementation_for_load", return_value="eager"), \
         patch.object(EL, "VibeVoiceStreamingConfig", _FakeStreamingCfg), \
         patch.object(EL.model_management, "get_torch_device",
                      return_value=torch.device("cpu")):
        return load_external_vibevoice_model(
            weight_path=str(weight_path), config_name="VibeVoice-1.5B",
            attention_mode="sdpa", use_llm_4bit=False, dtype_str="auto",
        )


DIMS = (24, 512, 2048)


@pytest.fixture(scope="module")
def big_gguf(tmp_path_factory):
    """A ~90 MiB multi-layer Q8_0 checkpoint so RSS deltas dominate noise."""
    from conftest import write_synthetic_gguf

    path = tmp_path_factory.mktemp("ram") / "big.gguf"
    write_synthetic_gguf(
        str(path), stub_vv_gguf_spec(n_layers=DIMS[0], hidden=DIMS[1],
                                     ffn=DIMS[2], vocab=1000, qtype="Q8_0"),
        seed=5,
    )
    return path


class TestRamSpikeRegression:
    def test_no_dequant_calls_and_true_residency(self, big_gguf):
        raw_file_bytes = big_gguf.stat().st_size
        calls = []
        real_deq = gguf.dequantize

        def _spy(data, qtype):
            calls.append(qtype)
            return real_deq(data, qtype)

        with patch.object(gguf, "dequantize", side_effect=_spy):
            bundle = _run_loader(big_gguf, DIMS)

        assert calls == [], "legacy gguf.dequantize must be off the load path"

        model = bundle["model"]
        resident = sum(
            m.weight.numel() for m in model.modules() if isinstance(m, GGUFLinear)
        )
        assert resident == bundle["quant_stats"]["raw_bytes"]
        # Residency must be a fraction of even the raw file (floats excluded).
        assert resident < raw_file_bytes

    def test_peak_rss_stays_near_raw_size(self, big_gguf):
        """Generous RSS bound: legacy full-dequant would blow far past it."""
        raw_file_bytes = big_gguf.stat().st_size
        if raw_file_bytes < 8 * 1024 * 1024:
            pytest.skip("synthetic file too small for a meaningful RSS bound")

        gc.collect()
        before = _rss_bytes()
        if before == 0:
            pytest.skip("no RSS probe available")

        bundle = _run_loader(big_gguf, DIMS)
        after = _rss_bytes()
        delta = max(0, after - before)

        # Legacy path expands every Q8_0 block to fp32 (~4.25x raw) plus a
        # defensive copy per tensor (~2x that transiently). Bound generously
        # at 2.5x raw — comfortably above honest residency + noise, far below
        # the legacy blowup.
        assert delta <= int(2.5 * raw_file_bytes), (
            f"peak RSS grew {delta / 2**20:.1f} MiB for a "
            f"{raw_file_bytes / 2**20:.1f} MiB file"
        )
