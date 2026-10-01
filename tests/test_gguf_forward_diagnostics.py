"""Regression tests for WHERE the GGUF forward counters get reported.

WHY this file exists: the counters themselves were right, but they were read
inside ``load_external_vibevoice_model`` (modules/external_loader.py) — i.e.
BEFORE any forward pass. On the first load in a process the line therefore
always read ``gguf_forward_fast=0 gguf_forward_streamed=0``, which is exactly
the value a reader takes as "the weight_function hook path is not poisoning
this run". The instrument was structurally guaranteed to produce the answer
it was mandated to measure, and no production code ever zeroed the cumulative
totals between loads.

The corrected contract pinned here:

1. each external load RESETS the counters, so a run's numbers describe that
   run and not the whole process history;
2. the load-time diagnostics line reports load facts only — it no longer
   prints a fast/streamed split that cannot mean anything yet;
3. the generation paths REPORT the counters after forwards have run.
"""

import inspect
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from gguf.constants import GGMLQuantizationType as T
from gguf.quants import quantize as oracle_quantize

from ComfyUI_VibeVoice.modules import gguf_quant as G
from ComfyUI_VibeVoice.modules.external_loader import _log_load_diagnostics


def _resident_gguf_linear(out_f=64, in_f=32):
    """A CPU GGUFLinear with real Q8_0 raw blocks installed."""
    lin = G.GGUFLinear(in_f, out_f, bias=False, ggml_type=T.Q8_0)
    rng = np.random.default_rng(11)
    raw = oracle_quantize(
        (rng.standard_normal((out_f, in_f)) * 0.05).astype(np.float32), T.Q8_0
    )
    lin.set_raw_weight(
        torch.from_numpy(np.ascontiguousarray(raw)).view(torch.uint8).reshape(-1)
    )
    return lin


def _patch_model_management():
    """Real torch.device from the mocked runtime module (a bare MagicMock
    breaks ``Tensor.to()`` — generate_audio uses get_torch_device())."""
    mm = MagicMock()
    mm.get_torch_device.return_value = torch.device("cpu")
    return patch("ComfyUI_VibeVoice.modules.generation.model_management", mm)


def _bump_both_counters():
    """One resident forward + one hook-forced (streamed) forward."""
    G.reset_gguf_forward_counters()
    lin = _resident_gguf_linear()
    x = torch.randn(4, 32, dtype=torch.bfloat16)
    lin(x)
    lin.weight_function = [lambda t: t]
    lin(x)
    assert G.gguf_forward_counters() == {"fast": 1, "streamed": 1}


class TestLoadResetsCounterScope:
    """(1) the per-load counter window actually exists."""

    def test_tts_loader_resets_counters(self, tmp_path):
        from ComfyUI_VibeVoice.modules.external_loader import (
            load_external_vibevoice_model,
        )

        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"not a real checkpoint")
        _bump_both_counters()

        def _stop(weight_path, config_name):
            raise RuntimeError("stop after the entry bookkeeping")

        with patch(
            "ComfyUI_VibeVoice.modules.external_loader._load_weight_state_dict",
            return_value={},
        ), patch(
            "ComfyUI_VibeVoice.modules.external_loader.resolve_sidecar_config",
            side_effect=_stop,
        ):
            with pytest.raises(RuntimeError, match="stop after the entry"):
                load_external_vibevoice_model(
                    weight_path=str(weights), config_name="VibeVoice-7B"
                )

        assert G.gguf_forward_counters() == {"fast": 0, "streamed": 0}, (
            "a fresh load must zero the counters, or the post-run line "
            "attributes an earlier model's forwards to this one"
        )

    def test_asr_loader_resets_counters(self, tmp_path):
        from ComfyUI_VibeVoice.modules.external_loader import (
            load_external_vibevoice_asr_model,
        )

        weights = tmp_path / "asr.safetensors"
        weights.write_bytes(b"not a real checkpoint")
        _bump_both_counters()

        def _stop(weight_path, config_name):
            raise RuntimeError("stop after the entry bookkeeping")

        with patch(
            "ComfyUI_VibeVoice.modules.external_loader._load_weight_state_dict",
            return_value={},
        ), patch(
            "ComfyUI_VibeVoice.modules.external_loader.resolve_sidecar_config",
            side_effect=_stop,
        ):
            with pytest.raises(RuntimeError, match="stop after the entry"):
                load_external_vibevoice_asr_model(
                    weight_path=str(weights), config_name="VibeVoice-ASR"
                )

        assert G.gguf_forward_counters() == {"fast": 0, "streamed": 0}


class TestLoadLineNoLongerFakesACounter:
    """(2) the load-time line must not carry a structurally-zero readout."""

    def test_load_diagnostics_line_omits_forward_counters(self, caplog, monkeypatch):
        monkeypatch.setenv("VIBEVOICE_DIAGNOSTICS", "1")
        _bump_both_counters()
        with caplog.at_level(logging.INFO, logger="ComfyUI_VibeVoice.modules.external_loader"):
            _log_load_diagnostics(
                config_name="VibeVoice-7B",
                requested_attention_mode="eager",
                resolved_attention_mode="eager",
                weight_family="gguf",
                load_device=torch.device("cpu"),
            )
        line = "\n".join(r.getMessage() for r in caplog.records)
        assert "gguf_forward" not in line, (
            "forward counters read at LOAD time are always 0 and read as "
            "'the hook path is not firing'; they are reported after generation"
        )
        # The load facts the line exists for are still there.
        assert "VibeVoice-7B" in line and "resolved_attention=eager" in line


class TestPostForwardReport:
    """(3) the real readout, taken where it can actually be non-zero."""

    def test_reports_counts_after_forwards(self, caplog, monkeypatch):
        monkeypatch.setenv("VIBEVOICE_DIAGNOSTICS", "1")
        _bump_both_counters()
        with caplog.at_level(logging.INFO, logger="ComfyUI_VibeVoice.modules.gguf_quant"):
            G.log_gguf_forward_counters("tts_generate")
        line = "\n".join(r.getMessage() for r in caplog.records)
        assert "gguf_forward_fast=1" in line
        assert "gguf_forward_streamed=1" in line
        assert "tts_generate" in line

    def test_silent_in_production(self, caplog, monkeypatch):
        """With no env var set the counters must not reach the console."""
        monkeypatch.delenv("VIBEVOICE_DIAGNOSTICS", raising=False)
        _bump_both_counters()
        with caplog.at_level(logging.INFO, logger="ComfyUI_VibeVoice.modules.gguf_quant"):
            G.log_gguf_forward_counters("tts_generate")
        assert not [r for r in caplog.records
                    if "GGUF forward diagnostics" in r.getMessage()]

    def test_silent_when_no_gguf_forward_happened(self, caplog):
        """A non-GGUF model must not get a meaningless line per generation."""
        G.reset_gguf_forward_counters()
        with caplog.at_level(logging.INFO, logger="ComfyUI_VibeVoice.modules.gguf_quant"):
            G.log_gguf_forward_counters("tts_generate")
        assert not [r for r in caplog.records if "GGUF forward diagnostics" in r.getMessage()]

    def test_tts_generate_reports_after_the_forwards(self):
        from ComfyUI_VibeVoice.modules.generation import generate_audio

        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        out = MagicMock()
        out.speech_outputs = [torch.randn(24000)]
        mock_model.generate.return_value = out

        proc = MagicMock()
        proc.return_value = {"input_ids": torch.randint(0, 100, (1, 10))}
        proc.tokenizer = MagicMock()

        with patch("ComfyUI_VibeVoice.modules.generation.ProgressBarWithConsole"), \
             _patch_model_management(), \
             patch("ComfyUI_VibeVoice.modules.generation.log_gguf_forward_counters") as report, \
             patch("ComfyUI_VibeVoice.modules.generation.preprocess_comfy_audio",
                   return_value=np.random.randn(24000).astype(np.float32)):
            generate_audio(
                model=mock_model,
                processor=proc,
                text="[1] Hello world",
                voice_samples=[{"waveform": torch.randn(1, 1, 24000),
                                "sample_rate": 24000}],
                speaker_ids=[1],
            )

        assert report.call_count == 1, (
            "generate_audio must report the forward counters, otherwise the "
            "user's slow GGUF run has no usable evidence in the log"
        )
        assert report.call_args[0][0] == "tts_generate"

    @pytest.mark.parametrize(
        "module_name, func_names",
        [
            ("asr_generation", ("_transcribe_streaming", "_transcribe_native",
                                "transcribe_audio")),
            ("realtime_generation", ("generate_realtime_audio",)),
        ],
    )
    def test_other_generation_paths_are_wired(self, module_name, func_names):
        """Every path that runs model forwards reports the counters."""
        mod = __import__(
            f"ComfyUI_VibeVoice.modules.{module_name}", fromlist=[func_names[0]]
        )
        assert hasattr(mod, "log_gguf_forward_counters"), (
            f"{module_name} must import log_gguf_forward_counters"
        )
        for name in func_names:
            assert "log_gguf_forward_counters(" in inspect.getsource(
                getattr(mod, name)
            ), f"{module_name}.{name} never reports the forward counters"
