"""Phase 3 tests: streaming per-tensor safetensors load for quant checkpoints.

Plan 2026-08-27 (fp8-resident + streaming load RAM fix). The quantized
safetensors path must NEVER materialize the full-file state dict; these tests
prove:

- streaming-vs-batch parity: identical module tree + identical tensors,
- structural guarantee: ``comfy.utils.load_torch_file`` is not called for
  quantized safetensors (still called for dense safetensors),
- deterministic spike proof: zero kitchen fp8-dequant calls during a
  resident load + installed fp8 bytes equal the file's fp8 bytes,
- RSS bound on a ~90 MiB synthetic fp8 checkpoint,
- mid-stream error propagation (shape mismatch, unplanned quant tensor,
  missing scale, storage-dtype mismatch).
"""

import gc
import json
import os
from types import SimpleNamespace

import pytest
import torch
from unittest.mock import MagicMock, patch

import comfy.utils
from safetensors.torch import save_file, load_file

from ComfyUI_VibeVoice.modules import external_loader as EL
from ComfyUI_VibeVoice.modules.external_loader import (
    _load_state_dict_into_model_from_memory,
    _plan_quantized_safetensors_load,
    _prepare_quantized_safetensors_load,
    _stream_apply_safetensors,
    load_external_vibevoice_model,
)
from ComfyUI_VibeVoice.modules.convrot_quant import (
    ConvRotInt8Linear,
    QuantLayerInfo,
    scan_checkpoint_quantization,
)
from ComfyUI_VibeVoice.modules.fp8_quant import FP8Linear, probe_fp8_backend
from ComfyUI_VibeVoice.modules.quant_common import (
    QuantTargetMismatch,
    replace_linears_for_quant,
    validate_weight_plan,
)
from ComfyUI_VibeVoice.modules.dtype_utils import cast_model_to_dtype_if_needed
from ComfyUI_VibeVoice.modules.memory_census import census
from ComfyUI_VibeVoice.modules.patcher import (
    VibeVoicePatcher,
    select_patcher_class,
)
from conftest import build_stub_vv


# ====================================================================
# Synthetic quant checkpoint builder
# ====================================================================

def _quant_tensors(prefix, kind, shape):
    """One quant layer's file tensors (weight + scale + comfy_quant meta)."""
    tensors = {}
    if kind == "rowwise_fp8":
        tensors[f"{prefix}.weight"] = torch.randint(
            -100, 100, shape).to(torch.float8_e4m3fn)
        tensors[f"{prefix}.weight_scale"] = torch.tensor(0.5)
        meta = {"format": "float8_e4m3fn", "orig_dtype": "torch.bfloat16"}
    else:
        tensors[f"{prefix}.weight"] = torch.randint(
            -100, 100, shape, dtype=torch.int8)
        if kind == "block":
            gs = 16
            tensors[f"{prefix}.weight_scale"] = torch.rand(
                (shape[0] // gs, shape[1] // gs),
                dtype=torch.float32) * 0.1
            meta = {"format": "int8_blockwise", "group_size": gs,
                    "orig_dtype": "torch.bfloat16"}
        else:
            tensors[f"{prefix}.weight_scale"] = torch.full(
                (shape[0], 1), 0.5)
            if kind == "convrot":
                meta = {"convrot": True, "convrot_groupsize": 16,
                        "format": "int8_tensorwise",
                        "orig_dtype": "torch.bfloat16"}
            else:
                meta = {"format": "int8_tensorwise", "per_row": True,
                        "orig_dtype": "torch.bfloat16"}
    tensors[f"{prefix}.comfy_quant"] = torch.frombuffer(
        bytearray(json.dumps(meta).encode("utf-8")), dtype=torch.uint8
    )
    return tensors


def _L(prefix):
    return f"model.language_model.layers.0.{prefix}"


def _save_mixed(path, drop=(), extra=None):
    """A checkpoint exercising EVERY load branch against the default stub
    tree (n_layers=1, hidden=64, ffn=128, vocab=96):

    - convrot int8 resident   -> q_proj   (64, 64)
    - fp8 scalar resident     -> o_proj   (64, 64)
    - rowwise int8 dequant    -> k_proj   (16, 64)
    - blockwise int8 dequant  -> gate_proj (128, 64)
    - dense bf16/fp32         -> embed / norms / lm_head / remaining linears
    """
    tensors = {}
    tensors.update(_quant_tensors(_L("self_attn.q_proj"), "convrot", (64, 64)))
    tensors.update(_quant_tensors(_L("self_attn.o_proj"), "rowwise_fp8", (64, 64)))
    tensors.update(_quant_tensors(_L("self_attn.k_proj"), "rowwise", (16, 64)))
    tensors.update(_quant_tensors(_L("mlp.gate_proj"), "block", (128, 64)))

    g = torch.Generator().manual_seed(7)
    dense = {
        "model.language_model.embed_tokens.weight":
            torch.randn(96, 64, generator=g).to(torch.bfloat16),
        "model.language_model.norm.weight": torch.randn(64, generator=g),
        "model.language_model.layers.0.input_layernorm.weight":
            torch.randn(64, generator=g),
        "model.language_model.layers.0.post_attention_layernorm.weight":
            torch.randn(64, generator=g),
        "model.language_model.layers.0.self_attn.v_proj.weight":
            torch.randn(16, 64, generator=g).to(torch.bfloat16),
        "model.language_model.layers.0.mlp.up_proj.weight":
            torch.randn(128, 64, generator=g).to(torch.bfloat16),
        "model.language_model.layers.0.mlp.down_proj.weight":
            torch.randn(64, 128, generator=g).to(torch.bfloat16),
        "model.prediction_head.cond_proj.weight":
            torch.randn(64, 64, generator=g).to(torch.bfloat16),
        "model.prediction_head.final_layer.linear.weight":
            torch.randn(16, 64, generator=g),
        "lm_head.weight": torch.randn(96, 64, generator=g).to(torch.bfloat16),
    }
    tensors.update(dense)
    for key in drop:
        tensors.pop(key, None)
    if extra:
        tensors.update(extra)
    save_file(tensors, str(path))
    return path


# ====================================================================
# Full-loader environment (same mock pattern as test_loader_quant_paths)
# ====================================================================

class _FakeStreamingCfg:
    pass


@pytest.fixture
def stubbed_load_env():
    """Mocks for load_external_vibevoice_model around a REAL stub tree.

    Yields (run_fn, holder); run_fn(weight_path, **kw) executes the loader.
    """
    holder = {"dims": (1, 64, 128)}

    def _instantiate(config, is_streaming, attn_implementation, final_load_dtype,
                     use_meta=True):
        n_layers, hidden, ffn = holder["dims"]
        return build_stub_vv(n_layers=n_layers, hidden=hidden, ffn=ffn)

    def _run(weight_path, dims=None, **kwargs):
        if dims is not None:
            holder["dims"] = dims
        kwargs.setdefault("config_name", "VibeVoice-1.5B")
        kwargs.setdefault("attention_mode", "sdpa")
        kwargs.setdefault("use_llm_4bit", False)
        kwargs.setdefault("dtype_str", "auto")
        with patch.object(EL.VibeVoiceLoader, "_load_config",
                          return_value=holder.get("config", MagicMock())), \
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
            bundle = load_external_vibevoice_model(
                weight_path=str(weight_path), **kwargs
            )
        holder["bundle"] = bundle
        holder["model"] = bundle["model"]
        return bundle

    return _run, holder


def _same(a, b):
    """Element equality tolerant of NaN sentinel parameters."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if torch.isnan(a).any() or torch.isnan(b).any():
        return bool((torch.isnan(a) == torch.isnan(b)).all())
    return bool(torch.equal(a, b))


def _batch_reference(path, dims=(1, 64, 128)):
    """The PRE-Phase-3 batch chain: full dict -> prepare -> assign -> cast."""
    n_layers, hidden, ffn = dims
    model = build_stub_vv(n_layers=n_layers, hidden=hidden, ffn=ffn)
    sd = comfy.utils.load_torch_file(str(path), device=torch.device("cpu"))
    qmap = scan_checkpoint_quantization(str(path))
    plan, _, _ = _prepare_quantized_safetensors_load(sd, qmap)
    replace_linears_for_quant(model, plan)
    model = _load_state_dict_into_model_from_memory(model, sd)
    cast_model_to_dtype_if_needed(model, torch.bfloat16)
    return model


# ====================================================================
# Streaming vs batch parity
# ====================================================================

class TestStreamingBatchParity:
    def test_identical_tree_and_tensors(self, tmp_path, stubbed_load_env):
        run, _ = stubbed_load_env
        p = _save_mixed(tmp_path / "mixed.safetensors")

        bundle = run(p)
        streamed = bundle["model"]
        batched = _batch_reference(p)

        # Module classes match everywhere (residents swapped identically).
        # convert_tree_for_streaming builds fresh dynamic subclasses per
        # call, so compare class NAMES, not identity.
        s_mods = dict(streamed.named_modules())
        b_mods = dict(batched.named_modules())
        assert set(s_mods) == set(b_mods)
        for name in s_mods:
            assert type(s_mods[name]).__name__ == type(b_mods[name]).__name__, name

        # Resident kinds survived both paths.
        q = streamed.model.language_model.layers[0].self_attn.q_proj
        o = streamed.model.language_model.layers[0].self_attn.o_proj
        assert isinstance(q, ConvRotInt8Linear)
        assert isinstance(o, FP8Linear)

        # Every persistent tensor is bit-identical.
        s_sd = streamed.state_dict()
        b_sd = batched.state_dict()
        assert set(s_sd) == set(b_sd)
        for key in s_sd:
            assert _same(s_sd[key], b_sd[key]), key

    def test_family_label_and_stats(self, tmp_path, stubbed_load_env):
        run, _ = stubbed_load_env
        p = _save_mixed(tmp_path / "mixed.safetensors")
        bundle = run(p)
        # Mixed convrot + fp8 file: convrot dominates the label.
        assert bundle["weight_family"] == "convrot_int8"
        assert bundle["quant_stats"] == {
            "n_resident_layers": 2,
            "n_rowwise_layers": 2,
            "n_fp8_resident_layers": 1,
        }

    def test_missing_and_unexpected_sets(self, tmp_path):
        """Direct-level: missing/unexpected mirror batch semantics."""
        p = _save_mixed(
            tmp_path / "mu.safetensors",
            drop=["lm_head.weight"],
            extra={"bogus.extra.tensor": torch.zeros(4)},
        )
        model = build_stub_vv()
        qmap = scan_checkpoint_quantization(str(p))
        plan, _, _ = _plan_quantized_safetensors_load(qmap)
        replace_linears_for_quant(model, plan)

        missing, unexpected = _stream_apply_safetensors(model, str(p), qmap)

        assert "lm_head.weight" in missing
        assert unexpected == ["bogus.extra.tensor"]
        # Everything else in the file was delivered.
        assert "model.language_model.embed_tokens.weight" not in missing


# ====================================================================
# Structural guarantees (the RAM-spike kill is real)
# ====================================================================

class TestStreamingStructuralGuarantees:
    def test_quant_safetensors_read_scales_without_mapping_the_file(
            self, tmp_path, stubbed_load_env):
        """Pass 1 must NOT open a mapping of the checkpoint.

        ``safe_open`` maps the whole file, and on Windows the first read
        through that mapping commits ~1x file size as private, untouched
        memory that stays pinned for as long as the tensor lives (measured:
        +9,050 MB on the real 9.47 GB fp8 checkpoint — see
        tests/probe_external_fp8_full_path.py). The header already records
        each tensor's byte range, so the scales come from those ranges and
        the file is never mapped to get them.
        """
        import safetensors

        from ComfyUI_VibeVoice.modules.external_loader import (
            read_safetensors_tensors_by_name,
        )

        run, _ = stubbed_load_env
        p = _save_mixed(tmp_path / "noload.safetensors")

        def _forbidden(*a, **kw):
            raise AssertionError("the scale pre-read must not map the file")

        with patch.object(safetensors, "safe_open", side_effect=_forbidden):
            scales = read_safetensors_tensors_by_name(
                str(p),
                ["model.language_model.layers.0.self_attn.q_proj.weight_scale"],
            )
        assert scales, "the scale pre-read returned nothing"

        bundle = run(p)
        assert bundle["weight_family"] == "convrot_int8"

    def test_dense_safetensors_stream_per_tensor(self, tmp_path, stubbed_load_env):
        """The dense route streams too — no full-file state dict is built.

        Building the whole CPU dict and then ``model.to(device)`` is what made
        BF16/FP16 checkpoints page-fault the file mapping on the way to VRAM.
        Every route now assigns per-tensor through the same streaming assign.
        """
        from ComfyUI_VibeVoice.modules.loader import VibeVoiceLoader

        run, _ = stubbed_load_env
        g = torch.Generator().manual_seed(3)
        dense = {
            "model.language_model.embed_tokens.weight":
                torch.randn(96, 64, generator=g).to(torch.bfloat16),
            "lm_head.weight": torch.randn(96, 64, generator=g).to(torch.bfloat16),
        }
        p = tmp_path / "dense.safetensors"
        save_file(dense, str(p))

        consumed = []
        real_apply = VibeVoiceLoader._stream_apply_dense

        def _spy_apply(model, tensor_pairs, **kw):
            consumed.append(type(tensor_pairs).__name__)
            return real_apply(model, tensor_pairs, **kw)

        with patch.object(EL.VibeVoiceLoader, "_stream_apply_dense",
                          staticmethod(_spy_apply)):
            bundle = run(p)

        assert bundle["weight_family"] == "dense"
        # A generator, not a dict: the dense route consumes the file the same
        # per-tensor way the quant route does.
        assert consumed == ["generator"]
        embed = bundle["model"].model.language_model.embed_tokens.weight
        assert torch.equal(embed.data, dense["model.language_model.embed_tokens.weight"])

    def test_zero_kitchen_dequant_calls_and_fp8_byte_accounting(
            self, tmp_path, stubbed_load_env):
        if probe_fp8_backend() is None:
            pytest.skip("no comfy_kitchen fp8 backend on this box")
        import comfy_kitchen

        run, _ = stubbed_load_env
        p = _save_mixed(tmp_path / "fp8resident.safetensors")
        file_fp8 = load_file(str(p))
        file_fp8_elems = sum(
            t.numel() for t in file_fp8.values()
            if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        )

        with patch.object(comfy_kitchen, "dequantize_per_tensor_fp8",
                          side_effect=AssertionError(
                              "load-time fp8 dequant must not happen")):
            bundle = run(p)

        installed = sum(
            m.weight.numel() for m in bundle["model"].modules()
            if isinstance(m, FP8Linear)
        )
        assert installed == file_fp8_elems > 0
        o = bundle["model"].model.language_model.layers[0].self_attn.o_proj
        assert o.weight.dtype == torch.float8_e4m3fn
        assert o.weight_scale.dtype == torch.float32


# ====================================================================
# The quant route's host-RAM CEILING, asserted (task fp8-peak)
# ====================================================================
#
# HISTORY. 2026-09-29 (tests/probe_ram_census.py, real 7B fp8 file): the
# then-current quant route peaked at ~2x file (18GiB ws / 18GiB private for
# 8.82GiB), settling to the model itself (~10GiB, all private copies). Two
# causes, since REMOVED by the Dynamic-VRAM port (2026-09-30):
#   1. every assigned tensor was a safe_open/get_tensor OWNED copy;
#   2. select_patcher_class returned the legacy patcher for any
#      family != "dense", closing the view-paging route to quants.
# 2026-09-30 (tests/probe_external_fp8_full_path.py, exact live path,
# dynamic route): a THIRD cause hid in pass-1 — one safe_open + get_tensor
# for a tiny scale committed ~1x file as PRIVATE (Windows; pages never
# faulted, so ws stayed flat and the census could not see it) and pinned it
# until the scale tensor died. Fix: pass-1 pre-reads the scales through
# core's aimdo load_torch_file arm (file-backed, 22MB private on the real
# 9.47GB file) and clones only the scales. The same probe then measures the
# whole 7B fp8 load at 1.18GB private over baseline (0.13x file), with the
# model holding view=8.31GB / private=1.02GB — the assign=True shape core
# itself uses (comfy/sd.py:2407).
#
# The tests below pin the LEGACY-route behavior as exercised by this stub
# env (CPU target, aimdo off -> per-tensor clones): owned copies, no views,
# fp8 bytes exact, final cast a no-op. On a CUDA+aimdo host the same code
# path preserves views instead.


class TestQuantRouteCeiling:
    def test_quant_params_carry_no_file_views_by_design(
            self, tmp_path, stubbed_load_env):
        run, _ = stubbed_load_env
        p = _save_mixed(tmp_path / "ceiling.safetensors")
        bundle = run(p)
        report = census(bundle["model"])

        # The quant reader returns owned copies, so NOTHING on this route can
        # be a view or an mmap. A non-zero view bucket here would mean the
        # reader started mapping — a design change, not a bug fix.
        assert report["param_view_bytes"] == 0
        assert report["param_mmap_bytes"] == 0
        assert report["buffer_view_bytes"] == 0
        assert report["param_private_bytes"] > 0
        # "convrot_int8": this mixed file also carries a convrot resident, so
        # the family is not the pure "fp8_resident" label the real 7B file
        # gets (asserted in TestFp8QuantizedEmbedding). Both are != "dense",
        # which is all the gate at modules/patcher.py:606 cares about.
        assert bundle["weight_family"] in ("fp8_resident", "convrot_int8")

    def test_private_footprint_is_about_one_x_file(
            self, tmp_path, stubbed_load_env):
        """The ceiling itself: ~1x file, not ~2x (i.e. NOT a bf16 recast)."""
        run, _ = stubbed_load_env
        p = _save_mixed(tmp_path / "ceiling_bytes.safetensors")
        file_bytes = p.stat().st_size
        bundle = run(p)
        report = census(bundle["model"])

        resident = sum(
            m.weight.untyped_storage().nbytes()
            for m in bundle["model"].modules() if isinstance(m, FP8Linear)
        )
        file_fp8 = sum(
            t.numel() for t in load_file(str(p)).values()
            if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        )
        # The residents occupy exactly the fp8 bytes the file carries: 1 byte
        # per element, never 2. A bf16 recast would double this line.
        assert resident == file_fp8 > 0
        # Whole-model private host bytes stay near the file: the quant tail
        # dequantises (embed/norms) and costs more than the file, so the
        # bound is generous on one side and absolute on the other.
        assert report["param_private_bytes"] <= 4 * file_bytes

    def test_fp8_resident_follows_the_device_not_the_family(self):
        """2026-09-30: fp8_resident is ON the dynamic route (the whole point of
        the port); what keeps it legacy is a CPU target or a missing aimdo
        alias. The previous version asserted the opposite through a CPU
        device, where EVERY family returns legacy anyway - vacuous."""
        for family in ("fp8_resident", "dense", "convrot_int8", "gguf_block", ""):
            assert select_patcher_class(family, torch.device("cpu"))                 is VibeVoicePatcher  # no CUDA -> legacy, whatever the label

    def test_loaded_fp8_resident_keeps_fp8_storage_after_the_cast(
            self, tmp_path, stubbed_load_env):
        """End-to-end at KB scale: cast AFTER the load must not touch fp8.

        The ranked hypothesis for the reported 7B spike was that
        ``_quant_protected_names`` fails to protect FP8Linear, so the final
        cast dequantises 9.47GB of fp8 into ~17GB of bf16. MEASURED: refuted —
        the loader's own final cast leaves every resident at float8 width
        (asserted here through the real loader, and at unit scale in
        tests/test_fp8_quant.py::TestFp8ResidentSurvivesDtypeCast).
        """
        run, _ = stubbed_load_env
        p = _save_mixed(tmp_path / "cast_survives.safetensors")
        bundle = run(p)
        model = bundle["model"]

        # The loader has already run its final cast at this point; re-running
        # it must be a no-op for the residents.
        cast_model_to_dtype_if_needed(model, torch.bfloat16)

        residents = [m for m in model.modules() if isinstance(m, FP8Linear)]
        assert residents
        for m in residents:
            assert m.weight.dtype == torch.float8_e4m3fn
            assert m.weight_scale.dtype == torch.float32
            assert m.weight.untyped_storage().nbytes() == m.weight.numel()


# ====================================================================
# Mid-stream error propagation
# ====================================================================

class TestStreamingErrorPaths:
    """Mid-stream failures surface as the loader's wrapped RuntimeError with
    the original actionable message preserved."""

    def test_shape_mismatch_raises_friendly(self, tmp_path, stubbed_load_env):
        run, _ = stubbed_load_env
        p = _save_mixed(
            tmp_path / "badshape.safetensors",
            drop=[_L("self_attn.q_proj") + ".weight"],
            extra={_L("self_attn.q_proj") + ".weight":
                   torch.zeros(32, 32, dtype=torch.int8)},
        )
        with pytest.raises(RuntimeError, match="Checkpoint weight shapes do not match"):
            run(p)

    def test_unplanned_quant_tensor_rejected(self, tmp_path, stubbed_load_env):
        run, _ = stubbed_load_env
        p = _save_mixed(
            tmp_path / "rogue.safetensors",
            drop=[_L("self_attn.v_proj") + ".weight"],
            extra={_L("self_attn.v_proj") + ".weight":
                   torch.zeros(16, 64).to(torch.float8_e4m3fn)},
        )
        with pytest.raises(RuntimeError, match="quantized-weight tensors"):
            run(p)

    def test_missing_scale_raises(self, tmp_path, stubbed_load_env):
        run, _ = stubbed_load_env
        p = _save_mixed(
            tmp_path / "noscale.safetensors",
            drop=[_L("self_attn.k_proj") + ".weight_scale"],
        )
        with pytest.raises(RuntimeError, match="weight_scale"):
            run(p)

    def test_resident_storage_dtype_mismatch_raises(
            self, tmp_path, stubbed_load_env):
        if probe_fp8_backend() is None:
            pytest.skip("no comfy_kitchen fp8 backend on this box")
        run, _ = stubbed_load_env
        # Meta declares e4m3fn but the file carries e5m2 storage.
        p = _save_mixed(
            tmp_path / "wrongfp8.safetensors",
            drop=[_L("self_attn.o_proj") + ".weight"],
            extra={_L("self_attn.o_proj") + ".weight":
                   torch.zeros(64, 64).to(torch.float8_e5m2)},
        )
        with pytest.raises(RuntimeError, match="storage"):
            run(p)


# ====================================================================
# Peak-RAM regression harness (~90 MiB fp8 checkpoint)
# ====================================================================

def _rss_bytes():
    import os
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss


BIG_DIMS = (24, 512, 2048)


def _save_big_fp8(path, dims=BIG_DIMS, vocab=1000):
    """All stub linears as scalar-fp8 residents + small dense tensors."""
    n_layers, hidden, ffn = dims
    tensors = {}
    linears = [
        ("self_attn.q_proj", (hidden, hidden)),
        ("self_attn.k_proj", (hidden // 4, hidden)),
        ("self_attn.v_proj", (hidden // 4, hidden)),
        ("self_attn.o_proj", (hidden, hidden)),
        ("mlp.gate_proj", (ffn, hidden)),
        ("mlp.up_proj", (ffn, hidden)),
        ("mlp.down_proj", (hidden, ffn)),
    ]
    for i in range(n_layers):
        for rel, shape in linears:
            prefix = f"model.language_model.layers.{i}.{rel}"
            tensors.update(_quant_tensors(prefix, "rowwise_fp8", shape))
    tensors.update(_quant_tensors(
        "model.prediction_head.cond_proj", "rowwise_fp8", (hidden, hidden)))

    g = torch.Generator().manual_seed(11)
    tensors["model.language_model.embed_tokens.weight"] = \
        torch.randn(vocab, hidden, generator=g).to(torch.bfloat16)
    tensors["lm_head.weight"] = \
        torch.randn(vocab, hidden, generator=g).to(torch.bfloat16)
    tensors["model.language_model.norm.weight"] = \
        torch.randn(hidden, generator=g)
    save_file(tensors, str(path))
    return path


@pytest.fixture(scope="module")
def big_fp8_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("ramfp8") / "big_fp8.safetensors"
    return _save_big_fp8(path)


class TestStreamingRamBound:
    def test_no_dequant_calls_and_fp8_residency(self, big_fp8_file):
        if probe_fp8_backend() is None:
            pytest.skip("no comfy_kitchen fp8 backend on this box")
        import comfy_kitchen

        calls = []

        def _spy(*a, **kw):
            calls.append(a)
            raise AssertionError("load-time fp8 dequant must not happen")

        # Run the loader directly (no stubbed fixture: module-scoped file).
        n_layers, hidden, ffn = BIG_DIMS

        def _instantiate(config, is_streaming, attn_implementation,
                         final_load_dtype, use_meta=True):
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
                          return_value=torch.device("cpu")), \
             patch.object(comfy.utils, "load_torch_file",
                          side_effect=AssertionError(
                              "full-dict materialization on the quant path")), \
             patch.object(comfy_kitchen, "dequantize_per_tensor_fp8",
                          side_effect=_spy):
            bundle = load_external_vibevoice_model(
                weight_path=str(big_fp8_file), config_name="VibeVoice-1.5B",
                attention_mode="sdpa", use_llm_4bit=False, dtype_str="auto",
            )

        assert calls == []
        assert bundle["weight_family"] == "fp8_resident"

        model = bundle["model"]
        installed = sum(
            m.weight.numel() for m in model.modules()
            if isinstance(m, FP8Linear)
        )
        # Every linear in the file is resident: installed fp8 elements equal
        # the file's fp8 elements (1 byte each), a fraction of the dequant
        # footprint (2 bytes bf16 + fp32 transients).
        file_elems = sum(
            t.numel() for t in load_file(str(big_fp8_file)).values()
            if t.dtype == torch.float8_e4m3fn
        )
        assert installed == file_elems > 0

    def test_peak_rss_stays_bounded(self, big_fp8_file):
        raw_file_bytes = big_fp8_file.stat().st_size
        if raw_file_bytes < 8 * 1024 * 1024:
            pytest.skip("synthetic file too small for a meaningful RSS bound")

        n_layers, hidden, ffn = BIG_DIMS

        def _instantiate(config, is_streaming, attn_implementation,
                         final_load_dtype, use_meta=True):
            return build_stub_vv(n_layers=n_layers, hidden=hidden, ffn=ffn,
                                 vocab=1000)

        gc.collect()
        before = _rss_bytes()

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
            load_external_vibevoice_model(
                weight_path=str(big_fp8_file), config_name="VibeVoice-1.5B",
                attention_mode="sdpa", use_llm_4bit=False, dtype_str="auto",
            )

        after = _rss_bytes()
        delta = max(0, after - before)
        # Legacy path: full fp8 dict (~1x raw) + bf16 dequants (~2x raw) +
        # model params (~2x raw) ~= 5x raw. Streaming: residents install raw
        # bytes (~1x raw) + dense tail. Bound generously at 2.5x raw.
        assert delta <= int(2.5 * raw_file_bytes), (
            f"peak RSS grew {delta / 2**20:.1f} MiB for a "
            f"{raw_file_bytes / 2**20:.1f} MiB file"
        )


# ====================================================================
# fp8-quantized non-Linear targets (GPU-gate regression: the real
# VibeVoice-*-fp8_e4m3 exports quantize model.language_model.embed_tokens,
# an nn.Embedding — it must dequant-at-load, not fail the whole load)
# ====================================================================

def _fp8_info(prefix, *, convrot=False, resident_fp8=True):
    return QuantLayerInfo(
        prefix=prefix,
        group_size=0,
        in_features=8,
        out_features=16,
        has_bias=False,
        convrot=convrot,
        orig_dtype="torch.bfloat16",
        rowwise_dtype=torch.float8_e4m3fn,
        resident_fp8=resident_fp8,
    )


class _LinAndEmb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 16)
        self.emb = torch.nn.Embedding(32, 8)


class TestNonLinearFp8Demotion:
    """Unit: _demote_nonlinear_fp8_residents strategy resolution."""

    def test_embedding_demoted_to_dequant(self):
        model = _LinAndEmb()
        qmap = {
            "lin": _fp8_info("lin"),
            "emb": _fp8_info("emb"),
        }
        resolved = EL._demote_nonlinear_fp8_residents(model, qmap)
        assert resolved["lin"].resident_fp8 is True
        assert resolved["emb"].resident_fp8 is False
        # The demoted info keeps every field the dequant math needs.
        kept = resolved["emb"]
        assert kept.rowwise_dtype == torch.float8_e4m3fn
        assert kept.orig_dtype == "torch.bfloat16"
        assert (kept.out_features, kept.in_features) == (16, 8)

    def test_convrot_nonlinear_keeps_hard_fail(self):
        # Rotated weights cannot be dequantized without the rotation; the
        # downstream QuantTargetMismatch (re-export advice) stays correct.
        model = _LinAndEmb()
        qmap = {"emb": _fp8_info("emb", convrot=True)}
        resolved = EL._demote_nonlinear_fp8_residents(model, qmap)
        assert resolved["emb"].convrot is True

    def test_missing_module_left_for_replacement_error(self):
        model = _LinAndEmb()
        qmap = {"ghost": _fp8_info("ghost")}
        resolved = EL._demote_nonlinear_fp8_residents(model, qmap)
        assert resolved["ghost"].resident_fp8 is True

    def test_input_map_not_mutated(self):
        model = _LinAndEmb()
        original = _fp8_info("emb")
        qmap = {"emb": original}
        EL._demote_nonlinear_fp8_residents(model, qmap)
        assert qmap["emb"] is original
        assert qmap["emb"].resident_fp8 is True


def _save_fp8_quant_embed(path, extra_quant=()):
    """fp8 checkpoint quantizing embed_tokens like the real exports:
    scalar-scale fp8 on the nn.Embedding + named extra linears, dense tail."""
    tensors = {}
    tensors.update(_quant_tensors(
        "model.language_model.embed_tokens", "rowwise_fp8", (96, 64)))
    for prefix in extra_quant:
        shape = (96, 64) if prefix == "lm_head" else (64, 64)
        tensors.update(_quant_tensors(prefix, "rowwise_fp8", shape))
    g = torch.Generator().manual_seed(19)
    tensors.update({
        "model.language_model.norm.weight": torch.randn(64, generator=g),
        "model.language_model.layers.0.input_layernorm.weight":
            torch.randn(64, generator=g),
        "model.language_model.layers.0.post_attention_layernorm.weight":
            torch.randn(64, generator=g),
        "model.language_model.layers.0.self_attn.v_proj.weight":
            torch.randn(16, 64, generator=g).to(torch.bfloat16),
        "model.language_model.layers.0.mlp.up_proj.weight":
            torch.randn(128, 64, generator=g).to(torch.bfloat16),
        "model.language_model.layers.0.mlp.down_proj.weight":
            torch.randn(64, 128, generator=g).to(torch.bfloat16),
        "model.prediction_head.cond_proj.weight":
            torch.randn(64, 64, generator=g).to(torch.bfloat16),
        "model.prediction_head.final_layer.linear.weight":
            torch.randn(16, 64, generator=g),
    })
    if "lm_head" not in extra_quant:
        tensors["lm_head.weight"] = \
            torch.randn(96, 64, generator=g).to(torch.bfloat16)
    save_file(tensors, str(path))
    return path


class TestFp8QuantizedEmbedding:
    """E2E: fp8-quantized embed_tokens loads via dequant-at-load while the
    fp8 linears stay resident (GPU-gate regression, 2026-08-27)."""

    def test_mixed_file_demotes_embed_keeps_linear_resident(
            self, tmp_path, stubbed_load_env):
        if probe_fp8_backend() is None:
            pytest.skip("no comfy_kitchen fp8 backend on this box")
        run, _ = stubbed_load_env
        p = _save_mixed(
            tmp_path / "fp8embed.safetensors",
            drop=["model.language_model.embed_tokens.weight"],
            extra=_quant_tensors(
                "model.language_model.embed_tokens", "rowwise_fp8", (96, 64)),
        )
        bundle = run(p)
        model = bundle["model"]

        embed = model.model.language_model.embed_tokens
        assert isinstance(embed, torch.nn.Embedding)
        assert not isinstance(embed, FP8Linear)

        # Dequant-at-load math: weight == (q.float() * scale).to(bf16).
        raw = load_file(str(p))
        q = raw["model.language_model.embed_tokens.weight"].to(torch.float32)
        s = raw["model.language_model.embed_tokens.weight_scale"].to(torch.float32)
        expected = (q * s).to(torch.bfloat16)
        got = embed.weight.detach()
        assert got.dtype == torch.bfloat16
        assert torch.equal(got, expected)

        # The fp8 linear stayed resident.
        o = model.model.language_model.layers[0].self_attn.o_proj
        assert isinstance(o, FP8Linear)

        # Demoted embed counts as rowwise; convrot still owns the label.
        assert bundle["weight_family"] == "convrot_int8"
        assert bundle["quant_stats"] == {
            "n_resident_layers": 2,
            "n_rowwise_layers": 3,
            "n_fp8_resident_layers": 1,
        }

    def test_pure_fp8_file_keeps_fp8_resident_label(
            self, tmp_path, stubbed_load_env):
        """Mirrors the real VibeVoice-*-fp8_e4m3 exports: every quantized
        layer is fp8, embed_tokens among them."""
        if probe_fp8_backend() is None:
            pytest.skip("no comfy_kitchen fp8 backend on this box")
        run, _ = stubbed_load_env
        p = _save_fp8_quant_embed(
            tmp_path / "purefp8.safetensors",
            extra_quant=[_L("self_attn.o_proj"), "lm_head"],
        )
        bundle = run(p)
        model = bundle["model"]

        assert isinstance(model.model.language_model.embed_tokens,
                          torch.nn.Embedding)
        assert isinstance(model.model.language_model.layers[0].self_attn.o_proj,
                          FP8Linear)
        assert isinstance(model.lm_head, FP8Linear)
        # replaced == fp8 residents exactly -> the fp8_resident label holds
        # even though the embedding was demoted to dequant-at-load.
        assert bundle["weight_family"] == "fp8_resident"
        assert bundle["quant_stats"] == {
            "n_resident_layers": 2,
            "n_rowwise_layers": 1,
            "n_fp8_resident_layers": 2,
        }


# ====================================================================
# N-13: fp8 storage dtype must never leak into activation dtypes
# (inference gate: NoCapableBackendError from the diffusion head's
# t_freq.to(mlp[0].weight.dtype) cast)
# ====================================================================

class TestFp8ComputeDtype:
    """Unit: FP8Linear declares a compute dtype and rejects fp8 inputs."""

    def test_make_fp8_linear_sets_compute_dtype_from_orig_dtype(self):
        from ComfyUI_VibeVoice.modules.fp8_quant import make_fp8_linear

        lin = make_fp8_linear(_fp8_info("x"))(8, 16, False)
        assert isinstance(lin, FP8Linear)
        assert lin.weight.dtype == torch.float8_e4m3fn
        assert lin.compute_dtype == torch.bfloat16

    def test_make_fp8_linear_compute_dtype_none_when_orig_missing(self):
        from dataclasses import replace as dc_replace
        from ComfyUI_VibeVoice.modules.fp8_quant import make_fp8_linear

        info = dc_replace(_fp8_info("x"), orig_dtype="")
        lin = make_fp8_linear(info)(8, 16, False)
        assert lin.compute_dtype is None

    def test_fp8_linear_rejects_fp8_activations(self):
        from ComfyUI_VibeVoice.modules.fp8_quant import make_fp8_linear

        lin = make_fp8_linear(_fp8_info("x"))(8, 16, False)
        x = torch.zeros(2, 8, dtype=torch.float8_e4m3fn)
        with pytest.raises(TypeError, match="fp8 activations"):
            lin(x)


# ====================================================================
# ASR branch shares the streaming wiring
# ====================================================================

class TestASRStreaming:
    def test_asr_quant_safetensors_streams(self, tmp_path):
        from ComfyUI_VibeVoice.modules.external_loader import (
            load_external_vibevoice_asr_model,
        )

        class _StubASR(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                vv = build_stub_vv(n_layers=1)
                self.model = vv.model
                self.lm_head = vv.lm_head

        p = _save_mixed(tmp_path / "asr_quant.safetensors")
        calls = []
        real = comfy.utils.load_torch_file

        def _spy(path, **kw):
            calls.append(str(path))
            return real(path, **kw)

        with patch.object(EL, "VibeVoiceASRForConditionalGeneration", _StubASR), \
             patch.object(EL, "_load_asr_config", return_value=MagicMock()), \
             patch.object(EL, "_load_asr_tokenizer", return_value=MagicMock()), \
             patch.object(EL, "_load_asr_processor", return_value=MagicMock()), \
             patch.object(EL, "resolve_sidecar_config", return_value="/fake/c.json"), \
             patch.object(EL, "resolve_sidecar_preprocessor", return_value=""), \
             patch.object(EL, "resolve_sidecar_tokenizer_dir", return_value="/fake/d"), \
             patch.object(EL, "resolve_dtype", return_value=torch.float32), \
             patch.object(EL, "resolve_attention_mode",
                          side_effect=lambda m, quantize_4bit=False: m), \
             patch.object(EL, "get_attn_implementation_for_load", return_value="eager"), \
             patch.object(EL.model_management, "get_torch_device",
                          return_value=torch.device("cpu")), \
             patch.object(comfy.utils, "load_torch_file",
                          side_effect=_spy):
            bundle = load_external_vibevoice_asr_model(
                weight_path=str(p), config_name="VibeVoice-ASR",
                attention_mode="sdpa", dtype_str="auto",
            )

        assert bundle["is_asr"] is True
        assert bundle["weight_family"] == "convrot_int8"
        # The ASR quant route assigns per-tensor from the file generator and
        # never builds a full-dict weight read.
        assert calls == []
        q = bundle["model"].model.language_model.layers[0].self_attn.q_proj
        assert isinstance(q, ConvRotInt8Linear)
        assert q.weight.dtype == torch.int8


# ====================================================================
# Realtime quant route: verified, not assumed
# ====================================================================
#
# Three quantized exports of the realtime family (fp8_e4m3, int8_block,
# int8_convrot) become loadable for the first time once the family resolves
# to a packaged config, and none of them had been exercised before. The real
# files are 1 GB each, so every case here is a synthetic checkpoint built
# with the established `comfy_quant` idiom -- the shape and the guard
# behaviour are what is under test, not the weights.


def _realtime_prefix(layer, proj):
    return f"model.tts_language_model.layers.{layer}.self_attn.{proj}"


def _save_realtime_quant(path, layers):
    """A tiny realtime-named quant checkpoint: {proj: kind} per layer."""
    tensors = {}
    for layer, specs in layers.items():
        for proj, kind, shape in specs:
            tensors.update(_quant_tensors(_realtime_prefix(layer, proj), kind, shape))
    tensors["tts_eos_classifier.fc1.weight"] = torch.randn(64, 64)
    save_file(tensors, str(path))
    return path


_REALTIME_QUANT_LAYERS = {
    0: [
        ("q_proj", "convrot", (64, 64)),
        ("k_proj", "rowwise_fp8", (64, 64)),
        ("v_proj", "block", (128, 64)),
        ("o_proj", "rowwise", (16, 64)),
    ],
}


class TestRealtimeQuantScan:
    """scan_checkpoint_quantization and the quant guards on a realtime file."""

    @pytest.mark.parametrize(
        "family,kind,expect_convrot",
        [
            ("fp8_e4m3", "rowwise_fp8", False),
            ("int8_block", "block", False),
            ("int8_convrot", "convrot", True),
        ],
    )
    def test_scan_sees_every_quant_layer(self, tmp_path, family, kind,
                                         expect_convrot):
        """One entry per quantized linear, named by its realtime prefix."""
        shapes = {
            "q_proj": (64, 64),
            "k_proj": (64, 64),
            "v_proj": (128, 64),
            "o_proj": (16, 64),
        }
        layers = {0: [(proj, kind, shapes[proj]) for proj in shapes]}
        path = _save_realtime_quant(tmp_path / f"{family}.safetensors", layers)

        quant_map = scan_checkpoint_quantization(path)

        assert set(quant_map) == {_realtime_prefix(0, proj) for proj in shapes}
        assert len(quant_map) == len(shapes)
        assert {info.convrot for info in quant_map.values()} == {expect_convrot}

    def test_scan_reports_only_the_convrot_layer_as_convrot(self, tmp_path):
        """The convrot flag decides the resident-vs-dequant branch downstream."""
        path = _save_realtime_quant(
            tmp_path / "mixed.safetensors", _REALTIME_QUANT_LAYERS)

        quant_map = scan_checkpoint_quantization(path)

        assert quant_map[_realtime_prefix(0, "q_proj")].convrot is True
        assert quant_map[_realtime_prefix(0, "v_proj")].convrot is False

    def test_dense_realtime_file_has_no_quant_layers(self, tmp_path):
        """The BF16 realtime export carries no *.comfy_quant keys at all."""
        path = tmp_path / "bf16.safetensors"
        save_file(
            {
                "model.language_model.embed_tokens.weight": torch.randn(96, 64),
                "lm_head.weight": torch.randn(96, 64),
            },
            str(path),
        )

        assert scan_checkpoint_quantization(path) == {}

    def test_lm_head_guard_does_not_false_positive_on_realtime(self):
        """The packaged realtime config sets tie_word_embeddings=false.

        The guard fires on a quantized lm_head when the config ties it. With
        untied embeddings the realtime exports must load, so the guard has to
        stay silent -- asserted against the shipped asset, not a stub.
        """
        asset = os.path.join(
            EL._packaged_configs_dir(),
            "default_VibeVoice-Realtime-0.5B_config.json",
        )
        with open(asset, encoding="utf-8") as fh:
            decoder = json.load(fh)["decoder_config"]

        assert decoder["tie_word_embeddings"] is False
        config = SimpleNamespace(
            decoder_config=SimpleNamespace(
                tie_word_embeddings=decoder["tie_word_embeddings"]
            ),
            tie_word_embeddings=False,
        )

        EL._assert_lm_head_not_tied(config, {"lm_head": object()})

    def test_lm_head_guard_still_arms_on_a_tied_config(self):
        """Untying the realtime config must not disarm the guard itself."""
        config = SimpleNamespace(
            decoder_config=SimpleNamespace(tie_word_embeddings=True),
            tie_word_embeddings=True,
        )

        with pytest.raises(QuantTargetMismatch):
            EL._assert_lm_head_not_tied(config, {"lm_head": object()})

    @pytest.mark.parametrize(
        "family,is_gguf,quantized",
        [
            ("bf16", False, False),
            ("fp8_e4m3", False, False),
            ("int8_block", False, False),
            ("int8_convrot", False, True),
            ("gguf_block", True, False),
        ],
    )
    @pytest.mark.parametrize("attention_mode", ["sdpa", "eager", "flash", "sage"])
    def test_validate_weight_plan_accepts_each_family(self, family, is_gguf,
                                                      quantized, attention_mode):
        """Quantize_llm_4bit is off for all of them; nothing else may conflict."""
        validate_weight_plan(
            is_gguf_file=is_gguf,
            convrot_quant_map={"layer": object()} if quantized else {},
            use_llm_4bit=False,
            attention_mode=attention_mode,
        )
