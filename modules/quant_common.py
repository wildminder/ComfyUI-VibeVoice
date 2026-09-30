"""Shared quantized-linear replacement and weight-plan validation.

Used by BOTH quant-resident families:

- GGUF raw-block residency (:mod:`modules.gguf_quant`) — ``nn.Linear`` targets
  are swapped for :class:`~modules.gguf_quant.GGUFLinear` before any weights
  are assigned.
- ConvRot INT8 checkpoints (:mod:`modules.convrot_quant`) — targets are swapped
  for :class:`~modules.convrot_quant.ConvRotInt8Linear`.

Replacement must run BEFORE weight assignment (the swapped modules own
differently-shaped/dtyped parameters). All checks are strict: a target that is
missing, not an ``nn.Linear``, or shape-inconsistent raises immediately rather
than silently misloading weights.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class QuantTargetMismatch(RuntimeError):
    """Raised when a planned quantized-linear replacement cannot be applied."""


def resolve_module(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """Resolve a dotted module path against ``model`` (KeyError if absent)."""
    mod = model
    for part in path.split("."):
        if not hasattr(mod, part):
            raise KeyError(path)
        mod = getattr(mod, part)
    return mod


def replace_linears_for_quant(model: torch.nn.Module, layer_plan: dict) -> list:
    """Swap every module named in ``layer_plan`` for a factory-built module.

    Args:
        model: The model tree to mutate.
        layer_plan: mapping of dotted module path ->
            callable ``(in_features, out_features, has_bias) -> nn.Module``.

    Returns:
        List of replaced module paths (order follows the plan).

    Raises:
        QuantTargetMismatch: On missing / non-Linear / shape-mismatched
            targets. Nothing is replaced unless ALL targets validate first.
    """
    modules = dict(model.named_modules())
    built = []
    for prefix, factory in layer_plan.items():
        if prefix not in modules:
            raise QuantTargetMismatch(
                f"Quantized layer '{prefix}' not found in model tree"
            )
        target = modules[prefix]
        if isinstance(target, torch.nn.Linear):
            in_f = target.in_features
            out_f = target.out_features
            bias = target.bias is not None
        else:
            kind = type(target).__name__
            raise QuantTargetMismatch(
                f"Quantized layer '{prefix}' is {kind}, expected nn.Linear. "
                f"Rotated INT8 quantization can only execute on Linear "
                f"modules. If this checkpoint quantizes/rotates non-Linear "
                f"modules (e.g. embeddings, norms), re-export it excluding "
                f"those layers (e.g. the converter's skip-embeddings / "
                f"--heur option)."
            )
        try:
            # META-ONLY construction (2026-09-30). The model tree was built
            # under torch.device("meta") so every parameter is virtual; these
            # replacements were NOT, so each factory's torch.empty(...)
            # allocated REAL host memory for the whole resident — 8.08 GB for
            # the 7B fp8 file, committed but never touched (which is why the
            # live sampler showed peak_ws 5.6 GB beside peak_private 27.94
            # GB). The checkpoint assign that follows replaces every one of
            # those parameters (fp8 storage + fp32 scale + bias), so the real
            # allocation was pure waste. GGUF's set_raw_weight installs its own
            # real raw bytes afterwards, which is unaffected.
            with torch.device("meta"):
                new_mod = factory(in_f, out_f, bias)
        except Exception as e:
            raise QuantTargetMismatch(
                f"Failed to build quantized replacement for '{prefix}': {e}"
            ) from e
        built.append((prefix, new_mod))

    parent_cache = {}
    replaced = []
    for prefix, new_mod in built:
        parent_name, _, child_name = prefix.rpartition(".")
        if parent_name not in parent_cache:
            parent_cache[parent_name] = (
                model if not parent_name else resolve_module(model, parent_name)
            )
        setattr(parent_cache[parent_name], child_name, new_mod)
        replaced.append(prefix)

    logger.debug(f"Replaced {len(replaced)} nn.Linear(s) with quant-resident modules")
    return replaced


def validate_weight_plan(
    *,
    is_gguf_file: bool,
    convrot_quant_map: dict | None,
    use_llm_4bit: bool,
    attention_mode: str,
    gguf_kquant_present: bool = False,
) -> None:
    """Single choke point for quantization-family exclusivity rules (D4).

    Rules:
    - GGUF file + ConvRot metadata present -> hard error (mutually exclusive).
    - ConvRot checkpoint + bnb 4-bit requested -> hard error.
    - SageAttention over GGUF K-quants -> allowed, warning only (sage patches
      attention math only; K-quant linears are unaffected).
    """
    if is_gguf_file and convrot_quant_map:
        raise ValueError(
            "Weight plan conflict: the file contains both GGUF tensors and "
            "ConvRot (*.comfy_quant) metadata. These formats are mutually "
            "exclusive; the checkpoint is likely corrupted or misconverted."
        )
    if convrot_quant_map and use_llm_4bit:
        raise ValueError(
            "Weight plan conflict: the checkpoint already carries ConvRot INT8 "
            "quantized linears; 'quantize_llm_4bit' cannot be combined with it."
        )
    if attention_mode == "sage" and is_gguf_file and gguf_kquant_present:
        logger.warning(
            "SageAttention requested alongside GGUF K-quants: sage patches "
            "attention computation only, quantized linear layers still run "
            "through per-matmul dequantization."
        )
