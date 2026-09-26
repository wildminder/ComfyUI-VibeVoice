"""Regression tests for ``VibeVoiceStreamingPreTrainedModel._init_weights``.

transformers 5.x builds the model on ``meta`` inside ``from_pretrained``, loads
the checkpoint, and then calls ``PreTrainedModel._initialize_missing_keys`` ->
``initialize_weights()`` -> ``self._init_weights(module)`` over the whole module
graph. ``_initialize_weights`` skips a module only when ``module._is_hf_initialized``
is set; the per-parameter ``_is_hf_initialized`` flag is honoured *only* for remote
code. So our own ``_init_weights`` runs unconditionally on every ``nn.Linear``.

For every ``nn.Linear`` in the graph the checkpoint tensor is then overwritten
with ``normal_(0, initializer_range)`` and its bias with ``0``. Measured against
VibeVoice-Realtime-0.5B this silently destroyed exactly 8 tensors:
``acoustic_connector.{fc1,fc2}.{weight,bias}`` and
``tts_eos_classifier.{fc1,fc2}.{weight,bias}`` — while reporting
``missing=276 unexpected=0 mismatched=0``. The node loader bypasses
``from_pretrained`` for this reason and is unaffected.

These tests pin the contract ``from_pretrained`` relies on: an already-loaded
tensor must survive ``_init_weights``.
"""

import importlib
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).parent.parent

# The exact names ``conftest.py`` MagicMocks. Teardown must remove real modules
# under these prefixes and restore exactly what conftest registered — deleting
# the wider ``ComfyUI_VibeVoice.src*`` set would also drop the conftest
# ``ComfyUI_VibeVoice.src`` package alias, which is not restorable from ``saved``.
_MOCKED_PREFIXES = ("src.vibevoice", "ComfyUI_VibeVoice.src.vibevoice")


@pytest.fixture(scope="module")
def streaming():
    """Import the genuine vendored streaming base module.

    ``conftest.py`` replaces the whole ``src.vibevoice`` tree with MagicMocks to
    keep the default suite lightweight, so the mocked entries are removed for
    this module and restored on teardown.
    """
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name.startswith(_MOCKED_PREFIXES)
    }
    for name in saved:
        del sys.modules[name]
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        yield importlib.import_module(
            "ComfyUI_VibeVoice.src.vibevoice.modular.modeling_vibevoice_streaming"
        )
    finally:
        for name in [
            n for n in list(sys.modules) if n.startswith(_MOCKED_PREFIXES)
        ]:
            del sys.modules[name]
        sys.modules.update(saved)


class _Cfg:
    class decoder_config:
        initializer_range = 0.02


class _Holder:
    """Minimal stand-in exposing what ``_init_weights`` reads off ``self``."""

    config = _Cfg()


def _loaded_linear(streaming, in_f=8, out_f=4):
    """An ``nn.Linear`` already holding checkpoint values, as from_pretrained leaves it."""
    lin = nn.Linear(in_f, out_f)
    weight = torch.full((out_f, in_f), 0.5)
    bias = torch.full((out_f,), 0.25)
    lin.weight.data.copy_(weight)
    lin.bias.data.copy_(bias)
    # transformers marks successfully loaded params with this flag.
    lin.weight._is_hf_initialized = True
    lin.bias._is_hf_initialized = True
    return lin, weight, bias


def test_init_weights_preserves_loaded_linear(streaming):
    """A checkpoint-loaded Linear must not be re-randomised."""
    lin, weight, bias = _loaded_linear(streaming)

    # Act
    streaming.VibeVoiceStreamingPreTrainedModel._init_weights(_Holder(), lin)

    # Assert
    assert torch.equal(lin.weight.data, weight), (
        "from_pretrained loaded acoustic_connector/tts_eos_classifier weights were "
        "overwritten with normal_(0, initializer_range) by _init_weights"
    )
    assert torch.equal(lin.bias.data, bias), "loaded bias was zeroed by _init_weights"


def test_init_weights_still_initialises_unloaded_linear(streaming):
    """Guard against 'fixing' the above by making _init_weights a no-op.

    An unflagged Linear is genuinely new and must still get the configured
    distribution, otherwise the 276 checkpoint-absent acoustic-tokenizer encoder
    tensors would keep whatever ``torch.empty`` handed back.
    """
    lin = nn.Linear(64, 64)
    lin.weight.data.zero_()
    lin.bias.data.fill_(3.0)

    # Act
    streaming.VibeVoiceStreamingPreTrainedModel._init_weights(_Holder(), lin)

    # Assert
    assert not torch.equal(lin.weight.data, torch.zeros(64, 64))
    assert float(lin.weight.data.std()) == pytest.approx(0.02, rel=0.05)
    assert torch.equal(lin.bias.data, torch.zeros(64))
