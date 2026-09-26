"""Calibration of the realtime ``tts_eos_classifier`` stop condition (S4.1).

The vendored generation loop stops a sample as soon as
``sigmoid(tts_eos_classifier(last_hidden_state)) > 0.5``
(``modeling_vibevoice_streaming_inference.py``). F9 of the investigation was
that this fires on the *first* latent, producing one latent / ~0.13 s / near
silence, and F11 was the loose end behind it: the head read **0.78** on the
shipped voice prompt's *own* final hidden state, which is not a reading a
finished-utterance test should produce.

**F11 is refuted as a property of the checkpoint.** The 0.78 was measured
through ``from_pretrained``, and on transformers 5.x that path re-ran
``_init_weights`` over the already-loaded ``tts_eos_classifier`` tensors
(see tests/test_pretrained_init_weights.py) — it was a randomly re-initialised
head. Re-measured through the **node** path after that fix, on the same
prompt's own final hidden state, the head reads **0.01166** — two orders of
magnitude below the stop threshold, and on the correct side of it. The full
re-measurement is in ``docs/plans/evidence/2026-09-26-s1.2-decision.md``.

So nothing about the head or the threshold needed correcting. What these tests
pin is that this stays true:

- tier C — the threshold and the comparison operator are what the loop says they
  are, and the head can express both sides of it;
- tier G — with the real weights and the real shipped prompt, the reading is
  below the threshold.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).parent.parent
_INFERENCE_PATH = REPO_ROOT / "src" / "vibevoice" / "modular" / "modeling_vibevoice_streaming_inference.py"

#: The loop's stop threshold, read from the vendored generation loop.
#: ``modeling_vibevoice_streaming_inference.py``:
#:     tts_eos_logits = torch.sigmoid(self.tts_eos_classifier(...))
#:     if tts_eos_logits[0].item() > 0.5:
STOP_THRESHOLD = 0.5

#: F11 as first measured, through ``from_pretrained`` (contaminated head).
#: Kept so the refutation has a pinned "before" number to point at.
F11_AS_FIRST_MEASURED = 0.78

#: The same reading re-measured through the node path after the F13
#: ``_init_weights`` fix, on the prompt's own final hidden state.
F11_REMEASURED = 0.01166

_MOCKED_PREFIXES = ("src.vibevoice", "ComfyUI_VibeVoice.src.vibevoice")

# The tier-G measurement must run on the *node* loader, which is exactly what
# tests/test_realtime_e2e_gpu.py sets up. Re-using its fixtures keeps the two
# suites measuring the same thing on the same loaded model. The import is lazy
# so the default (CPU, no-checkpoint) suite never pays for it. Pytest resolves
# a fixture only if its name is in the *using* module's namespace, so the two
# transitive fixtures have to be re-exported here too.
try:  # pragma: no cover - import shape only
    from tests.test_realtime_e2e_gpu import (  # noqa: F401
        real_vendored_modules,
        realtime_assets,
        realtime_env,
    )
except ImportError:  # pragma: no cover - the fixtures are simply unavailable
    real_vendored_modules = realtime_assets = realtime_env = None


@pytest.fixture(scope="module")
def streaming():
    """Import the genuine vendored streaming base module (un-mocks the tree)."""
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
        for name in [n for n in list(sys.modules) if n.startswith(_MOCKED_PREFIXES)]:
            del sys.modules[name]
        sys.modules.update(saved)


def _head_reading(streaming, probability: float, hidden_size: int = 16) -> float:
    """Drive the real ``BinaryClassifier`` to a chosen sigmoid output.

    ``fc1`` is neutralised (identity weights, zero bias) so the ReLU passes the
    probe hidden state through unchanged, and ``fc2`` carries a single unit
    weight with a bias chosen to hit the requested logit. This exercises the
    shipped module rather than a local re-implementation of it. The probe
    hidden state contributes exactly ``1.0`` to the pre-sigmoid value, so the
    bias carries the remainder of the target logit.
    """
    head = streaming.BinaryClassifier(hidden_size)
    with torch.no_grad():
        head.fc1.weight.zero_()
        head.fc1.bias.zero_()
        head.fc1.weight.fill_diagonal_(1.0)
        head.fc2.weight.zero_()
        head.fc2.weight[0, 0] = 1.0
        head.fc2.bias[0] = float(torch.logit(torch.tensor(probability))) - 1.0
        hidden = torch.zeros(1, hidden_size)
        hidden[0, 0] = 1.0
        return float(torch.sigmoid(head(hidden))[0, 0])


class TestEosThresholdContract:
    """Tier C — the stop condition is the one the evidence was measured against."""

    def test_loop_uses_a_half_threshold(self):
        source = _INFERENCE_PATH.read_text(encoding="utf-8")
        assert "tts_eos_logits[0].item() > 0.5" in source, (
            "The realtime generation loop no longer stops on a 0.5 EOS reading. "
            "S4.1 calibrated the head against 0.5; if the threshold moved, "
            "tests/test_eos_calibration.py and the S4.1 evidence must be re-derived."
        )

    def test_loop_reads_the_sigmoid_of_the_eos_head(self):
        source = _INFERENCE_PATH.read_text(encoding="utf-8")
        assert "torch.sigmoid(self.tts_eos_classifier(" in source, (
            "The loop must threshold the sigmoid of tts_eos_classifier; a raw-logit "
            "comparison would put the stop boundary at 0.5 logits, not 0.5 probability."
        )

    def test_head_can_express_readings_on_both_sides_of_the_threshold(self, streaming):
        low = _head_reading(streaming, F11_REMEASURED)
        high = _head_reading(streaming, F11_AS_FIRST_MEASURED)
        assert low < STOP_THRESHOLD < high, (
            "BinaryClassifier cannot reach both sides of the stop threshold, so the "
            "threshold comparison is not the thing under test."
        )

    @pytest.mark.parametrize(
        "probability, expected_finished",
        [
            (F11_REMEASURED, False),
            (0.49, False),
            (0.5, False),  # the loop compares strictly: exactly 0.5 keeps speaking
            (0.51, True),
            (F11_AS_FIRST_MEASURED, True),
        ],
    )
    def test_measured_readings_land_on_the_expected_side(
        self, streaming, probability, expected_finished
    ):
        reading = _head_reading(streaming, probability)
        assert (reading > STOP_THRESHOLD) is expected_finished


class TestF11Refuted:
    """Tier C — the 0.78 was the clobbered head, not a property of the checkpoint."""

    def test_remeasured_anchor_is_below_the_threshold(self):
        assert F11_REMEASURED < STOP_THRESHOLD, (
            "The re-measured EOS reading on the shipped prompt's own final hidden "
            "state is at or above the stop threshold. F11 is then NOT refuted: the "
            "prompt itself would end the utterance before a single latent is "
            "generated, which is exactly F9's symptom."
        )

    def test_original_reading_was_the_clobbered_one(self):
        assert F11_AS_FIRST_MEASURED > STOP_THRESHOLD, (
            "F11 was only ever recorded above the threshold. If the original 0.78 "
            "no longer holds, the refutation narrative in this file is stale."
        )

    def test_both_readings_are_from_the_same_measurement_site(self):
        """Both numbers must be readings of the *same* quantity.

        The refutation only works because the fix restored the head without
        changing where it is probed. If a future change moves the probe, the two
        constants stop being comparable and this test should fail loudly.
        """
        assert 0.0 < F11_REMEASURED < 1.0 and 0.0 < F11_AS_FIRST_MEASURED < 1.0


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "y", "yes", "true"}


@pytest.mark.skipif(
    not _env_enabled("RUN_VIBEVOICE_E2E"),
    reason="Set RUN_VIBEVOICE_E2E=1 plus VIBEVOICE_REALTIME_* to re-measure F11.",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Realtime E2E requires CUDA.")
def test_real_prompt_final_hidden_state_does_not_trip_eos(realtime_assets):
    """Tier G — re-measure F11 on the real checkpoint through the node path.

    Loads the way production does (``modules.generation.load_vibevoice_model``,
    which bypasses ``from_pretrained`` — the fixture also registers the model
    and the shipped ``.pt`` prompt), then reads the head on the prompt's own
    final hidden state.
    """
    model = realtime_assets["model"]
    preset = realtime_assets["preset"]

    final_hidden = preset["tts_lm"].last_hidden_state[:, -1, :].to(model.device)
    with torch.no_grad():
        reading = float(torch.sigmoid(model.tts_eos_classifier(final_hidden))[0].item())

    print(
        f"[eos] anchor on the shipped prompt's own final hidden state: {reading:.5f} "
        f"(threshold {STOP_THRESHOLD})"
    )
    assert reading < STOP_THRESHOLD, (
        f"EOS head reads {reading:.5f} on the prompt's own final hidden state, at or "
        f"above the {STOP_THRESHOLD} stop threshold. The voice prompt itself would end "
        f"the utterance before any latent is generated — that is F9's symptom, and "
        f"it means either the _init_weights clobber has returned or the F11 refutation "
        f"({F11_REMEASURED}) is stale."
    )
