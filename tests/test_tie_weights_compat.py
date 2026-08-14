"""Tests for tie_weights() transformers 5.x compatibility.

Transformers 5.x's init_weights() calls self.tie_weights(recompute_mapping=False),
but the VibeVoice models originally overrode tie_weights() without this parameter.
These tests verify that all three model files have the correct signature by
reading the source files directly (since the vendored modules are mocked in tests).
"""

import os
import re
import pytest


def _get_source_path(filename):
    """Get the full path to a vendored source file."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "src", "vibevoice", "modular", filename)


def _extract_tie_weights_signature(source_text):
    """Extract the tie_weights method signature from source text.

    Returns the parameter list as a string, or None if not found.
    """
    # Match: def tie_weights(self, ...):
    match = re.search(r'def\s+tie_weights\s*\(([^)]*)\)', source_text)
    if match:
        return match.group(1)
    return None


class TestTieWeightsSignature:
    """Verify tie_weights() accepts transformers 5.x kwargs in source files."""

    def test_modeling_vibevoice_tie_weights_accepts_kwargs(self):
        """VibeVoiceForConditionalGeneration.tie_weights accepts missing_keys and recompute_mapping."""
        path = _get_source_path("modeling_vibevoice.py")
        with open(path, 'r', encoding='utf-8') as f:
            source = f.read()
        sig = _extract_tie_weights_signature(source)
        assert sig is not None, "tie_weights method not found in modeling_vibevoice.py"
        assert "missing_keys" in sig, f"missing_keys not in tie_weights signature: {sig}"
        assert "recompute_mapping" in sig, f"recompute_mapping not in tie_weights signature: {sig}"
        assert "kwargs" in sig or "**" in sig, f"**kwargs not in tie_weights signature: {sig}"

    def test_modeling_vibevoice_streaming_tie_weights_accepts_kwargs(self):
        """VibeVoiceStreamingForConditionalGenerationInference.tie_weights accepts kwargs."""
        path = _get_source_path("modeling_vibevoice_streaming_inference.py")
        with open(path, 'r', encoding='utf-8') as f:
            source = f.read()
        sig = _extract_tie_weights_signature(source)
        assert sig is not None, "tie_weights method not found in modeling_vibevoice_streaming_inference.py"
        assert "missing_keys" in sig, f"missing_keys not in tie_weights signature: {sig}"
        assert "recompute_mapping" in sig, f"recompute_mapping not in tie_weights signature: {sig}"
        assert "kwargs" in sig or "**" in sig, f"**kwargs not in tie_weights signature: {sig}"

    def test_modeling_vibevoice_asr_tie_weights_accepts_kwargs(self):
        """VibeVoiceASRForConditionalGeneration.tie_weights accepts kwargs."""
        path = _get_source_path("modeling_vibevoice_asr.py")
        with open(path, 'r', encoding='utf-8') as f:
            source = f.read()
        sig = _extract_tie_weights_signature(source)
        assert sig is not None, "tie_weights method not found in modeling_vibevoice_asr.py"
        assert "missing_keys" in sig, f"missing_keys not in tie_weights signature: {sig}"
        assert "recompute_mapping" in sig, f"recompute_mapping not in tie_weights signature: {sig}"
        assert "kwargs" in sig or "**" in sig, f"**kwargs not in tie_weights signature: {sig}"

    def test_tie_weights_can_be_called_with_recompute_mapping_false(self):
        """Verify the signature pattern matches what transformers 5.x expects.

        Transformers 5.x calls: self.tie_weights(recompute_mapping=False)
        and also: self.tie_weights(missing_keys=set(), recompute_mapping=False)

        The signature must accept both call patterns.
        """
        path = _get_source_path("modeling_vibevoice.py")
        with open(path, 'r', encoding='utf-8') as f:
            source = f.read()
        sig = _extract_tie_weights_signature(source)
        assert sig is not None

        # The signature should contain: self, missing_keys=None, recompute_mapping=True, **kwargs
        # This allows both call patterns:
        #   tie_weights(recompute_mapping=False)
        #   tie_weights(missing_keys=set(), recompute_mapping=False)
        assert "self" in sig
        assert "missing_keys" in sig
        assert "recompute_mapping" in sig
