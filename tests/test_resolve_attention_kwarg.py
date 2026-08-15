"""Regression tests for AUD-009: resolve_attention_mode kwarg contract.

The signature is ``resolve_attention_mode(requested_mode, quantize_4bit=False)``.
Two ASR call sites previously passed ``use_llm_4bit=`` as a keyword, raising
``TypeError`` on every ASR load. These tests lock the contract so the drift
cannot silently return.
"""

import inspect

from ComfyUI_VibeVoice.modules.attention_utils import resolve_attention_mode


class TestResolveAttentionModeSignature:
    """The public signature must expose ``quantize_4bit`` (not ``use_llm_4bit``)."""

    def test_signature_has_quantize_4bit_param(self):
        sig = inspect.signature(resolve_attention_mode)
        assert "quantize_4bit" in sig.parameters

    def test_signature_has_no_use_llm_4bit_param(self):
        sig = inspect.signature(resolve_attention_mode)
        assert "use_llm_4bit" not in sig.parameters

    def test_keyword_call_with_quantize_4bit_works(self):
        # Must not raise TypeError.
        assert resolve_attention_mode("sdpa", quantize_4bit=False) == "sdpa"
        assert resolve_attention_mode("eager", quantize_4bit=True) == "sdpa"


class TestCallSitesUseCorrectKwarg:
    """Every production call site must use the ``quantize_4bit`` keyword."""

    def test_asr_generation_uses_quantize_4bit_kwarg(self):
        import ComfyUI_VibeVoice.modules.asr_generation as asr_gen

        source = inspect.getsource(asr_gen.load_asr_model_patched)
        assert "use_llm_4bit=" not in source
        assert "quantize_4bit=False" in source

    def test_asr_loader_uses_quantize_4bit_kwarg(self):
        import ComfyUI_VibeVoice.modules.asr_loader as asr_loader

        source = inspect.getsource(asr_loader.VibeVoiceASRLoader.load_model)
        assert "use_llm_4bit=" not in source
        assert "quantize_4bit=False" in source

    def test_tts_loader_call_is_valid(self):
        """loader.py passes positionally; verify the call shape still binds."""
        import ComfyUI_VibeVoice.modules.loader as loader_mod

        source = inspect.getsource(loader_mod.VibeVoiceLoader.load_model)
        # Positional call: resolve_attention_mode(attention_mode, use_llm_4bit)
        # binds use_llm_4bit's VALUE to the quantize_4bit parameter — valid.
        assert "resolve_attention_mode(attention_mode, use_llm_4bit)" in source
        # And no invalid keyword usage:
        assert "resolve_attention_mode(attention_mode, use_llm_4bit=" not in source
