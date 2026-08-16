"""Doc/code consistency tests (CRIT-003).

These tests guard against the README / node docstrings re-asserting the
removed "zero-shot / leave empty -> a unique voice is generated" claim that
contradicts ``modules.generation.generate_audio`` (which raises ``ValueError``
unless at least one valid voice sample is supplied).

The files are read as plain text so the tests do not require torch / comfy
to be importable (consistent with the rest of the offline test harness).
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README_PATH = os.path.join(ROOT, "README.md")
TTS_NODE_PATH = os.path.join(ROOT, "nodes", "tts_node.py")


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


class TestReadmeNoZeroShotClaim:
    """README must require a reference and not promise zero-shot generation."""

    def test_readme_requires_reference_audio(self):
        """README explicitly states at least one reference audio is required."""
        text = _read(README_PATH).lower()
        assert "at least one" in text, (
            "README must state that at least one reference audio is required."
        )
        assert "reference audio is required" in text, (
            "README must state that reference audio is required."
        )

    def test_readme_no_unique_voice_generated_claim(self):
        """README must NOT promise that an empty speaker input yields a unique voice."""
        text = _read(README_PATH).lower()
        assert "unique voice will be generated" not in text, (
            "README still claims an empty speaker input generates a unique voice, "
            "which contradicts generate_audio's requirement of >=1 reference."
        )
        assert "leave the speaker" not in text or "required" in text, (
            "README should not tell users they can simply leave a speaker empty "
            "without also stating that a reference is required."
        )

    def test_readme_no_automatic_zero_shot_clone(self):
        """README changelog / feature list must not advertise auto zero-shot cloning."""
        text = _read(README_PATH).lower()
        assert "a unique voice will be generated for them automatically" not in text


class TestNodeDocstringNoZeroShot:
    """Node docstring must not advertise zero-shot TTS without a reference."""

    def test_tts_node_docstring_no_zero_shot(self):
        text = _read(TTS_NODE_PATH)
        # The exact removed phrase from the original module / class docstring.
        assert "zero-shot TTS for speakers without reference audio" not in text, (
            "tts_node.py docstring still advertises zero-shot TTS without a reference."
        )
        assert "generated via zero-shot TTS" not in text, (
            "tts_node.py tooltip still claims a speaker is generated via zero-shot TTS."
        )


class TestReadmeExternalModelLoading:
    """Phase 6.1: README must document external model loading."""

    def test_readme_mentions_external_model_loading(self):
        """README documents the 'Load VibeVoice Model' node and external_model input."""
        text = _read(README_PATH)
        assert "Load VibeVoice Model" in text, (
            "README must document the 'Load VibeVoice Model' node."
        )
        assert "external_model" in text, (
            "README must document the external_model input."
        )

    def test_readme_mentions_sidecar_configs(self):
        """README documents the sidecar config binding convention."""
        text = _read(README_PATH)
        assert ".config.json" in text, (
            "README must document the <weight>.config.json sidecar convention."
        )
        assert "preprocessor" in text, (
            "README must document the preprocessor sidecar config."
        )
        assert "tokenizer.json" in text, (
            "README must document the tokenizer.json sidecar."
        )

    def test_readme_mentions_diffusion_models_folder(self):
        """README tells users where to place external weight files."""
        text = _read(README_PATH)
        assert "diffusion_models" in text, (
            "README must state that external weights go in models/diffusion_models."
        )
