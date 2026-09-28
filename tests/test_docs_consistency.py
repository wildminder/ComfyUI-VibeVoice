"""Doc/code consistency tests (CRIT-003).

These tests guard against the README / node docstrings re-asserting the
removed "zero-shot / leave empty -> a unique voice is generated" claim that
contradicts ``modules.generation.generate_audio`` (which raises ``ValueError``
unless at least one valid voice sample is supplied).

The files are read as plain text so the tests do not require torch / comfy
to be importable (consistent with the rest of the offline test harness).
"""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README_PATH = os.path.join(ROOT, "README.md")
TTS_NODE_PATH = os.path.join(ROOT, "nodes", "tts_node.py")
REALTIME_NODE_PATH = os.path.join(ROOT, "nodes", "realtime_node.py")


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _declared_dependencies() -> list[str]:
    """The raw ``project.dependencies`` strings from pyproject.toml."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        import re

        raw = _read(os.path.join(ROOT, "pyproject.toml"))
        match = re.search(r"^dependencies = \[(.*)\]$", raw, re.M)
        assert match, "pyproject.toml dependencies array not found"
        return [
            entry.strip().strip('"')
            for entry in match.group(1).split(",")
            if entry.strip()
        ]
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
        return tomllib.load(fh)["project"]["dependencies"]


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


class TestNodeSchemaDocs:
    """The canonical node and deprecated shim document the split behavior."""

    def test_tts_node_documents_both_families(self):
        text = _read(TTS_NODE_PATH)
        assert "standard and realtime" in text.lower()
        assert "voice_preset" in text

    def test_realtime_node_source_is_gone(self):
        """The removed node must not creep back as a source file.

        Its absence is the whole point of the removal: a reintroduced
        ``realtime_node.py`` would be a second way to run realtime models, and
        the two would drift apart again the way the shim did.
        """
        assert not os.path.exists(REALTIME_NODE_PATH), (
            "nodes/realtime_node.py was removed; realtime models run on "
            "VibeVoiceTTSNode. Re-adding the file reintroduces a second, "
            "separately-maintained generation path."
        )


# ---------------------------------------------------------------------------
# S6.1 — the support-matrix section. One test per claim, so a claim cannot be
# edited out of the README without a test failing and naming what went missing.
# ---------------------------------------------------------------------------

STREAMING_INFERENCE_PATH = os.path.join(
    ROOT, "src", "vibevoice", "modular", "modeling_vibevoice_streaming_inference.py"
)


