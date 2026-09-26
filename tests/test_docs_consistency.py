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


class TestReadmeUnifiedTtsNode:
    """README must document one canonical TTS node for both model families."""

    def test_readme_presents_one_canonical_tts_node(self):
        text = _read(README_PATH)
        assert "single canonical TTS node" in text
        assert "`VibeVoice TTS`" in text

    def test_legacy_realtime_mentions_are_deprecation_adjacent(self):
        """Current-usage sections must present the old node as deprecated.

        Historical changelog entries describe past releases and are exempt; the
        newest changelog entry must mark the shim deprecated.
        """
        text = _read(README_PATH)
        usage, _sep, changelog = text.partition("<!-- CHANGELOG -->")
        lines = usage.splitlines()
        found = False
        for index, line in enumerate(lines):
            if "VibeVoiceRealtime" not in line and "VibeVoice Realtime TTS" not in line:
                continue
            found = True
            window = "\n".join(lines[max(0, index - 5): index + 6]).lower()
            assert "deprecat" in window, (
                "README mentions the legacy realtime node outside a deprecation note."
            )
            assert "vibevoice tts" in window, (
                "README must point the legacy realtime node at the canonical TTS node."
            )
        assert found, "README must document the deprecated legacy realtime node."

    def test_latest_changelog_entry_marks_shim_deprecated(self):
        text = _read(README_PATH)
        _sep, _marker, changelog = text.partition("<!-- CHANGELOG -->")
        newest = changelog.split("<summary>", 1)[1].split("</details>", 1)[0]
        assert "2.9.0" in newest
        newest_lower = newest.lower()
        assert "deprecat" in newest_lower
        assert "vibevoice tts" in newest_lower

    def test_readme_documents_realtime_preset_folder_and_pt_contract(self):
        text = _read(README_PATH)
        assert "models/tts/VibeVoice/voices" in text
        assert "vibevoice_voices" in text
        assert ".pt" in text
        assert "voice_preset" in text

    def test_readme_distinguishes_diffusion_steps_from_generation_length(self):
        text = _read(README_PATH).lower()
        assert "diffusion quality/time" in text
        assert "max_new_tokens" in text
        assert "0 = model default" in text or "0` = model default" in text

    def test_readme_states_realtime_has_no_reference_cloning(self):
        text = _read(README_PATH).lower()
        assert "reference-audio cloning is not supported" in text
        assert "single-speaker" in text

    def test_readme_states_realtime_output_is_completed_audio_not_streaming(self):
        text = _read(README_PATH).lower()
        assert "no live pcm" in text
        assert "completed comfyui `audio`" in text

    def test_readme_omits_realtime_as_generic_fallback(self):
        text = _read(README_PATH).lower()
        assert "realtime-as-standard-fallback" not in text
        assert "as a generic fallback" not in text


class TestNodeSchemaDocs:
    """The canonical node and deprecated shim document the split behavior."""

    def test_tts_node_documents_both_families(self):
        text = _read(TTS_NODE_PATH)
        assert "standard and realtime" in text.lower()
        assert "voice_preset" in text

    def test_realtime_shim_is_documented_as_deprecated_delegate(self):
        text = _read(REALTIME_NODE_PATH)
        assert "deprecated" in text.lower()
        assert "VibeVoiceTTSNode" in text


# ---------------------------------------------------------------------------
# S6.1 — the support-matrix section. One test per claim, so a claim cannot be
# edited out of the README without a test failing and naming what went missing.
# ---------------------------------------------------------------------------

MATRIX_DOC_PATH = os.path.join(
    ROOT, "docs", "plans", "2026-09-26-two-version-matrix.md"
)
STREAMING_INFERENCE_PATH = os.path.join(
    ROOT, "src", "vibevoice", "modular", "modeling_vibevoice_streaming_inference.py"
)


class TestReadmeTransformersSupportMatrix:
    """The declared transformers range and what each family actually supports."""

    def test_readme_has_a_support_matrix_section(self):
        assert "### Support Matrix" in _read(README_PATH), (
            "README must carry a Support Matrix section: the three model families "
            "do not share a supported transformers range, and users need to see it."
        )

    def test_readme_states_the_families_do_not_share_a_range(self):
        text = _read(README_PATH).lower()
        assert "do **not** share a supported `transformers` range" in text, (
            "README must state that the model families differ in transformers support; "
            "a single unqualified range is what let 5.3 install silently (F12)."
        )

    def test_readme_states_the_declared_range(self):
        """The README must quote the range pyproject.toml actually declares.

        Hard-coding the specifier here is what let a `<5` cap outlive the green
        5.3.0 row it was written to describe, so the expectation is derived from
        the manifest instead of being pinned in the test.
        """
        text = _read(README_PATH)
        spec = next(
            dep
            for dep in _declared_dependencies()
            if dep.replace(" ", "").startswith("transformers")
        )
        assert spec in text, (
            f"README must state the range the package actually declares ({spec}); "
            f"it currently states a different one."
        )

    def test_readme_states_realtime_is_validated_on_both_rows(self):
        text = _read(README_PATH)
        assert "10/11 acceptance tests pass" in text, (
            "README must record that the 4.5x realtime row is not fully green. "
            "Claiming a green 4.5x row that was not run is exactly what S5.2 "
            "forbids."
        )
        assert "11/11" in text, "README must record the 5.3.0 realtime result."

    def test_readme_points_at_the_recorded_matrix(self):
        text = _read(README_PATH)
        assert "2026-09-26-two-version-matrix.md" in text, (
            "README must link the recorded two-version matrix so the numbers above "
            "can be checked."
        )

    def test_readme_says_asr_requires_transformers_5_3(self):
        text = _read(README_PATH)
        assert "Requires `transformers >= 5.3.0`" in text, (
            "README must state the ASR family's own transformers requirement. It is "
            "the one family that cannot run on the declared range, and leaving it "
            "unsaid turns the pin into an install-time conflict."
        )

    def test_readme_says_asr_is_impossible_on_4x(self):
        text = _read(README_PATH)
        assert "does not exist in any 4.x release" in text, (
            "README must say why the ASR row is blank on 4.5x — an empty cell reads "
            "as untested rather than as structurally impossible."
        )

    def test_readme_does_not_claim_a_green_4_5x_row(self):
        text = _read(README_PATH)
        for line in text.splitlines():
            if "Realtime TTS" in line and "4.5x" in line:
                assert "✅" not in line, (
                    "README claims a fully green 4.5x realtime row; the matrix "
                    "records one failing test there."
                )


class TestReadmeAttentionGuidance:
    """S3.2 excluded sage for realtime — a user-visible default change."""

    def test_readme_states_sage_is_excluded_for_realtime(self):
        text = _read(README_PATH).lower()
        assert "excluded for realtime models" in text, (
            "README must say sage is excluded for realtime models. It silently "
            "falls back to sdpa, so a user who picks sage deserves to know."
        )

    def test_readme_input_docs_flag_the_realtime_sage_fallback(self):
        text = _read(README_PATH)
        assert "**Realtime models do not use `sage`**" in text, (
            "The attention_mode input bullet must point at the support matrix; a "
            "user reads the input list, not the matrix."
        )

    def test_every_sage_row_of_the_quantize_table_carries_the_caveat(self):
        """The quantize/attention table must not read as "sage is recommended".

        The table is the first thing a user reads and it recommends `sage` in
        bold for the quantize-4bit row; without the realtime caveat on the row
        itself, a realtime user pairs "recommended" with a backend the support
        matrix excludes. Only the Feature Compatibility & VRAM table is in
        scope — the attention-parity table further down already carries a
        "❌ excluded" verdict column.
        """
        text = _read(README_PATH)
        section = text.partition("### Feature Compatibility & VRAM Matrix")[2]
        table = section.partition("### Support Matrix")[0]
        rows = [
            line
            for line in table.splitlines()
            if line.lstrip().startswith("|") and "`sage`" in line
        ]
        assert len(rows) == 2, (
            f"expected the quantize table's two sage rows (4-bit off/on), found "
            f"{len(rows)}: {rows}"
        )
        for line in rows:
            assert "Realtime models do not use `sage`" in line, (
                f"sage row without a realtime caveat: {line.strip()!r}. The support "
                f"matrix excludes sage for realtime; this table must say so too."
            )

    def test_readme_reports_the_measured_cosines(self):
        text = _read(README_PATH)
        for value in ("0.999962", "0.999940", "0.994651"):
            assert value in text, (
                f"README must report the measured attention parity ({value}); an "
                f"exclusion without its number is not reviewable."
            )

    def test_readme_names_the_parity_gate(self):
        text = _read(README_PATH)
        assert "gate `0.999`" in text or "gate 0.999" in text, (
            "README must state the cosine gate the backends were measured against."
        )

    def test_readme_says_standard_tts_still_uses_sage(self):
        text = _read(README_PATH).lower()
        assert "standard tts family\nis unaffected" in text or (
            "standard tts family is unaffected" in text
        ), (
            "The sage exclusion is realtime-only; README must not read as a "
            "repo-wide removal of the backend."
        )


class TestReadmePrecisionGuidance:
    """S3.3: bf16 and fp32 agree, so there is no dtype caveat to give."""

    def test_readme_states_no_dtype_caveat(self):
        text = _read(README_PATH)
        assert "**No dtype caveat.**" in text, (
            "README must state the measured dtype conclusion explicitly. Silence "
            "reads as 'unmeasured' and users invent workarounds."
        )

    def test_readme_reports_the_dtype_eos_readings(self):
        text = _read(README_PATH)
        assert "0.000017" in text and "0.000016" in text, (
            "README must report both first-latent EOS readings so the 'precision "
            "does not shift the decision' claim is checkable."
        )
        assert "0.999968" in text, "README must report the bf16/fp32 conditioning cosine."


class TestDualApiCacheShimIsDocumented:
    """S6.1: the vendored file must explain why the dual-API shim exists."""

    def test_streaming_inference_explains_the_legacy_pickle(self):
        text = _read(STREAMING_INFERENCE_PATH)
        assert "key_cache" in text and "value_cache" in text, (
            "The shim note must name the legacy list attributes the released .pt "
            "prompts are pickled with."
        )
        assert "cache.layers" in text, (
            "The shim note must name the modern container attribute that replaced "
            "them."
        )

    def test_streaming_inference_warns_the_failure_is_silent(self):
        text = _read(STREAMING_INFERENCE_PATH).lower()
        assert "silently" in text, (
            "The shim note must say the failure mode is silent, not loud. That is "
            "the only reason a dual-API wrapper is justified here."
        )

    def test_readme_points_at_the_shim_explanation(self):
        text = _read(README_PATH)
        assert "modeling_vibevoice_streaming_inference.py" in text, (
            "README must point users at the shim's in-source explanation."
        )

    def test_readme_explains_the_legacy_prompt_format(self):
        text = _read(README_PATH).lower()
        assert "legacy `dynamiccache`" in text, (
            "README must explain that the official .pt prompts carry a legacy "
            "DynamicCache, since that is what forces the shim."
        )


class TestDocsBackedByTheMatrixFile:
    """The numbers quoted in the README must exist in the recorded matrix."""

    def test_matrix_doc_exists(self):
        assert os.path.isfile(MATRIX_DOC_PATH), (
            "The recorded two-version matrix is the authority for every version "
            "claim in the README; it must not be deleted."
        )

    def test_readme_version_claims_appear_in_the_matrix(self):
        matrix = _read(MATRIX_DOC_PATH)
        for claim in ("4.57.6", "5.3.0", "11 passed"):
            assert claim in matrix, (
                f"The README quotes '{claim}'; the matrix must record it too, or "
                f"the README is citing a number nobody measured."
            )
