"""The console-logging contract this package holds itself to.

Background. ComfyUI core installs ``app.logger.ColoredFormatter`` on the ROOT
logger (``app/logger.py``: ``setup_logger`` does ``logging.getLogger()`` then
``addHandler``). The formatter wraps every record in a coloured ``[LEVEL]`` tag
and formats the body as bare ``%(message)s``. So a line reaches the console as::

    [INFO] [ComfyUI-VibeVoice] Audio generation complete. Sample rate: 24000Hz

The whole idiom is therefore: a BARE ``logging.<level>(...)`` call on the root
logger whose message begins with the literal ``"[ComfyUI-VibeVoice] "``. Core
owns the level, the handler and the colour; this package owns nothing but the
prefix. There is deliberately no ``vv_logging`` module, no ``Logger`` subclass,
no ``LoggerAdapter``, no ``Formatter`` and no ``setLevel`` anywhere.

Why these tests are AST-based and not regex: the call sites are multi-line
f-strings and implicit literal concatenations (``gguf_quant.py``'s mixed-naming
message is one call spanning four lines; ``external_loader.py``'s load message
was one call spanning four). A regex cannot see those and would pass vacuously
against a deleted line -- which is exactly how the previous review found four
"done" items that were not done. Every assertion below reads the real source
through ``ast``.

Read the source tree, never the imported module: ``conftest.py`` mocks the
``src.vibevoice`` submodules with ``MagicMock()``, so a runtime assertion about
them is impossible.
"""

import ast
import io
import logging
import os

import pytest

PREFIX = "[ComfyUI-VibeVoice] "
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(PKG_DIR, "tests")

# First-party source trees. `src/vibevoice` is vendored upstream Microsoft code
# but is locally tracked and locally modified, so it is in scope; `tests/` is
# excluded because test helpers legitimately build log lines of their own.
SOURCE_ROOTS = ("modules", "nodes", "src")
SOURCE_FILES = ("__init__.py", "vibevoice_nodes.py")

# `logger.` -> `logging.` was applied to every one of these; anything still
# present means a call site was missed.
LOG_LEVELS = ("debug", "info", "warning", "error", "critical", "exception")


# ---------------------------------------------------------------------------
# Source walking helpers
# ---------------------------------------------------------------------------
def _iter_source_files():
    for root in SOURCE_ROOTS:
        base = os.path.join(PKG_DIR, root)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames.sort()
            for fn in sorted(filenames):
                if fn.endswith(".py"):
                    yield os.path.join(dirpath, fn)
    for fn in SOURCE_FILES:
        yield os.path.join(PKG_DIR, fn)


def _rel(path):
    return os.path.relpath(path, PKG_DIR).replace("\\", "/")


def _tree(path):
    with open(path, encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=path)


def _literal_text(node):
    """Best-effort literal message of a logging call's first argument.

    Handles plain strings, implicit concatenation (which Python folds into one
    ``Constant``) and f-strings. Returns ``None`` when the first argument is a
    computed expression -- those must be wrapped in an f-string at the call
    site, which the prefix test asserts separately.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                out.append(part.value)
            else:
                out.append("\x00")  # stand-in for a FormattedValue hole
        return "".join(out)
    return None


def _collect_log_calls():
    """Yield ``(relpath, lineno, level, first_arg_node)`` for every log call."""
    out = []
    for path in _iter_source_files():
        for node in ast.walk(_tree(path)):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr not in LOG_LEVELS:
                continue
            recv = node.func.value
            if isinstance(recv, ast.Name) and recv.id == "logging":
                out.append((_rel(path), node.lineno, node.func.attr,
                            node.args[0] if node.args else None))
    return out


def _all_log_messages(level=None):
    """Every literal message in the package, optionally filtered by level."""
    msgs = []
    for rel, lineno, lvl, arg in _collect_log_calls():
        if level is not None and lvl != level:
            continue
        text = _literal_text(arg) if arg is not None else None
        if text is not None:
            msgs.append((rel, lineno, lvl, text))
    return msgs


def _messages_containing(needle):
    """``(relpath, lineno, level, text)`` for every literal message with needle.

    ``None`` level means the needle appears at several levels.
    """
    return [m for m in _all_log_messages() if needle in m[3]]


# ---------------------------------------------------------------------------
# 1. Rendering: exact bytes out of ComfyUI's own ColoredFormatter
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not os.environ.get("COMFYUI_ROOT"),
    reason="ColoredFormatter lives in the ComfyUI checkout (COMFYUI_ROOT).",
)
class TestRenderedBytes:
    """Byte-exact rendering through core's formatter.

    This is the only check that exercises the real ``app.logger`` code. It
    asserts BYTES, not intent: if core changes its colour table this fails and
    someone looks, which is the correct outcome.
    """

    @staticmethod
    def _render(level, message, *args):
        from app.logger import ColoredFormatter

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(ColoredFormatter("%(message)s"))
        root = logging.getLogger()
        root.addHandler(handler)
        prev = root.level
        root.setLevel(logging.DEBUG)
        try:
            logging.log(level, message, *args)
        finally:
            root.setLevel(prev)
            root.removeHandler(handler)
        # StreamHandler terminates every record with the stream terminator;
        # the formatter's own output is what is under test here.
        return stream.getvalue().rstrip("\n")

    def test_info_renders_green_tag_then_prefixed_message(self):
        out = self._render(logging.INFO, PREFIX + "Audio generation complete.")
        assert out == "\x1b[32m[INFO]\x1b[0m [ComfyUI-VibeVoice] Audio generation complete."

    def test_warning_renders_bold_yellow_tag(self):
        out = self._render(logging.WARNING, PREFIX + "sage is not usable on this machine")
        assert out == (
            "\x1b[1m\x1b[33m[WARNING]\x1b[0m "
            "[ComfyUI-VibeVoice] sage is not usable on this machine"
        )

    def test_error_renders_bold_red_tag(self):
        out = self._render(logging.ERROR, PREFIX + "boom")
        assert out == "\x1b[1m\x1b[31m[ERROR]\x1b[0m [ComfyUI-VibeVoice] boom"

    def test_prefix_is_part_of_the_message_not_a_format_string(self):
        """The prefix must survive ``%``-style lazy args intact."""
        out = self._render(logging.WARNING, PREFIX + "model '%s' -> '%s'", "a", "b")
        assert out.endswith("[ComfyUI-VibeVoice] model 'a' -> 'b'")


# ---------------------------------------------------------------------------
# 2. No leftover machinery
# ---------------------------------------------------------------------------
class TestNoLeftoverMachinery:
    def test_rejected_helper_module_does_not_exist(self):
        assert not os.path.exists(os.path.join(PKG_DIR, "modules", "vv_logging.py"))

    def test_no_stdlib_getlogger_bindings(self):
        """No `logging.getLogger(__name__)` anywhere in first-party code."""
        offenders = []
        for path in _iter_source_files():
            for node in ast.walk(_tree(path)):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "getLogger"):
                    offenders.append(f"{_rel(path)}:{node.lineno}")
        assert offenders == [], f"bare getLogger bindings survived: {offenders}"

    def test_no_transformers_get_logger_bindings(self):
        """The vendored tree used transformers' get_logger; it is gone too."""
        offenders = []
        for path in _iter_source_files():
            for node in ast.walk(_tree(path)):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "get_logger"):
                    offenders.append(f"{_rel(path)}:{node.lineno}")
        assert offenders == [], f"transformers get_logger survived: {offenders}"

    def test_no_module_level_set_level_or_propagate(self):
        """`logger.setLevel(INFO)` pinned the whole subtree and `propagate =
        False` severed pytest's caplog. Neither may come back."""
        offenders = []
        for path in _iter_source_files():
            for node in ast.walk(_tree(path)):
                if isinstance(node, ast.Attribute) and node.attr in (
                    "setLevel", "propagate", "addHandler", "setFormatter",
                    "basicConfig", "Formatter", "StreamHandler", "FileHandler",
                ):
                    if isinstance(node.ctx, ast.Store):
                        offenders.append(f"{_rel(path)}:{node.lineno} .{node.attr}")
        assert offenders == [], f"logging machinery survived: {offenders}"

    def test_no_module_level_logging_import_from_transformers(self):
        """`from transformers.utils import logging` has no module-level
        info/warning/debug, so a converted call site would raise
        AttributeError at runtime."""
        offenders = []
        for path in _iter_source_files():
            for node in ast.walk(_tree(path)):
                if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(
                    "utils"
                ):
                    for alias in node.names:
                        if alias.name == "logging":
                            offenders.append(f"{_rel(path)}:{node.lineno}")
        assert offenders == [], f"transformers logging import survived: {offenders}"

    def test_diagnostics_gate_module_is_untouched(self):
        """The env gate for diagnostic CONTENT is a separate, merged mechanism."""
        path = os.path.join(PKG_DIR, "modules", "diagnostics.py")
        assert os.path.isfile(path)
        names = {n.name for n in ast.walk(_tree(path))
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert "diagnostics_enabled" in names
        assert "census_enabled" in names


# ---------------------------------------------------------------------------
# 3. Every emitted line carries the prefix
# ---------------------------------------------------------------------------
class TestEveryMessageIsPrefixed:
    def test_every_log_call_is_a_bare_root_call(self):
        """No `logger.x(...)` receivers left, and no other logging alias."""
        offenders = []
        for path in _iter_source_files():
            for node in ast.walk(_tree(path)):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr in LOG_LEVELS):
                    continue
                recv = ast.unparse(node.func.value)
                if recv != "logging":
                    offenders.append(f"{_rel(path)}:{node.lineno} {recv}.{node.func.attr}")
        assert offenders == [], f"non-root log calls survived: {offenders}"

    def test_every_literal_message_starts_with_the_prefix(self):
        """The likeliest silent failure is a missing prefix on one site."""
        unprefixed = [
            f"{rel}:{lineno} [{lvl}] {text!r}"
            for rel, lineno, lvl, text in _all_log_messages()
            if not text.startswith(PREFIX)
        ]
        assert unprefixed == [], f"unprefixed messages: {unprefixed}"

    def test_computed_first_arguments_wrap_the_prefix_in_an_fstring(self):
        """A computed message cannot be prefixed inside a literal, so the call
        must interpolate it into a prefixed f-string instead."""
        offenders = []
        for path in _iter_source_files():
            for node in ast.walk(_tree(path)):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in LOG_LEVELS):
                    continue
                if not (isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "logging"):
                    continue
                if not node.args:
                    offenders.append(f"{_rel(path)}:{node.lineno} (no message)")
                    continue
                first = node.args[0]
                # A `f"..." + (x if y else z)` concatenation is still prefixed
                # as long as its LEFTMOST fragment carries the prefix.
                leftmost = first
                while isinstance(leftmost, ast.BinOp) and isinstance(leftmost.op, ast.Add):
                    leftmost = leftmost.left
                # A literal leftmost fragment -- including the folded implicit
                # concatenation of several f-strings, which `ast` collapses into
                # ONE JoinedStr -- is still REQUIRED to carry the prefix. It is
                # never an exemption.
                text = _literal_text(leftmost)
                if text is None or not text.startswith(PREFIX):
                    offenders.append(
                        f"{_rel(path)}:{node.lineno} "
                        f"{ast.unparse(first)[:60]!r} is not prefix-wrapped"
                    )
        assert offenders == [], f"unprefixed computed messages: {offenders}"

    def test_prefix_appears_exactly_once_per_message(self):
        doubled = [
            f"{rel}:{lineno} {text!r}"
            for rel, lineno, lvl, text in _all_log_messages()
            if text.count(PREFIX.strip()) > 1
        ]
        assert doubled == [], f"prefix duplicated: {doubled}"


# ---------------------------------------------------------------------------
# 4. Level policy, asserted in BOTH directions
# ---------------------------------------------------------------------------
def _assert_level(needle, expected):
    """A message must exist at `expected` AND nowhere else.

    Both halves matter. Present-at-expected alone is satisfiable by a site that
    also still fires at INFO; absent-INFO alone is satisfiable by a deleted
    line. Only the pair pins the decision.
    """
    hits = _messages_containing(needle)
    assert hits, (
        f"no message containing {needle!r} exists any more -- if it was deleted "
        f"on purpose, this test needs updating with the reason"
    )
    levels = sorted({h[2] for h in hits})
    assert levels == [expected], (
        f"{needle!r} is emitted at {levels}, expected exactly [{expected!r}]: "
        f"{[(h[0], h[1]) for h in hits]}"
    )


class TestExplicitLevelDecisions:
    def test_generation_complete_stays_info(self):
        """The user named this one explicitly: it stays INFO."""
        _assert_level("Audio generation complete. Sample rate:", "info")

    def test_mixed_gguf_naming_is_a_warning(self):
        """A mixed-naming GGUF is resolved by heuristic majority vote and the
        remainder silently aliased -- if a checkpoint sounds subtly wrong this
        is the only evidence a heuristic ran."""
        _assert_level("GGUF file mixes tensor naming conventions", "warning")

    @pytest.mark.parametrize("arch", ["SM80+", "SM89", "SM90", "SM120"])
    def test_sage_kernel_selection_is_debug(self, arch):
        """Kernel/kernel-selection trivia is diagnostics, not user-facing."""
        _assert_level(f"SageAttention: Using {arch}", "debug")

    def test_sage_unsupported_arch_stays_a_warning(self):
        """The sibling line is a genuine degradation, not trivia."""
        _assert_level("SageAttention has no kernel for SM", "warning")

    def test_model_discovery_is_debug(self):
        _assert_level("Scanning for VibeVoice models in:", "debug")

    def test_discovered_model_registry_dump_is_debug(self):
        _assert_level("Discovered VibeVoice models:", "debug")

    @pytest.mark.parametrize("needle", [
        "Loading VibeVoice models for",
    ])
    def test_load_start_stays_info(self, needle):
        """Disagrees with the ask's example list, deliberately: this is the
        package's only 'a load is starting' line, and it names both the model
        and the target device."""
        _assert_level(needle, "info")

    def test_download_progress_stays_info(self):
        _assert_level("Downloading official VibeVoice model:", "info")

    def test_transcription_result_stays_info(self):
        _assert_level("ASR transcription complete.", "info")

    def test_shard_count_is_debug(self):
        _assert_level("shards from", "debug")

    def test_attention_configuration_is_debug(self):
        _assert_level("Successfully configured model", "debug")

    def test_tokenizer_download_attempts_are_debug(self):
        _assert_level("Attempting to download 'tokenizer.json'", "debug")
        _assert_level("Download successful.", "debug")

    def test_nan_scrub_is_an_error(self):
        """A genuine data fault the user may need to act on."""
        _assert_level("Audio contains NaN or Inf values", "error")

    @pytest.mark.parametrize("needle", [
        "falling back to 'sdpa'",
        "Attention mode",
        "Could not parse speaker marker",
    ])
    def test_fallbacks_are_warnings(self, needle):
        """Spot-check that the degradation class kept its level."""
        found = _messages_containing(needle)
        assert found, f"expected a warning about {needle!r}"
        for rel, lineno, lvl, _ in found:
            assert lvl == "warning", f"{rel}:{lineno} {needle!r} is {lvl}"

    def test_internals_are_debug(self):
        """Spot-check that diagnostics kept their level."""
        for needle in ("Skipping non-type streaming entry",
                       "Loading shard:",
                       "Cleaned up cached models:"):
            found = _messages_containing(needle)
            assert found, f"expected a debug line about {needle!r}"
            for rel, lineno, lvl, _ in found:
                assert lvl == "debug", f"{rel}:{lineno} {needle!r} is {lvl}"


class TestDeletedMessagesStayGone:
    """Both messages were noise; deleting them is a decision, not an accident."""

    def test_resample_notice_is_absent_everywhere(self):
        assert _messages_containing("Resampling reference audio") == []

    def test_external_load_confirmations_are_absent_everywhere(self):
        assert _messages_containing("Successfully loaded external VibeVoice") == []

    def test_asr_load_confirmation_is_absent_everywhere(self):
        assert _messages_containing(
            "Successfully loaded external VibeVoice ASR model"
        ) == []

    def test_the_strings_are_not_in_the_source_at_all(self):
        """Belt and braces: not merely unlogged, not present as dead strings."""
        for path in _iter_source_files():
            with open(path, encoding="utf-8") as fh:
                body = fh.read()
            for needle in ("Resampling reference audio",
                           "Successfully loaded external VibeVoice"):
                assert needle not in body, f"{_rel(path)} still contains {needle!r}"


# ---------------------------------------------------------------------------
# 5. Presentation-only: no loading/forward logic moved
# ---------------------------------------------------------------------------
class TestLogicDidNotMove:
    def test_extension_node_ids_unchanged(self):
        """The node IDs are a compatibility contract with saved workflows."""
        import tests.test_extension as ext
        assert ext.LEGACY_EXTENSION_NODE_IDS == (
            "VibeVoiceTTS",
            "VibeVoiceASR",
            "VibeVoiceLoadExternalModel",
        )

    def test_comfy_entrypoint_importable(self):
        from ComfyUI_VibeVoice.vibevoice_nodes import comfy_entrypoint
        assert callable(comfy_entrypoint)

    def test_legacy_input_id_tuples_still_bind(self):
        """Saved workflows bind widget values by input position, so the input
        id ORDER is part of the node's contract, not an implementation detail."""
        import tests.test_node_schema as schema
        assert schema.LEGACY_MAIN_TTS_INPUT_IDS[0] == "model_name"
        assert schema.LEGACY_MAIN_TTS_INPUT_IDS[1] == "text"
        assert schema.LEGACY_REALTIME_INPUT_IDS[0] == "model_name"

    def test_resample_still_runs_inside_its_branch(self):
        """The deletion removed a LOG LINE, not the resample call."""
        path = os.path.join(PKG_DIR, "modules", "audio_utils.py")
        tree = _tree(path)
        call = None
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "resample_audio_tensor"):
                call = node
                break
        assert call is not None, "resample_audio_tensor call is gone"
        # Walk up to the `if original_sr != int(target_sr):` that must still
        # guard it. `ast` has no parent pointers, so search for the statement.
        for parent in ast.walk(tree):
            if not isinstance(parent, ast.If):
                continue
            for stmt in parent.body:
                for sub in ast.walk(stmt):
                    if sub is call:
                        src = ast.unparse(parent.test)
                        assert "original_sr" in src and "target_sr" in src, src
                        return
        raise AssertionError("resample_audio_tensor is no longer under an `if`")

    def test_no_numerics_file_gained_a_log_call(self):
        """Only the presentation layer may have gained call sites. The
        forward/backward maths in these files must be untouched by the audit."""
        for rel in ("src/vibevoice/modular/sage_attention_patch.py",):
            tree = _tree(os.path.join(PKG_DIR, rel))
            levels = [
                n.func.attr for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in LOG_LEVELS
            ]
            # 4 kernel-selection lines (now DEBUG) + the unsupported-arch warning
            assert sorted(levels) == ["debug"] * 4 + ["warning"], levels


class TestLogCountsAsABudget:
    #: The audited user-facing INFO set is 45 literal sites. The ceiling allows
    #: a handful of genuinely new progress lines but catches a wholesale
    #: reversal of the audit (every demoted diagnostic promoted back to INFO).
    INFO_CEILING = 55

    def test_package_emits_a_bounded_number_of_user_facing_lines(self):
        info = _all_log_messages(level="info")
        assert len(info) <= self.INFO_CEILING, (
            f"{len(info)} INFO lines in the package, ceiling is "
            f"{self.INFO_CEILING}; re-audit the new ones"
        )
        assert len(info) >= 40, (
            f"only {len(info)} INFO lines -- several may have been demoted "
            f"past the point of usefulness; check the level decisions"
        )

    def test_warning_lines_stay_far_below_error_lines(self):
        """Sanity check that levels are not all collapsed to one value."""
        counts = {lvl: len(_all_log_messages(level=lvl)) for lvl in LOG_LEVELS}
        assert counts["debug"] > 0
        assert counts["info"] > 0
        assert counts["warning"] > 0
        assert counts["error"] > 0