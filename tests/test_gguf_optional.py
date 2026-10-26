"""Acceptance core: the ``gguf`` package is OPTIONAL (plan section 4.5, task T5).

``gguf`` is needed only by the ``.gguf`` single-file path, yet it used to be
imported at module scope in modules/gguf_quant.py, so an install without it
died with an ImportError that took the whole node package down ("Load Failed"
in Manager). The contract pinned here:

- the hot import chain imports cleanly with ``gguf`` marked absent (no
  ImportError) — the module-level matrices degrade to the ``None`` sentinel and
  the availability cache reports absent;
- the runtime paths that genuinely need it refuse with one canonical,
  actionable message (``RuntimeError``, naming the package and the pip
  command), and the node gate emits ``vibevoice.gguf_missing`` exactly once;
- the frontend listener for that event is registered on disk (WEB_DIRECTORY +
  the js file), otherwise the event is dropped as "Unknown message type";
- with ``gguf`` actually installed, ``gguf_available()`` says so.

Blocking is done the stdlib way — ``sys.modules["gguf"] = None`` — so the
blocked cases pass on a machine WITHOUT ``gguf`` too: nothing here requires the
package to be present or absent. The gguf-dependent test files
(test_gguf_linear.py, test_quant_groundwork.py, ...) degrade to SKIP via
module-level ``pytest.importorskip("gguf")`` instead of erroring on such a
machine.

Note on spelling: the package is loaded by ComfyUI from a directory whose name
("ComfyUI-VibeVoice") is not a Python identifier, so conftest.py registers it
under the stable alias ``ComfyUI_VibeVoice`` and every test imports through
that. A plain ``import modules.generation`` would build a SECOND module tree
whose ``from ..src.vibevoice`` imports fail ("attempted relative import beyond
top-level package"), which is why the hot chain is re-imported through the alias
below.
"""

import contextlib
import importlib
import importlib.util
import os
import sys
import types
from unittest.mock import patch

import pytest

PKG = "ComfyUI_VibeVoice"
NODE = f"{PKG}.nodes.external_loader_node"
GGUF_MODULE = f"{PKG}.modules.gguf_quant"

#: The import chain that used to die without ``gguf``. ``nodes/__init__.py``
#: pulls the TTS/ASR nodes too, so importing ``external_loader_node`` alone
#: covers tts_node as well; the rest are named so a failure says which link broke.
HOT_CHAIN = (
    "modules.gguf_quant",
    "modules.generation",
    "modules.external_loader",
    "modules.notify",
    "nodes.external_loader_node",
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INIT_FILE = os.path.join(REPO_ROOT, "__init__.py")
JS_FILE = os.path.join(REPO_ROOT, "js", "gguf_missing_modal.js")

#: Strings the frontend must contain for the notice to reach the browser at all.
JS_REQUIRED_STRINGS = (
    "vibevoice.gguf_missing",     # the event type the backend emits
    "ComfyAsyncDialog",           # the modal primitive
    "Abort",                      # dismiss action
    "Install missing package",    # one-click install action
    "/customnode/install/pip",    # ComfyUI-Manager's endpoint
    "manager_available",          # payload key the install branch reads
    "install_cmd",                # payload key (copyable command)
)

EVENT = "vibevoice.gguf_missing"
PAYLOAD_KEYS = {"package", "message", "install_cmd", "manager_available"}


def _chain_names():
    """Fully-qualified names of the modules on the hot import chain."""
    return [f"{PKG}.{submodule}" for submodule in HOT_CHAIN]


@contextlib.contextmanager
def _chain_reimported():
    """Force the next import of the chain to re-execute its modules.

    conftest.py imports the whole node chain as it starts (via
    ``nodes/__init__.py``), so a plain ``import_module`` would hand back the copy
    built WITH ``gguf`` and prove nothing. Removing just these leaf modules
    forces a genuine fresh import; the package roots stay registered because the
    ``ComfyUI_VibeVoice`` alias is installed by conftest (the directory name is
    not importable from ``sys.path``). The originals are restored afterwards so
    the rest of the session keeps the module objects carrying the real
    gguf-built type matrices (``G._T``).

    Both lookup paths are restored, because ``import a.b.c`` and
    ``from a.b import c`` do not read the same slot. ``from a.b import c``
    reads ``sys.modules["a.b"].c`` — the PARENT's attribute — while
    ``import a.b.c`` binds ``sys.modules["a.b.c"]``. A fresh submodule import
    only sets the parent attribute when it was missing (CPython leaves an
    existing attribute alone), so restoring ``sys.modules`` alone can leave the
    parent pointing at the module built under the gguf block. That stale
    attribute is what later tests see, so it is snapshotted and restored too.
    """
    names = _chain_names()
    original = {name: sys.modules.get(name) for name in names}
    # ``parent module -> {attribute name: original module object}``
    original_attrs = {}
    for name in names:
        parent_name, _, leaf = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and hasattr(parent, leaf):
            original_attrs.setdefault(parent_name, {})[leaf] = getattr(
                parent, leaf
            )
    for name in names:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        # Drop whatever this import built, then put the originals back.
        for name in names:
            sys.modules.pop(name, None)
        for name, mod in original.items():
            if mod is not None:
                sys.modules[name] = mod
        # And make every parent attribute point at the original module again,
        # so ``from <parent> import <leaf>`` cannot hand out the temporary copy.
        for parent_name, attrs in original_attrs.items():
            parent = sys.modules.get(parent_name)
            if parent is None:
                continue
            for leaf, mod in attrs.items():
                setattr(parent, leaf, mod)
        # Any attribute a fresh import added where the parent had none must go.
        for name in names:
            parent_name, _, leaf = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is None or leaf in original_attrs.get(parent_name, {}):
                continue
            if hasattr(parent, leaf):
                delattr(parent, leaf)


@pytest.fixture
def gguf_blocked(monkeypatch):
    """Mark ``gguf`` absent for one test and reset the availability cache.

    ``sys.modules["gguf"] = None`` is the stdlib "absent module" sentinel:
    ``import gguf`` raises ImportError and ``find_spec("gguf")`` reports None
    (both verified on this interpreter in TestPackageImportsWithoutGguf).

    The ``gguf.*`` submodules are dropped alongside it, because a submodule
    still sitting in ``sys.modules`` satisfies ``from gguf.constants import ...``
    even with the parent blocked (CPython hands out cached submodules without
    re-running the parent's import). A real install without ``gguf`` has none of
    them, so dropping them is what the block is meant to model.

    The availability cache is cleared on the availability module so the next
    ``gguf_available()`` re-probes instead of answering from a value cached
    while the package was present.
    """
    for name in [n for n in sys.modules if n.startswith("gguf.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "gguf", None)
    gguf_quant = importlib.import_module(GGUF_MODULE)
    gguf_quant._GGUF_SPEC_CACHE.clear()
    return gguf_quant


def _assert_stdlib_absent_semantics():
    """``gguf`` blocked (``sys.modules["gguf"] = None``) reads as absent.

    Both consumers must agree: ``import gguf`` raises ImportError and
    ``find_spec("gguf")`` reports None — that second one is what
    :func:`modules.gguf_quant.gguf_available` probes.
    """
    with pytest.raises(ImportError):
        import gguf  # noqa: F401  (local import on purpose)
    assert importlib.util.find_spec("gguf") is None, (
        "sys.modules['gguf'] = None must make find_spec() report the package "
        "absent; if the stdlib semantics changed, gguf_available() no longer "
        "detects the block."
    )


def _import_hot_chain():
    """Import every module on the chain that used to require ``gguf``."""
    for name in _chain_names():
        importlib.import_module(name)


def _execute_node_gate():
    """Run the loader node's execute() against a single ``.gguf`` weight."""
    from ComfyUI_VibeVoice.nodes.external_loader_node import (
        VibeVoiceExternalLoaderNode,
    )

    with patch(f"{NODE}.folder_paths.get_full_path_or_raise",
               return_value="/models/diffusion_models/vibe-q8_0.gguf"):
        VibeVoiceExternalLoaderNode.execute(
            model_file="vibe-q8_0.gguf",
            config_name="VibeVoice-1.5B",
            attention_mode="sdpa",
            quantize_llm_4bit=False,
            dtype="auto",
        )


class TestPackageImportsWithoutGguf:
    """AC1: nothing at import time may need ``gguf``."""

    def test_package_imports_without_gguf(self, gguf_blocked):
        """The hot chain imports with ``gguf`` blocked — no ImportError."""
        # The block itself must read as "absent module" to the stdlib, or
        # gguf_available() would keep answering True.
        _assert_stdlib_absent_semantics()
        assert gguf_blocked.gguf_available() is False

        with _chain_reimported():
            # A fresh tree, imported with the package marked absent: the
            # module-level try/except in gguf_quant must degrade to the
            # sentinel instead of raising.
            _import_hot_chain()

            gguf_quant = importlib.import_module(GGUF_MODULE)
            assert gguf_quant._T is None, (
                "modules/gguf_quant.py must not need gguf at import time: the "
                "type matrix should stay None (lazily built on first GGUF use) "
                "when the package is absent."
            )
            assert gguf_quant._GGUF_SPEC_CACHE == {}, (
                "the availability cache must start empty so gguf_available() "
                "probes instead of answering a stale value."
            )
            assert gguf_quant.gguf_available() is False
            # gguf-free symbols must stay importable without the package.
            assert callable(gguf_quant.log_gguf_forward_counters)
            assert callable(gguf_quant.clear_dequant_cache)

    def test_package_imports_with_gguf_available(self):
        """The other half: the same chain imports WITH ``gguf`` present."""
        pytest.importorskip("gguf", reason="gguf package not installed")

        with _chain_reimported():
            _import_hot_chain()
            gguf_quant = importlib.import_module(GGUF_MODULE)
            assert gguf_quant._T is not None, (
                "with gguf installed the type matrix is built at import time."
            )
            assert gguf_quant.gguf_available() is True

    def test_reimport_leaves_rest_of_suite_untouched(self):
        """The temporary leaf swap is invisible to every later test."""
        gguf_quant = importlib.import_module(GGUF_MODULE)
        originals = {name: sys.modules.get(name) for name in _chain_names()}
        sentinel = gguf_quant._T

        with _chain_reimported():
            pass

        for name, mod in originals.items():
            assert sys.modules.get(name) is mod, (
                f"{name} was not restored to its original module object; a "
                "fresh copy would leave later tests without the gguf-built "
                "type matrices."
            )
        assert gguf_quant._T is sentinel


class TestGgufRefusalMessage:
    """AC2: one friendly refusal from every path that needs ``gguf``."""

    def test_require_gguf_raises_friendly_error(self, gguf_blocked):
        """require_gguf() names the package and the pip install command."""
        with pytest.raises(RuntimeError) as excinfo:
            gguf_blocked.require_gguf()

        text = str(excinfo.value)
        assert "gguf" in text
        assert "pip install" in text

    def test_open_gguf_reader_raises_friendly_error(self, gguf_blocked):
        """open_gguf_reader() refuses BEFORE importing the package."""
        assert gguf_blocked.gguf_available() is False
        with pytest.raises(RuntimeError) as excinfo:
            gguf_blocked.open_gguf_reader("/models/nope.gguf")

        text = str(excinfo.value)
        assert "gguf" in text
        assert "pip install" in text
        assert gguf_blocked.GGUF_MISSING_MESSAGE in text


class TestNodeGateEmitsEventOnce:
    """AC2: the node refuses AND pushes exactly one UI notice."""

    def test_node_gate_emits_event_once(self, gguf_blocked, mock_prompt_server):
        """A blocked ``.gguf`` selection raises + sends one gguf_missing event."""
        assert gguf_blocked.gguf_available() is False

        with pytest.raises(RuntimeError) as excinfo:
            _execute_node_gate()

        text = str(excinfo.value)
        assert "gguf" in text and "pip install" in text, (
            "the node's refusal must carry the canonical message, not a bare "
            "module error."
        )

        mock_prompt_server.send_sync.assert_called_once()
        event, payload = mock_prompt_server.send_sync.call_args[0]
        assert event == EVENT
        assert set(payload) == PAYLOAD_KEYS, (
            f"payload keys drifted from the frontend contract: {sorted(payload)}"
        )
        assert payload["package"] == "gguf"
        assert payload["message"] == gguf_blocked.GGUF_MISSING_MESSAGE
        assert payload["install_cmd"].endswith("-m pip install gguf")
        # conftest's server stand-in is a MagicMock, so the route scan cannot
        # see a real Manager route: False unless the endpoint is registered.
        assert payload["manager_available"] is False

    def test_gguf_missing_notice_is_best_effort(self, gguf_blocked, monkeypatch):
        """Headless (or pre-boot) sends must stay silent, never fatal."""
        from ComfyUI_VibeVoice.modules import notify

        message = gguf_blocked.GGUF_MISSING_MESSAGE

        # No ``server`` module at all (headless): the import fails, no-op.
        monkeypatch.delitem(sys.modules, "server")
        notify.notify_gguf_missing(message)

        # Server module present but nothing booted: instance is None, no-op.
        class _NotBooted:
            instance = None

        booting = types.ModuleType("server")
        booting.PromptServer = _NotBooted
        monkeypatch.setitem(sys.modules, "server", booting)
        notify.notify_gguf_missing(message)


class TestFrontendAssets:
    """AC3: the listener for the event is actually shipped."""

    def test_frontend_assets_registered(self):
        """WEB_DIRECTORY is declared and the modal js wires the whole contract."""
        import ComfyUI_VibeVoice

        with open(INIT_FILE, encoding="utf-8") as fh:
            init_source = fh.read()
        assert 'WEB_DIRECTORY = "./js"' in init_source, (
            "__init__.py must declare WEB_DIRECTORY so ComfyUI's node loader "
            "globs js/ and serves gguf_missing_modal.js (nodes.py:2299-2302)."
        )
        assert getattr(ComfyUI_VibeVoice, "WEB_DIRECTORY", None) == "./js", (
            "ComfyUI globs the WEB_DIRECTORY module attribute for **/*.js "
            "(nodes.py:2299-2302); without it the modal never reaches the browser."
        )
        web_root = os.path.abspath(
            os.path.join(REPO_ROOT, ComfyUI_VibeVoice.WEB_DIRECTORY)
        )
        assert os.path.isdir(web_root)
        assert os.path.isfile(JS_FILE), (
            f"{JS_FILE} must exist for ComfyUI's web glob to serve it."
        )

        with open(JS_FILE, encoding="utf-8") as fh:
            source = fh.read()
        for needle in JS_REQUIRED_STRINGS:
            assert needle in source, (
                f"js/gguf_missing_modal.js is missing {needle!r}: the notice "
                "would be dropped or the install button could not work."
            )


class TestGgufAvailableWhenInstalled:
    """AC1/AC2 other half: presence must be reported truthfully."""

    def test_gguf_available_when_installed(self):
        """With ``gguf`` importable, gguf_available() is True (cache reset first)."""
        pytest.importorskip("gguf", reason="gguf package not installed")

        gguf_quant = importlib.import_module(GGUF_MODULE)
        gguf_quant._GGUF_SPEC_CACHE.clear()
        assert gguf_quant.gguf_available() is True

    def test_manager_pip_route_detection(self, monkeypatch):
        """manager_available follows the registered PromptServer route."""
        from ComfyUI_VibeVoice.modules import notify

        class _Route:
            def __init__(self, path):
                self.path = path

        class _FakePromptServer:
            pass

        instance = _FakePromptServer()
        instance.routes = [_Route("/customnode/install/gguf")]
        _FakePromptServer.instance = instance

        # notify reaches the class through ``from server import PromptServer``.
        monkeypatch.setattr(sys.modules["server"], "PromptServer", _FakePromptServer)

        assert notify.manager_pip_install_available() is False
        instance.routes.append(_Route(notify._MANAGER_PIP_ROUTE))
        assert notify.manager_pip_install_available() is True

        instance.routes = [_Route("/customnode/install/gguf")]
        assert notify.manager_pip_install_available() is False
