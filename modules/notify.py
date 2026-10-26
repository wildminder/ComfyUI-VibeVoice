"""Best-effort browser notification for the optional ``gguf`` dependency.

``gguf`` is an optional package: only the ``.gguf`` single-file path needs it
(see :mod:`modules.gguf_quant`). When a user selects a ``.gguf`` checkpoint
without it installed, the loader refuses with a ``RuntimeError``; this module
layers a UI notice on top of that error so the browser can open the
"Abort / Install missing package" modal (frontend listener for the
``vibevoice.gguf_missing`` event lives in ``js/gguf_missing_modal.js``).

Two rules shape the implementation:

- **Never break the error path.** The ``RuntimeError`` is the user-visible
  outcome; this notification is a convenience. Every step (importing
  PromptServer, reaching its instance, the send itself) is guarded so a
  headless run — no ``server`` module, no booted PromptServer, no running
  loop, or a send failure — degrades to a no-op with a debug log rather than
  masking or replacing the error.
- **PromptServer is imported lazily, inside the functions.** This module must
  stay importable in headless/test contexts where ``server`` is absent or a
  test double (``conftest.py`` installs a MagicMock ``server``).
"""

import logging
import sys

#: Wire event name; the frontend registers exactly this type
#: (``api.addEventListener``), otherwise ComfyUI drops it as unknown.
_GGUF_MISSING_EVENT = "vibevoice.gguf_missing"

#: ComfyUI-Manager's one-click pip-install endpoint (registered on
#: ``PromptServer.instance.routes`` by ComfyUI-Manager's manager_server.py
#: when the Manager is active). Its presence is the availability signal.
_MANAGER_PIP_ROUTE = "/customnode/install/pip"

#: Copyable install command, built from this interpreter so the button never
#: pipes into the wrong pip. Mirrors the hint appended to
#: ``modules.gguf_quant.GGUF_MISSING_MESSAGE``.
_INSTALL_COMMAND = f"{sys.executable} -m pip install gguf"


def manager_pip_install_available() -> bool:
    """Whether ComfyUI-Manager's pip-install endpoint is currently registered.

    ComfyUI-Manager registers ``POST /customnode/install/pip`` on
    ``PromptServer.instance.routes`` (an aiohttp ``RouteTableDef``) when it is
    loaded, so scanning that table for the route is the robust signal — it is
    True exactly when the Manager can actually handle the install click.
    ``find_spec("comfyui_manager")`` is not used: the folder name varies
    across Manager installs, a registered route does not.

    Defaults to False on any surprise (headless, server still booting, or a
    test-double ``server`` module whose mock routes must not be walked).
    """
    try:
        from server import PromptServer

        # Check the class before the instance: ``isinstance`` against a
        # MagicMock attribute raises TypeError ("arg 2 must be a type"), which
        # is exactly how a mocked ``server`` (conftest, headless probes)
        # reveals itself.
        if not isinstance(PromptServer, type):
            return False
        instance = PromptServer.instance
        if not isinstance(instance, PromptServer):
            # None before the server boots, or a non-PromptServer stand-in.
            return False
        return any(
            getattr(route, "path", None) == _MANAGER_PIP_ROUTE
            for route in instance.routes
        )
    except Exception:
        return False


def notify_gguf_missing(message: str) -> None:
    """Push the missing-``gguf`` notice to connected browsers (best effort).

    Builds the payload ``{"package", "message", "install_cmd",
    "manager_available"}`` and sends it as ``vibevoice.gguf_missing`` via
    ``PromptServer.instance.send_sync``. The frontend uses
    ``manager_available`` to pick between ComfyUI-Manager's one-click install
    and the copyable command, without probing the backend itself.

    Args:
        message: The full user-facing text (normally
            ``modules.gguf_quant.GGUF_MISSING_MESSAGE``, which already names
            the package and the install command).
    """
    try:
        from server import PromptServer
    except Exception:
        # Headless run (no ComfyUI ``server`` module to import) — nothing to
        # notify; the caller's RuntimeError is the whole outcome.
        return

    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        # Server has not booted (yet) in this process.
        return
    send_sync = getattr(instance, "send_sync", None)
    if send_sync is None:
        return

    payload = {
        "package": "gguf",
        "message": message,
        "install_cmd": _INSTALL_COMMAND,
        "manager_available": manager_pip_install_available(),
    }
    try:
        send_sync(_GGUF_MISSING_EVENT, payload)
    except Exception as e:
        # send_sync hops onto the server's asyncio loop; with no running loop
        # (headless, pre-boot) that raises. Never silent, never fatal: the
        # RuntimeError raised alongside this call is what the user sees.
        logging.debug(
            f"[VibeVoice TTS] Could not deliver the {_GGUF_MISSING_EVENT} notice "
            f"({e}); the missing-gguf RuntimeError still propagates."
        )
