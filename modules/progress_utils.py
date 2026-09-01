"""Progress reporting helpers for VibeVoice nodes.

ComfyUI surfaces inference progress in two places:

* the **frontend** bar, driven by ``comfy.utils.ProgressBar`` (it posts
  updates to a server hook which relays them to the UI), and
* the **console** bar, the standard ``tqdm`` bar that every sampler prints to
  the terminal (the k-diffusion loops run under ``comfy.utils.model_trange``).

``comfy.utils.ProgressBar`` only drives the frontend bar — it never writes to
the terminal. VibeVoice's long-running ``generate()`` loops therefore showed a
UI bar but no console bar, unlike the samplers. This module provides a
drop-in wrapper that drives both.
"""

import comfy.utils
from tqdm.auto import tqdm


class ProgressBarWithConsole:
    """``comfy.utils.ProgressBar`` plus the standard tqdm console bar.

    Mirrors the ``ProgressBar`` interface (``update_absolute`` / ``update`` /
    ``total`` / ``current``) so it can replace ``ProgressBar`` at the call
    sites unchanged. Every update is forwarded to the real frontend bar and
    also mirrored to a lazily-created ``tqdm`` console bar, matching the
    sampler bar's look (no description, absolute position, auto total
    re-sync).

    The console bar is created lazily on the first update so loops that only
    learn their real total at runtime (the streaming path starts at
    ``total=1`` and corrects it on the first callback) do not flash a wrong
    total. It is gated on ``comfy.utils.PROGRESS_BAR_ENABLED`` — the same flag
    ``nodes.py`` reads to build the samplers' ``disable_pbar`` — so disabling
    the ComfyUI progress bar silences the console bar here too.
    """

    def __init__(self, total, node_id=None):
        self._ui = comfy.utils.ProgressBar(total, node_id=node_id)
        self._console = None

    @property
    def total(self):
        return self._ui.total

    @property
    def current(self):
        return self._ui.current

    def update_absolute(self, value, total=None, preview=None):
        self._ui.update_absolute(value, total=total, preview=preview)
        self._sync_console()

    def update(self, value):
        self.update_absolute(self._ui.current + value)

    def _sync_console(self):
        if self._console is None:
            if not comfy.utils.PROGRESS_BAR_ENABLED:
                return
            self._console = tqdm(total=self._ui.total)
        elif self._console.total != self._ui.total:
            self._console.total = self._ui.total
        self._console.n = self._ui.current
        self._console.refresh()

    def close(self):
        """Finalize the console bar. Safe to call when no bar was created."""
        if self._console is not None:
            self._console.close()
            self._console = None
