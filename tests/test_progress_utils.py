"""Tests for modules/progress_utils.py.

``ProgressBarWithConsole`` must drive both ComfyUI progress surfaces:

* the **frontend** bar (``comfy.utils.ProgressBar``), and
* the **standard tqdm console bar** that every sampler prints to the terminal.

The frontend bar is exercised with the *real* ``comfy.utils.ProgressBar``
(its server hook is ``None`` under test, so ``update_absolute`` only tracks
``current``/``total``). The console bar is asserted through a patched
``tqdm`` so no real terminal output is produced. The global
``comfy.utils.PROGRESS_BAR_ENABLED`` flag — the same one ``nodes.py`` reads to
build the samplers' ``disable_pbar`` — must gate the console bar.
"""

import pytest
from unittest.mock import patch

import comfy.utils
from ComfyUI_VibeVoice.modules import progress_utils
from ComfyUI_VibeVoice.modules.progress_utils import ProgressBarWithConsole


@pytest.fixture
def mock_tqdm():
    """Patch the tqdm class referenced inside progress_utils.

    No real console bar is created; assertions go through
    ``mock_tqdm.return_value`` (the would-be bar instance).
    """
    with patch.object(progress_utils, "tqdm") as m:
        yield m


class TestFrontendBar:
    """The wrapper must forward every update to the real frontend bar."""

    def test_ui_bar_tracks_absolute_update(self, mock_tqdm):
        bar = ProgressBarWithConsole(10)
        bar.update_absolute(7)
        assert bar._ui.current == 7
        assert bar._ui.total == 10

    def test_ui_bar_total_reset_via_update_absolute(self, mock_tqdm):
        bar = ProgressBarWithConsole(10)
        bar.update_absolute(3, total=100)
        assert bar._ui.total == 100
        assert bar._ui.current == 3

    def test_ui_bar_clamps_value_to_total(self, mock_tqdm):
        bar = ProgressBarWithConsole(10)
        bar.update_absolute(999)
        assert bar._ui.current == 10

    def test_update_relative_accumulates(self, mock_tqdm):
        bar = ProgressBarWithConsole(10)
        bar.update(3)
        bar.update(4)
        assert bar._ui.current == 7

    def test_total_and_current_properties_delegate(self, mock_tqdm):
        bar = ProgressBarWithConsole(10)
        bar.update_absolute(6, total=42)
        assert bar.total == 42
        assert bar.current == 6

    def test_node_id_forwarded_to_ui_bar(self, mock_tqdm):
        with patch.object(comfy.utils, "ProgressBar") as mock_ui_cls:
            ProgressBarWithConsole(10, node_id="node_7")
            mock_ui_cls.assert_called_once_with(10, node_id="node_7")


class TestConsoleBar:
    """The wrapper must mirror updates to a lazily-created tqdm console bar."""

    def test_console_bar_created_lazily_on_first_update(self, mock_tqdm):
        bar = ProgressBarWithConsole(10)
        mock_tqdm.assert_not_called()
        bar.update_absolute(1)
        mock_tqdm.assert_called_once_with(total=10)

    def test_console_bar_tracks_position(self, mock_tqdm):
        bar = ProgressBarWithConsole(10)
        bar.update_absolute(4)
        console = mock_tqdm.return_value
        assert console.n == 4
        console.refresh.assert_called()

    def test_console_bar_uses_corrected_total_on_first_update(self, mock_tqdm):
        # Streaming path: constructed with a total=1 placeholder, corrected on
        # the first callback. Lazy creation must see the corrected total.
        bar = ProgressBarWithConsole(1)
        bar.update_absolute(1, total=512)
        mock_tqdm.assert_called_once_with(total=512)
        assert mock_tqdm.return_value.n == 1

    def test_console_bar_resyncs_total_after_creation(self, mock_tqdm):
        bar = ProgressBarWithConsole(10)
        bar.update_absolute(1)
        bar.update_absolute(2, total=50)
        console = mock_tqdm.return_value
        assert console.total == 50
        assert console.n == 2

    def test_console_bar_n_clamped_via_ui(self, mock_tqdm):
        bar = ProgressBarWithConsole(10)
        bar.update_absolute(999)
        assert mock_tqdm.return_value.n == 10

    def test_no_console_bar_when_disabled(self, mock_tqdm):
        with patch.object(comfy.utils, "PROGRESS_BAR_ENABLED", False):
            bar = ProgressBarWithConsole(10)
            bar.update_absolute(5)
            mock_tqdm.assert_not_called()

    def test_frontend_bar_still_updates_when_console_disabled(self, mock_tqdm):
        with patch.object(comfy.utils, "PROGRESS_BAR_ENABLED", False):
            bar = ProgressBarWithConsole(10)
            bar.update_absolute(5)
            assert bar._ui.current == 5


class TestClose:
    """close() finalizes the console bar and must be safe to call anytime."""

    def test_close_closes_console_bar(self, mock_tqdm):
        bar = ProgressBarWithConsole(10)
        bar.update_absolute(1)
        console = mock_tqdm.return_value
        bar.close()
        console.close.assert_called_once()

    def test_close_safe_when_no_bar_created(self, mock_tqdm):
        bar = ProgressBarWithConsole(10)
        bar.close()
        mock_tqdm.return_value.close.assert_not_called()

    def test_close_idempotent(self, mock_tqdm):
        bar = ProgressBarWithConsole(10)
        bar.update_absolute(1)
        console = mock_tqdm.return_value
        bar.close()
        bar.close()
        console.close.assert_called_once()

    def test_close_safe_when_disabled(self, mock_tqdm):
        with patch.object(comfy.utils, "PROGRESS_BAR_ENABLED", False):
            bar = ProgressBarWithConsole(10)
            bar.update_absolute(5)
            bar.close()
            mock_tqdm.assert_not_called()
