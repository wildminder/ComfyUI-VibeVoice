"""Tests for nodes/external_loader_node.py - Load VibeVoice Model node."""

import contextlib
import logging
import os

import pytest
from unittest.mock import MagicMock, patch

import comfy.model_management as comfy_mm
from comfy_api.latest import io

from ComfyUI_VibeVoice.modules import model_registry
from ComfyUI_VibeVoice.modules.attention_utils import resolve_attention_mode
from ComfyUI_VibeVoice.modules.external_loader import (
    clear_reconciliation_memo,
    reconciled_config_name,
    remember_reconciled_config,
)
from ComfyUI_VibeVoice.modules.model_registry import (
    FAMILY_ASR,
    FAMILY_TTS,
    clear_active_keys,
    get_active,
    identity_for_external,
    set_active,
)
from ComfyUI_VibeVoice.modules.utils import VIBEVOICE_ASR_PATCHER_CACHE, VIBEVOICE_PATCHER_CACHE
from ComfyUI_VibeVoice.nodes.external_loader_node import VibeVoiceExternalLoaderNode

NODE = "ComfyUI_VibeVoice.nodes.external_loader_node"


class TestExternalLoaderNodeSchema:
    """Test the VibeVoiceExternalLoaderNode schema."""

    def test_node_schema_id(self):
        """define_schema().node_id == 'VibeVoiceLoadExternalModel'."""
        schema = VibeVoiceExternalLoaderNode.define_schema()
        assert schema.node_id == "VibeVoiceLoadExternalModel"

    def test_node_display_name(self):
        """Display name is 'Load VibeVoice Model'."""
        schema = VibeVoiceExternalLoaderNode.define_schema()
        assert schema.display_name == "Load VibeVoice Model"

    def test_node_has_model_file_input(self):
        """'model_file' is in the input ids."""
        schema = VibeVoiceExternalLoaderNode.define_schema()
        input_ids = [inp.id for inp in schema.inputs]
        assert "model_file" in input_ids

    def test_node_has_config_name_input(self):
        """'config_name' is in the input ids."""
        schema = VibeVoiceExternalLoaderNode.define_schema()
        input_ids = [inp.id for inp in schema.inputs]
        assert "config_name" in input_ids

    def test_node_has_attention_mode_input(self):
        """'attention_mode' is in the input ids."""
        schema = VibeVoiceExternalLoaderNode.define_schema()
        input_ids = [inp.id for inp in schema.inputs]
        assert "attention_mode" in input_ids

    def test_node_has_quantize_input(self):
        """'quantize_llm_4bit' is in the input ids."""
        schema = VibeVoiceExternalLoaderNode.define_schema()
        input_ids = [inp.id for inp in schema.inputs]
        assert "quantize_llm_4bit" in input_ids

    def test_node_has_dtype_input(self):
        """'dtype' is in the input ids."""
        schema = VibeVoiceExternalLoaderNode.define_schema()
        input_ids = [inp.id for inp in schema.inputs]
        assert "dtype" in input_ids

    def test_node_output_is_vibevoice_model(self):
        """Output type string is 'VIBEVOICE_MODEL'."""
        schema = VibeVoiceExternalLoaderNode.define_schema()
        assert len(schema.outputs) == 1
        assert schema.outputs[0].io_type == "VIBEVOICE_MODEL"

    def test_node_category(self):
        """Node category is 'WMNodes/sound/tts'."""
        assert VibeVoiceExternalLoaderNode.CATEGORY == "WMNodes/sound/tts"


class TestExternalLoaderNodeExecute:
    """Test the VibeVoiceExternalLoaderNode.execute() method."""

    def test_node_execute_calls_load_external(self):
        """execute() calls load_external_vibevoice_model with correct kwargs."""
        fake_bundle = {"model": MagicMock(), "processor": MagicMock()}

        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.load_external_vibevoice_model",
            return_value=fake_bundle,
        ) as mock_load, patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_full_path_or_raise",
            return_value="/fake/path/model.safetensors",
        ):
            result = VibeVoiceExternalLoaderNode.execute(
                model_file="model.safetensors",
                config_name="VibeVoice-1.5B",
                attention_mode="sdpa",
                quantize_llm_4bit=False,
                dtype="auto",
            )

        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["weight_path"] == "/fake/path/model.safetensors"
        assert call_kwargs["config_name"] == "VibeVoice-1.5B"
        assert call_kwargs["attention_mode"] == "sdpa"
        assert call_kwargs["use_llm_4bit"] is False
        assert call_kwargs["dtype_str"] == "auto"

    def test_node_execute_resolves_path_via_folder_paths(self):
        """execute() resolves the path via folder_paths.get_full_path_or_raise."""
        fake_bundle = {"model": MagicMock()}

        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.load_external_vibevoice_model",
            return_value=fake_bundle,
        ), patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_full_path_or_raise",
            return_value="/fake/path/model.safetensors",
        ) as mock_resolve:
            VibeVoiceExternalLoaderNode.execute(
                model_file="model.safetensors",
                config_name="VibeVoice-1.5B",
                attention_mode="sdpa",
                quantize_llm_4bit=False,
                dtype="auto",
            )

        mock_resolve.assert_called_once_with("diffusion_models", "model.safetensors")

    def test_node_execute_returns_node_output(self):
        """execute() returns an io.NodeOutput wrapping the bundle dict."""
        fake_bundle = {"model": MagicMock(), "processor": MagicMock()}

        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.load_external_vibevoice_model",
            return_value=fake_bundle,
        ), patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_full_path_or_raise",
            return_value="/fake/path/model.safetensors",
        ):
            result = VibeVoiceExternalLoaderNode.execute(
                model_file="model.safetensors",
                config_name="VibeVoice-1.5B",
                attention_mode="sdpa",
                quantize_llm_4bit=False,
                dtype="auto",
            )

        assert isinstance(result, io.NodeOutput)
        # The bundle should be the first output value
        assert result[0] is fake_bundle

    def test_node_execute_passes_quantize_flag(self):
        """execute() passes quantize_llm_4bit=True through to the loader."""
        fake_bundle = {"model": MagicMock()}

        with patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.load_external_vibevoice_model",
            return_value=fake_bundle,
        ) as mock_load, patch(
            "ComfyUI_VibeVoice.nodes.external_loader_node.folder_paths.get_full_path_or_raise",
            return_value="/fake/path/model.safetensors",
        ):
            VibeVoiceExternalLoaderNode.execute(
                model_file="model.safetensors",
                config_name="VibeVoice-7B",
                attention_mode="eager",
                quantize_llm_4bit=True,
                dtype="bf16",
            )

        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["use_llm_4bit"] is True
        assert call_kwargs["dtype_str"] == "bf16"
        assert call_kwargs["config_name"] == "VibeVoice-7B"


# ====================================================================
# Reconciliation memo: request key == consumer key across runs
# ====================================================================
#
# The loader node's unload-before-load gate runs BEFORE the model is built,
# so it can only know the post-reconciliation config_name by remembering
# what the previous load produced. Without that memo the node keys the gate
# on the REQUESTED name while the consumer keys the patcher on the
# bundle's EFFECTIVE name — every re-run then evicts and rebuilds the live
# model. All fixtures here are tmp_path files; no real checkpoint is read.


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch):
    """Isolate the active-key registry, patcher caches and the memo."""
    VIBEVOICE_PATCHER_CACHE.clear()
    VIBEVOICE_ASR_PATCHER_CACHE.clear()
    clear_active_keys()
    clear_reconciliation_memo()
    monkeypatch.setattr(comfy_mm, "soft_empty_cache", lambda: None)
    monkeypatch.setattr(comfy_mm, "current_loaded_models", [])
    yield
    VIBEVOICE_PATCHER_CACHE.clear()
    VIBEVOICE_ASR_PATCHER_CACHE.clear()
    clear_active_keys()
    clear_reconciliation_memo()


def _write_weights(tmp_path, tag, payload=b"fake-weights-"):
    f = tmp_path / f"weights_{tag}.safetensors"
    f.write_bytes(payload + tag.encode())
    return f


def _make_bundle(f, name, is_asr=False):
    """A bundle shaped exactly like load_external_vibevoice_model output."""
    st = os.stat(f)
    return {
        "model": MagicMock(),
        "processor": MagicMock(),
        "config": object(),
        "model_name": name,
        "source_path": str(f),
        "source_mtime_ns": st.st_mtime_ns,
        "source_size": st.st_size,
        "attention_mode": resolve_attention_mode("sdpa", False),
        "use_llm_4bit": False,
        "dtype_str": "auto",
        "is_streaming": False,
        "is_asr": is_asr,
    }


@contextlib.contextmanager
def _run_node(weight_path, bundle, recorded, config_name="VibeVoice-1.5B"):
    """Run execute() with path resolution + the heavy build stubbed.

    ``recorded`` collects every key handed to the real ``evict_if_changed``,
    so the assertions see the actual gate argument rather than a re-derivation.
    """
    real_evict = model_registry.evict_if_changed

    def _spy_evict(family, new_key, caches):
        recorded.append(new_key)
        return real_evict(family, new_key, caches)

    with patch(f"{NODE}.folder_paths.get_full_path_or_raise",
               return_value=str(weight_path)), \
         patch(f"{NODE}.load_external_vibevoice_model", return_value=bundle), \
         patch(f"{NODE}.evict_if_changed", side_effect=_spy_evict):
        yield


def _execute(weight_path, config_name="VibeVoice-1.5B", q4=False, dtype="auto"):
    return VibeVoiceExternalLoaderNode.execute(
        model_file=os.path.basename(weight_path),
        config_name=config_name,
        attention_mode="sdpa",
        quantize_llm_4bit=q4,
        dtype=dtype,
    )


def _expected_key(weight_path, config_name, q4=False, dtype="auto", prefix="external"):
    return identity_for_external(
        str(weight_path),
        config_name,
        resolve_attention_mode("sdpa", q4),
        use_llm_4bit=q4,
        dtype_str=dtype,
        prefix=prefix,
    )


class TestReconciledRequestKey:
    def test_request_key_uses_reconciled_name_on_second_run(self, tmp_path, caplog):
        """Run 2 must key the gate on the name the bundle actually carries.

        The bundle reports "VibeVoice-7B" (reconcile_config corrected the
        widget's "VibeVoice-1.5B"), so the consumer's patcher lives under the
        7B key. Run 2 must reproduce that key exactly and leave the live
        patcher alone.
        """
        f = _write_weights(tmp_path, "reconciled")
        bundle = _make_bundle(f, "VibeVoice-7B")
        effective_key = _expected_key(f, "VibeVoice-7B")

        recorded = []
        with caplog.at_level(logging.INFO), _run_node(f, bundle, recorded):
            _execute(f)
        # Run 1: memo is cold, the gate uses the REQUESTED name (pre-fix
        # behaviour preserved) and the load teaches the memo the real one.
        assert recorded == [_expected_key(f, "VibeVoice-1.5B")]

        # The consumer has now cached the patcher under the reconciled key and
        # recorded it as active — the state a real graph reaches after run 1.
        live_patcher = MagicMock(cache_key=effective_key)
        VIBEVOICE_PATCHER_CACHE[effective_key] = live_patcher
        set_active(FAMILY_TTS, effective_key)

        recorded.clear()
        with caplog.at_level(logging.INFO), _run_node(f, bundle, recorded):
            _execute(f)

        assert recorded == [effective_key], (
            "run 2 must key the gate on the reconciled name the consumer uses"
        )
        assert reconciled_config_name(str(f), "VibeVoice-1.5B") == "VibeVoice-7B"
        assert get_active(FAMILY_TTS) == effective_key
        assert VIBEVOICE_PATCHER_CACHE == {effective_key: live_patcher}
        live_patcher.unpatch_model.assert_not_called()
        assert "Destroying VibeVoice models" not in caplog.text

    def test_memo_invalidated_when_weight_file_changes(self, tmp_path):
        """A changed checkpoint must not stay pinned to a stale reconciled name.

        The memo key carries mtime_ns + size, so rewriting the file drops the
        entry and the gate falls back to the requested name — the safe
        direction, because a genuinely different model is still evicted.
        """
        f = _write_weights(tmp_path, "changing")
        bundle = _make_bundle(f, "VibeVoice-7B")

        recorded = []
        with _run_node(f, bundle, recorded):
            _execute(f)
        assert reconciled_config_name(str(f), "VibeVoice-1.5B") == "VibeVoice-7B"

        # 1) Same path, same size, new mtime.
        st = os.stat(f)
        os.utime(f, ns=(st.st_mtime_ns + 2_000_000_000, st.st_mtime_ns + 2_000_000_000))
        assert reconciled_config_name(str(f), "VibeVoice-1.5B") == "VibeVoice-1.5B"

        recorded.clear()
        with _run_node(f, bundle, recorded):
            _execute(f)
        assert recorded == [_expected_key(f, "VibeVoice-1.5B")]

        # 2) Same mtime, different size.
        with _run_node(f, bundle, recorded):
            _execute(f)
        assert reconciled_config_name(str(f), "VibeVoice-1.5B") == "VibeVoice-7B"
        f.write_bytes(b"fake-weights-" + b"changing" + b"-and-more-bytes")
        assert reconciled_config_name(str(f), "VibeVoice-1.5B") == "VibeVoice-1.5B"

    def test_first_run_still_evicts_the_previous_model(self, tmp_path, caplog):
        """The gate must not be 'fixed' by disabling it.

        With a cold memo and a DIFFERENT model active, one execute() still
        evicts and destroys the previous patcher before building.
        """
        f = _write_weights(tmp_path, "first_run")
        bundle = _make_bundle(f, "VibeVoice-1.5B")
        old_patcher = MagicMock(cache_key="previous_model_key")
        VIBEVOICE_PATCHER_CACHE["previous_model_key"] = old_patcher
        set_active(FAMILY_TTS, "previous_model_key")

        recorded = []
        with caplog.at_level(logging.INFO), _run_node(f, bundle, recorded):
            _execute(f)

        assert recorded == [_expected_key(f, "VibeVoice-1.5B")]
        old_patcher.unpatch_model.assert_called_once_with(
            unpatch_weights=True, destroy=True
        )
        assert "previous_model_key" not in VIBEVOICE_PATCHER_CACHE
        assert get_active(FAMILY_TTS) == _expected_key(f, "VibeVoice-1.5B")

    def test_asr_namespace_unaffected(self, tmp_path):
        """ASR still keys into its own "asr_external" namespace / FAMILY_ASR."""
        f = _write_weights(tmp_path, "asr_memo")
        bundle = _make_bundle(f, "VibeVoice-ASR", is_asr=True)

        recorded = []
        with _run_node(f, bundle, recorded, config_name="VibeVoice-ASR"):
            _execute(f, config_name="VibeVoice-ASR")

        expected = _expected_key(f, "VibeVoice-ASR", prefix="asr_external")
        assert recorded == [expected]
        assert expected.startswith("asr_external_VibeVoice-ASR@")
        assert get_active(FAMILY_ASR) == expected
        assert get_active(FAMILY_TTS) is None
        assert VIBEVOICE_PATCHER_CACHE == {}
        # The ASR branch never reconciles, so the memo only ever round-trips
        # the requested name — it cannot pull a TTS family into this namespace.
        assert reconciled_config_name(str(f), "VibeVoice-ASR") == "VibeVoice-ASR"

    def test_missing_model_name_does_not_break_execute(self, tmp_path):
        """A hand-built/mock bundle without 'model_name' must not raise."""
        f = _write_weights(tmp_path, "no_name")
        bundle = {"model": MagicMock(), "processor": MagicMock()}

        recorded = []
        with _run_node(f, bundle, recorded):
            result = _execute(f)

        assert result[0] is bundle
        assert recorded == [_expected_key(f, "VibeVoice-1.5B")]
        assert reconciled_config_name(str(f), "VibeVoice-1.5B") == "VibeVoice-1.5B"


class TestReconciliationMemo:
    def test_memo_roundtrip_unit(self, tmp_path):
        f = _write_weights(tmp_path, "unit")
        path = str(f)

        # Cold miss returns the requested name unchanged.
        assert reconciled_config_name(path, "VibeVoice-1.5B") == "VibeVoice-1.5B"

        remember_reconciled_config(path, "VibeVoice-1.5B", "VibeVoice-7B")
        assert reconciled_config_name(path, "VibeVoice-1.5B") == "VibeVoice-7B"
        # Keyed per requested name: a different selection is a different entry.
        assert reconciled_config_name(path, "VibeVoice-7B") == "VibeVoice-7B"
        # Keyed per file.
        other = str(_write_weights(tmp_path, "unit_other"))
        assert reconciled_config_name(other, "VibeVoice-1.5B") == "VibeVoice-1.5B"

        # Keyed on the absolute path: a relative spelling of the same file
        # resolves to the same entry.
        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            assert reconciled_config_name(os.path.basename(path), "VibeVoice-1.5B") == "VibeVoice-7B"
        finally:
            os.chdir(cwd)

        # A changed file invalidates the entry (mtime_ns + size).
        st = os.stat(f)
        os.utime(f, ns=(st.st_mtime_ns + 5_000_000_000, st.st_mtime_ns + 5_000_000_000))
        assert reconciled_config_name(path, "VibeVoice-1.5B") == "VibeVoice-1.5B"

        # clear() drops everything.
        remember_reconciled_config(path, "VibeVoice-1.5B", "VibeVoice-7B")
        clear_reconciliation_memo()
        assert reconciled_config_name(path, "VibeVoice-1.5B") == "VibeVoice-1.5B"

    def test_memo_degrades_on_unreadable_path(self, tmp_path):
        """A nonexistent / unstattable path never raises, it just misses."""
        missing = str(tmp_path / "does_not_exist.safetensors")
        assert not os.path.exists(missing)
        assert reconciled_config_name(missing, "VibeVoice-1.5B") == "VibeVoice-1.5B"
        # Remembering for a missing file is still allowed (keyed on placeholders).
        remember_reconciled_config(missing, "VibeVoice-1.5B", "VibeVoice-7B")
        assert reconciled_config_name(missing, "VibeVoice-1.5B") == "VibeVoice-7B"
        # An empty path is handled the same way.
        assert reconciled_config_name("", "VibeVoice-1.5B") == "VibeVoice-1.5B"
        # An empty effective name is never stored.
        remember_reconciled_config(missing, "VibeVoice-1.5B", "")
        assert reconciled_config_name(missing, "VibeVoice-1.5B") == "VibeVoice-7B"
