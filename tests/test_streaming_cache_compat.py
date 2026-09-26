"""Regression tests for the realtime voice-prompt KV cache compatibility layer.

The released ``.pt`` voice prompts store their cache in the pre-4.57
``key_cache`` / ``value_cache`` lists. transformers 5.x expects a list of cache
*layer objects* and reads ``layer.keys`` / ``layer.values`` plus
``get_seq_length()`` to size the attention mask.

A wrapper that only implemented the 4.x names did not raise — the voice
prefill was simply invisible to the model, so it conditioned on nothing and
produced garbled syllables (or an immediate end-of-speech). These tests pin the
5.x interface to the same tensors as the legacy ones.
"""

import importlib
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def mod():
    """Import the genuine vendored streaming module.

    ``conftest.py`` replaces the whole ``src.vibevoice`` tree with MagicMocks to
    keep the default suite lightweight, so the mocked entries are removed for
    this module and restored on teardown.
    """
    mocked_prefixes = ("src.vibevoice", "ComfyUI_VibeVoice.src.vibevoice")
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name.startswith(mocked_prefixes)
    }
    for name in saved:
        del sys.modules[name]
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        yield importlib.import_module(
            "ComfyUI_VibeVoice.src.vibevoice.modular."
            "modeling_vibevoice_streaming_inference"
        )
    finally:
        for name in [
            n
            for n in list(sys.modules)
            if n.startswith("src.vibevoice") or n.startswith("ComfyUI_VibeVoice.src")
        ]:
            del sys.modules[name]
        sys.modules.update(saved)


class _LegacyCache:
    """Stands in for a DynamicCache unpickled from a released .pt voice prompt."""

    def __init__(self, key_cache, value_cache):
        self.key_cache = key_cache
        self.value_cache = value_cache


def _cache(n_layers=2, seq=8, heads=2, dim=4, batch=1):
    keys = [torch.randn(batch, heads, seq, dim) for _ in range(n_layers)]
    values = [torch.randn(batch, heads, seq, dim) for _ in range(n_layers)]
    return _LegacyCache(keys, values)


def _real_dynamic_cache(n_layers=3, seq=8, heads=2, dim=4, batch=1):
    """A genuine 5.x ``DynamicCache`` carrying the *legacy* pickle shape.

    The released prompts are a ``DynamicCache`` pickled by an older
    transformers: the object is of that class, but its state lives in
    ``key_cache`` / ``value_cache`` and it has no ``layers`` at all. Building
    one here means the container-level assertions run against the installed
    ``DynamicCache`` methods rather than against a stand-in.
    """
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    for layer_idx in range(n_layers):
        cache.update(
            torch.randn(batch, heads, seq, dim),
            torch.randn(batch, heads, seq, dim),
            layer_idx,
        )
    cache.key_cache = [layer.keys for layer in cache.layers]
    cache.value_cache = [layer.values for layer in cache.layers]
    del cache.layers
    return cache


class TestCacheLayerSurfaceMatchesTheInstalledMixin:
    """S2.1 anti-regression: the wrapper must cover the *installed* mixin API.

    The comparison is made against ``transformers.cache_utils.CacheLayerMixin``
    at test time, so a transformers bump that adds (or renames) an accessor
    fails here loudly instead of degrading silently into "the prefill is
    invisible" - the failure mode that produced F2.
    """

    @staticmethod
    def _required_surface():
        from transformers.cache_utils import CacheLayerMixin

        return {name for name in dir(CacheLayerMixin) if not name.startswith("_")}

    @staticmethod
    def _public(obj):
        return {name for name in dir(obj) if not name.startswith("_")}

    def _missing(self, implemented):
        return self._required_surface() - implemented

    def test_required_surface_is_not_empty(self):
        # Guards the test itself: an empty requirement would make the check
        # below vacuously pass on a broken import.
        assert {"offload", "prefetch", "reorder_cache", "update"} <= self._required_surface()

    def test_mock_layer_implements_every_mixin_accessor(self, mod):
        missing = self._missing(self._public(mod.MockCacheLayer))
        assert not missing, f"MockCacheLayer is missing {sorted(missing)}"

    def test_the_check_bites_when_an_accessor_is_deleted(self, mod):
        """Mutation guard: drop one accessor and the surface check must fail.

        A contract test that cannot fail proves nothing, so the same predicate
        is re-run against (a) the real surface minus one member and (b) a class
        that lost the whole mixin contract by no longer inheriting it.
        """
        complete = self._public(mod.MockCacheLayer)
        assert not self._missing(complete)
        assert self._missing(complete - {"prefetch"}) == {"prefetch"}

        class _NotACacheLayer:
            """Everything deleted - the degenerate case of the same bug."""

        assert self._missing(self._public(_NotACacheLayer)) == self._required_surface()

    def test_the_5x_only_accessors_are_present(self, mod):
        # Named explicitly so the intent survives a mixin reshuffle: these are
        # the three that were missing and that ``Cache`` calls on the offload,
        # prefetch and beam-search paths.
        for name in ("offload", "prefetch", "reorder_cache"):
            assert callable(getattr(mod.MockCacheLayer, name, None)), name


class TestCacheLayerExposesModernApi:
    def test_layers_are_built_from_the_legacy_lists(self, mod):
        cache = mod._ensure_cache_has_layers(_cache(n_layers=3))
        assert len(cache.layers) == 3

    def test_5x_and_legacy_names_alias_the_same_tensor(self, mod):
        cache = mod._ensure_cache_has_layers(_cache())
        layer = cache.layers[0]
        assert layer.keys is layer.key_cache
        assert layer.values is layer.value_cache

    def test_get_seq_length_reports_the_prefill_length(self, mod):
        cache = mod._ensure_cache_has_layers(_cache(seq=108))
        # This is what the 5.x attention path calls to build the attention mask.
        assert cache.layers[0].get_seq_length() == 108

    def test_get_mask_sizes_includes_the_query_length(self, mod):
        cache = mod._ensure_cache_has_layers(_cache(seq=316))
        kv_len, offset = cache.layers[0].get_mask_sizes(torch.arange(5))
        assert kv_len == 316 + 5
        assert offset == 0

    def test_is_initialized_reflects_presence(self, mod):
        populated = mod._ensure_cache_has_layers(_cache(seq=8))
        empty = mod._ensure_cache_has_layers(_cache(n_layers=1, seq=0))
        assert populated.layers[0].is_initialized is True
        assert empty.layers[0].is_initialized is False

    def test_empty_cache_reports_zero_length(self, mod):
        empty = mod._ensure_cache_has_layers(_cache(n_layers=1, seq=0))
        assert empty.layers[0].get_seq_length() == 0

    def test_sliding_and_compileable_flags_exist(self, mod):
        layer = mod._ensure_cache_has_layers(_cache()).layers[0]
        assert layer.is_sliding is False
        assert hasattr(layer, "is_compileable")

    def test_lazy_initialization_is_a_noop(self, mod):
        layer = mod._ensure_cache_has_layers(_cache()).layers[0]
        before = layer.get_seq_length()
        layer.lazy_initialization(torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4))
        assert layer.get_seq_length() == before


class TestCacheLayerUpdate:
    def test_update_appends_to_the_parent_cache(self, mod):
        cache = mod._ensure_cache_has_layers(_cache(seq=8))
        layer = cache.layers[0]
        new_k = torch.randn(1, 2, 1, 4)
        new_v = torch.randn(1, 2, 1, 4)

        keys, values = layer.update(new_k, new_v)

        assert keys.shape[-2] == 9
        assert values.shape[-2] == 9
        # The parent list is the single source of truth for both APIs.
        assert cache.key_cache[0].shape[-2] == 9
        assert layer.get_seq_length() == 9

    def test_update_on_a_detached_layer_returns_its_own_tensors(self, mod):
        keys = torch.randn(1, 2, 3, 4)
        values = torch.randn(1, 2, 3, 4)
        layer = mod.MockCacheLayer(keys, values)

        out_k, out_v = layer.update(torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4))

        assert out_k.shape[-2] == 3
        assert out_v.shape[-2] == 3

    def test_legacy_attribute_writes_stay_visible_to_5x_readers(self, mod):
        layer = mod._ensure_cache_has_layers(_cache(seq=4)).layers[0]
        replacement = torch.randn(1, 2, 11, 4)
        layer.key_cache = replacement
        assert layer.keys is replacement
        assert layer.get_seq_length() == 11

    def test_update_concatenates_on_the_cache_axis(self, mod):
        cache = mod._ensure_cache_has_layers(_cache(seq=3, heads=2, dim=5))
        layer = cache.layers[0]
        before = layer.keys.clone()

        out_k, out_v = layer.update(torch.randn(1, 2, 2, 5), torch.randn(1, 2, 2, 5))

        # dim=-2 is the sequence axis for [batch, heads, seq, head_dim]; a
        # concatenation on any other axis would silently corrupt RoPE alignment.
        assert out_k.shape == (1, 2, 5, 5)
        assert out_v.shape == (1, 2, 5, 5)
        assert torch.equal(out_k[..., :3, :], before)


class TestCacheLayerOffloadPrefetchReorder:
    """S2.1 behaviour for the three accessors the wrapper was missing."""

    def test_device_and_dtype_come_from_the_key_tensor(self, mod):
        layer = mod.MockCacheLayer(
            torch.randn(1, 2, 4, 5, dtype=torch.float16), torch.randn(1, 2, 4, 5)
        )
        assert layer.device == torch.device("cpu")
        assert layer.dtype == torch.float16

    def test_offload_moves_both_tensors_to_cpu_and_keeps_the_parent_in_sync(self, mod):
        cache = mod._ensure_cache_has_layers(_cache(seq=6))
        layer = cache.layers[0]
        original_keys = layer.keys.clone()

        layer.offload()

        assert layer.keys.device.type == "cpu"
        assert layer.values.device.type == "cpu"
        # No data loss, and the legacy spelling sees the same object.
        assert torch.equal(layer.keys, original_keys)
        assert cache.key_cache[0] is layer.keys
        assert cache.value_cache[0] is layer.values

    def test_prefetch_is_a_noop_when_already_on_the_layer_device(self, mod):
        cache = mod._ensure_cache_has_layers(_cache(seq=6))
        layer = cache.layers[0]
        before = layer.keys.clone()

        layer.prefetch()

        assert layer.keys.device == layer.device
        assert torch.equal(layer.keys, before)

    def test_prefetch_brings_an_offloaded_layer_back_to_its_device(self, mod):
        if not torch.cuda.is_available():
            pytest.skip("a real offload/prefetch move needs a second device")
        layer = mod.MockCacheLayer(torch.randn(1, 2, 4, 5), torch.randn(1, 2, 4, 5))
        layer.device = torch.device("cuda")
        before = layer.keys.clone()

        layer.offload()
        assert layer.keys.device.type == "cpu"

        layer.prefetch()
        assert layer.keys.device.type == "cuda"
        assert torch.equal(layer.keys.cpu(), before)

    def test_offload_then_prefetch_round_trips_without_data_loss(self, mod):
        cache = mod._ensure_cache_has_layers(_cache(seq=6))
        layer = cache.layers[0]
        before_k, before_v = layer.keys.clone(), layer.values.clone()

        layer.offload()
        layer.prefetch()

        assert torch.equal(layer.keys, before_k)
        assert torch.equal(layer.values, before_v)

    def test_reorder_cache_selects_the_new_batch_order(self, mod):
        cache = mod._ensure_cache_has_layers(_cache(seq=6, batch=3))
        layer = cache.layers[0]
        rows = layer.keys[:, :, 0, 0].clone()

        layer.reorder_cache(torch.tensor([2, 0, 1]))

        assert torch.equal(layer.keys[:, :, 0, 0], rows[[2, 0, 1]])
        assert cache.key_cache[0] is layer.keys

    def test_reorder_cache_on_an_empty_layer_is_a_noop(self, mod):
        layer = mod.MockCacheLayer(None, None)
        layer.reorder_cache(torch.tensor([0]))
        assert layer.keys is None

    def test_crop_and_batch_helpers_keep_the_parent_in_sync(self, mod):
        cache = mod._ensure_cache_has_layers(_cache(seq=6, batch=2))
        layer = cache.layers[0]

        layer.crop(4)
        assert layer.get_seq_length() == 4
        assert cache.key_cache[0].shape[-2] == 4

        layer.batch_repeat_interleave(2)
        assert layer.keys.shape[0] == 4
        assert cache.key_cache[0].shape[0] == 4

        layer.batch_select_indices(torch.tensor([2, 3]))
        assert layer.keys.shape[0] == 2
        assert cache.key_cache[0].shape[0] == 2


class TestContainerParityOnARealDynamicCache:
    """S2.2: the *container* must behave like a 5.x cache, not just its layers.

    Every assertion below runs on an object whose class is the installed
    ``transformers.cache_utils.DynamicCache``, so it is the real container
    implementation being exercised - ``_ensure_cache_has_layers`` only adapts
    it (populates ``layers`` and the stateful offload bookkeeping) rather than
    reimplementing it.
    """

    def test_layers_are_populated_before_any_forward(self, mod):
        cache = _real_dynamic_cache(n_layers=3, seq=316)
        assert getattr(cache, "layers", None) in (None, [])

        wrapped = mod._ensure_cache_has_layers(cache)

        # The ordering is the whole point: ``Cache.get_mask_sizes`` answers
        # ``(query_length, 0)`` while ``layers`` is short, which makes the voice
        # prefill invisible with no exception raised.
        assert len(wrapped.layers) == 3
        assert wrapped.layers[0].get_seq_length() == 316

    def test_container_get_seq_length_returns_the_prefill(self, mod):
        wrapped = mod._ensure_cache_has_layers(_real_dynamic_cache(seq=316))
        assert wrapped.get_seq_length() == 316

    def test_container_get_mask_sizes_returns_prefill_plus_query(self, mod):
        wrapped = mod._ensure_cache_has_layers(_real_dynamic_cache(seq=316))
        kv_length, kv_offset = wrapped.get_mask_sizes(torch.arange(316, 321), 0)
        assert kv_length == 316 + 5
        assert kv_offset == 0

    def test_container_get_max_cache_shape_matches_the_layer(self, mod):
        wrapped = mod._ensure_cache_has_layers(_real_dynamic_cache(seq=8))
        assert wrapped.get_max_cache_shape() == -1

    def test_container_offload_then_prefetch_round_trips(self, mod):
        wrapped = mod._ensure_cache_has_layers(_real_dynamic_cache(seq=6))
        before_k = wrapped.layers[0].keys.clone()
        before_v = wrapped.layers[0].values.clone()

        wrapped.offload(0)
        assert wrapped.layers[0].keys.device.type == "cpu"
        assert wrapped.key_cache[0].device.type == "cpu"

        wrapped.prefetch(0)
        assert torch.equal(wrapped.layers[0].keys, before_k)
        assert torch.equal(wrapped.layers[0].values, before_v)
        assert wrapped.get_seq_length() == 6

    def test_container_crop_shortens_the_prefill(self, mod):
        wrapped = mod._ensure_cache_has_layers(_real_dynamic_cache(seq=10))
        wrapped.crop(4)
        assert wrapped.get_seq_length() == 4
        assert wrapped.key_cache[0].shape[-2] == 4
        assert wrapped.get_mask_sizes(torch.arange(4, 9), 0)[0] == 9

    def test_container_batch_repeat_interleave_and_select_indices(self, mod):
        wrapped = mod._ensure_cache_has_layers(
            _real_dynamic_cache(seq=5, batch=2)
        )
        wrapped.batch_repeat_interleave(3)
        assert wrapped.layers[0].keys.shape[0] == 6
        assert wrapped.key_cache[0].shape[0] == 6

        wrapped.batch_select_indices(torch.tensor([4, 5]))
        assert wrapped.layers[0].keys.shape[0] == 2
        assert wrapped.key_cache[0].shape[0] == 2

    def test_container_reorder_cache_follows_the_beam_index(self, mod):
        wrapped = mod._ensure_cache_has_layers(_real_dynamic_cache(seq=5, batch=3))
        rows = wrapped.layers[0].keys[:, :, 0, 0].clone()
        wrapped.reorder_cache(torch.tensor([1, 2, 0]))
        assert torch.equal(wrapped.layers[0].keys[:, :, 0, 0], rows[[1, 2, 0]])

    def test_container_is_sliding_and_is_compileable_agree_with_the_layers(self, mod):
        wrapped = mod._ensure_cache_has_layers(_real_dynamic_cache(n_layers=2))
        assert wrapped.is_sliding == [False, False]
        assert wrapped.is_compileable is False
        assert wrapped.is_initialized is True

    def test_legacy_pickle_carrying_offloading_true_is_corrected(self, mod):
        """A stale ``offloading=True`` is a live AttributeError on the offload path.

        ``Cache.__init__`` creates ``prefetch_stream`` only when ``offloading``
        is truthy, and ``Cache.update`` reads that stream on every step. A
        pickle that carries the flag without the stream therefore fails inside
        the model's first forward, so the adapter has to drop the flag.
        """
        cache = _real_dynamic_cache(seq=4)
        cache.offloading = True
        assert not hasattr(cache, "prefetch_stream")

        wrapped = mod._ensure_cache_has_layers(cache)

        assert wrapped.offloading is False
        # One real Cache.update must not raise.
        keys, values = wrapped.update(
            torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4), 0
        )
        assert keys.shape[-2] == 5
        assert values.shape[-2] == 5

    def test_offloading_with_a_real_stream_is_left_alone(self, mod):
        cache = _real_dynamic_cache(seq=4)
        cache.offloading = True
        cache.prefetch_stream = torch.Stream()

        wrapped = mod._ensure_cache_has_layers(cache)

        assert wrapped.offloading is True
        assert wrapped.prefetch_stream is cache.prefetch_stream

    def test_adapted_cache_exposes_a_prefetch_stream(self, mod):
        # ``Cache.offload``/``Cache.prefetch`` open ``prefetch_stream``
        # unconditionally, so it has to exist even with offloading off.
        wrapped = mod._ensure_cache_has_layers(_real_dynamic_cache(seq=4))
        assert hasattr(wrapped, "prefetch_stream")

    def test_adapted_cache_stays_deep_copyable(self, mod):
        # A cached voice prompt is deep-copied once per generation. A real
        # torch.Stream cannot be copied ("cannot pickle 'torch.Stream' object"),
        # so the stream the adapter provides must not make the cache uncopyable.
        import copy as _copy

        wrapped = mod._ensure_cache_has_layers(_real_dynamic_cache(seq=4))
        clone = _copy.deepcopy(wrapped)

        assert clone.get_seq_length() == 4
        assert len(clone.layers) == len(wrapped.layers)
        assert clone.layers[0].keys is clone.key_cache[0]

    def test_adaptation_preserves_a_preconfigured_offloading_flag(self, mod):
        cache = _real_dynamic_cache(seq=4)
        cache.offloading = False

        wrapped = mod._ensure_cache_has_layers(cache)

        assert wrapped.offloading is False


class TestMaskConstructionConsumesThePrefill:
    """S2.3 tier C: the mask 5.3 actually hands to attention must be 321 wide.

    No weights are needed: a two-layer Qwen2 is built from a tiny config on CPU,
    exactly as the vendored model builds its LMs (``AutoModel.from_config`` on a
    Qwen2 config), and the mask is captured with a forward pre-hook registered
    with ``with_kwargs=True`` because ``modeling_qwen2`` passes it by keyword.

    The arithmetic that matters: ``cache_position`` is ``arange(316, 321)`` and
    ``masking_utils`` uses those *real* positions as the query indices, so every
    prefill key is visible by construction. What has to be checked is that the
    kv length is ``316 + 5`` and not the ``cache_position`` length - an
    unpopulated ``layers`` list makes ``Cache.get_mask_sizes`` answer
    ``(query_length, 0)`` silently.
    """

    PREFILL = 316
    QUERY = 5
    LAYERS = 2
    HEADS = 4
    KV_HEADS = 2
    HEAD_DIM = 16

    def _tiny_qwen2(self):
        from transformers import AutoModel
        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

        config = Qwen2Config(
            vocab_size=256,
            hidden_size=self.HEADS * self.HEAD_DIM,
            intermediate_size=64,
            num_hidden_layers=self.LAYERS,
            num_attention_heads=self.HEADS,
            num_key_value_heads=self.KV_HEADS,
            max_position_embeddings=1024,
        )
        return AutoModel.from_config(config).eval()

    def _prefilled_cache(self, mod, n_layers=None):
        return mod._ensure_cache_has_layers(
            _real_dynamic_cache(
                n_layers=n_layers or self.LAYERS,
                seq=self.PREFILL,
                heads=self.KV_HEADS,
                dim=self.HEAD_DIM,
            )
        )

    def _run(self, model, cache, cache_position):
        """One forward, returning every mask handed to self-attention."""
        seen = []

        def record(module, args, kwargs):
            seen.append(kwargs.get("attention_mask"))

        handles = [
            layer.self_attn.register_forward_pre_hook(record, with_kwargs=True)
            for layer in model.layers
        ]
        try:
            with torch.no_grad():
                model(
                    input_ids=torch.randint(0, 256, (1, cache_position.shape[0])),
                    attention_mask=torch.ones(
                        1, self.PREFILL + cache_position.shape[0], dtype=torch.long
                    ),
                    past_key_values=cache,
                    cache_position=cache_position,
                    use_cache=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        return seen

    def test_container_reports_prefill_plus_query_before_the_forward(self, mod):
        cache = self._prefilled_cache(mod)
        cache_position = torch.arange(self.PREFILL, self.PREFILL + self.QUERY)

        assert len(cache.layers) == self.LAYERS
        assert cache.get_seq_length() == self.PREFILL
        assert cache.get_mask_sizes(cache_position, 0) == (self.PREFILL + self.QUERY, 0)

    def test_an_unadapted_cache_would_silently_lose_the_prefill(self, mod):
        """The contrast case: this is what F2 looks like from the mask side.

        With ``layers`` empty, ``Cache.get_mask_sizes`` falls back to the query
        length. No exception, no warning - just a model that conditions on
        nothing. Pinning this makes the "no bypass" claim above falsifiable.
        """
        cache = _real_dynamic_cache(n_layers=self.LAYERS, seq=self.PREFILL, dim=self.HEAD_DIM)
        cache.layers = []
        cache_position = torch.arange(self.PREFILL, self.PREFILL + self.QUERY)

        assert cache.get_mask_sizes(cache_position, 0) == (self.QUERY, 0)

    def test_attention_receives_a_321_wide_mask(self, mod):
        model = self._tiny_qwen2()
        cache = self._prefilled_cache(mod)
        cache_position = torch.arange(self.PREFILL, self.PREFILL + self.QUERY)

        masks = self._run(model, cache, cache_position)

        assert len(masks) == self.LAYERS
        for mask in masks:
            assert mask is not None, "a 5-token window must not take the is_causal_skip path"
            assert tuple(mask.shape) == (1, 1, self.QUERY, self.PREFILL + self.QUERY)

    def test_the_mask_is_causal_against_the_real_cache_positions(self, mod):
        model = self._tiny_qwen2()
        cache = self._prefilled_cache(mod)
        cache_position = torch.arange(self.PREFILL, self.PREFILL + self.QUERY)

        mask = self._run(model, cache, cache_position)[0][0, 0].bool()

        for row, query_position in enumerate(cache_position.tolist()):
            visible = mask[row]
            # A query may attend to itself and to everything before it, and to
            # nothing after it.
            expected = torch.arange(self.PREFILL + self.QUERY) <= query_position
            assert torch.equal(visible, expected), f"row {row} (cache_position {query_position})"

    def test_no_prefill_position_is_masked_out(self, mod):
        model = self._tiny_qwen2()
        cache = self._prefilled_cache(mod)
        cache_position = torch.arange(self.PREFILL, self.PREFILL + self.QUERY)

        mask = self._run(model, cache, cache_position)[0][0, 0].bool()

        # Every one of the 316 cached voice-prompt positions stays visible to
        # every query row: that is the whole claim of F3.
        assert mask[:, : self.PREFILL].all()


class TestReleasedVoicePromptPickle:
    """S2.2: the real .pt preset, not a hand-built stand-in.

    Skipped when the local checkpoint/preset is absent, because the default
    suite must stay runnable without a model download.
    """

    @staticmethod
    def _preset_path():
        """Resolve a real preset the way the node does, or ``None``.

        Registration is the production path (``folder_paths`` only learns about
        the TTS roots once ``extra_model_paths.yaml`` has been read and the
        node has registered them), so anything less would skip the test on a
        machine that in fact has the voices on disk.
        """
        import os

        import folder_paths

        from modules.folder_registration import (
            register_voice_preset_folder,
            register_vibevoice_folders,
        )
        from modules.voice_presets import resolve_voice_preset_path

        name = os.environ.get("VIBEVOICE_REALTIME_VOICE_PRESET", "en-Carter_man")
        try:
            config = Path(os.environ.get("COMFYUI_ROOT", "")) / "extra_model_paths.yaml"
            if config.is_file():
                import utils.extra_config as extra_config

                extra_config.load_extra_path_config(str(config))
            roots = register_vibevoice_folders(folder_paths)
            register_voice_preset_folder(folder_paths, roots[0], roots[1:])
            return resolve_voice_preset_path(name)
        except Exception:  # noqa: BLE001 - no local voices is a skip, not a failure
            return None

    def test_real_preset_reports_sane_cache_state(self, mod):
        path = self._preset_path()
        if path is None:
            pytest.skip("no local voice preset available")

        from modules.voice_presets import load_voice_preset

        preset = load_voice_preset(path, torch.device("cpu"))
        cache = mod._ensure_cache_has_layers(preset["tts_lm"].past_key_values)

        import transformers.cache_utils as cache_utils

        # The unpickled object IS the installed DynamicCache - the adapter
        # only populates state, it does not reimplement the container.
        assert isinstance(preset["tts_lm"].past_key_values, cache_utils.DynamicCache)
        assert len(cache.layers) == len(cache.key_cache) > 0

        cache_position = torch.arange(cache.get_seq_length(), cache.get_seq_length() + 5)
        assert cache.get_seq_length() > 0
        assert cache.get_mask_sizes(cache_position, 0) == (cache.get_seq_length() + 5, 0)
        assert cache.is_sliding == [False] * len(cache.layers)
        assert cache.is_initialized is True
        assert cache.offloading is False
        assert cache.layers[0].keys is cache.key_cache[0]


class TestEnsureCacheHasLayers:
    def test_none_passes_through(self, mod):
        assert mod._ensure_cache_has_layers(None) is None

    def test_is_idempotent(self, mod):
        cache = _cache(n_layers=2)
        first = mod._ensure_cache_has_layers(cache)
        second = mod._ensure_cache_has_layers(cache)
        assert first is second
        assert len(second.layers) == 2

    def test_a_cache_with_no_legacy_lists_gets_an_empty_layer_list(self, mod):
        class Bare:
            pass

        cache = mod._ensure_cache_has_layers(Bare())
        assert cache.layers == []

    def test_an_object_that_rejects_attributes_is_left_alone(self, mod):
        # ``object()`` has no __dict__, so every setattr raises and the
        # defensive branches swallow it rather than breaking the caller.
        cache = mod._ensure_cache_has_layers(object())
        assert not hasattr(cache, "layers")


class TestCachePositionContract:
    """``cache_position`` must name the tokens about to be fed, and only those.

    This is the contract transformers 4.57.6 implemented and 5.3.0 changed.
    4.57.6 returned ``cache_position[-1:] + num_new_tokens`` — a tensor of
    exactly ``num_new_tokens`` entries. 5.3.0 returns
    ``torch.cat((cache_position, next_cache_position))``
    (``transformers/generation/utils.py:938``), i.e. the whole history with the
    new positions appended, so it grows by one entry per step.

    The vendored ``prepare_inputs_for_generation`` slices its inputs with
    ``input_ids[:, -cache_position.shape[0]:]``, so the widened tensor silently
    re-feeds every already-cached token: measured on VibeVoice-Realtime-0.5B the
    KV cache grew 316 -> 321 -> 327 -> 334 -> 342 -> 351 -> 361 -> 372 instead
    of by one per speech token. The mask is built correctly *for the wrong
    query*, so nothing raises — the model just never reaches its own
    end-of-speech and every clip runs to the length budget.
    """

    @staticmethod
    def _update(mod, cache_position, num_new_tokens=1):
        """Drive the vendored override with a stand-in for the base class.

        The override delegates to ``super()._update_model_kwargs_for_generation``
        and then trims. Faking the super() call is what makes this a pure unit
        test: the assertion is on what the override hands back, not on which
        transformers is installed.
        """
        import transformers

        owner = mod.VibeVoiceStreamingForConditionalGenerationInference
        installed = transformers.generation.utils.GenerationMixin._update_model_kwargs_for_generation

        def fake_super(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
            # Exactly transformers 5.3.0's implementation, lines 932-939.
            cache_position = model_kwargs.get("cache_position")
            if cache_position is not None:
                model_kwargs["cache_position"] = torch.cat(
                    (
                        cache_position,
                        torch.arange(num_new_tokens, dtype=torch.long) + cache_position[-1] + 1,
                    )
                )
            return model_kwargs

        try:
            transformers.generation.utils.GenerationMixin._update_model_kwargs_for_generation = fake_super
            instance = owner.__new__(owner)
            return owner._update_model_kwargs_for_generation(
                instance, None, {"cache_position": cache_position}, num_new_tokens=num_new_tokens
            )
        finally:
            transformers.generation.utils.GenerationMixin._update_model_kwargs_for_generation = installed

    def test_a_single_new_token_yields_a_single_position(self, mod):
        out = self._update(mod, torch.tensor([318, 319, 320]))
        assert out["cache_position"].tolist() == [321]

    def test_a_text_window_yields_exactly_that_many_positions(self, mod):
        out = self._update(mod, torch.tensor([318, 319, 320]), num_new_tokens=5)
        assert out["cache_position"].tolist() == [321, 322, 323, 324, 325]

    def test_the_history_is_never_carried_forward(self, mod):
        """The whole point: the tensor must not start at position 0.

        5.3.0 hands back ``[0, 1, ..., 320, 321]``. Feeding that to
        ``prepare_inputs_for_generation`` makes the query window 322 tokens
        wide, which is the defect this pins.
        """
        history = torch.arange(0, 321)
        out = self._update(mod, history)
        assert out["cache_position"].numel() == 1
        assert int(out["cache_position"][0]) > int(history[-1])

    def test_it_stays_correct_over_many_consecutive_steps(self, mod):
        """Feed the result back in, as the generation loop does.

        The 4.x contract is a fixed point: each step advances the position by
        exactly one, whatever history preceded it.
        """
        cache_position = torch.tensor([318, 319, 320])
        for expected in range(321, 360):
            cache_position = self._update(mod, cache_position)["cache_position"]
            assert cache_position.tolist() == [expected]

    def test_model_kwargs_without_a_cache_position_are_left_alone(self, mod):
        out = self._update(mod, None)
        assert out["cache_position"] is None

    def test_zero_new_tokens_does_not_slice_the_whole_history(self, mod):
        """``tensor[-0:]`` is the entire tensor, not an empty tail.

        The generation loop never asks for zero new tokens, but a guard that
        silently inverted on that input would be a trap for the next caller.
        """
        history = torch.arange(0, 321)
        out = self._update(mod, history, num_new_tokens=0)
        assert out["cache_position"].tolist() == history.tolist()
