"""C3: Weight-visibility & VRAM accounting audit (AUD-002).

Determines empirically whether the real VibeVoice model's parameters are
visible to ComfyUI's ``ModelPatcher._load_list()`` / ``module_size()`` after
the handler's lazy ``load_model`` assigns the inner model.

Key mechanism under test: ``torch.nn.Module.__setattr__`` auto-registers any
``nn.Module``-valued attribute as a child module. ``VibeVoiceModelHandler``
sets ``self.model = None`` in ``__init__`` (plain attribute) and later assigns
the real model in ``load_model`` — at which point it should become a child and
its parameters should appear in ``handler.parameters()`` /
``handler.named_modules()``.
"""

import torch
from unittest.mock import patch

from ComfyUI_VibeVoice.modules.patcher import VibeVoicePatcher


class _InnerModel(torch.nn.Module):
    """A stand-in for the heavy VibeVoice model with real parameters."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)


class _HandlerLikeReal(torch.nn.Module):
    """Mirrors VibeVoiceModelHandler's lazy-load attribute pattern exactly."""

    def __init__(self):
        super().__init__()
        self.model = None  # plain attribute pre-load (like the real handler)
        self.processor = None
        self.model_pack_name = "vis"
        self.cache_key = "vis"
        self.size = 1024

    def load_model(self, device, attention_mode: str = "sdpa"):
        # Lazy instantiation, exactly like the real handler's load_model.
        if self.model is None:
            self.model = _InnerModel()
        self.model.to(device)


class TestWeightVisibility:
    """Does the inner model's weight become visible to the patcher?"""

    def test_pre_load_handler_has_no_parameters(self):
        handler = _HandlerLikeReal()
        # Before load_model, model is None → no registered children.
        assert list(handler.parameters()) == []
        assert handler.model is None

    def test_post_load_inner_model_is_registered_child(self):
        handler = _HandlerLikeReal()
        handler.load_model(torch.device("cpu"))
        # nn.Module.__setattr__ must have registered the inner model as a child.
        assert "model" in dict(handler.named_children())
        # And its parameters are now visible on the handler.
        params = list(handler.parameters())
        assert len(params) > 0

    def test_post_load_named_modules_includes_inner_weights(self):
        handler = _HandlerLikeReal()
        handler.load_model(torch.device("cpu"))
        module_names = [n for n, _ in handler.named_modules()]
        # The inner model and its submodule must be enumerable.
        assert any("model" in n for n in module_names)
        assert any("linear" in n for n in module_names)

    def test_module_size_reports_real_weights_post_load(self):
        import comfy.model_management as mm

        handler = _HandlerLikeReal()
        handler.load_model(torch.device("cpu"))
        size = mm.module_size(handler)
        # 16x16 weight + 16 bias, float32 → > 0 bytes.
        assert size > 0


class TestPatcherLoadListVisibility:
    """Does ModelPatcher._load_list() see the inner weights after load?"""

    def _build_patcher(self, handler):
        with patch("comfy.model_patcher.ModelPatcher.__init__"):
            patcher = VibeVoicePatcher(
                handler,
                attention_mode="sdpa",
                load_device=torch.device("cpu"),
                offload_device=torch.device("cpu"),
                size=1024,
            )
        patcher.load_device = torch.device("cpu")
        patcher.offload_device = torch.device("cpu")
        patcher.model = handler
        patcher.pinned = set()  # attr normally set by ModelPatcher.__init__
        return patcher

    def test_load_list_empty_before_load(self):
        handler = _HandlerLikeReal()
        patcher = self._build_patcher(handler)
        # Pre-load: no weights visible to the patcher's load list.
        assert patcher._load_list() == []

    def test_load_list_sees_weights_after_load(self):
        handler = _HandlerLikeReal()
        patcher = self._build_patcher(handler)
        handler.load_model(torch.device("cpu"))
        loading = patcher._load_list()
        # Post-load: the inner model's parameterized modules are enumerable.
        assert len(loading) > 0

    def test_loaded_size_reflects_model_loaded_weight_memory(self):
        handler = _HandlerLikeReal()
        patcher = self._build_patcher(handler)
        # ModelPatcher.__init__ normally seeds this attribute; replicate it.
        handler.model_loaded_weight_memory = 0
        assert patcher.loaded_size() == 0
        handler.model_loaded_weight_memory = 12345
        assert patcher.loaded_size() == 12345
