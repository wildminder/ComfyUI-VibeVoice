"""End-to-end DynamicVRAM mechanism test on a tiny synthetic checkpoint.

This is the structural proof of the RAM-spike fix, run against the REAL
comfy-aimdo stack (the same one ``main.py`` initialises) — not a mock:

1. ``comfy.utils.load_torch_file`` under ``aimdo_enabled`` returns zero-copy
   views into a ``ModelMMAP`` mapping, each storage tagged
   ``_comfy_tensor_file_slice`` (comfy/utils.py:105-154).
2. Our dense loader with ``preserve_file_views=True`` keeps those views in
   the parameters (assign semantics, no clone) — the exact thing the native
   Load Diffusion Model node does via ``assign=is_dynamic()``
   (comfy/sd.py:2407).
3. ``convert_tree_for_streaming`` installs ``comfy_cast_weights`` — the
   attribute core's dynamic ``load()`` gates on to allocate vbar ranges
   (comfy/model_patcher.py:1967) instead of stashing host backups
   (:2000-2009).
4. ``load_to_device`` -> ``load_models_gpu`` pages the weights, and a real
   forward reads disk->VRAM through ``read_tensor_file_slice_into``
   (comfy/memory_management.py:57-63).

Everything here is KB-scale synthetic tensors (the user's standing rule:
never load huge models in tests). The whole test is skipped when the native
stack refuses to initialise headless (no GPU, driver state) — it must never
fail a suite for environmental reasons.
"""

import contextlib
import gc
import os
import tempfile

import pytest
import torch
from torch import nn

import comfy.memory_management
import comfy.model_management
import comfy.model_patcher
import comfy.utils
from comfy_aimdo import control

from ComfyUI_VibeVoice.modules.comfy_stream import convert_tree_for_streaming
from ComfyUI_VibeVoice.modules.external_loader import _load_state_dict_into_model_from_memory
from ComfyUI_VibeVoice.modules.memory_census import census
from ComfyUI_VibeVoice.modules.patcher import (
    load_to_device,
    make_dynamic_patcher_class,
    select_patcher_class,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DynamicVRAM needs a real CUDA device"
)


def _aimdo_ready() -> bool:
    """True when control.init + init_devices both succeed (main.py:74/:278).

    Also repairs comfy_aimdo's own import-order trap: ``host_buffer`` /
    ``model_mmap`` / ``model_vbar`` / ``vram_buffer`` / ``storage`` snapshot
    ``lib = control.lib`` AT IMPORT TIME (e.g. host_buffer.py:6). Under
    pytest those modules are imported (via ``comfy.memory_management``)
    before ``control.init()`` loads the native library, so their ``lib`` is
    None forever and ``register_load_device`` would crash on
    ``hostbuf_allocate``. Production never sees this because ``main.py``
    calls ``control.init()`` before anything imports the submodules. A
    reload re-runs each module's top-level binding and argtypes setup after
    the library exists; callers resolve ``host_buffer.read_file_to_device``
    through the module at call time, so the reload is transparent.
    """
    import importlib

    try:
        if not control.init():
            return False
        if not bool(
            control.init_devices(
                (d.index, int(2 * 1024 ** 3))
                for d in comfy.model_management.get_all_torch_devices()
            )
        ):
            return False
        import comfy_aimdo

        for name in ("host_buffer", "model_mmap", "model_vbar", "vram_buffer", "storage"):
            submodule = getattr(comfy_aimdo, name, None)
            if submodule is not None and getattr(submodule, "lib", None) is None:
                importlib.reload(submodule)
        return True
    except Exception:
        return False


def _tiny_handler(module):
    """The pack's handler (ExternalVibeVoiceModelHandler) is itself an
    nn.Module that registers the real model as a child — that is how
    core's model_size()/module_size() walk it — and exposes .model,
    .model_pack_name, .processor and .load_model for patch_model's
    lazy-build branch. Minimal stand-in with the same shape.
    """

    class TinyHandler(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = module
            self.processor = None
            self.model_pack_name = "tiny-probe"
            self.attention_mode = "eager"
            self.cache_key = "tiny-probe"
            self.source_path = ""

        def load_model(self, device, attention_mode: str = "eager"):
            pass

    return TinyHandler()


@pytest.fixture(scope="module")
def aimdo_runtime():
    """Flip the two globals main.py:300-301 sets, for real, then undo it.

    Restores ``aimdo_enabled`` and the ``CoreModelPatcher`` alias in
    ``finally``: other tests in this process read both, and
    ``comfy.utils.load_torch_file`` branches on the flag.
    """
    if not _aimdo_ready():
        pytest.skip("comfy-aimdo native stack unavailable headless")
    old_enabled = comfy.memory_management.aimdo_enabled
    old_alias = comfy.model_patcher.CoreModelPatcher
    comfy.memory_management.aimdo_enabled = True
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    yield
    comfy.memory_management.aimdo_enabled = old_enabled
    comfy.model_patcher.CoreModelPatcher = old_alias
    gc.collect()
    with contextlib.suppress(Exception):
        control.deinit()


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(256, 256, dtype=torch.bfloat16)
        self.norm = nn.LayerNorm(256, dtype=torch.bfloat16)
        self.emb = nn.Embedding(64, 256, dtype=torch.bfloat16)


@pytest.fixture
def tiny_checkpoint():
    d = tempfile.mkdtemp(prefix="vibevoice_dynvram_")
    path = os.path.join(d, "tiny.safetensors")
    import safetensors.torch as st

    st.save_file(
        {
            "lin.weight": torch.randn(256, 256, dtype=torch.bfloat16),
            "lin.bias": torch.randn(256, dtype=torch.bfloat16),
            "norm.weight": torch.ones(256, dtype=torch.bfloat16),
            "norm.bias": torch.zeros(256, dtype=torch.bfloat16),
            "emb.weight": torch.randn(64, 256, dtype=torch.bfloat16),
        },
        path,
    )
    yield path


@pytest.mark.usefixtures("aimdo_runtime")
class TestDynamicVRAMMechanism:
    def test_load_torch_file_returns_tagged_views(self, tiny_checkpoint):
        """Step 1: the aimdo branch hands back file views, not owned copies."""
        sd = comfy.utils.load_torch_file(tiny_checkpoint)
        info = getattr(sd["lin.weight"].untyped_storage(), "_comfy_tensor_file_slice", None)
        assert info is not None
        assert info.size == 256 * 256 * 2

    def test_dense_loader_preserves_views_into_parameters(self, tiny_checkpoint):
        """Step 2: preserve_file_views=True — the tag reaches the parameter."""
        sd = comfy.utils.load_torch_file(tiny_checkpoint)
        tag = sd["lin.weight"].untyped_storage()._comfy_tensor_file_slice
        with torch.device("meta"):
            model = Tiny()
        loaded = _load_state_dict_into_model_from_memory(
            model, sd, preserve_file_views=True,
        )
        param_info = getattr(
            loaded.lin.weight.untyped_storage(), "_comfy_tensor_file_slice", None
        )
        assert param_info is not None
        assert param_info == tag

    def test_conversion_marks_modules_for_vbar(self, tiny_checkpoint):
        """Step 3: the conversion is what makes core take the vbar branch."""
        sd = comfy.utils.load_torch_file(tiny_checkpoint)
        with torch.device("meta"):
            model = Tiny()
        loaded = _load_state_dict_into_model_from_memory(
            model, sd, preserve_file_views=True,
        )
        convert_tree_for_streaming(loaded)
        assert getattr(loaded.lin, "comfy_cast_weights", False) is True

    def test_full_route_vbar_alloc_and_disk_to_vram_forward(self, tiny_checkpoint):
        """Steps 4-6: dynamic patcher, vbar alloc, real forward, correct math.

        This is the acceptance probe for the whole port. If any link breaks,
        one of the asserts names it.
        """
        sd = comfy.utils.load_torch_file(tiny_checkpoint)
        tag = sd["lin.weight"].untyped_storage()._comfy_tensor_file_slice
        with torch.device("meta"):
            model = Tiny()
        loaded = _load_state_dict_into_model_from_memory(
            model, sd, preserve_file_views=True,
        )
        convert_tree_for_streaming(loaded)

        cls = select_patcher_class("dense", torch.device("cuda"))
        assert cls.__mro__[1] is comfy.model_patcher.ModelPatcherDynamic

        lin = loaded.lin
        patcher = cls(
            _tiny_handler(loaded),
            load_device=comfy.model_management.get_torch_device(),
            offload_device=torch.device("cpu"),
        )
        assert patcher.is_dynamic()
        try:
            load_to_device(patcher)

            # vbar range allocated for the lin; NO host backup stash for the
            # converted module — that stash is the eager branch this whole
            # port exists to avoid. (Modules convert_tree_for_streaming did
            # NOT convert — here possibly the LayerNorm — legitimately take
            # core's eager path and land in backup; the real VibeVoice trees
            # convert every leaf including norms.)
            assert hasattr(lin, "_v")
            assert patcher.loaded_size() > 0
            assert not any(k.startswith("lin") for k in getattr(patcher, "backup", {}))

            # The same fact, measured over the whole tree rather than
            # spot-checked: the host-RAM census after a real load. ZERO
            # private host bytes is the machine-checkable form of "these
            # weights are still file views, nothing was copied into host
            # RAM" — the property a RAM spike would break. The only
            # non-view bytes are offhost: core force-loads the sub-16KB
            # LayerNorm to the device on purpose
            # (comfy/model_patcher.py:1975-1989) and stashes its view.
            report = census(loaded, patcher)
            assert report["param_private_bytes"] == 0
            assert report["param_offhost_bytes"] == 256 * 2 * 2  # norm.weight+bias
            assert report["backup_private_bytes"] == 0

            # The view must STILL be the parameter's storage after load: core
            # reads disk->VRAM from it at every forward.
            assert (
                getattr(
                    loaded.lin.weight.untyped_storage(),
                    "_comfy_tensor_file_slice",
                    None,
                )
                == tag
            )

            with torch.no_grad():
                probe_input = torch.randn(2, 256, dtype=torch.bfloat16, device="cuda")
                out = lin(probe_input)
            torch.cuda.synchronize()
            ref = torch.nn.functional.linear(
                probe_input.float(),
                sd["lin.weight"].float().cuda(),
                sd["lin.bias"].float().cuda(),
            )
            # bf16 granularity at |x|~50 is ~0.25 per ulp, so the comparison
            # needs a relative term, not just atol.
            assert torch.allclose(out.float(), ref, atol=1e-2, rtol=2e-2)
        finally:
            with contextlib.suppress(Exception):
                patcher.unpatch_model(destroy=True)
            gc.collect()
            comfy.model_management.soft_empty_cache()

    def test_vbar_embedding_keeps_weight_dtype_with_int64_indices(
        self, tiny_checkpoint
    ):
        """The int64-embedding regression, on the REAL vbar route.

        The reported crash was a bf16 model whose LM embeddings came back
        int64: the streaming wrapper handed the int64 index tensor to
        ``cast_bias_weight`` as its dtype source, so the whole embedding
        table was cast to int64 (core never does this — it passes only
        ``device=input.device``, comfy/ops.py:793). The int64 embeds then
        propagated through ``speech_tensors.type_as(x)`` into the acoustic
        tokenizer. This asserts the vbar-paged table keeps bf16 and returns
        exact rows.
        """
        sd = comfy.utils.load_torch_file(tiny_checkpoint)
        with torch.device("meta"):
            model = Tiny()
        loaded = _load_state_dict_into_model_from_memory(
            model, sd, preserve_file_views=True,
        )
        convert_tree_for_streaming(loaded)

        cls = select_patcher_class("dense", torch.device("cuda"))
        patcher = cls(
            _tiny_handler(loaded),
            load_device=comfy.model_management.get_torch_device(),
            offload_device=torch.device("cpu"),
        )
        try:
            load_to_device(patcher)
            ids = torch.tensor([0, 7, 63], dtype=torch.int64, device="cuda")
            with torch.no_grad():
                out = loaded.emb(ids)
            torch.cuda.synchronize()
            assert out.dtype == torch.bfloat16, f"embedding returned {out.dtype}"
            ref = torch.nn.functional.embedding(
                ids, sd["emb.weight"].cuda()
            )
            assert torch.equal(out, ref)
        finally:
            with contextlib.suppress(Exception):
                patcher.unpatch_model(destroy=True)
            gc.collect()
            comfy.model_management.soft_empty_cache()


class TinyFp8(nn.Module):
    """Tiny fp8-resident tree — the 7B fp8 route's shape at KB scale."""

    def __init__(self):
        super().__init__()
        from ComfyUI_VibeVoice.modules.fp8_quant import FP8Linear

        # 256x256 fp8 = 64 KiB — deliberately over core's 16 KiB
        # force-load threshold (model_patcher.py:1975), below which the
        # dynamic load eagerly moves a weight instead of paging it.
        self.proj = FP8Linear(
            256, 256, bias=True, fp8_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
        )

    def forward(self, x):
        return self.proj(x)


@pytest.fixture
def fp8_checkpoint():
    d = tempfile.mkdtemp(prefix="vibevoice_fp8vbar_")
    path = os.path.join(d, "fp8.safetensors")
    import safetensors.torch as st

    st.save_file(
        {
            "proj.weight": (torch.randn(256, 256) * 0.05).to(torch.float8_e4m3fn),
            "proj.weight_scale": torch.tensor(1.0, dtype=torch.float32),
            "proj.bias": torch.randn(256, dtype=torch.bfloat16) * 0.01,
        },
        path,
    )
    yield path


@pytest.mark.usefixtures("aimdo_runtime")
class TestFp8ResidentVbar:
    """Stage-1 probe: can an fp8 resident live on the dynamic (vbar) path?

    The quant families were kept off the dynamic branch by a stop condition
    inherited from the pre-aimdo port. This probe decides it with evidence:
    the dynamic load must vbar-allocate the resident (no host backup stash)
    and the forward must dequantize the paged bytes correctly.
    """

    def test_resident_takes_vbar_and_forwards_correctly(
        self, fp8_checkpoint, monkeypatch
    ):
        import comfy.ops

        # Spy on core's vbar staging: participation is the whole point of the
        # port, so it is asserted, not inferred (a forward that simply copied
        # the weight would also produce correct math).
        vbar_calls = []
        _real = comfy.ops.cast_modules_with_vbar

        def _spy(mods, dtype, device, bias_dtype, non_blocking):
            vbar_calls.append((dtype, device, bias_dtype))
            return _real(mods, dtype, device, bias_dtype, non_blocking)

        monkeypatch.setattr(comfy.ops, "cast_modules_with_vbar", _spy)
        import comfy_kitchen

        device = comfy.model_management.get_torch_device()
        sd = comfy.utils.load_torch_file(fp8_checkpoint)
        tag = sd["proj.weight"].untyped_storage()._comfy_tensor_file_slice
        with torch.device("meta"):
            model = TinyFp8()
        model.load_state_dict(sd, strict=False, assign=True)
        assert getattr(
            model.proj.weight.untyped_storage(), "_comfy_tensor_file_slice", None
        ) == tag

        cls = make_dynamic_patcher_class()
        patcher = cls(
            _tiny_handler(model),
            load_device=device,
            offload_device=torch.device("cpu"),
        )
        assert patcher.is_dynamic()
        try:
            load_to_device(patcher)
            proj = model.proj
            assert hasattr(proj, "_v"), "fp8 resident was not vbar-allocated"
            # The resident weight itself must NOT be stashed — that stash is
            # the eager branch. (A 4-byte ``weight_scale`` backup is core's
            # normal force-load of a sub-16KiB param and is expected.)
            assert not any(
                k.endswith(("proj.weight", "proj.bias")) for k in patcher.backup
            ), f"resident weight stashed a host backup: {list(patcher.backup)}"

            x = torch.randn(4, 256, dtype=torch.bfloat16, device=device)
            out = model(x)

            # The resident's forward went through core's vbar staging with
            # the STORAGE dtype for the weight and the ACTIVATION dtype for
            # the bias (comfy/ops.py:350-351, :375-393).
            assert vbar_calls, "forward bypassed cast_bias_weight's vbar branch"
            stage_dtype, stage_device, stage_bias_dtype = vbar_calls[-1]
            assert stage_dtype == torch.float8_e4m3fn, stage_dtype
            assert stage_bias_dtype == torch.bfloat16, stage_bias_dtype
            assert stage_device.type == device.type

            w_deq = comfy_kitchen.dequantize_per_tensor_fp8(
                sd["proj.weight"], sd["proj.weight_scale"], torch.bfloat16
            )
            ref = torch.nn.functional.linear(
                x.cpu(), w_deq, sd["proj.bias"].cpu()
            )
            assert torch.allclose(
                out.float().cpu(), ref.float(), atol=2e-2, rtol=5e-2
            )
        finally:
            with contextlib.suppress(Exception):
                patcher.unpatch_model(destroy=True)
            gc.collect()
            comfy.model_management.soft_empty_cache()
