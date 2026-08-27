"""Phase 0: characterization of core's lowvram protocol on FOREIGN trees.

These tests drive the REAL ``comfy.model_patcher.ModelPatcher.load`` /
``partially_unload`` on a small foreign (plain nn.Module) tree using CPU
devices only. They encode the protocol facts that motivate the streaming
integration (plan 2026-08-26, §1):

- modules beyond a tight lowvram budget are silently SKIPPED by ``load()``
  (never placed, never flagged);
- placed modules are flagged ``comfy_patched_weights`` regardless of
  castability;
- ``partially_unload`` strips flagged modules (flag cleared) with no
  streaming hook to bring them back — the production bug's precondition.

Device differentials are covered by the manual GPU gate (plan §5.4); here we
pin flags, ordering, and accounting deterministically.
"""

import pytest
import torch
from torch import nn

import comfy.model_management as comfy_mm
from comfy.model_patcher import ModelPatcher


def _foreign_tree():
    class _Tree(nn.Module):
        def __init__(self):
            super().__init__()
            self.hot = nn.Linear(128, 128)   # ~66 KB bf16 / 66K elems f32
            self.cold_tail = nn.Linear(4, 4)  # tiny

    return _Tree()


def _budget_for(tree, which="hot"):
    """A lowvram budget that admits `which` module but not the other."""
    mods = {n: m for n, m in tree.named_modules()}
    hot = comfy_mm.module_size(mods["hot"])
    tail = comfy_mm.module_size(mods["cold_tail"])
    # Sorted descending by size: hot loads first; tail must NOT fit.
    return hot + max(hot // 8, 1)


@pytest.fixture
def patcher():
    tree = _foreign_tree()
    mp = ModelPatcher(
        tree,
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        size=comfy_mm.module_size(tree),
    )
    yield mp, tree


class TestCoreLowvramProtocolOnForeignTree:
    """Core sorts modules LARGEST-FIRST; under a tight budget the biggest
    uncastable modules fail `lowvram_fits` and are silently skipped."""

    def test_tight_budget_skips_all_uncastable_modules(self, patcher):
        """Uncastable + over-budget: the FIRST failing module sets an
        offload_buffer it can never repay, so every later module fails
        `lowvram_fits` too -> NOTHING is placed, nothing streamed, no flags.
        On GPU this is exactly where CPU strays come from at first load."""
        mp, tree = patcher
        hot_mem = comfy_mm.module_size(tree.hot)
        tail_mem = comfy_mm.module_size(tree.cold_tail)
        budget = tail_mem + max(tail_mem // 8, 1)

        mp.patch_model(device_to=torch.device("cpu"),
                       lowvram_model_memory=budget)

        for m in (tree.hot, tree.cold_tail):
            assert not hasattr(m, "comfy_patched_weights"), m
            assert not hasattr(m, "weight_function"), m
        assert mp.model.model_loaded_weight_memory == 0

    def test_full_load_flags_everything(self, patcher):
        mp, tree = patcher
        mp.patch_model(device_to=torch.device("cpu"),
                       lowvram_model_memory=0)  # 0 => full_load
        assert getattr(tree.hot, "comfy_patched_weights", False) is True
        assert getattr(tree.cold_tail, "comfy_patched_weights", False) is True

    def test_partially_unload_strips_flagged_foreign_modules(self, patcher):
        mp, tree = patcher
        mp.patch_model(device_to=torch.device("cpu"),
                       lowvram_model_memory=0)
        assert getattr(tree.hot, "comfy_patched_weights", False) is True

        freed = mp.partially_unload(torch.device("cpu"), memory_to_free=1)
        assert freed > 0
        # Flag cleared: core considers it "offloaded"; a foreign tree has NO
        # mechanism to pull the weights back during forward — this is the
        # production crash precondition (plan §1 step 4).
        assert getattr(tree.hot, "comfy_patched_weights", False) is False
