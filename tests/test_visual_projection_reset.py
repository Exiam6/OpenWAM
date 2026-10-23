from types import SimpleNamespace

import torch
from torch import nn

from openwam.train.utils.visual_projection_reset import reset_visual_projections


def test_reset_changes_only_visual_projections_and_preserves_global_rng():
    dit = SimpleNamespace(
        patch_embedding=nn.Conv3d(48, 16, (1, 2, 2)),
        head=SimpleNamespace(head=nn.Linear(16, 192)),
        trunk=nn.Linear(16, 16),
    )
    architecture = SimpleNamespace(video_backbone=SimpleNamespace(dit=dit, dit2=None))
    trunk_before = dit.trunk.weight.detach().clone()
    patch_before = dit.patch_embedding.weight.detach().clone()
    rng_before = torch.random.get_rng_state().clone()

    names = reset_visual_projections(architecture, 1051)
    patch_once = dit.patch_embedding.weight.detach().clone()
    head_once = dit.head.head.weight.detach().clone()

    assert names == ["dit.patch_embedding", "dit.head.head"]
    assert torch.equal(dit.trunk.weight, trunk_before)
    assert not torch.equal(patch_once, patch_before)
    assert torch.equal(torch.random.get_rng_state(), rng_before)

    reset_visual_projections(architecture, 1051)
    assert torch.equal(dit.patch_embedding.weight, patch_once)
    assert torch.equal(dit.head.head.weight, head_once)
