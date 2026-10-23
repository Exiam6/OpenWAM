"""Deterministic visual projection reset for matched encoder comparisons."""

from __future__ import annotations

import torch


def reset_visual_projections(architecture, seed: int) -> list[str]:
    """Reset only the DiT latent input/output projections with a local RNG seed."""
    backbone = architecture.video_backbone
    names = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for dit_name in ("dit", "dit2"):
            dit = getattr(backbone, dit_name, None)
            if dit is None:
                continue
            head = getattr(getattr(dit, "head", None), "head", None)
            patch = getattr(dit, "patch_embedding", None)
            if patch is None or head is None:
                raise AttributeError(f"{dit_name} has no patch_embedding or head.head")
            patch.reset_parameters()
            head.reset_parameters()
            names.extend((f"{dit_name}.patch_embedding", f"{dit_name}.head.head"))
    if not names:
        raise ValueError("Architecture has no DiT visual projections")
    return names
