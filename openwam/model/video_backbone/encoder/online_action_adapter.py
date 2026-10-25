"""A small causal DINO feature adapter trained by the action objective.

This is an online action-grounding ablation, not the 2x2x2 token-compressing
TRB from ReWAM. It preserves the DINO latent width and Wan token geometry so
the existing matched model can be used as the initialization.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CausalActionAdapter(nn.Module):
    """Residual temporal bottleneck with no access to future latent frames."""

    def __init__(self, channels: int, hidden_channels: int = 128):
        super().__init__()
        if channels <= 0 or hidden_channels <= 0:
            raise ValueError("Adapter channel counts must be positive")
        self.down = nn.Conv3d(channels, hidden_channels, 1)
        self.temporal = nn.Conv3d(hidden_channels, hidden_channels, (3, 1, 1), groups=hidden_channels)
        self.up = nn.Conv3d(hidden_channels, channels, 1)
        # Exact identity at initialization. The output projection learns first;
        # gradients then reach the temporal and input projections.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 5:
            raise ValueError(f"Expected (B,C,T,H,W), got {tuple(features.shape)}")
        hidden = self.down(features)
        hidden = F.pad(hidden, (0, 0, 0, 0, 2, 0))
        hidden = F.silu(self.temporal(hidden))
        return features + self.up(hidden)


def action_only_current_latent_gradient(action_loss: Tensor, current_latent: Tensor) -> None:
    """Route only action-loss gradients through the clean current latent.

    The caller uses a detached adapter output for video noise and targets, and
    passes ``current_latent`` as the clean TI2V conditioning frame. The shared
    world/action forward still trains the common trunk on their total loss.
    This boundary hook replaces the total gradient entering the adapter with
    the gradient of the weighted action loss alone. It requires a normal
    autograd graph (including non-reentrant activation checkpointing).
    """
    if not current_latent.requires_grad:
        raise RuntimeError("Current latent must require grad for online action adaptation")
    action_grad = torch.autograd.grad(
        action_loss, current_latent, retain_graph=True, allow_unused=False
    )[0].detach()
    if not torch.isfinite(action_grad).all():
        raise FloatingPointError("Action gradient entering visual adapter is non-finite")
    current_latent.register_hook(lambda _combined_grad: action_grad)


__all__ = ["CausalActionAdapter", "action_only_current_latent_gradient"]
