"""Untrained prototype: restore corrupted observations in an existing latent space.

This is not a new RAE, a trained model, or an integrated policy change. It acts
only on the first (observed) latent frame, after the original encoder output.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ObservationAdapter(nn.Module):
    def __init__(self, channels=48, hidden=96):
        super().__init__()
        self.channels = channels
        self.residual = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU(), nn.Conv2d(hidden, channels, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, latent):
        if latent.ndim != 5 or latent.shape[1] != self.channels or latent.shape[2] < 1:
            raise ValueError("Expected (B, channels, T>=1, H, W)")
        observed = latent[:, :, 0]
        restored = observed + self.residual(observed)
        return torch.cat((restored.unsqueeze(2), latent[:, :, 1:]), dim=2)


def restoration_loss(adapter, clean_latent, corrupt_latent, clean_weight=1.0):
    clean, corrupt = clean_latent.detach(), corrupt_latent.detach()
    restore = F.mse_loss(adapter(corrupt)[:, :, :1].float(), clean[:, :, :1].float())
    preserve = F.mse_loss(adapter(clean)[:, :, :1].float(), clean[:, :, :1].float())
    return restore + clean_weight * preserve
