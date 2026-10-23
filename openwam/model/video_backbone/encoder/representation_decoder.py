"""Pixel decoder for a frozen representation encoder's Wan-grid features.

This decoder is trained separately from the world-action model. Its input is
the exact normalized latent emitted by ``DinoV2VideoEncoder.batch_encode``;
its output is an ImageNet-normalized RGB canvas at the original resolution.
No action or future image enters the decoder.
"""

from __future__ import annotations

import math

import torch.nn as nn
from einops import rearrange
from torch import Tensor


class RepresentationPixelDecoder(nn.Module):
    """Four 2x upsampling stages, mapping a 16x-compressed grid to pixels."""

    def __init__(self, in_channels: int = 768, width: int = 256) -> None:
        super().__init__()
        if in_channels <= 0 or width < 32 or width % 16:
            raise ValueError("in_channels must be positive and width >=32 divisible by 16")
        channels = [width, width, width // 2, width // 4, width // 8]
        self.in_channels = in_channels
        self.stem = nn.Conv2d(in_channels, channels[0], kernel_size=1)
        stages = []
        for current, following in zip(channels[:-1], channels[1:]):
            stages.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(current, following, kernel_size=3, padding=1),
                    nn.GroupNorm(math.gcd(8, following), following),
                    nn.SiLU(),
                    nn.Conv2d(following, following, kernel_size=3, padding=1),
                    nn.GroupNorm(math.gcd(8, following), following),
                    nn.SiLU(),
                ]
            )
        self.stages = nn.Sequential(*stages)
        self.output = nn.Conv2d(channels[-1], 3, kernel_size=3, padding=1)

    def forward(self, latents: Tensor) -> Tensor:
        if latents.ndim != 5 or latents.shape[1] != self.in_channels:
            raise ValueError(f"Expected (B, {self.in_channels}, T, H, W); got {tuple(latents.shape)}")
        batch, _, time, _, _ = latents.shape
        frames = rearrange(latents, "B C T H W -> (B T) C H W")
        pixels = self.output(self.stages(self.stem(frames)))
        return rearrange(pixels, "(B T) C H W -> B C T H W", B=batch, T=time)


__all__ = ["RepresentationPixelDecoder"]
