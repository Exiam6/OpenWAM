"""DINOv2 frame encoder with Wan-compatible video token geometry.

DINOv2 Base uses 14-pixel image patches, while the released RoboTwin model
consumes a 16-pixel latent grid. Padding before the frozen ViT and resampling
its patch features to the existing 16-pixel grid keeps the camera pixels and
the DiT token count fixed. The resampling has no trainable parameters.
"""

from __future__ import annotations

import math

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from openwam.model.video_backbone.encoder.base import VideoEncoderProperties
from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder, _causal_temporal_pool
from openwam.model.video_backbone.encoder.registry import register_video_encoder


@register_video_encoder("dinov2")
class DinoV2VideoEncoder(DinoV3VideoEncoder):
    """Frozen DINOv2 patches, resampled to the Wan VAE spatial grid.

    The inherited deploy path saves the HF config under ``dinov2/`` and the
    full DINO weights remain in the OpenWAM checkpoint state dict. A separately
    trained pixel decoder can be added later; this class alone is the raw
    representation baseline, not a complete RAE.
    """

    _checkpoint_subdir = "dinov2"
    _target_spatial_compression = 16

    def __init__(
        self,
        vit: nn.Module,
        *,
        embed_dim: int,
        patch_size: int,
        num_register_tokens: int,
        svae_path: str | None = None,
        svae_target_dim: int | None = None,
        svae_config: dict | None = None,
    ) -> None:
        super().__init__(
            vit,
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            svae_path=svae_path,
            svae_target_dim=svae_target_dim,
            svae_config=svae_config,
        )
        self._vit_patch_size = int(patch_size)
        self._m.requires_grad_(False)
        self._spec = VideoEncoderProperties(
            z_dim=self._spec.z_dim,
            spatial_compression=self._target_spatial_compression,
            temporal_compression=4,
            causal_temporal=True,
            pixel_decode=False,
            dit_patch_size=(1, 2, 2),
        )

    def _batch_encode_pooled_raw(self, video: Tensor) -> Tensor:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"DinoV2VideoEncoder expects (B, 3, T, H, W); got {tuple(video.shape)}")
        b, _, t, h, w = video.shape
        target_patch = self._target_spatial_compression
        if h % target_patch or w % target_patch:
            raise ValueError(f"H and W must be divisible by {target_patch}; got {(h, w)}")
        if (t - 1) % 4:
            raise ValueError(f"T must be 1 modulo 4; got {t}")

        flat = rearrange(video, "B C T H W -> (B T) C H W")
        patch = self._vit_patch_size
        padded_h = math.ceil(h / patch) * patch
        padded_w = math.ceil(w / patch) * patch
        # Normalized input zero corresponds to the ImageNet mean color.
        flat = F.pad(flat, (0, padded_w - w, 0, padded_h - h))
        outputs = self._m(flat, interpolate_pos_encoding=True)
        tokens = getattr(outputs, "last_hidden_state", None)
        if tokens is None:
            raise TypeError("DINOv2 forward must return last_hidden_state")
        patch_tokens = tokens[:, 1 + self._num_register_tokens :, :]
        grid_h, grid_w = padded_h // patch, padded_w // patch
        if patch_tokens.shape[1] != grid_h * grid_w:
            raise ValueError(f"DINOv2 returned {patch_tokens.shape[1]} patch tokens; expected {grid_h * grid_w}")
        grid = rearrange(patch_tokens, "BT (H W) D -> BT D H W", H=grid_h, W=grid_w)
        grid = F.interpolate(grid, size=(h // target_patch, w // target_patch), mode="bilinear", align_corners=False)
        grid = rearrange(grid, "(B T) D H W -> B D T H W", B=b, T=t)
        return _causal_temporal_pool(grid)


__all__ = ["DinoV2VideoEncoder"]
