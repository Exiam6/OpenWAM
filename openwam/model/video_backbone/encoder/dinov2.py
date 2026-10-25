"""DINOv2 frame encoder with Wan-compatible video token geometry.

DINOv2 Base uses 14-pixel image patches, while the released RoboTwin model
consumes a 16-pixel latent grid. Padding before the frozen ViT and resampling
its patch features to the existing 16-pixel grid keeps the camera pixels and
the DiT token count fixed. The resampling has no trainable parameters.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from openwam.model.video_backbone.encoder.base import VideoEncoderProperties
from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder, _causal_temporal_pool
from openwam.model.video_backbone.encoder.online_action_adapter import CausalActionAdapter
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
        online_action_adapter_hidden_dim: int | None = None,
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
        self.register_buffer("_feature_mean", None)
        self.register_buffer("_feature_std", None)
        self._m.requires_grad_(False)
        self._online_action_adapter = (
            CausalActionAdapter(int(self._spec.z_dim), int(online_action_adapter_hidden_dim))
            if online_action_adapter_hidden_dim is not None
            else None
        )
        self._spec = VideoEncoderProperties(
            z_dim=self._spec.z_dim,
            spatial_compression=self._target_spatial_compression,
            temporal_compression=4,
            causal_temporal=True,
            pixel_decode=False,
            dit_patch_size=(1, 2, 2),
        )

    def _set_feature_stats(self, mean: Tensor, std: Tensor) -> None:
        channels = self.properties.z_dim
        if tuple(mean.shape) != (channels,) or tuple(std.shape) != (channels,):
            raise ValueError(f"Feature statistics must have {channels} channels")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Feature statistics must be finite with positive standard deviations")
        self._feature_mean = mean.detach().float().view(1, channels, 1, 1, 1)
        self._feature_std = std.detach().float().view(1, channels, 1, 1, 1)

    def batch_encode(self, video: Tensor) -> Tensor:
        z = self._apply_svae(self._batch_encode_pooled_raw(video))
        if self._feature_mean is None:
            z = self._apply_feature_norm(z)
        else:
            z = ((z.float() - self._feature_mean) / self._feature_std).to(z.dtype)
        # Train preprocessing runs under @no_grad. Keep the adapter's graph in
        # architecture.compute_loss; deploy/evaluation use the same weights here.
        if self._online_action_adapter is not None and not self.training:
            z = self._online_action_adapter(z)
        return z

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        feature_stats_path: str | None = None,
        online_action_adapter_hidden_dim: int | None = None,
        **kwargs,
    ) -> "DinoV2VideoEncoder":
        model = super().from_pretrained(model_path, **kwargs)
        if online_action_adapter_hidden_dim is not None:
            model._online_action_adapter = CausalActionAdapter(
                model.properties.z_dim, int(online_action_adapter_hidden_dim)
            )
        if feature_stats_path is not None:
            stats = torch.load(Path(feature_stats_path), map_location="cpu", weights_only=True)
            model._set_feature_stats(stats["mean"], stats["std"])
        return model

    @classmethod
    def from_skeleton(cls, components_entry: dict, *, device: str = "cpu", encoder_cfg=None, ckpt_dir=None):
        model = super().from_skeleton(
            components_entry,
            device=device,
            encoder_cfg=encoder_cfg,
            ckpt_dir=ckpt_dir,
        )
        stats_path = (
            encoder_cfg.get("feature_stats_path")
            if isinstance(encoder_cfg, dict)
            else getattr(encoder_cfg, "feature_stats_path", None)
        )
        if stats_path is not None:
            channels = model.properties.z_dim
            model._set_feature_stats(torch.zeros(channels), torch.ones(channels))
        adapter_width = (
            encoder_cfg.get("online_action_adapter_hidden_dim")
            if isinstance(encoder_cfg, dict)
            else getattr(encoder_cfg, "online_action_adapter_hidden_dim", None)
        )
        if adapter_width is not None:
            model._online_action_adapter = CausalActionAdapter(model.properties.z_dim, int(adapter_width))
        return model

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
