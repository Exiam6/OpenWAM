"""Frozen train-only PCA compression for the native DINO video encoder."""
from dataclasses import replace
import json
from pathlib import Path
import torch
from torch import nn
from .dinov3 import DinoV3VideoEncoder
from .registry import register_video_encoder


class FrozenPCA(nn.Module):
    """Standardize channels, project, and whiten using training statistics only."""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        if not 1 < latent_dim <= input_dim:
            raise ValueError('Require 1 < latent_dim <= input_dim')
        self.register_buffer('mean', torch.zeros(input_dim))
        self.register_buffer('std', torch.ones(input_dim))
        self.register_buffer('basis', torch.zeros(input_dim, latent_dim))
        self.register_buffer('latent_std', torch.ones(latent_dim))

    def _apply(self, fn, recurse=True):
        # Model-wide BF16 conversion must not quantize fitted statistics.
        original = dict(self._buffers)
        super()._apply(fn, recurse=recurse)
        for name, value in original.items():
            self._buffers[name] = value.to(device=self._buffers[name].device, dtype=torch.float32)
        return self

    @classmethod
    def from_file(cls, path):
        d = torch.load(path, map_location='cpu', weights_only=True)
        mean, std, basis = [d[k].float() for k in ['mean', 'std', 'basis']]
        scale = d['eigenvalues'][:basis.shape[1]].float().clamp_min(1e-12).sqrt()
        if mean.ndim != 1 or std.shape != mean.shape or basis.ndim != 2 or basis.shape[0] != mean.numel():
            raise ValueError('Invalid PCA dimensions')
        if not all(torch.isfinite(x).all() for x in [mean, std, basis, scale]) or not (std > 0).all():
            raise ValueError('Invalid PCA statistics')
        obj = cls(mean.numel(), basis.shape[1])
        obj.load_state_dict(dict(mean=mean, std=std, basis=basis, latent_std=scale), strict=True)
        return obj

    def forward(self, z):
        if z.ndim != 5 or z.shape[1] != self.mean.numel():
            raise ValueError('Expected B,C,T,H,W with PCA input channels')
        with torch.autocast(device_type=z.device.type, enabled=False):
            x = z.movedim(1, -1).float()
            x = ((x - self.mean) / self.std) @ self.basis / self.latent_std
        return x.movedim(-1, 1).to(z.dtype)


@register_video_encoder('dinov3_pca')
class DinoV3PCAEncoder(DinoV3VideoEncoder):
    """Causal DINO + frozen PCA whitening + native per-token LayerNorm.

    Weights/statistics travel in the main checkpoint; the sidecar contains
    dimensions only. This encoder has no pixel decoder and is not a full RAE.
    """
    def __init__(self, vit, *, embed_dim, patch_size, num_register_tokens, pca):
        super().__init__(vit, embed_dim=embed_dim, patch_size=patch_size,
                         num_register_tokens=num_register_tokens)
        if pca.mean.numel() != embed_dim:
            raise ValueError('PCA and DINO widths differ')
        self._pca = pca
        dim = pca.basis.shape[1]
        self._spec = replace(self._spec, z_dim=dim)
        self._out_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def batch_encode(self, video):
        return self._apply_feature_norm(self._pca(self._batch_encode_pooled_raw(video)))

    @classmethod
    def from_pretrained(cls, model_path, *, pca_path, **kwargs):
        if any(kwargs.get(k) is not None for k in ['svae_path', 'svae_target_dim', 'svae_config']):
            raise ValueError('PCA and S-VAE are mutually exclusive')
        raw = DinoV3VideoEncoder.from_pretrained(model_path)
        return cls(raw._m, embed_dim=raw._raw_embed_dim,
                   patch_size=raw.properties.spatial_compression,
                   num_register_tokens=raw._num_register_tokens,
                   pca=FrozenPCA.from_file(pca_path))

    @classmethod
    def from_skeleton(cls, components_entry, *, device='cpu', encoder_cfg=None, ckpt_dir=None):
        if ckpt_dir is None:
            raise ValueError('PCA deployment requires checkpoint-local sidecar')
        d = json.loads((Path(ckpt_dir) / 'pca_config.json').read_text())
        if d.get('format_version') != 1 or d.get('whiten') is not True:
            raise ValueError('Unsupported PCA sidecar')
        if (Path(ckpt_dir) / 'svae_config.json').exists():
            raise ValueError('Ambiguous PCA/S-VAE checkpoint')
        raw = DinoV3VideoEncoder.from_skeleton(components_entry, device=device,
                                              encoder_cfg=None, ckpt_dir=ckpt_dir)
        with torch.device(device):
            pca = FrozenPCA(d['input_dim'], d['latent_dim'])
        return cls(raw._m, embed_dim=raw._raw_embed_dim,
                   patch_size=raw.properties.spatial_compression,
                   num_register_tokens=raw._num_register_tokens, pca=pca)

    def save_deploy_assets(self, output_dir, cfg):
        super().save_deploy_assets(output_dir, cfg)
        p = Path(output_dir) / 'pca_config.json'
        p.write_text(json.dumps({'format_version': 1, 'input_dim': self._raw_embed_dim,
                               'latent_dim': self.properties.z_dim, 'whiten': True}, indent=2) + '\n')
