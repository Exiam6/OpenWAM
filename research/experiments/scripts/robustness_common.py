"""Pure observation perturbations and coordinate-preserving latent adapters."""
import hashlib
import numpy as np
import torch
from torch import nn


def noise_image(image, seed, step, sigma=.10):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(step), 9173]))
    return np.rint(np.clip(image.astype(np.float32) + rng.normal(0, 255*sigma, image.shape), 0, 255)).astype(np.uint8)


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def change_observation(observation, condition, seed, step):
    if condition == 'clean':
        return observation
    cameras = observation['observation']
    if condition == 'noise':
        rgb = noise_image(cameras['head_camera']['rgb'], seed, step)
    elif condition == 'front':
        rgb = cameras['front_camera']['rgb']
    else:
        raise ValueError(condition)
    return {**observation, 'observation': {**cameras, 'head_camera': {**cameras['head_camera'], 'rgb': rgb}}}


class Adapter(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind == 'linear':
            self.residual = nn.Sequential(nn.Conv2d(48, 48, 1))
        elif kind == 'mlp':
            self.residual = nn.Sequential(nn.Conv2d(48, 96, 1), nn.GELU(), nn.Conv2d(96, 48, 1))
        else:
            raise ValueError(kind)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, z):
        assert z.ndim == 5 and z.shape[1] == 48 and z.shape[2] >= 1
        first = z[:, :, 0].float()
        restored = (first + self.residual(first)).to(z.dtype)
        return torch.cat([restored[:, :, None], z[:, :, 1:]], dim=2)
