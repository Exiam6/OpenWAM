"""Geometry and deployment checks for the DINOv2 representation path."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openwam.model.video_backbone.encoder import _VIDEO_ENCODER_REGISTRY
from openwam.model.video_backbone.encoder.dinov2 import DinoV2VideoEncoder


class FakeDinoV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.last_shape = None
        self.interpolate = None

    def forward(self, x, *, interpolate_pos_encoding=False):
        self.last_shape = tuple(x.shape)
        self.interpolate = interpolate_pos_encoding
        n = (x.shape[-2] // 14) * (x.shape[-1] // 14)
        # Each frame has a constant patch value, so spatial resizing cannot
        # change the expected first-frame/next-four-frame temporal mean.
        v = x[:, 0, 0, 0].view(-1, 1, 1)
        patches = v.expand(-1, n, 8).clone()
        cls = torch.full((x.shape[0], 1, 8), -100, dtype=x.dtype)
        return SimpleNamespace(last_hidden_state=torch.cat([cls, patches], dim=1))


def make_encoder():
    vit = FakeDinoV2()
    return DinoV2VideoEncoder(vit, embed_dim=8, patch_size=14, num_register_tokens=0), vit


def test_registry_and_wan_geometry():
    assert _VIDEO_ENCODER_REGISTRY["dinov2"] is DinoV2VideoEncoder
    encoder, vit = make_encoder()
    assert encoder.properties.spatial_compression == 16
    assert encoder.properties.dit_patch_size == (1, 2, 2)
    assert encoder.properties.z_dim == 8
    assert not vit.weight.requires_grad

    x = torch.zeros(2, 3, 5, 384, 320)
    x[:, :, 0] = 1
    x[:, :, 1:] = 3
    raw = encoder._batch_encode_pooled_raw(x)
    assert raw.shape == (2, 8, 2, 24, 20)
    assert vit.last_shape == (10, 3, 392, 322)
    assert vit.interpolate is True
    torch.testing.assert_close(raw[:, :, 0], torch.ones_like(raw[:, :, 0]))
    torch.testing.assert_close(raw[:, :, 1], torch.full_like(raw[:, :, 1], 3))
    assert encoder.batch_encode(x).shape == raw.shape


def test_rejects_invalid_temporal_and_spatial_shapes():
    encoder, _ = make_encoder()
    with pytest.raises(ValueError, match="modulo 4"):
        encoder.batch_encode(torch.zeros(1, 3, 4, 384, 320))
    with pytest.raises(ValueError, match="divisible by 16"):
        encoder.batch_encode(torch.zeros(1, 3, 5, 383, 320))


def test_deploy_config_uses_dinov2_namespace(tmp_path):
    encoder, _ = make_encoder()
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "dinov2"}))
    target = tmp_path / "checkpoint"
    encoder_cfg = SimpleNamespace(model_path=source)
    cfg = SimpleNamespace(model=SimpleNamespace(video_backbone=SimpleNamespace(encoder=encoder_cfg)))
    encoder.save_deploy_assets(str(target), cfg)
    assert (target / "dinov2" / "config.json").is_file()
    assert DinoV2VideoEncoder._resolve_config_dir(str(target)) == str(target / "dinov2")
