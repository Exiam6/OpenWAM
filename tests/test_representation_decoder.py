"""The RAE pixel decoder must preserve frame order and admit gradients."""

import pytest
import torch

from openwam.model.video_backbone.encoder.representation_decoder import RepresentationPixelDecoder


def test_decoder_shape_and_gradients():
    model = RepresentationPixelDecoder(in_channels=8, width=32)
    latents = torch.randn(2, 8, 2, 2, 3, requires_grad=True)
    output = model(latents)
    assert output.shape == (2, 3, 2, 32, 48)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert latents.grad is not None and latents.grad.abs().sum() > 0
    assert model.stem.weight.grad is not None and model.stem.weight.grad.abs().sum() > 0


def test_decoder_rejects_wrong_channels():
    model = RepresentationPixelDecoder(in_channels=8, width=32)
    with pytest.raises(ValueError, match="Expected"):
        model(torch.zeros(1, 9, 1, 2, 2))
