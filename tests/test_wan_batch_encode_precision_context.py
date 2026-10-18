"""Cached and online frozen VAE encoding share an explicit precision context."""
from types import SimpleNamespace
import torch
from openwam.model.video_backbone.wan.models.vae import WanVideoVAE


import unittest


class PrecisionContextTests(unittest.TestCase):
    def test_batch_encode_does_not_inherit_outer_autocast(self):
        observed=[]
        class Encoder:
            def encode(self, pixels, scale):
                observed.append((torch.is_autocast_enabled('cpu'),torch.is_grad_enabled(),pixels.dtype))
                assert scale=='fixed-scale'
                return pixels+1
        vae=SimpleNamespace(model=Encoder(),scale='fixed-scale')
        pixels=torch.ones(1,3,1,4,4,dtype=torch.bfloat16,requires_grad=True)
        direct=WanVideoVAE.batch_encode(vae,pixels,'cpu')
        with torch.autocast('cpu',dtype=torch.bfloat16):
            nested=WanVideoVAE.batch_encode(vae,pixels,'cpu')
            assert torch.is_autocast_enabled('cpu')
        assert observed==[(False,False,torch.bfloat16)]*2
        assert torch.equal(direct,nested)
        assert not direct.requires_grad and not nested.requires_grad
        assert torch.is_grad_enabled()


    def test_batch_encode_keeps_explicit_fp32_input(self):
        observed=[]
        class Encoder:
            def encode(self,pixels,scale):
                observed.append(pixels.dtype)
                assert not torch.is_autocast_enabled('cpu')
                return pixels
        pixels=torch.randn(1,3,1,4,4,dtype=torch.float32)
        vae=SimpleNamespace(model=Encoder(),scale=None)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            actual=WanVideoVAE.batch_encode(vae,pixels,'cpu')
        assert observed==[torch.float32]
        assert torch.equal(actual,pixels)
