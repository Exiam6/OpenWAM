"""CPU contract tests for the optional encoder, using a deterministic tiny ViT."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from omegaconf import OmegaConf
from openwam.model.video_backbone.encoder.dinov3_pca import FrozenPCA, DinoV3PCAEncoder
from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder
from openwam.model.video_backbone.encoder.registry import _VIDEO_ENCODER_REGISTRY

class TinyViT(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Conv2d(3,8,2,2)
    def forward(self,x):
        z=self.proj(x).flatten(2).transpose(1,2)
        return SimpleNamespace(last_hidden_state=torch.cat([z[:,:1]*0,z],1))


def test_frozen_pca_native_encoder_contract(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(9)
    root=tmp_path;basis=torch.linalg.qr(torch.randn(8,4)).Q
    stats={'mean':torch.randn(8),'std':torch.rand(8)+.1,'basis':basis,'eigenvalues':torch.tensor([4.,3.,2.,1.,.7,.5,.3,.1])}
    f=root/'pca.pt';torch.save(stats,f);p=FrozenPCA.from_file(f)
    x=torch.randn(2,8,3,4,4);expected=((x.movedim(1,-1)-stats['mean'])/stats['std'])@basis/torch.tensor([2.,3**.5,2**.5,1.])
    torch.testing.assert_close(p(x).movedim(1,-1),expected,rtol=1e-6,atol=1e-6)
    original={k:v.clone() for k,v in p.state_dict().items()};p.to(dtype=torch.bfloat16)
    assert all(v.dtype==torch.float32 and torch.equal(v,original[k]) for k,v in p.state_dict().items())
    assert sum(v.numel() for v in p.parameters())==0
    enc=DinoV3PCAEncoder(TinyViT(),embed_dim=8,patch_size=2,num_register_tokens=0,pca=p).eval()
    video=torch.randn(2,3,9,8,8);z=enc.batch_encode(video);assert z.shape==(2,4,3,4,4)
    changed=video.clone();changed[:,:,1:]=torch.randn_like(changed[:,:,1:])*5
    torch.testing.assert_close(z[:,:,:1],enc.batch_encode(changed)[:,:,:1],rtol=0,atol=0)
    torch.testing.assert_close(z[:,:,:1],enc.batch_encode(video[:,:,:1]),rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(z,torch.cat([enc.batch_encode(v[None]) for v in video]),rtol=1e-6,atol=1e-6)
    assert z.mean(dim=1).abs().max()<1e-6
    assert _VIDEO_ENCODER_REGISTRY['dinov3_pca'] is DinoV3PCAEncoder
    # Real sidecar + strict state_dict round trip; stub only the large ViT constructor.
    (root/'pca_config.json').write_text(json.dumps({'format_version':1,'input_dim':8,'latent_dim':4,'whiten':True}))
    raw=DinoV3VideoEncoder(TinyViT(),embed_dim=8,patch_size=2,num_register_tokens=0)
    with patch.object(DinoV3VideoEncoder,'from_skeleton',return_value=raw):
        restored=DinoV3PCAEncoder.from_skeleton({},ckpt_dir=str(root))
    restored.load_state_dict(enc.state_dict(),strict=True);restored.eval()
    torch.testing.assert_close(z,restored.batch_encode(video),rtol=0,atol=0)
    (root/'svae_config.json').write_text('{}')
    try:DinoV3PCAEncoder.from_skeleton({},ckpt_dir=str(root));raise AssertionError('ambiguous sidecar accepted')
    except ValueError:pass
    bad=dict(stats,std=torch.zeros(8));torch.save(bad,f)
    try:FrozenPCA.from_file(f);raise AssertionError('zero std accepted')
    except ValueError:pass
    try:enc.batch_encode(video[:,:,:2]);raise AssertionError('noncausal length accepted')
    except ValueError:pass
