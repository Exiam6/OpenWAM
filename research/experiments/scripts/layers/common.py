"""Fixed-source layer study, isolated from prior frozen experiments."""
import hashlib, json, os, sys, types
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel
ROOT = Path(os.environ['WAM_ROOT'])
REPO = ROOT/'OpenWAM'
for name in ['openwam','openwam.model','openwam.model.video_backbone','openwam.model.video_backbone.encoder']:
    mod=types.ModuleType(name);mod.__path__=[str(REPO.joinpath(*name.split('.')))];sys.modules[name]=mod
from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder, _causal_temporal_pool
from openwam.model.video_backbone.encoder.svae.model import SVAE, svae_loss
TASKS=['adjust_bottle','handover_block','place_object_basket']
SOURCES=['L12','L6','K4']

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def encoder():
    asset=ROOT/'assets/dinov3-study';cfg=AutoConfig.from_pretrained(asset/'dinov3',local_files_only=True)
    assert cfg.hidden_size==768 and cfg.num_hidden_layers==12
    model=AutoModel.from_config(cfg,dtype=torch.bfloat16)
    weights=load_file(asset/'encoder.safetensors');prefix='video_backbone.video_encoder._m.'
    model.load_state_dict({k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)},strict=True)
    model.eval().requires_grad_(False)
    enc=DinoV3VideoEncoder(model,embed_dim=768,patch_size=16,num_register_tokens=cfg.num_register_tokens).cuda().eval().requires_grad_(False)
    return enc

@torch.inference_mode()
def encode(enc,pix,check=False):
    b,c,t,h,w=pix.shape;captured={};handles=[]
    for n in [6,9,10,11,12]:
        def save(module,args,out,n=n):captured[n]=out
        handles.append(enc._m.model.layer[n-1].register_forward_hook(save))
    try:out=enc._m(pix.permute(0,2,1,3,4).reshape(b*t,c,h,w))
    finally:
        for handle in handles:handle.remove()
    # The HF final output is normalized exactly once. Apply that same learned
    # norm once to selected intermediate block outputs before fixed averaging.
    tokens={'L12':out.last_hidden_state,'L6':enc._m.norm(captured[6])}
    tokens['K4']=torch.stack([enc._m.norm(captured[n]).float() for n in [9,10,11,12]]).mean(0).to(out.last_hidden_state.dtype)
    if check:assert torch.equal(enc._m.norm(captured[12]),out.last_hidden_state)
    result={}
    for source,z in tokens.items():
        z=z[:,1+enc._num_register_tokens:]
        z=z.reshape(b,t,h//16,w//16,768).permute(0,4,1,2,3)
        result[source]=_causal_temporal_pool(z)
        assert torch.isfinite(result[source]).all()
    return result

def resize_head(arr):
    return Image.fromarray(arr).resize((320,384),resample=Image.Resampling.BILINEAR)

def split_map():
    return {int(e):'train' if i<35 else 'development' for i,e in enumerate(np.random.default_rng(42).permutation(50))}
