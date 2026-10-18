"""Extract only frozen VAE weights; no forward pass, CUDA or policy weights."""
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from openwam.model.video_backbone.wan.models.vae import WanVideoVAE38

R=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')
SOURCE=R.parent/'ex4-h_officialcalibration_1017/checkpoint/checkpoint_step_118655.safetensors'
DEST=R/'assets/wan-vae48.safetensors'
torch.set_num_threads(4)
assert not DEST.exists()
with torch.device('meta'):
    model=WanVideoVAE38()
expected=model.state_dict();prefix='video_backbone.vae.'
with safe_open(str(SOURCE),framework='pt',device='cpu') as f:
    names={k[len(prefix):] for k in f.keys() if k.startswith(prefix)}
    assert names==set(expected)
    values={}
    for name in sorted(names):
        x=f.get_tensor(prefix+name)
        assert x.shape==expected[name].shape and torch.isfinite(x).all()
        values[name]=x.contiguous()
DEST.parent.mkdir(exist_ok=True)
save_file(values,str(DEST))
with safe_open(str(DEST),framework='pt',device='cpu') as f:
    assert set(f.keys())==names
    assert all(torch.equal(values[k],f.get_tensor(k)) for k in names)
assert not torch.cuda.is_initialized()
result=dict(passed=True,source=str(SOURCE),source_prefix=prefix,destination=str(DEST),bytes=DEST.stat().st_size,
    sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(),tensors=len(names),parameters=sum(x.numel() for x in values.values()),
    shape_contract='Every source tensor exactly matches WanVideoVAE38 meta skeleton',serialization_exact=True,cuda_initialized=False,
    policy_weights_extracted=False,forward_executed=False)
(R/'vae-extraction.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
