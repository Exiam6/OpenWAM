"""Fixed six-run adapter study; select with validation, never demonstration test."""
import hashlib
import json
import sys
import time
from pathlib import Path
import cv2
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from safetensors.torch import load_file
from robustness_common import Adapter, noise_image

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/robustness-20260921'
sys.path.insert(0,str(ROOT/'OpenWAM'))
from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder
from openwam.dataloader.transforms.multiview import assemble_multiview_layout

def write(name,obj):
    tmp=OUT/(name+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(OUT/name)

def extract():
    enc=DinoV3VideoEncoder.from_skeleton({},ckpt_dir=str(ROOT/'assets/dinov3-policy'))
    prefix='video_backbone.video_encoder.'
    weights=load_file(ROOT/'assets/dinov3-study/encoder.safetensors')
    enc.load_state_dict({k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)},strict=True)
    enc=enc.to(device='cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
    reference=torch.load(OUT/'encoder-reference.pt',map_location='cpu',weights_only=True)
    assert {k:str(v.dtype) for k,v in enc.state_dict().items()}==reference['dtypes']
    with torch.inference_mode():
        z=enc.batch_encode(reference['pixels'].cuda()).cpu()
    delta=float((z-reference['latent']).abs().max());assert delta==0,delta
    write('standalone-encoder-check.json',{'max_abs_difference':delta,'state_dtypes_match':True})
    files=sorted((ROOT/'data/pick_dual_bottles').rglob('*.hdf5'),key=lambda p:int(p.stem.replace('episode','')))
    assert len(files)==50
    order=np.random.default_rng(42).permutation(50)
    splits={int(e):'train' if i<35 else 'val' if i<40 else 'test' for i,e in enumerate(order)}
    clean=[];noisy=[];rows=[];references=[];camera_names=['head_camera','left_camera','right_camera']
    started=time.monotonic()
    for episode,path in enumerate(files):
        if splits[episode]=='test':continue
        with h5py.File(path,'r') as f:
            starts=np.unique(np.linspace(0,len(f['endpose/left_endpose'])-33,12,dtype=int))
            for frame in starts:
                images={cam:Image.fromarray(cv2.imdecode(np.frombuffer(bytes(f[f'observation/{cam}/rgb'][frame]),np.uint8),cv2.IMREAD_COLOR)) for cam in camera_names}
                layout=assemble_multiview_layout(images,camera_names,384,320)
                with torch.inference_mode():
                    target=enc.batch_encode(enc.preprocess_video([layout]))[0].cpu()
                index=len(clean);clean.append(target)
                for replica in range(2 if splits[episode]=='train' else 1):
                    altered={**images,'head_camera':Image.fromarray(noise_image(np.asarray(images['head_camera']),500000+episode,2*int(frame)+replica))}
                    noisy_layout=assemble_multiview_layout(altered,camera_names,384,320)
                    with torch.inference_mode():
                        corrupt=enc.batch_encode(enc.preprocess_video([noisy_layout]))[0].cpu()
                    noisy.append(corrupt);references.append(index)
                    rows.append({'episode':episode,'frame':int(frame),'split':splits[episode],'replica':replica})
        print('EXTRACT',episode,splits[episode],len(rows),round(time.monotonic()-started,1),flush=True)
    data={'clean':torch.stack(clean),'noisy':torch.stack(noisy),'references':torch.tensor(references),'manifest':rows}
    torch.save(data,OUT/'adapter-cache.pt')
    write('adapter-extraction.json',{'clean_frames':len(clean),'corrupt_frames':len(noisy),'shape':list(data['clean'].shape),'manifest':rows,'seconds':time.monotonic()-started,'test_episodes_opened':False})
    del enc;torch.cuda.empty_cache()
    return data

@torch.no_grad()
def score(model,clean,noisy):
    restore=[];distort=[]
    for x,y in zip(noisy.split(8),clean.split(8)):
        restore.append((model(x).float()-y.float()).square().flatten(1).mean(1))
        distort.append((model(y).float()-y.float()).square().flatten(1).mean(1))
    return {'restoration_mse':torch.cat(restore).mean().item(),'clean_distortion_mse':torch.cat(distort).mean().item()}

def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    for _ in range(300):
        if (OUT/'encoder-reference.json').is_file():break
        time.sleep(1)
    assert (OUT/'encoder-reference.json').exists(),'Full policy reference did not arrive'
    data=extract()
    clean=data['clean'][data['references']].cuda();noisy=data['noisy'].cuda()
    train=torch.tensor([i for i,r in enumerate(data['manifest']) if r['split']=='train'],device='cuda')
    val=torch.tensor([i for i,r in enumerate(data['manifest']) if r['split']=='val'],device='cuda')
    identity=float((clean[val].float()-noisy[val].float()).square().mean())
    results=[];start=time.monotonic()
    for kind in ['linear','mlp']:
        for seed in [42,43,44]:
            torch.manual_seed(seed);model=Adapter(kind).cuda()
            assert torch.equal(model(noisy[:2]),noisy[:2])
            # Future frames untouched, input untouched, gradients confined to adapter.
            probe=torch.randn(2,48,3,4,4,device='cuda',requires_grad=True);original=probe.detach().clone()
            check=model(probe.detach());check.square().mean().backward()
            assert probe.grad is None and torch.equal(probe.detach(),original) and torch.equal(check[:,:,1:],probe[:,:,1:])
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            rng=torch.Generator(device='cuda').manual_seed(seed)
            torch.cuda.reset_peak_memory_stats();t=time.monotonic()
            for step in range(2000):
                ids=train[torch.randint(len(train),(8,),device='cuda',generator=rng)]
                x=noisy[ids].float();y=clean[ids].float()
                opt.zero_grad(set_to_none=True)
                loss=F.mse_loss(model(x),y)+F.mse_loss(model(y),y)
                assert torch.isfinite(loss)
                loss.backward();opt.step()
            model.eval();metrics=score(model,clean[val],noisy[val])
            row={'kind':kind,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),'steps':2000,**metrics,'seconds':time.monotonic()-t,'peak_gb':torch.cuda.max_memory_allocated()/1e9}
            results.append(row);write('adapter-training.json',results)
            torch.save({'kind':kind,'seed':seed,'state_dict':{k:v.cpu() for k,v in model.state_dict().items()}},OUT/f'adapter-{kind}-{seed}.pt')
            print('TRAINED',json.dumps(row),flush=True)
    candidates={}
    for kind in ['linear','mlp']:
        rows=[r for r in results if r['kind']==kind]
        restore=float(np.mean([r['restoration_mse'] for r in rows]));distort=float(np.mean([r['clean_distortion_mse'] for r in rows]))
        candidates[kind]={'restoration_mse':restore,'clean_distortion_mse':distort,'noise_ratio':restore/identity,'clean_ratio':distort/identity,'eligible':restore<=.9*identity and distort<=.1*identity}
    eligible=[k for k,v in candidates.items() if v['eligible']]
    selected=min(eligible,key=lambda k:candidates[k]['restoration_mse']) if eligible else None
    if selected:
        import shutil
        shutil.copyfile(OUT/f'adapter-{selected}-42.pt',OUT/'adapter-selected.pt')
    write('adapter-selection.json',{'identity_noise_mse':identity,'candidates':candidates,'selected':selected,'deployment_seed':42,'validation_only':True,'seconds_training':time.monotonic()-start,'selected_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    print('ADAPTER COMPLETE',selected,flush=True)

if __name__=='__main__':main()
