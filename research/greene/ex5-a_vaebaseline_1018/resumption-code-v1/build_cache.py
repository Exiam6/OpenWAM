"""Gated VAE encoding job; importing/checking admission never initializes CUDA."""
import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import signal
import time

ROOT=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')
BASE=ROOT.parent/'rae-factorial-20261010'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def admission():
    path=ROOT/'gpu-registration.json'
    assert path.exists(),'GPU registration absent; current stage is CPU-only'
    p=json.loads(path.read_text())
    assert p['gpu_launch_permitted'] and p['cache_seconds_limit']==7200
    assert 'cache' in p['released_stages']
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(p['deadline'])
    # The launcher must freeze both the reviewed gates and the actual runtime code.
    for name,digest in p['cache_files'].items():assert sha(name)==digest,name
    assert str(Path(__file__).resolve()) in p['cache_files']
    for name in p['pause_paths']:assert not Path(name).exists(),name
    return p

def main():
    p=admission()
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from openwam.model.video_backbone.wan.models.vae import WanVideoVAE38
    from openwam.model.video_backbone.encoder.wan22_vae import WanVideoVAEEncoder
    from openwam.model.video_backbone.wan.preprocess import preprocess_video
    assert not torch.cuda.is_initialized()
    out=ROOT/'vae-cache';out.mkdir(exist_ok=True)
    assert not (out/'started.json').exists(),'Previous attempt must be preserved'
    start=time.monotonic();result=dict(passed=False,completed=0,preflight=[])
    (out/'started.json').write_text(json.dumps(dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),seconds_limit=7200,pid=__import__('os').getpid()))+'\n')
    def guard():
        assert time.monotonic()-start<7140,'Cache cap reached; cleanup reserve'
        assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(p['deadline'])
        for name in p['pause_paths']:assert not Path(name).exists()
    def stop(*_):raise TimeoutError('Cache job deadline or termination')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(7140)
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        audit=json.loads((ROOT/'attempt-data-restored/input-audit.json').read_text());assert audit['passed']
        manifest_path=ROOT/'attempt-data-restored/window-manifest.json';assert sha(manifest_path)==audit['window_manifest_sha256']
        manifest=json.loads(manifest_path.read_text());windows=manifest['windows'];assert len(windows)==6424
        by_episode=collections.defaultdict(list)
        for m in windows:by_episode[m['episode']].append(m)
        sources={x['episode']:x for x in manifest['sources']}
        assert set(sources)==set(by_episode)
        for x in sources.values():guard();assert sha(x['path'])==x['sha256']
        original=torch.load(BASE/'cache.pt',map_location='cpu',weights_only=False,mmap=True)
        assert [dict((k,m[k]) for k in original['metadata'][0]) for m in windows]==original['metadata']
        vae=WanVideoVAE38().to(dtype=torch.bfloat16)
        values=load_file(str(ROOT/'assets/wan-vae48.safetensors'),device='cpu')
        vae.load_state_dict(values,strict=True);del values
        vae.requires_grad_(False).eval();encoder=WanVideoVAEEncoder(vae).to('cuda')
        def pixels(images):return encoder.preprocess_video([Image.fromarray(x) for x in images])
        def encode(images):
            guard()
            with torch.no_grad():z=encoder.batch_encode(pixels(images))
            assert torch.isfinite(z).all();return z
        # Training-only selection by immutable metadata, before any model outcome.
        for task in ['adjust_bottle','handover_block','place_object_basket']:
            m=next(x for x in windows if x['task']==task and x['split']=='train')
            with np.load(sources[m['episode']]['path'],allow_pickle=False) as f:images=f['images'][m['rgb_indices']]
            frames=[Image.fromarray(x) for x in images];native=preprocess_video(frames,dtype=torch.bfloat16,device='cuda')
            assert torch.equal(native,pixels(images))
            full=encode(images);first=encode(images[:1]);changed=images.copy();changed[1:]=255-changed[1:];altered=encode(changed)
            with torch.no_grad():native_z=vae.batch_encode(native,device='cuda')
            assert tuple(full.shape)==(1,48,3,24,20)
            native_diff=float((native_z-full).abs().max());first_diff=float((first-full[:,:,:1]).abs().max());causal_diff=float((altered[:,:,:1]-full[:,:,:1]).abs().max())
            future_diff=float((altered[:,:,1:]-full[:,:,1:]).abs().max())
            assert native_diff<=1e-6 and first_diff<=1e-6 and causal_diff<=1e-6 and future_diff>1e-6
            result['preflight'].append(dict(task=task,cache_index=m['cache_index'],native_difference=native_diff,first_frame_difference=first_diff,future_to_first_difference=causal_diff,future_difference=future_diff))
        (out/'preflight.json').write_text(json.dumps(dict(passed=True,rows=result['preflight']),indent=2)+'\n')
        latents=torch.empty((6424,48,3,24,20),dtype=torch.bfloat16,device='cpu')
        encoding_start=time.monotonic();sampled=[]
        for episode,items in by_episode.items():
            with np.load(sources[episode]['path'],allow_pickle=False) as f:images=f['images']
            for m in items:
                z=encode(images[m['rgb_indices']]).squeeze(0).cpu();latents[m['cache_index']].copy_(z);result['completed']+=1
                if result['completed']<=24:sampled.append(time.monotonic()-encoding_start)
                if result['completed']==24:
                    # Include initial work, observed encoding rate and120s serialization/cleanup reserve.
                    projected=time.monotonic()-start+(6424-24)*sampled[-1]/24+120
                    (out/'cost-review.json').write_text(json.dumps(dict(passed=projected<7140,projected_seconds=projected,cap=7200))+'\n')
                    assert projected<7140,'Full cache does not fit its registered budget'
                if result['completed']%64==0:
                    (out/'progress.json').write_text(json.dumps(dict(completed=result['completed'],total=6424,seconds=time.monotonic()-start))+'\n')
            del images
        guard();assert result['completed']==6424
        payload={k:original[k].clone() if torch.is_tensor(original[k]) else original[k] for k in ['context','seq_lens','actions','proprio','task_ids','metadata']};payload['vae']=latents
        dest=out/'cache.pt';assert not dest.exists();temp=out/'cache.pt.partial';torch.save(payload,temp);temp.rename(dest)
        saved=torch.load(dest,map_location='cpu',weights_only=False,mmap=True)
        for k in ['context','seq_lens','actions','proprio','task_ids']:assert torch.equal(saved[k],original[k])
        assert saved['metadata']==original['metadata'] and torch.equal(saved['vae'],latents)
        result.update(passed=True,cache=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,latents_shape=list(latents.shape),serialization_exact=True)
    except Exception as error:
        result['error']=repr(error);raise
    finally:
        signal.alarm(0);result['occupied_seconds']=time.monotonic()-start
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='preflight'}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-admission',action='store_true');args=parser.parse_args()
    if args.check_admission:
        try:admission();print('admitted')
        except AssertionError as e:print(str(e));raise SystemExit(2)
    else:main()
