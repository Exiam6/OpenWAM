"""Independent CPU audit before constructing a matched VAE cache."""
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from safetensors import safe_open
from openwam.dataloader.utils.normalization import apply_normalization

ROOT = Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')
BASE = ROOT.parent/'rae-factorial-20261010'
DATA = ROOT.parent/'rae-policy-expanded-20261006'
OFFICIAL = ROOT.parent/'ex4-h_officialcalibration_1017/checkpoint/checkpoint_step_118655.safetensors'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()

def write(path, obj):
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(obj, indent=2)+'\n'); temp.replace(path)

def main():
    torch.set_num_threads(4)
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT/'input-audit.json').exists(), 'Preserve prior audit evidence'
    began = time.monotonic()
    result = dict(started=dt.datetime.now(dt.timezone.utc).isoformat(),passed=False,source=str(DATA),gpu_initialized=False)
    try:
        old = json.loads((DATA/'dataset-audit.json').read_text())
        p = json.loads((BASE/'protocol.json').read_text())
        cache = torch.load(DATA/'cache.pt', map_location='cpu', weights_only=False, mmap=True)
        assert cache['metadata'] == old['metadata']
        metadata = cache['metadata']; assert len(metadata) == 6424
        result['cache_sha256'] = sha(DATA/'cache.pt')
        assert result['cache_sha256'] == old['cache_sha256']
        shapes = dict(raw=(6424,768,3,24,20),svae=(6424,48,3,24,20),actions=(6424,32,20),proprio=(6424,20))
        for key, shape in shapes.items(): assert tuple(cache[key].shape) == shape
        stats = np.load(DATA/'normalization_stats.npy',allow_pickle=True).item()['eef']
        by_episode = collections.defaultdict(list)
        for i, m in enumerate(metadata): by_episode[m['episode']].append((i,m))
        assert set(by_episode) == set(old['source_hashes'])
        train = {m['episode'] for m in metadata if m['split']=='train'}
        dev = {m['episode'] for m in metadata if m['split']=='dev'}
        assert not train & dev and len(train)==765 and len(dev)==76
        manifest=[]; sources=[]; max_action=max_proprio=0.
        for n,(episode,windows) in enumerate(by_episode.items()):
            assert time.monotonic()-began < 3*3600
            path=DATA/'data'/episode/'full-trajectory.npz'
            digest=sha(path); assert digest == old['source_hashes'][episode], episode
            with zipfile.ZipFile(path) as z, z.open('images.npy') as f:
                version=np.lib.format.read_magic(f)
                header=np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f)
            shape,fortran,dtype=header
            assert shape[1:]==(384,320,3) and dtype==np.uint8 and not fortran
            with np.load(path,allow_pickle=False) as data:
                raw=data['proprio'];ticks=data['physics_ticks'];timestep=float(data['timestep'])
            assert raw.shape==(shape[0],20) and np.isfinite(raw).all()
            normalized=apply_normalization(raw,stats,'min-max')
            assert np.isfinite(normalized).all()
            expected_starts=np.linspace(0,len(raw)-33,8 if windows[0][1]['split']=='train' else 4).astype(int)
            assert [m['start'] for _,m in windows]==expected_starts.tolist(), episode
            for i,m in windows:
                start=m['start'];indices=list(range(start,start+33,4)); assert len(indices)==9 and indices[-1]<shape[0]
                a=float(np.max(np.abs(cache['actions'][i].numpy()-normalized[start+1:start+33])))
                q=float(np.max(np.abs(cache['proprio'][i].numpy()-normalized[start])))
                assert a==0. and q==0., (i,a,q)
                max_action=max(max_action,a);max_proprio=max(max_proprio,q)
                assert int(cache['task_ids'][i])==p['tasks'].index(m['task'])
                duration=float((ticks[start+32]-ticks[start])*timestep)
                assert abs(duration-m['horizon_seconds'])<1e-7
                manifest.append(dict(cache_index=i,**m,rgb_indices=indices,action_indices=[start+1,start+32],source_sha256=digest))
            sources.append(dict(episode=episode,path=str(path),sha256=digest,frames=shape[0]))
            if n%100==0:print(json.dumps(dict(episodes=n+1,total=len(by_episode),seconds=time.monotonic()-began)),flush=True)
        schedule=[]
        for seed in p['seeds']:
            rng=np.random.default_rng(seed)
            indices={t:np.array([i for i,m in enumerate(metadata) if m['split']=='train' and m['task']==t]) for t in p['tasks']}
            rebuilt=np.array([[[rng.choice(indices[t]) for t in p['tasks']] for _ in range(2)] for _ in range(20000)])
            path=BASE/'runs'/f'raw768_native-seed{seed}'/'batches.npy'
            assert np.array_equal(rebuilt,np.load(path)), seed
            init=BASE/'initialization'/f'common-seed{seed}.safetensors'
            with safe_open(str(init),framework='pt',device='cpu') as f: tensor_count=len(f.keys())
            schedule.append(dict(seed=seed,batches_path=str(path),batches_sha256=sha(path),exact_replay=True,initialization_path=str(init),initialization_sha256=sha(init),initialization_tensors=tensor_count))
        with safe_open(str(OFFICIAL),framework='pt',device='cpu') as f:
            tensors=[dict(key=k,shape=f.get_slice(k).get_shape()) for k in f.keys() if k.startswith('video_backbone.vae.')]
        assert len(tensors)==196
        write(ROOT/'window-manifest.json',dict(windows=sorted(manifest,key=lambda x:x['cache_index']),sources=sources))
        result.update(passed=True,episodes=len(sources),train_episodes=len(train),dev_episodes=len(dev),windows=len(manifest),max_action_difference=max_action,max_proprio_difference=max_proprio,
                      schedules=schedule,official_vae_tensor_inventory=tensors,normalization_sha256=sha(DATA/'normalization_stats.npy'),
                      window_manifest_sha256=sha(ROOT/'window-manifest.json'),
                      note='No VAE encoding/model execution; header inventory is not weight extraction or GPU validation. Equal data indices and labels do not alone establish equal RGB preprocessing.')
    except Exception as e:
        result['error']=repr(e)
        raise
    finally:
        result['gpu_initialized']=torch.cuda.is_initialized();result['seconds']=time.monotonic()-began
        write(ROOT/'input-audit.json',result)
    assert not result['gpu_initialized']
    print(json.dumps({k:v for k,v in result.items() if k not in ['schedules','official_vae_tensor_inventory']}))

if __name__=='__main__':main()
