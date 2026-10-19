"""Real CPU model construction, initial-weight pairing and native RGB parity."""
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from matched_model import ROOT,build_cpu
from openwam.model.video_backbone.wan.preprocess import preprocess_video

path=ROOT/'model-cpu-audit.json';assert not path.exists()
start=time.monotonic();result=dict(passed=False,seeds=[],preprocessing=[])
try:
    metadata=json.loads((ROOT/'attempt-data-restored/window-manifest.json').read_text())['windows']
    for seed in [42,43,44]:
        model,cfg,row=build_cpu(seed);result['seeds'].append(row)
        if seed==42:
            encoder=model.video_backbone.video_encoder
            assert encoder.properties.z_dim==48 and encoder.properties.spatial_compression==16 and encoder.properties.temporal_compression==4
            for task in ['adjust_bottle','handover_block','place_object_basket']:
                m=next(x for x in metadata if x['split']=='dev' and x['task']==task)
                with np.load(ROOT/'source-data/data'/m['episode']/'full-trajectory.npz',allow_pickle=False) as f:
                    images=f['images'][m['rgb_indices']]
                frames=[Image.fromarray(x) for x in images]
                wrapped=encoder.preprocess_video(frames)
                native=preprocess_video(frames,dtype=wrapped.dtype,device='cpu')
                assert torch.equal(wrapped,native)
                assert tuple(wrapped.shape)==(1,3,9,384,320)
                assert torch.isfinite(wrapped).all() and wrapped.min()>=-1 and wrapped.max()<=1
                result['preprocessing'].append(dict(task=task,cache_index=m['cache_index'],native_pixel_max_difference=float((wrapped-native).abs().max()),shape=list(wrapped.shape),dtype=str(wrapped.dtype)))
        del model;gc.collect()
        print(json.dumps(dict(seed=seed,matched=row['matched_trainable_tensors'],seconds=time.monotonic()-start)),flush=True)
    assert not torch.cuda.is_initialized();result['passed']=True
except Exception as e:
    result['error']=repr(e);raise
finally:
    result.update(seconds=time.monotonic()-start,cuda_initialized=torch.cuda.is_initialized(),forward_executed=False,
        limitation='Pixel parity on three fixed development windows, not encoder-latent or training/deployment forward parity. GPU encoding and training remain unstarted.')
    path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(passed=result['passed'],seeds=len(result['seeds']),pixel_checks=len(result['preprocessing']),seconds=result['seconds'])))
