"""First-hour interface gate; no fitting and no held-out scoring."""
import os,time,json,sys
from pathlib import Path
import cv2,h5py,numpy as np,torch
import common as c
out=Path(os.environ['LAYER_RUN']);out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(6);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
inventory={}
for task in c.TASKS:
    files=sorted((c.ROOT/'data'/task).rglob('*.hdf5'),key=lambda p:int(p.stem.replace('episode','')))
    assert len(files)==50,(task,len(files));counts=[]
    for p in files:
        with h5py.File(p,'r') as f:
            n=len(f['endpose/left_endpose']);assert n>=33
            for name in ['endpose/right_endpose','endpose/left_gripper','endpose/right_gripper','observation/head_camera/rgb','observation/left_camera/rgb','observation/right_camera/rgb']:
                assert len(f[name])==n,(p,name)
            counts.append(n)
    inventory[task]={'episodes':len(files),'frames_min':min(counts),'frames_max':max(counts),'sample':str(files[0].relative_to(c.ROOT))}
c.write(out/'data-inventory.json',inventory)
# First training episode only; previous hold-outs are not read for scores.
epi=next(e for e,s in c.split_map().items() if s=='train')
p=next((c.ROOT/'data'/c.TASKS[0]).rglob(f'episode{epi}.hdf5'))
with h5py.File(p,'r') as f:
    images=[c.resize_head(cv2.imdecode(np.frombuffer(bytes(f['observation/head_camera/rgb'][i]),np.uint8),cv2.IMREAD_COLOR)) for i in range(0,33,4)]
enc=c.encoder();pix=enc.preprocess_video(images);torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    baseline=enc.batch_encode_pooled_for_svae_training(pix)
    selected=c.encode(enc,pix,check=True)
    assert torch.equal(selected['L12'],baseline),'K1 must match exact existing path'
    altered=pix.clone();altered[:,:,1:]=altered[:,:,1:].flip(2)
    changed=c.encode(enc,altered)
    for k in c.SOURCES:assert torch.equal(selected[k][:,:,0],changed[k][:,:,0]),k
    torch.cuda.synchronize();t=time.perf_counter()
    for _ in range(5):c.encode(enc,pix)
    torch.cuda.synchronize();seconds=(time.perf_counter()-t)/5
result={'passed':True,'strict_weights':True,'K1_exact':True,'condition_future_invariance_all_sources':True,'source_shapes':{k:list(v.shape) for k,v in selected.items()},'seconds_per_9frame_clip':seconds,'peak_allocated_GB':torch.cuda.max_memory_allocated()/1e9,'input':'head_only_384x320_native_single_view','normalization':'same learned final norm once per selected block; K4 fp32 mean cast to bf16','encoder_sha256':c.sha(c.ROOT/'assets/dinov3-study/encoder.safetensors'),'files':{p.name:c.sha(p) for p in Path(__file__).parent.glob('*.py')},'no_model_fit':True,'no_heldout_scoring':True}
c.write(out/'interface-preflight.json',result);print(json.dumps(result),flush=True)
