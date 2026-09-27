#!/usr/bin/env python3
"""Repeat corruption evaluation at the exact clean encoder batch geometry.

The initial single-frame extraction was not numerically identical to nine-frame
DINO extraction in bf16. Preserve those outputs as an audit trail, but exclude
them from the final report. All perturbations are re-encoded in batches of nine
identical frames and only the conditioning latent is used (DINO is per-frame).
"""
import gc,time
import torch,numpy as np,h5py,cv2
from PIL import Image
import night as n
p=n.p

@torch.inference_mode()
def matched(data,folder):
    path=folder/'corruptions-matched-batch.pt'
    if path.exists():return torch.load(path,weights_only=False)
    enc=p.build_encoder();test=p.ids(data,'test').tolist();out={'test_ids':test,'noise10':[],'brightness06':[]};diff=[];mse=[];start=time.perf_counter()
    for index in test:
        meta=data['manifest'][index]
        with h5py.File(p.ROOT/'data'/meta['episode_file'],'r') as f:
            images={c:Image.fromarray(cv2.imdecode(np.frombuffer(bytes(f[f'observation/{c}/rgb'][meta['start_frame']]),np.uint8),cv2.IMREAD_COLOR)) for c in p.CAMS}
        frame=p.assemble_multiview_layout(images,p.CAMS,384,320);arr=np.asarray(frame).astype(np.float32)
        rng=np.random.default_rng(10000+meta['episode']*1000+meta['start_frame'])
        variants={'clean':frame,'noise10':Image.fromarray(np.clip(arr+rng.normal(0,10,arr.shape),0,255).astype(np.uint8)),'brightness06':Image.fromarray(np.clip(arr*.6,0,255).astype(np.uint8))}
        for name,im in variants.items():
            z=enc.batch_encode_pooled_for_svae_training(enc.preprocess_video([im]*9))[0,:,:1].cpu().half()
            if name=='clean':
                error=z.float()-data['features'][index,:,:1].float();diff.append(float(error.abs().max()));mse.append(float(error.square().mean()))
            else:out[name].append(z)
    assert max(diff)<1e-6,('Clean encoding mismatch',max(diff))
    for name in ['noise10','brightness06']:out[name]=torch.stack(out[name])
    p.write_json(folder/'corruption-matched-batch-check.json',{'wall_seconds':time.perf_counter()-start,'all_test_clean_max_abs':max(diff),'all_test_clean_mean_mse':float(np.mean(mse)),'test_clips_checked':len(test),'frames_per_encoder_batch':9,'noise_sigma_pixels':10,'brightness_factor':.6,'old_single_frame_results_excluded':True})
    torch.save(out,path);del enc;gc.collect();torch.cuda.empty_cache();return out

def main():
    p.ensure_gpu();start=time.perf_counter()
    for task in n.TASKS:
        folder=n.RUNROOT/task;assert (folder/'complete.json').exists();p.OUT=folder;p.DATA_TASK=task
        data=p.get_data();pca=p.fit_pca(data);corrupt=matched(data,folder);dest=folder/'evaluation-matched-batch';dest.mkdir(exist_ok=True)
        n.evaluate(data,pca,corrupt,dest,'pca48')
        for seed in n.SEEDS:
            for weight in [1,4]:
                name=f'svae-w{weight}-s{seed}'
                n.evaluate(data,pca,corrupt,dest,name,folder/name/'final.pt')
        del data,pca,corrupt;gc.collect();torch.cuda.empty_cache()
    p.write_json(n.RUNROOT/'corrections-complete.json',{'wall_seconds':time.perf_counter()-start,'completed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    print('MATCHED-BATCH EVALUATION COMPLETE',flush=True)
if __name__=='__main__':main()
