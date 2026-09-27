"""One-time fresh-episode confirmation using only frozen reducers/readouts."""
import datetime,json,os,time
from pathlib import Path
import cv2,h5py,numpy as np,torch
from PIL import Image
import common as c
from readout_v2 import visual,predict,scores
RUN=Path(os.environ['LAYER_RUN']);OUT=RUN/'confirmation'

def main():
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
    assert not (OUT/'summary.json').exists(),'Never repeat completed confirmation'
    manifest=json.loads((RUN/'fresh/manifest-v2.json').read_text());assert manifest['complete']
    assert all(len(manifest['accepted'][t])==20 for t in c.TASKS)
    assert json.loads((RUN/'confirmation-parity.json').read_text())['passed']
    assert json.loads((RUN/'confirmation-identity.json').read_text())['passed']
    frozen=json.loads((RUN/'confirmation-freeze.json').read_text())
    for p,h in frozen['sha256'].items():assert c.sha(RUN/p)==h,p
    assert c.sha(c.ROOT/'assets/dinov3-study/encoder.safetensors')==frozen['encoder_sha256']
    OUT.mkdir(exist_ok=True);enc=c.encoder();rows=[];states=[];targets=[];refs=[];conds=[];features={s:[] for s in c.SOURCES};seen=set();start=time.monotonic()
    for ti,task in enumerate(c.TASKS):
        for record in manifest['accepted'][task]:
            seed=record['seed'];path=Path(record['hdf5']);assert c.sha(path)==record['hdf5_sha256'];assert seed not in seen;seen.add(seed)
            with h5py.File(path,'r') as f:
                n=len(f['endpose/left_endpose']);assert n>=44
                assert len(np.unique(np.linspace(0,n-33,12,dtype=int)))==12
                for st in np.unique(np.linspace(0,n-33,12,dtype=int)):
                    st=int(st);i=len(rows);lp=f['endpose/left_endpose'][st];rp=f['endpose/right_endpose'][st]
                    states.append(np.concatenate([lp,rp,[f['endpose/left_gripper'][st],f['endpose/right_gripper'][st]]]))
                    targets.append(np.concatenate([f['endpose/left_endpose'][st+4,:3]-lp[:3],f['endpose/right_endpose'][st+4,:3]-rp[:3],[f['endpose/left_gripper'][st+4],f['endpose/right_gripper'][st+4]]]))
                    arr=np.array(c.resize_head(cv2.imdecode(np.frombuffer(bytes(f['observation/head_camera/rgb'][st]),np.uint8),cv2.IMREAD_COLOR)))
                    left=c.resize_head(cv2.imdecode(np.frombuffer(bytes(f['observation/left_camera/rgb'][st]),np.uint8),cv2.IMREAD_COLOR))
                    variants=[('clean',Image.fromarray(arr)),('left_transfer',left)]
                    for sigma in [.1,.04]:
                        for draw in range(3):
                            rng=np.random.default_rng(20260922+seed*100000+st*10+draw)
                            variants.append((f'noise{sigma:.2f}',Image.fromarray(np.clip(arr.astype(np.float32)+rng.normal(0,255*sigma,arr.shape),0,255).astype(np.uint8))))
                    for name,im in variants:
                        z=c.encode(enc,enc.preprocess_video([im]*9));refs.append(i);conds.append(name)
                        for source in c.SOURCES:features[source].append(z[source][0,:,:1].cpu().half())
                    rows.append({'task':task,'seed':seed,'frame':st,'target_frame':st+4,'episode_sha256':record['hdf5_sha256'],'head_encoded_sha256':__import__('hashlib').sha256(bytes(f['observation/head_camera/rgb'][st])).hexdigest()})
            c.write(OUT/'progress.json',{'stage':'extracting','episodes':len(seen),'frames':len(rows),'elapsed_seconds':time.monotonic()-start})
    del enc;torch.cuda.empty_cache();arrays={}
    for source in c.SOURCES:
        arrays.update(visual(torch.stack(features[source]),source));del features[source]
    state=np.array(states,dtype=np.float32);target=np.array(targets,dtype=np.float32);refs=np.array(refs);conds=np.array(conds);tasks=np.array([r['task'] for r in rows]);ep=np.array([r['seed'] for r in rows])
    np.savez(OUT/'vectors.npz',**arrays,state=state,target=target,reference=refs,condition=conds);c.write(OUT/'sample-manifest.json',rows)
    coef=np.load(RUN/'readout/coefficients.npz');saved=torch.load(RUN/'readout/mlp/models.pt',map_location='cpu',weights_only=False)
    names=['proprio']+list(arrays);results={};preds={}
    for method in ['ridge','mlp']:
        result={}
        for task in c.TASKS:
            for name in names:
                key=task+'/'+name
                if method=='ridge':
                    models=[{k:coef[f'{key}/{g}/{k}'] for k in ['xm','xs','ym','ys','w']} for g in [0,1]];scale=np.concatenate([m['ys'] for m in models])
                else:
                    xm,xs,ym,scale=[t.numpy() for t in saved['scales'][key]]
                    model=torch.nn.Sequential(torch.nn.Linear(len(xm),128),torch.nn.ReLU(),torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,8));model.load_state_dict(saved['states'][key]);model.eval()
                for condition in ['clean','noise0.10','noise0.04','left_transfer']:
                    ii=np.flatnonzero((tasks[refs]==task)&(conds==condition));rr=refs[ii]
                    x=state[rr] if name=='proprio' else np.concatenate([arrays[name][ii],state[rr]],1)
                    if method=='ridge':pred=np.concatenate([predict(m,x) for m in models],1);y=target[rr].astype(float)
                    else:
                        with torch.inference_mode():pred=model(torch.from_numpy((x-xm)/xs)).numpy()*scale+ym
                        y=target[rr]
                    k=key+'/'+condition;preds[method+'/'+k]=pred;result[k]=scores(pred,y,scale,ep[rr])
        results[method]=result
    paired={};rng=np.random.default_rng(20260922);ix=rng.integers(0,20,(2000,3,20))
    for method,result in results.items():
        paired[method]={}
        for condition in ['clean','noise0.10','noise0.04','left_transfer']:
            aa=[];bb=[]
            for task in c.TASKS:
                aa.append(np.mean([result[f'{task}/K4_svae{s}/{condition}']['per_episode_groups'] for s in [42,43,44]],0));bb.append(np.mean([result[f'{task}/L12_svae{s}/{condition}']['per_episode_groups'] for s in [42,43,44]],0))
            a=np.stack(aa);b=np.stack(bb);delta=a-b
            boot=[100*np.mean([delta[t,z[t]].mean() for t in range(3)])/np.mean([b[t,z[t]].mean() for t in range(3)]) for z in ix]
            paired[method][condition]={'E_relative_change_percent':float(100*delta.mean()/b.mean()),'relative_percent95':np.quantile(boot,[.025,.975]).tolist(),'per_task_relative_percent':[float(100*delta[t].mean()/b[t].mean()) for t in range(3)],'per_task_group_relative_percent':(100*delta.mean(1)/b.mean(1)).tolist(),'task_order':c.TASKS,'independent_episodes':60}
    noise=paired['ridge']['noise0.10'];clean=paired['ridge']['clean']
    passes=noise['E_relative_change_percent']<=-10 and noise['relative_percent95'][1]<0 and clean['relative_percent95'][1]<=3 and sum(x<0 for x in noise['per_task_relative_percent'])>=2 and np.max(noise['per_task_group_relative_percent'])<=5
    summary={'completed_at':datetime.datetime.now().astimezone().isoformat(),'scope':'fresh within-task expert-feasible scenes; offline control readout, not closed-loop policy','conditions':results,'primary_paired':paired,'predeclared_primary_success':bool(passes),'episodes':60,'frames':len(rows),'refit_models':0,'limitations':['not unseen tasks','prior encoder pretraining overlap beyond available episode corpus unknown','left_transfer is head-trained to wrist camera; no wrist-trained within-view baseline yet','not contact or controller-command labels','3reducer seeds averaged within same60episodes, not180 independent observations'],'elapsed_seconds':time.monotonic()-start}
    np.savez(OUT/'predictions.npz',**preds);c.write(OUT/'summary.json',summary);print(json.dumps({'primary':paired,'primary_success':bool(passes)}),flush=True)
if __name__=='__main__':main()
