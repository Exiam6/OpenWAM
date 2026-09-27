"""Fixed-capacity secondary check, no tuning or new confirmation scores."""
import json,os,time
from pathlib import Path
import numpy as np,torch
import common as c
from readout_v2 import scores
RUN=Path(os.environ['LAYER_RUN']);OUT=RUN/'readout';DEST=OUT/'mlp'

def main():
    assert (OUT/'development-summary.json').exists()
    assert not (DEST/'summary.json').exists()
    DEST.mkdir(exist_ok=True);torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
    data=np.load(OUT/'vectors.npz');rows=json.loads((OUT/'manifest.json').read_text());target=data['target'].astype(np.float32);state=data['state'].astype(np.float32)
    tasks=np.array([r['task'] for r in rows]);ep=np.array([r['episode'] for r in rows]);split=np.array([r['split'] for r in rows]);ref=data['noisy_reference'];cond=data['noisy_condition']
    names=['proprio']+[s+'_'+v for s in c.SOURCES for v in ['raw','pca','svae42','svae43','svae44']]
    models={};scales={};meta={};begin=time.monotonic()
    # Fit every frozen readout before scoring any development condition.
    for task in c.TASKS:
        tr=(tasks==task)&(split=='train')
        for name in names:
            x=state if name=='proprio' else np.concatenate([data[name],state],1)
            xm=x[tr].mean(0);xs=x[tr].std(0).clip(1e-6);ym=target[tr].mean(0);ys=target[tr].std(0).clip(1e-6)
            xx=torch.from_numpy((x[tr]-xm)/xs).cuda();yy=torch.from_numpy((target[tr]-ym)/ys).cuda()
            torch.manual_seed(42);m=torch.nn.Sequential(torch.nn.Linear(x.shape[1],128),torch.nn.ReLU(),torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,8)).cuda()
            opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4);rng=torch.Generator().manual_seed(42)
            for step in range(500):
                ii=torch.randint(len(xx),(64,),generator=rng).cuda();opt.zero_grad(set_to_none=True);pred=m(xx[ii]);e=(pred-yy[ii]).square();loss=.5*e[:,:6].mean()+.5*e[:,6:].mean();assert torch.isfinite(loss)
                loss.backward();opt.step()
            key=task+'/'+name;m.cpu().eval();models[key]=m;scales[key]=(xm,xs,ym,ys)
            meta[key]={'steps':500,'seed':42,'final_train_batch_loss':float(loss),'parameters':sum(p.numel() for p in m.parameters())}
            print('MLP_FIT',key,flush=True);del opt,xx,yy;torch.cuda.empty_cache()
    torch.save({'states':{k:m.state_dict() for k,m in models.items()},'scales':{k:[torch.from_numpy(a) for a in v] for k,v in scales.items()},'metadata':meta},DEST/'models.pt');c.write(DEST/'training.json',meta)
    results={};predictions={}
    with torch.inference_mode():
        for task in c.TASKS:
            va=np.flatnonzero((tasks==task)&(split=='development'));nv=np.flatnonzero(tasks[ref]==task)
            for name in names:
                key=task+'/'+name;m=models[key];xm,xs,ym,ys=scales[key]
                variants={'clean':(state[va] if name=='proprio' else np.concatenate([data[name][va],state[va]],1),target[va],ep[va])}
                for noise in ['noise0.00','noise0.10','noise0.04']:
                    ii=nv[cond[nv]==noise];rr=ref[ii];xx=state[rr] if name=='proprio' else np.concatenate([data['noisy_'+name][ii],state[rr]],1)
                    variants[noise]=(xx,target[rr],ep[rr])
                for variant,(xx,y,ee) in variants.items():
                    pred=m(torch.from_numpy((xx-xm)/xs)).numpy()*ys+ym;k=key+'/'+variant;predictions[k]=pred;results[k]=scores(pred,y,ys,ee)
    paired={}
    for variant in ['clean','noise0.10','noise0.04']:
        ds=[];bs=[]
        for task in c.TASKS:
            aa=np.mean([results[f'{task}/K4_svae{s}/{variant}']['per_episode_groups'] for s in [42,43,44]],0);bb=np.mean([results[f'{task}/L12_svae{s}/{variant}']['per_episode_groups'] for s in [42,43,44]],0);ds.append(aa-bb);bs.append(bb)
        d=np.stack(ds);b=np.stack(bs);ix=np.random.default_rng(20260922).integers(0,5,(2000,3,5));boot=[np.mean([d[t,z[t]].mean() for t in range(3)]) for z in ix]
        paired[variant]={'E_relative_change_percent':float(100*d.mean()/b.mean()),'delta95':np.quantile(boot,[.025,.975]).tolist(),'per_task_relative_percent':[float(100*d[t].mean()/b[t].mean()) for t in range(3)]}
    np.savez(DEST/'predictions.npz',**predictions);summary={'scope':'secondary fixed-MLP capacity check on15previously exposed development episodes','new_confirmation_scored':0,'conditions':results,'primary_paired':paired,'elapsed_seconds':time.monotonic()-begin,'readout_training_seed':42,'readout_seed_variance_not_measured':True};c.write(DEST/'summary.json',summary);print(json.dumps(paired),flush=True)
if __name__=='__main__':main()
