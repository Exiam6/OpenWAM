"""Predeclared task-specific grouped readouts; development data only."""
import argparse,json,os,time
from pathlib import Path
import cv2,h5py,numpy as np,torch
from torch.nn import functional as F
import common as c
from extract_train import load_source
RUN=Path(os.environ['LAYER_RUN']);OUT=RUN/'readout';ALPHAS=[1e-4,.01,1.,100.]

def fit(x,y,alpha):
    xm=x.mean(0);xs=x.std(0).clip(1e-6);ym=y.mean(0);ys=y.std(0).clip(1e-6)
    z=(x-xm)/xs/np.sqrt(x.shape[1]);t=(y-ym)/ys
    if z.shape[1]>z.shape[0]:w=z.T@np.linalg.solve(z@z.T+alpha*np.eye(len(z)),t)
    else:w=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@t)
    return dict(xm=xm,xs=xs,ym=ym,ys=ys,w=w)
def predict(m,x):return ((x-m['xm'])/m['xs']/np.sqrt(x.shape[1]))@m['w']*m['ys']+m['ym']
def select(x,y,ep):
    folds=np.array_split(np.random.default_rng(20260922).permutation(np.unique(ep)),5);scores=[]
    for alpha in ALPHAS:
        errors=[]
        for fold in folds:
            val=np.isin(ep,fold);m=fit(x[~val],y[~val],alpha);err=((predict(m,x[val])-y[val])/m['ys'])**2
            errors.append(float(np.mean([err[ep[val]==e].mean() for e in fold])))
        scores.append({'alpha':alpha,'fold_scores':errors,'score':float(np.mean(errors))})
    chosen=min(scores,key=lambda z:z['score']);return fit(x,y,chosen['alpha']),{'selected':chosen['alpha'],'all_scores':scores,'folds':[a.tolist() for a in folds]}

def checks():
    rng=np.random.default_rng(7);x=rng.normal(size=(40,4));y=x@rng.normal(size=(4,8));m=fit(x,y,0)
    assert np.max(np.abs(predict(m,x)-y))<1e-10
    ep=np.repeat(np.arange(10),4);tr=ep<5;m,a=select(x[tr],y[tr],ep[tr]);y[~tr]=1e9
    mm,b=select(x[tr],y[tr],ep[tr]);assert a==b and np.array_equal(m['w'],mm['w'])
    assert sorted(v for f in a['folds'] for v in f)==list(range(5))
    z=torch.arange(2*48*2*2).reshape(2,48,1,2,2).float();assert pool(z).shape==(2,192)
    return {'passed':True,'checks':['ridge_exact_fixture','holdout_label_mutation_no_fit_change','episode_disjoint_folds','spatial_pool_shape']}

def pool(z):
    z=F.layer_norm(z.permute(0,2,3,4,1).float(),(z.shape[1],),eps=1e-6).permute(0,4,1,2,3)
    return F.adaptive_avg_pool2d(z[:,:,0],(2,2)).flatten(1).cpu().numpy()

@torch.inference_mode()
def visual(x,source):
    folder=RUN/source;pca=torch.load(folder/'pca.pt',weights_only=True);mu=pca['mean'].cuda();std=pca['std'].cuda();basis=pca['basis'].cuda();out={}
    # Only current conditioning frame is supplied to all control readouts.
    out[source+'_raw']=np.concatenate([pool(p.cuda().float()) for p in x.split(8)])
    vs=[]
    for chunk in x.split(8):
        z=(chunk.cuda().float().permute(0,2,3,4,1)-mu)/std
        vs.append(pool((z@basis).permute(0,4,1,2,3)))
    out[source+'_pca']=np.concatenate(vs)
    for seed in [42,43,44]:
        ck=torch.load(folder/f'seed{seed}/final.pt',weights_only=True);m=c.SVAE(**ck['model_config']).cuda();m.load_state_dict(ck['state_dict']);m.eval();vs=[]
        for chunk in x.split(8):
            with torch.autocast('cuda',dtype=torch.bfloat16):z=m.encode_mean(chunk.cuda().float())
            vs.append(pool(z))
        out[f'{source}_svae{seed}']=np.concatenate(vs);del m;torch.cuda.empty_cache()
    return out

def vectors():
    OUT.mkdir(exist_ok=True);assert not (OUT/'vectors.npz').exists()
    cache_rows=json.loads((RUN/'features/manifest.json').read_text())
    for r in cache_rows:
        assert (c.ROOT/r['path']).is_file(),r['path']
        assert (c.ROOT/'data') in (c.ROOT/r['path']).parents,r['path']
    for source in c.SOURCES:assert (RUN/source/'training-complete.json').exists()
    begin=time.monotonic();arrays={};rows=None;states=None;targets=None
    for source in c.SOURCES:
        x,meta,state,target=load_source(source)
        if rows is None:rows=meta;states=state.numpy();targets=target.numpy()
        else:assert rows==meta
        arrays.update(visual(x[:,:,:1],source));del x
    dev=[i for i,r in enumerate(rows) if r['split']=='development'];enc=c.encoder();noise={s:[] for s in c.SOURCES};references=[];conditions=[]
    for i in dev:
        row=rows[i]
        with h5py.File(c.ROOT/row['path'],'r') as f:
            arr=np.array(c.resize_head(cv2.imdecode(np.frombuffer(bytes(f['observation/head_camera/rgb'][row['frame']]),np.uint8),cv2.IMREAD_COLOR)))
        for sigma in [0.,.1,.04]:
            for draw in range(1 if sigma==0 else 3):
                rng=np.random.default_rng(20260922+row['episode_uid']*100000+row['frame']*10+draw)
                from PIL import Image
                im=Image.fromarray(np.clip(arr.astype(np.float32)+rng.normal(0,255*sigma,arr.shape),0,255).astype(np.uint8))
                z=c.encode(enc,enc.preprocess_video([im]*9));references.append(i);conditions.append(f'noise{sigma:.2f}')
                for s in c.SOURCES:noise[s].append(z[s][0,:,:1].cpu().half())
    del enc;torch.cuda.empty_cache()
    for s in c.SOURCES:
        v=visual(torch.stack(noise[s]),s)
        arrays.update({'noisy_'+k:x for k,x in v.items()});del noise[s]
    zero=np.flatnonzero(np.array(conditions)=='noise0.00');refs=np.array(references)[zero]
    parity={key:float(np.max(np.abs(arrays['noisy_'+key][zero]-arrays[key][refs]))) for key in list(arrays) if not key.startswith('noisy_')}
    c.write(OUT/'zero-noise-parity.json',{'max_abs_vector_difference':parity,'matched_backbone_batch_frames':9})
    assert max(parity.values())==0.,'Zero noise pipeline differs; stop before scoring and inspect'
    arrays.update(state=states,target=targets,noisy_reference=np.array(references),noisy_condition=np.array(conditions))
    np.savez(OUT/'vectors.npz',**arrays);c.write(OUT/'manifest.json',rows)
    c.write(OUT/'vectors-complete.json',{'elapsed_seconds':time.monotonic()-begin,'clean_frames':len(rows),'development_frames':len(dev),'noise_copies':len(references),'fresh_confirmation_used':False,'checks':checks()})

def scores(pred,y,scale,ep):
    e=((pred-y)/scale)**2
    unique=np.unique(ep);per=np.array([[e[ep==i,:6].mean(),e[ep==i,6:].mean()] for i in unique])
    return {'episode_ids':unique.tolist(),'per_episode_groups':per.tolist(),'translation':float(per[:,0].mean()),'gripper':float(per[:,1].mean()),'E':float(per.mean())}

def probes():
    assert not (OUT/'development-summary.json').exists()
    data=np.load(OUT/'vectors.npz');rows=json.loads((OUT/'manifest.json').read_text());target=data['target'].astype(float);state=data['state'].astype(float)
    task=np.array([r['task'] for r in rows]);ep=np.array([r['episode'] for r in rows]);split=np.array([r['split'] for r in rows]);ref=data['noisy_reference'];cond=data['noisy_condition']
    names=['proprio']+[s+'_'+m for s in c.SOURCES for m in ['raw','pca','svae42','svae43','svae44']];results={};predictions={};selections={};coeff={}
    # All readouts/CV decisions are fitted on training episodes before any score.
    models={}
    for task_name in c.TASKS:
        tr=(task==task_name)&(split=='train')
        for name in names:
            x=state if name=='proprio' else np.concatenate([data[name],state],1).astype(float)
            mm=[];dec=[]
            for sl in [slice(0,6),slice(6,8)]:
                m,d=select(x[tr],target[tr,sl],ep[tr]);mm.append(m);dec.append(d)
            models[task_name,name]=mm;selections[task_name+'/'+name]=dec
            for gi,m in enumerate(mm):
                for k,v in m.items():coeff[f'{task_name}/{name}/{gi}/{k}']=v
    c.write(OUT/'selection.json',selections);np.savez(OUT/'coefficients.npz',**coeff)
    for task_name in c.TASKS:
        va=np.flatnonzero((task==task_name)&(split=='development'));nv=np.flatnonzero(task[ref]==task_name)
        for name in names:
            m=models[task_name,name];scale=np.concatenate([a['ys'] for a in m]);x=state[va] if name=='proprio' else np.concatenate([data[name][va],state[va]],1)
            variants={'clean':(x,target[va],ep[va])}
            for noise in ['noise0.00','noise0.10','noise0.04']:
                idx=nv[cond[nv]==noise];rr=ref[idx];xx=state[rr] if name=='proprio' else np.concatenate([data['noisy_'+name][idx],state[rr]],1)
                variants[noise]=(xx,target[rr],ep[rr])
            if name!='proprio':
                v=data[name][va];vv=v.reshape(len(v),-1,4).copy()
                for j,i in enumerate(va):vv[j]=vv[j][:,np.random.default_rng(777+i).permutation(4)]
                variants['spatial_shuffle']=(np.concatenate([vv.reshape(v.shape),state[va]],1),target[va],ep[va])
                eps=np.unique(ep[va]);other={e:eps[(j+1)%len(eps)] for j,e in enumerate(eps)};ids=np.empty(len(va),int)
                for e,n in other.items():ids[ep[va]==e]=np.flatnonzero(ep[va]==n)
                variants['visual_mismatch']=(np.concatenate([v[ids],state[va]],1),target[va],ep[va])
            for variant,(xx,y,ee) in variants.items():
                pred=np.concatenate([predict(a,xx) for a in m],1);key='/'.join([task_name,name,variant]);predictions[key]=pred;results[key]=scores(pred,y,scale,ee)
    paired={}
    for variant in ['clean','noise0.10','noise0.04']:
        per=[];base=[]
        for task_name in c.TASKS:
            aa=np.mean([results[f'{task_name}/K4_svae{s}/{variant}']['per_episode_groups'] for s in [42,43,44]],0)
            bb=np.mean([results[f'{task_name}/L12_svae{s}/{variant}']['per_episode_groups'] for s in [42,43,44]],0)
            per.append(aa-bb);base.append(bb)
        d=np.stack(per);b=np.stack(base);rng=np.random.default_rng(20260922);ix=rng.integers(0,5,(2000,3,5));boot=[]
        for z in ix:boot.append(np.mean([d[t,z[t]].mean() for t in range(3)]))
        paired[variant]={'E_relative_change_percent':float(100*d.mean()/b.mean()),'E_delta':float(d.mean()),'delta95':np.quantile(boot,[.025,.975]).tolist(),'per_task_relative_percent':[float(100*d[t].mean()/b[t].mean()) for t in range(3)],'task_order':c.TASKS,'independent_episodes':15,'seed_average_not_independent_replication':True}
    summary={'scope':'previously exposed development episodes only; not confirmation or policy benefit','conditions':results,'primary_paired':paired,'source_seeds_all_retained':[42,43,44],'development_gate_direction_only':paired['noise0.10']['E_relative_change_percent']<0 and paired['clean']['E_relative_change_percent']<=5,'fresh_episodes_scored':0,'mlp_capacity_check':'pending; fixed separate stage','readout':'task-specific, separate translation and gripper alpha, training-episode5fold CV'}
    np.savez(OUT/'predictions.npz',**predictions);c.write(OUT/'development-summary.json',summary);print(json.dumps({'primary':paired,'development_gate_direction_only':summary['development_gate_direction_only']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','vectors','probes']);a=p.parse_args();torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
    if a.mode=='check':print(json.dumps(checks()))
    elif a.mode=='vectors':vectors()
    else:probes()
