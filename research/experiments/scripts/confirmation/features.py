"""Frozen feature extraction and once-only independent confirmation; never fits."""
import datetime,gc,hashlib,json,os,sys,time,zipfile
from pathlib import Path
from worker import R,O,guards,now,sha,write
sys.path.insert(0,str(R/'scripts'))
import cv2,h5py,numpy as np,torch
from PIL import Image
import rae_noise_control as g
from readout_v2 import scores
TASKS=g.TASKS;NAMES=g.NAMES;CONDS=g.CONDS

def models():
    weights=torch.load(R/'assets/wan22-vae-baseline/Wan2.2_VAE.pth',map_location='cpu',weights_only=True,mmap=True)
    weights=weights.get('model_state',weights);model=g.interface.WanVideoVAE38()
    model.load_state_dict({'model.'+k:v for k,v in weights.items()},strict=True,assign=True)
    wan=g.interface.WanVideoVAEEncoder(model.to('cuda',dtype=torch.bfloat16)).eval().requires_grad_(False)
    return wan,g.c.encoder()

def parity(wan,dino):
    B=g.B;G=g.O;rows=json.loads((G/'clean-rows.json').read_text());br=json.loads((B/'train-rows.json').read_text());bi=[i for i,x in enumerate(br) if x['frame']==0]
    assert [br[i] for i in bi]==rows and len(rows)==105
    clean=dict(np.load(G/'clean-dino-goal-vectors.npz'));wv=np.load(B/'wan-training-vectors.npz');clean.update({k:wv[k][bi] for k in g.WN})
    tasks=np.array([x['task'] for x in rows]);ids=[int(np.flatnonzero(tasks==t)[0]) for t in TASKS];raw=[];arrays={k:[] for k in g.WN}
    with zipfile.ZipFile(B/'train-images.zip') as z:
        for i in ids:
            encoded=z.read(str(bi[i])+'.jpg');assert hashlib.sha256(encoded).hexdigest()==rows[i]['encoded_head_hashes'][0]
            im=g.c.resize_head(cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR))
            for k,x in zip(g.WN,g.wan_vectors(wan,im)): arrays[k].append(x)
            raw.append(g.c.encode(dino,dino.preprocess_video([im]*9))['L12'][0,:,:1].cpu().half())
    arrays.update(g.visual(torch.stack(raw),'L12'));result={}
    for k in NAMES:
        a=np.array(arrays[k]);b=clean[k][ids];std=np.array([clean[k][tasks==rows[i]['task']].astype(float).std(0).clip(1e-6) for i in ids]);d=(a-b)/std
        result[k]={'standardized_RMS':np.sqrt((d*d).mean(1)).tolist(),'standardized_max':abs(d).max(1).tolist(),'max_abs':float(abs(a-b).max())}
    passed=all(max(x['standardized_RMS'])<=.05 and max(x['standardized_max'])<=.5 for x in result.values())
    write(O/'features/hardware-parity.json',{'passed':passed,'training_only':True,'fixed_indices':ids,'RMS_limit':.05,'coordinate_limit':.5,'routes':result,'time':now().isoformat()})
    assert passed,'Hardware parity failed; no tolerance change'

def extract(record,wan,dino):
    guards();task=record['task'];seed=record['seed'];dest=O/'features'/task/f'seed-{seed}';dest.mkdir(parents=True,exist_ok=False)
    path=Path(record['hdf5']);assert sha(path)==record['hdf5_sha256'];rows=[];states=[];targets=[];refs=[];conds=[];draws=[];raw=[];arrays={k:[] for k in g.WN}
    with h5py.File(path,'r') as f:
        n=len(f['endpose/left_endpose']);assert n>=44
        frame_indices=np.unique(np.linspace(0,n-33,12,dtype=int));assert len(frame_indices)==12
        candidates=[]
        for ai,arm in enumerate(['left','right']):
            grip=f[f'endpose/{arm}_gripper'][:];cross=np.flatnonzero((grip[:-1]>.5)&(grip[1:]<=.5))+1
            if len(cross): candidates.append((int(cross[0]),ai,arm))
        assert candidates,'Missing first-close label; retain cohort without scoring or replacement'
        event,ai,arm=min(candidates);goal=np.asarray(f[f'endpose/{arm}_endpose'][event,:2],dtype=np.float32)
        label={'task':task,'seed':seed,'event_frame':event,'arm':arm,'target_xy':goal.tolist(),'simultaneous_crossing':len(candidates)>1 and candidates[0][0]==candidates[1][0]}
        for st in frame_indices:
            guards();st=int(st);i=len(rows);lp=f['endpose/left_endpose'][st];rp=f['endpose/right_endpose'][st]
            states.append(np.concatenate([lp,rp,[f['endpose/left_gripper'][st],f['endpose/right_gripper'][st]]]))
            targets.append(np.concatenate([f['endpose/left_endpose'][st+4,:3]-lp[:3],f['endpose/right_endpose'][st+4,:3]-rp[:3],[f['endpose/left_gripper'][st+4],f['endpose/right_gripper'][st+4]]]))
            encoded=bytes(f['observation/head_camera/rgb'][st]);arr=np.array(g.c.resize_head(cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR)))
            variants=[('clean',-1,Image.fromarray(arr))]
            for sigma in [.1,.04]:
                for draw in range(3):
                    rng=np.random.default_rng(20260922+seed*100000+st*10+draw)
                    variants.append((f'noise{sigma:.2f}',draw,Image.fromarray(np.clip(arr.astype(np.float32)+rng.normal(0,255*sigma,arr.shape),0,255).astype(np.uint8))))
            for condition,draw,im in variants:
                for k,x in zip(g.WN,g.wan_vectors(wan,im)): arrays[k].append(x)
                raw.append(g.c.encode(dino,dino.preprocess_video([im]*9))['L12'][0,:,:1].cpu().half())
                refs.append(i);conds.append(condition);draws.append(draw)
            rows.append({'task':task,'seed':seed,'frame':st,'target_frame':st+4,'episode_sha256':record['hdf5_sha256'],'head_encoded_sha256':hashlib.sha256(encoded).hexdigest()})
    arrays={k:np.asarray(v) for k,v in arrays.items()};arrays.update(g.visual(torch.stack(raw),'L12'))
    assert all(np.isfinite(x).all() for x in arrays.values())
    for k in NAMES: assert arrays[k].shape==(84,3072 if k=='L12_raw' else 192)
    np.savez(dest/'vectors.npz',**arrays,state=np.asarray(states,dtype=np.float32),target=np.asarray(targets,dtype=np.float32),reference=np.array(refs),condition=np.array(conds),draw=np.array(draws))
    write(dest/'rows.json',rows);write(dest/'goal.json',label)
    write(dest/'complete.json',{'time':now().isoformat(),'sha256':{p.name:sha(p) for p in [dest/'vectors.npz',dest/'rows.json',dest/'goal.json']},'readouts_scored':False})
    del raw;gc.collect()

def matrix_and_contrasts(result,kind):
    groups={g.WN[0]:[g.WN[0]],g.WN[1]:[g.WN[1]],'DINO_raw768':['L12_raw'],'DINO_PCA48':['L12_pca'],'DINO_SVAE48_seed_mean':['L12_svae42','L12_svae43','L12_svae44']}
    if kind=='state':groups['proprio']=['proprio']
    ix=np.random.default_rng(20260922).integers(0,20,(2000,3,20));matrix={};contrasts={}
    def contrast(a,b):
        d=a-b;ds=np.stack([d[t,ix[:,t]] for t in range(3)],1).mean((1,2,3));bs=np.stack([b[t,ix[:,t]] for t in range(3)],1).mean((1,2,3))
        return {'relative_change_percent':float(100*d.mean()/b.mean()),'paired95':np.quantile(100*ds/bs,[.025,.975]).tolist(),'absolute_difference':float(d.mean()),'absolute95':np.quantile(ds,[.025,.975]).tolist(),'per_task_relative_percent':(100*d.mean((1,2))/b.mean((1,2))).tolist()}
    for con in CONDS:
        a={}
        field='per_episode_groups' if kind=='state' else 'per_episode_xy_nmse'
        for label,names in groups.items():
            a[label]=np.array([np.mean([result[n+'/'+t+'/'+con][field] for n in names],0) for t in TASKS])
        matrix[con]={n:{'E':float(x.mean()),'per_task_E':x.mean((1,2)).tolist(),'per_task_components':x.mean(1).tolist()} for n,x in a.items()}
        bases=g.WN+(['proprio'] if kind=='state' else [])
        contrasts[con]={base:{n:contrast(a[n],a[base]) for n in ['DINO_raw768','DINO_PCA48','DINO_SVAE48_seed_mean']} for base in bases}
        if kind=='state':
            for base in bases:
                for n in ['DINO_raw768','DINO_PCA48','DINO_SVAE48_seed_mean']:
                    contrasts[con][base][n]['translation']=contrast(a[n][:,:,:1],a[base][:,:,:1])
                    contrasts[con][base][n]['gripper']=contrast(a[n][:,:,1:],a[base][:,:,1:])
    return matrix,contrasts

def score(records):
    guards();assert len(records)==60 and not (O/'result.json').exists()
    old=json.loads((O/'identity-reference.json').read_text());old_hashes=set(old['hashes']);seen_ep=set();seen_initial=set();arrays={k:[] for k in NAMES};rows=[];labels={};ss=[];ys=[];refs=[];conds=[]
    for rec in records:
        task,seed=rec['task'],rec['seed'];dest=O/'features'/task/f'seed-{seed}';complete=json.loads((dest/'complete.json').read_text())
        for name,h in complete['sha256'].items():assert sha(dest/name)==h
        rr=json.loads((dest/'rows.json').read_text());assert len(rr)==12 and rr[0]['frame']==0
        assert rec['hdf5_sha256'] not in seen_ep|old_hashes;seen_ep.add(rec['hdf5_sha256'])
        assert rr[0]['head_encoded_sha256'] not in seen_initial|old_hashes;seen_initial.add(rr[0]['head_encoded_sha256'])
        assert all(x['head_encoded_sha256'] not in old_hashes for x in rr)
        prov=json.loads((O/'fresh'/task/f'seed-{seed}/runtime-provenance.json').read_text());assert any(x['need_plan'] and all(y=='CuroboPlanner' for y in x['planners'].values()) for x in prov)
        v=np.load(dest/'vectors.npz');refs.extend((v['reference']+len(rows)).tolist());conds.extend(v['condition'].tolist());ss.extend(v['state']);ys.extend(v['target']);rows.extend(rr)
        for k in NAMES:arrays[k].append(v[k])
        labels[task,seed]=json.loads((dest/'goal.json').read_text())['target_xy']
    arrays={k:np.concatenate(x) for k,x in arrays.items()};state=np.asarray(ss,dtype=float);target=np.asarray(ys,dtype=float);ref=np.array(refs);cond=np.array(conds);task=np.array([x['task'] for x in rows]);ep=np.array([x['seed'] for x in rows]);frame=np.array([x['frame'] for x in rows])
    assert len(rows)==720 and len(ref)==5040
    write(O/'identity-audit.json',{'passed':True,'episodes':60,'frames':720,'encoded_hash_overlap':0,'prior_reference_hashes':len(old_hashes),'time':now().isoformat()})
    cf=np.load(g.O/'coefficients.npz');cs=np.load(R/'results/rae-state-control-20260923/coefficients.npz');cp=np.load(g.L/'readout/coefficients.npz');constants=json.loads((g.L/'initial-goal/constants.json').read_text());result={'goal':{},'state':{}};preds={}
    for kind in result:
        for name in NAMES+(['proprio'] if kind=='state' else []):
            for t in TASKS:
                if kind=='goal': models=[{k:cf[f'{name}/{t}/{k}'] for k in ['xm','xs','ym','ys','w']}];scale=np.array(constants[t]['std'])
                else:
                    coef=cp if name=='proprio' else cs;prefix=f'{t}/proprio' if name=='proprio' else f'{name}/{t}'
                    models=[{k:coef[f'{prefix}/{j}/{k}'] for k in ['xm','xs','ym','ys','w']} for j in [0,1]];scale=np.concatenate([cp[f'{t}/L12_raw/{j}/ys'] for j in [0,1]])
                    assert np.allclose(scale,np.concatenate([m['ys'] for m in models]),rtol=1e-12,atol=1e-12)
                for con in CONDS:
                    ii=np.flatnonzero((task[ref]==t)&(cond==con)&((frame[ref]==0) if kind=='goal' else True));rr=ref[ii]
                    x=state[rr] if name=='proprio' else arrays[name][ii].astype(float)
                    if kind=='state' and name!='proprio':x=np.concatenate([x,state[rr]],1)
                    pred=np.concatenate([g.predict(m,x) for m in models],1);assert np.isfinite(pred).all();key=name+'/'+t+'/'+con;preds[kind+'/'+key]=pred
                    if kind=='state':result[kind][key]=scores(pred,target[rr],scale,ep[rr])
                    else:
                        y=np.array([labels[t,int(e)] for e in ep[rr]]);err=((pred-y)/scale)**2;eps=np.unique(ep[rr]);assert len(eps)==20;per=np.array([err[ep[rr]==e].mean(0) for e in eps]);result[kind][key]={'E':float(per.mean()),'episode_ids':eps.tolist(),'per_episode_xy_nmse':per.tolist()}
    matrix={};paired={}
    for kind in result:matrix[kind],paired[kind]=matrix_and_contrasts(result[kind],kind)
    primary={k:paired[k]['noise0.10']['Wan48_native_scale']['DINO_PCA48'] for k in result}
    passed=all(x['absolute95'][1]<0 for x in primary.values())
    np.savez(O/'predictions.npz',**preds);write(O/'sample-manifest.json',rows)
    write(O/'result.json',{'completed_at':now().isoformat(),'scope':'new independent within-task offline readout confirmation','episodes':60,'state_frames':720,'feature_rows':5040,'readout_refits':0,'conditions':result,'matrix':matrix,'paired':paired,'co_primary':primary,'joint_primary_passed':bool(passed),'task_order':TASKS,'limits':['not fullRAE: no pixel decoder training','not policy improvement or true physical contact','known noise strengths and fixed tasks','rawDINO3072 versus192pooled dimensions','all3SVAEseeds averaged within scenes','pretraining, architecture and objectives not causally separated','hardware parity approximate, not bit identity']})
    print('CONFIRMATION_COMPLETE',json.dumps({'joint_primary_passed':bool(passed),'co_primary':primary}),flush=True)

def main():
    guards();assert not (O/'features/started.json').exists(),'No automatic feature rerun'
    for path,h in json.loads((O/'freeze.json').read_text())['files'].items():assert sha(path)==h,path
    write(O/'features/started.json',{'time':now().isoformat(),'gpu':os.environ['CUDA_VISIBLE_DEVICES']})
    torch.set_num_threads(4);torch.manual_seed(20260922);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    wan,dino=models();parity(wan,dino);seen=set();start=time.monotonic()
    while True:
        guards();complete=True;records=[]
        for t in TASKS:
            folder=O/'fresh'/t;manifest=folder/'manifest.json';progress=folder/'progress.json';exitfile=folder/'exit.json'
            if exitfile.exists():assert json.loads(exitfile.read_text())['exit_code']==0, f'{t} collector failed; no scoring'
            if manifest.exists():
                m=json.loads(manifest.read_text());assert m['complete'];accepted=m['accepted']
            else:
                complete=False;accepted=json.loads(progress.read_text())['accepted'] if progress.exists() else []
            records.extend(accepted)
            for rec in accepted:
                key=(t,rec['seed'])
                if key not in seen:
                    extract(rec,wan,dino);seen.add(key)
                    write(O/'features/progress.json',{'stage':'extracting_without_scoring','episodes':len(seen),'elapsed_seconds':time.monotonic()-start,'time':now().isoformat()})
                    print('EXTRACTED',t,rec['seed'],len(seen),flush=True)
        if complete:
            assert len(records)==60 and len(seen)==60
            # Wait for successful parent exits as well as complete manifests.
            if all((O/'fresh'/t/'exit.json').exists() for t in TASKS):break
        cutoff=datetime.datetime.fromisoformat(json.loads((O/'window.json').read_text())['collection_deadline'])
        assert now()<cutoff+datetime.timedelta(seconds=60) or complete,'Incomplete cohort at fixed cutoff; no scoring'
        time.sleep(20)
    del wan,dino;gc.collect();torch.cuda.empty_cache()
    for path,h in json.loads((O/'freeze.json').read_text())['files'].items():assert sha(path)==h,path
    score(records)

if __name__=='__main__':main()
