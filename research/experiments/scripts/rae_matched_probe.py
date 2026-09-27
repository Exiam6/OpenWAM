"""New Wan-only readouts, descriptive comparison to unchanged cached DINO results."""
import datetime, hashlib, json, os, sys, time, zipfile
from pathlib import Path
import cv2, h5py, numpy as np, torch
from PIL import Image
from torch.nn import functional as F
import rae_interface_preflight as interface
sys.path.insert(0, str(Path(__file__).parent/'layers'))
from readout_v2 import select, predict, scores
R=interface.ROOT; O=R/'results/rae-baseline-20260922'; L=R/'results/layers-20260922'
TASKS=['adjust_bottle','handover_block','place_object_basket']; NAMES=['Wan48_native_scale','Wan48_per_token_layernorm']; CONDITIONS=['clean','noise0.10','noise0.04']
def write(name,obj): interface.write(name,obj)
def image(raw): return Image.fromarray(cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)).resize((320,384),Image.Resampling.BILINEAR)

def main():
    assert not (O/'probe-result.json').exists()
    assert json.loads((O/'result.json').read_text())['passed']
    freeze=json.loads((O/'probe-freeze.json').read_text())
    for name,h in freeze['files'].items(): assert interface.sha(Path(name))==h,name
    torch.set_num_threads(4);torch.manual_seed(20260922);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    start=time.monotonic();weight=R/'assets/wan22-vae-baseline/Wan2.2_VAE.pth';assert interface.sha(weight)==interface.PROTOCOL['asset']['sha256']
    state=torch.load(weight,map_location='cpu',weights_only=True,mmap=True);state=state.get('model_state',state)
    model=interface.WanVideoVAE38();model.load_state_dict({'model.'+k:v for k,v in state.items()},strict=True,assign=True)
    enc=interface.WanVideoVAEEncoder(model.to(device='cuda',dtype=torch.bfloat16)).eval().requires_grad_(False);torch.cuda.reset_peak_memory_stats()
    @torch.inference_mode()
    def vectors(im):
        z=enc.batch_encode(enc.preprocess_video([im])).float();assert tuple(z.shape)==(1,48,1,24,20) and torch.isfinite(z).all()
        ln=F.layer_norm(z.permute(0,2,3,4,1),(48,),eps=1e-6).permute(0,4,1,2,3)
        return [F.adaptive_avg_pool2d(a[:,:,0],(2,2)).flatten(1)[0].cpu().numpy() for a in [z,ln]]
    trainrows=json.loads((O/'train-rows.json').read_text());assert len(trainrows)==1260 and all(x['split']=='train' for x in trainrows)
    train=[[],[]]
    with zipfile.ZipFile(O/'train-images.zip') as z:
        for i,row in enumerate(trainrows):
            raw=z.read(str(i)+'.jpg');assert hashlib.sha256(raw).hexdigest()==row['encoded_head_hashes'][0]
            for dest,v in zip(train,vectors(image(raw))): dest.append(v)
            if i%120==0: print('TRAIN_FEATURES',i,flush=True)
    train=[np.array(x) for x in train];np.savez(O/'wan-training-vectors.npz',**dict(zip(NAMES,train)))
    # Fit every new readout before accessing held-out images or outcome labels.
    lab=np.load(O/'train-labels.npz');state0=lab['state'].astype(float);target0=lab['target'].astype(float)
    tt=np.array([r['task'] for r in trainrows]);ee=np.array([r['episode'] for r in trainrows]);ff=np.array([r['frame'] for r in trainrows]);models={};coefs={};choices={}
    goals=json.loads((L/'initial-goal/training-samples.json').read_text());lookup={(r['task'],r['episode']):r['target_xy'] for r in goals}
    gtarget=np.array([lookup[r['task'],r['episode']] for r in trainrows],dtype=np.float32).astype(float)
    oldcoef=np.load(L/'readout/coefficients.npz');goalconstants=json.loads((L/'initial-goal/constants.json').read_text())
    for ni,name in enumerate(NAMES):
        for task in TASKS:
            idx=tt==task;assert idx.sum()==420 and len(np.unique(ee[idx]))==35
            x=np.concatenate([train[ni],state0],1).astype(float)
            for group,sl in [('translation',slice(0,6)),('gripper',slice(6,8))]:
                key='state/'+name+'/'+task+'/'+group;m,decision=select(x[idx],target0[idx,sl],ee[idx]);models[key]=m;choices[key]=decision
                for k,v in m.items():coefs[key+'/'+k]=v
                gi=0 if group=='translation' else 1;assert np.array_equal(m['ys'],oldcoef[f'{task}/L12_raw/{gi}/ys'])
            idx=idx&(ff==0);assert idx.sum()==35
            key='goal/'+name+'/'+task;m,decision=select(train[ni][idx].astype(float),gtarget[idx],ee[idx]);models[key]=m;choices[key]=decision
            assert np.array_equal(m['ys'],np.array(goalconstants[task]['std']))
            for k,v in m.items():coefs[key+'/'+k]=v
    np.savez(O/'wan-coefficients.npz',**coefs);write('wan-selection.json',choices);write('probe-fit-complete.json',{'at':datetime.datetime.now().astimezone().isoformat(),'ridge_fits':len(models),'heldout_images_accessed':False,'training_episodes':105})
    rows=json.loads((L/'confirmation/sample-manifest.json').read_text());assert len(rows)==720
    manifest=json.loads((L/'fresh/manifest-v2.json').read_text());paths={(x['task'],x['seed']):x['hdf5'] for task in TASKS for x in manifest['accepted'][task]}
    test=[[],[]];refs=[];conds=[]
    for i,row in enumerate(rows):
        with h5py.File(paths[row['task'],row['seed']],'r') as f: raw=bytes(f['observation/head_camera/rgb'][row['frame']])
        assert hashlib.sha256(raw).hexdigest()==row['head_encoded_sha256'];arr=np.array(image(raw));variants=[('clean',Image.fromarray(arr))]
        for sigma in [.1,.04]:
            for draw in range(3):
                rng=np.random.default_rng(20260922+row['seed']*100000+row['frame']*10+draw)
                variants.append((f'noise{sigma:.2f}',Image.fromarray(np.clip(arr.astype(np.float32)+rng.normal(0,255*sigma,arr.shape),0,255).astype(np.uint8))))
        for cond,im in variants:
            refs.append(i);conds.append(cond)
            for dest,v in zip(test,vectors(im)):dest.append(v)
        if i%60==0:
            write('probe-progress.json',{'stage':'extracting_exposed60','frames':i+1,'elapsed_seconds':time.monotonic()-start});print('TEST_FEATURES',i,flush=True)
    peak=torch.cuda.max_memory_allocated()/1e9
    del enc,model;torch.cuda.empty_cache()
    test=[np.array(x) for x in test];refs=np.array(refs);conds=np.array(conds)
    previous=np.load(L/'confirmation/vectors.npz');keep=np.isin(previous['condition'],CONDITIONS)
    assert np.array_equal(refs,previous['reference'][keep]) and np.array_equal(conds,previous['condition'][keep])
    states=previous['state'].astype(float);targets=previous['target'].astype(float)
    np.savez(O/'wan-evaluation-vectors.npz',**dict(zip(NAMES,test)),reference=refs,condition=conds)
    taskids=np.array([r['task'] for r in rows]);ep=np.array([r['seed'] for r in rows]);frame=np.array([r['frame'] for r in rows])
    glabel=json.loads((L/'initial-goal/fresh-labels.json').read_text());gl={(x['task'],x['seed']):x['target_xy'] for x in glabel};results={};predictions={}
    for ni,name in enumerate(NAMES):
        for task in TASKS:
            mm=[models['state/'+name+'/'+task+'/'+g] for g in ['translation','gripper']];scale=np.concatenate([m['ys'] for m in mm]);gm=models['goal/'+name+'/'+task]
            for condition in CONDITIONS:
                ix=np.flatnonzero((taskids[refs]==task)&(conds==condition));rr=refs[ix];x=np.concatenate([test[ni][ix],states[rr]],1)
                pred=np.concatenate([predict(m,x) for m in mm],1);key='state/'+name+'/'+task+'/'+condition;results[key]=scores(pred,targets[rr],scale,ep[rr]);predictions[key]=pred
                ix=np.flatnonzero((taskids[refs]==task)&(frame[refs]==0)&(conds==condition));rr=refs[ix];pred=predict(gm,test[ni][ix]);y=np.array([gl[task,int(ep[i])] for i in rr]);err=((pred-y)/gm['ys'])**2;unique=np.unique(ep[rr]);assert len(unique)==20
                per=np.array([err[ep[rr]==e].mean(0) for e in unique]);key='goal/'+name+'/'+task+'/'+condition;results[key]={'episode_ids':unique.tolist(),'per_episode_xy_nmse':per.tolist(),'E':float(per.mean())};predictions[key]=pred
    oldstate=json.loads((L/'confirmation/summary.json').read_text())['conditions']['ridge'];oldgoal=json.loads((L/'initial-goal/fresh-summary.json').read_text())['conditions'];bootstrap=np.random.default_rng(20260922).integers(0,20,(2000,3,20));matrix={};contrasts={}
    for endpoint in ['state','goal']:
        matrix[endpoint]={};contrasts[endpoint]={};field='per_episode_groups' if endpoint=='state' else 'per_episode_xy_nmse'
        for condition in CONDITIONS:
            arrays={}
            for name in NAMES:
                arrays[name]=np.array([results[f'{endpoint}/{name}/{t}/{condition}'][field] for t in TASKS])
            for label,source_names in [('DINO_raw768',['L12_raw']),('DINO_PCA48',['L12_pca']),('DINO_SVAE48_seed_mean',[f'L12_svae{s}' for s in [42,43,44]])]:
                collected=[]
                for task in TASKS:
                    ds=[oldstate[f'{task}/{n}/{condition}'] if endpoint=='state' else oldgoal[f'ridge/{task}/{n}/{condition}'] for n in source_names]
                    expected=results[f'{endpoint}/{NAMES[0]}/{task}/{condition}']['episode_ids'];assert all(x['episode_ids']==expected for x in ds)
                    collected.append(np.mean([x[field] for x in ds],axis=0))
                arrays[label]=np.array(collected)
            matrix[endpoint][condition]={n:{'E':float(a.mean()),'per_task_E':a.mean(axis=(1,2)).tolist()} for n,a in arrays.items()}
            contrasts[endpoint][condition]={}
            for base in NAMES:
                contrasts[endpoint][condition][base]={};b=arrays[base]
                for name in ['DINO_raw768','DINO_PCA48','DINO_SVAE48_seed_mean']:
                    d=arrays[name]-b;ds=np.stack([d[t,bootstrap[:,t]] for t in range(3)],axis=1).mean(axis=(1,2,3));bs=np.stack([b[t,bootstrap[:,t]] for t in range(3)],axis=1).mean(axis=(1,2,3));assert np.all(bs>0)
                    contrasts[endpoint][condition][base][name]={'DINO_minus_Wan_relative_percent':float(100*d.mean()/b.mean()),'descriptive_relative_percent95':np.quantile(100*ds/bs,[.025,.975]).tolist(),'per_task_relative_percent':(100*d.mean(axis=(1,2))/b.mean(axis=(1,2))).tolist()}
    np.savez(O/'wan-predictions.npz',**predictions)
    result={'completed_at':datetime.datetime.now().astimezone().isoformat(),'scope':'Exploratory Wan-only follow-up on exposed old60; cached DINO predictions reused, not rerun. Not a full RAE or policy evaluation.','passed_integrity_gates':True,'training_episodes':105,'evaluation_episodes':60,'new_ridge_fits':18,'new_Wan_training_frames':1260,'new_Wan_evaluation_frames_with_noise':len(refs),'task_order':TASKS,'conditions':results,'matrix':matrix,'contrasts':contrasts,'elapsed_seconds':time.monotonic()-start,'peak_allocated_GB':peak,'limits':['different encoders/objectives, no isolated causal pretraining effect','raw3072D versus compressed192D readout capacity differs; PCA48 is dimension control','readout target is short-horizon end-effector/gripper or first-close XY; not measured contact or controller commands','old60 already exposed; intervals descriptive, no independent confirmation','all Wan norms/endpoints/tasks retained; no threshold rescue or favorable normalization selection','DINO path has no pixel decoder; no full RAE reconstruction replication'],'protocol_sha256':interface.sha(O/'matched-probe-protocol.json')}
    write('probe-result.json',result);print(json.dumps({'matrix':matrix,'seconds':result['elapsed_seconds']}),flush=True)
if __name__=='__main__': main()
