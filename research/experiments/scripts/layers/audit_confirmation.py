"""Independent fresh-prediction/label/bootstrap audit; no fitting or simulation."""
import hashlib,json,os
from pathlib import Path
import h5py,numpy as np,torch
R=Path(os.environ['LAYER_RUN']);out=R/'confirmation';tasks=['adjust_bottle','handover_block','place_object_basket'];torch.set_num_threads(6)
rows=json.loads((out/'sample-manifest.json').read_text());d=np.load(out/'vectors.npz');predictions=np.load(out/'predictions.npz');summary=json.loads((out/'summary.json').read_text());coef=np.load(R/'readout/coefficients.npz');mlp=torch.load(R/'readout/mlp/models.pt',map_location='cpu',weights_only=False)
manifest=json.loads((R/'fresh/manifest-v2.json').read_text());assert summary['episodes']==60 and len(rows)==720 and summary['refit_models']==0
ref=d['reference'];condition=d['condition'];task=np.array([r['task'] for r in rows]);episode=np.array([r['seed'] for r in rows]);target=d['target'];state=d['state'];assert len(ref)==5760
for name,mult in [('clean',1),('left_transfer',1),('noise0.10',3),('noise0.04',3)]:assert np.array_equal(np.bincount(ref[condition==name],minlength=720),np.full(720,mult))
# Re-read true rawframe+4 labels and image hashes, independent of extraction arrays.
for t in tasks:
    for rec in manifest['accepted'][t]:
        rr=np.flatnonzero((task==t)&(episode==rec['seed']));assert len(rr)==12
        with h5py.File(rec['hdf5'],'r') as f:
            wanted=np.unique(np.linspace(0,len(f['endpose/left_endpose'])-33,12,dtype=int));assert wanted.tolist()==[rows[i]['frame'] for i in rr]
            for i in rr:
                a=rows[i]['frame'];b=rows[i]['target_frame'];assert b==a+4
                lp=f['endpose/left_endpose'][a];rp=f['endpose/right_endpose'][a]
                x=np.r_[lp,rp,f['endpose/left_gripper'][a],f['endpose/right_gripper'][a]].astype(np.float32)
                y=np.r_[f['endpose/left_endpose'][b,:3]-lp[:3],f['endpose/right_endpose'][b,:3]-rp[:3],f['endpose/left_gripper'][b],f['endpose/right_gripper'][b]].astype(np.float32)
                assert np.array_equal(x,state[i]) and np.array_equal(y,target[i])
                assert hashlib.sha256(bytes(f['observation/head_camera/rgb'][a])).hexdigest()==rows[i]['head_encoded_sha256']
computed={};counts={};model_errors={};rng=np.random.default_rng(20260922);indices=rng.integers(0,20,(2000,3,20));primary={};interaction={}
for method in ['ridge','mlp']:
    conds=summary['conditions'][method];assert {k[len(method)+1:] for k in predictions.files if k.startswith(method+'/')}==set(conds)
    for key,saved in conds.items():
        t,name,variant=key.split('/');ii=np.flatnonzero((task[ref]==t)&(condition==variant));rr=ref[ii];p=predictions[method+'/'+key];y=target[rr];e=episode[rr]
        x=state[rr] if name=='proprio' else np.concatenate([d[name][ii],state[rr]],axis=1)
        if method=='ridge':
            scale=np.concatenate([coef[f'{t}/{name}/{g}/ys'] for g in [0,1]]);y=y.astype(float);pp=[]
            for group in [0,1]:
                pars={k:coef[f'{t}/{name}/{group}/{k}'] for k in ['xm','xs','ym','ys','w']};z=(x-pars['xm'])/pars['xs']/np.sqrt(x.shape[1]);pp.append(z@pars['w']*pars['ys']+pars['ym'])
            calc=np.concatenate(pp,axis=1)
        else:
            xm,xs,ym,scale=[v.numpy() for v in mlp['scales'][t+'/'+name]]
            model=torch.nn.Sequential(torch.nn.Linear(len(xm),128),torch.nn.ReLU(),torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,8));model.load_state_dict(mlp['states'][t+'/'+name]);model.eval()
            with torch.inference_mode():calc=model(torch.from_numpy((x-xm)/xs)).numpy()*scale+ym
        err=float(np.max(np.abs(p-calc)));model_errors[method+'/'+key]=err;assert np.allclose(p,calc,rtol=1e-6,atol=1e-7),(method,key,err)
        normalized=((p-y)/scale)**2;per=np.array([[normalized[e==j,:6].mean(),normalized[e==j,6:].mean()] for j in np.unique(e)]);assert per.shape==(20,2)
        assert np.unique(e).tolist()==saved['episode_ids'];assert np.allclose(per,saved['per_episode_groups'],rtol=1e-7,atol=1e-10)
        assert np.isclose(per.mean(),saved['E'],rtol=1e-7,atol=1e-10);computed[method+'/'+key]=per.astype(np.float64)
    counts[method]=len(conds);primary[method]={};interaction[method]={}
    for variant in ['clean','noise0.10','noise0.04','left_transfer']:
        a=np.stack([np.mean([computed[f'{method}/{t}/K4_svae{s}/{variant}'] for s in [42,43,44]],axis=0) for t in tasks]);b=np.stack([np.mean([computed[f'{method}/{t}/L12_svae{s}/{variant}'] for s in [42,43,44]],axis=0) for t in tasks]);delta=a-b
        dd=np.stack([delta[t,indices[:,t]] for t in range(3)],axis=1).mean(axis=(1,2,3));bb=np.stack([b[t,indices[:,t]] for t in range(3)],axis=1).mean(axis=(1,2,3));ci=np.quantile(100*dd/bb,[.025,.975]);pct=100*delta.mean()/b.mean();saved=summary['primary_paired'][method][variant]
        assert np.isclose(pct,saved['E_relative_change_percent'],rtol=1e-8);assert np.allclose(ci,saved['relative_percent95'],rtol=1e-8,atol=1e-10)
        primary[method][variant]={'relative_percent':float(pct),'relative_percent95':ci.tolist()}
    for comp in ['raw','pca','svae']:
        interaction[method][comp]={}
        for variant in ['clean','noise0.10']:
            values={}
            for source in ['L12','K4']:
                kk=[f'{method}/{t}/{source}_{comp+str(s) if comp=="svae" else comp}/{variant}' for t in tasks for s in ([42,43,44] if comp=='svae' else [0])];values[source]=float(np.mean([computed[k] for k in kk]))
            interaction[method][comp][variant]={'L12_E':values['L12'],'K4_E':values['K4'],'relative_change_percent':100*(values['K4']/values['L12']-1)}
noise=summary['primary_paired']['ridge']['noise0.10'];clean=summary['primary_paired']['ridge']['clean'];success=noise['E_relative_change_percent']<=-10 and noise['relative_percent95'][1]<0 and clean['relative_percent95'][1]<=3 and sum(v<0 for v in noise['per_task_relative_percent'])>=2 and np.max(noise['per_task_group_relative_percent'])<=5
assert bool(success)==summary['predeclared_primary_success']
frozen=json.loads((R/'confirmation-freeze.json').read_text())
for name,expected in frozen['sha256'].items():
    h=hashlib.sha256()
    with (R/name).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==expected,name
result={'passed':True,'labels_and_images_rechecked':720,'fresh_unique_episodes':60,'prediction_conditions_recomputed':counts,'max_prediction_recompute_abs':max(model_errors.values()),'primary_bootstrap_recomputed':primary,'source_compression_interaction_descriptive':interaction,'predeclared_primary_success':bool(success),'frozen_assets_unchanged':True,'refits':0}
(out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
