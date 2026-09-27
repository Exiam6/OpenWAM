"""Recompute saved prediction metrics independently; no model inference."""
from pathlib import Path
import json,numpy as np
R=Path('/home/zifanz4/openwam-experiments/studies/layers-20260922/development')
rows=json.loads((R/'manifest.json').read_text());d=np.load(R/'audit-labels.npz');coef=np.load(R/'coefficients.npz')
tasks=np.array([r['task'] for r in rows]);eps=np.array([r['episode'] for r in rows]);train=np.array([r['split']=='train' for r in rows]);ref=d['noisy_reference'];noise=d['noisy_condition']
verified={};mechanism={}
for method,path in [('ridge',R),('mlp',R/'mlp')]:
    summary=json.loads((path/('development-summary.json' if method=='ridge' else 'summary.json')).read_text());pred=np.load(path/'predictions.npz');assert set(pred.files)==set(summary['conditions']);computed={}
    for key,p in pred.items():
        task,name,condition=key.split('/');tr=(tasks==task)&train;va=np.flatnonzero((tasks==task)&~train)
        if condition.startswith('noise'):ii=np.flatnonzero((tasks[ref]==task)&(noise==condition));inds=ref[ii]
        else:inds=va
        y=d['target'][inds];e=eps[inds]
        if method=='ridge':scale=np.concatenate([coef[f'{task}/{name}/{g}/ys'] for g in [0,1]]);y=y.astype(float)
        else:scale=d['target'][tr].std(0).clip(1e-6)
        errors=((p-y)/scale)**2;per=np.array([[errors[e==i,:6].mean(),errors[e==i,6:].mean()] for i in np.unique(e)]);saved=summary['conditions'][key]
        assert np.allclose(per,saved['per_episode_groups'],rtol=1e-7,atol=1e-10),key
        assert np.isclose(per.mean(),saved['E'],rtol=1e-7,atol=1e-10),key
        computed[key]=per.astype(np.float64)
    for variant in ['clean','noise0.10','noise0.04']:
        pairs=[];base=[]
        for task in ['adjust_bottle','handover_block','place_object_basket']:
            a=np.mean([computed[f'{task}/K4_svae{s}/{variant}'] for s in [42,43,44]],axis=0);b=np.mean([computed[f'{task}/L12_svae{s}/{variant}'] for s in [42,43,44]],axis=0);pairs.append(a-b);base.append(b)
        pct=100*np.mean(pairs)/np.mean(base);assert np.isclose(pct,summary['primary_paired'][variant]['E_relative_change_percent'],rtol=1e-7)
    mechanism[method]={}
    for comp in ['raw','pca','svae']:
        mechanism[method][comp]={}
        for variant in ['clean','noise0.10']:
            vals={}
            for source in ['L12','K4']:
                keys=[f'{task}/{source}_{comp+str(s) if comp=="svae" else comp}/{variant}' for task in ['adjust_bottle','handover_block','place_object_basket'] for s in ([42,43,44] if comp=='svae' else [0])]
                vals[source]=float(np.mean([computed[k] for k in keys]))
            mechanism[method][comp][variant]={'L12_E':vals['L12'],'K4_E':vals['K4'],'relative_change_percent':100*(vals['K4']/vals['L12']-1)}
    verified[method]=len(computed)
parity=json.loads((R/'zero-noise-parity.json').read_text());assert max(parity['max_abs_vector_difference'].values())==0
out={'passed':True,'prediction_conditions_recomputed':verified,'source_compression_interaction_descriptive':mechanism,'all15source_compressor_seed_zero_noise_paths_exact':True,'fresh_confirmation_scored':0,'limitations':['development reuse','raw-vs48 absolute comparisons differ in readout dimension; within-raw source contrast is dimension matched','no irreversible-information-loss or policy-benefit claim']}
(R/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
