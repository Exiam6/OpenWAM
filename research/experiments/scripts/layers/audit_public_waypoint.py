"""NumPy-only numerical audit of the secondary waypoint report, not raw provenance."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent);r=p.parse_args().directory
s=json.loads((r/'initial-goal-summary.json').read_text());labels=json.loads((r/'initial-goal-labels.json').read_text());constants=json.loads((r/'initial-goal-training-scales.json').read_text());pred=np.load(r/'initial-goal-predictions.npz',allow_pickle=False);tasks=['adjust_bottle','handover_block','place_object_basket'];computed={}
for key,p in pred.items():
 method,task,name,variant=key.split('/');sample=[a for a in labels if a['task']==task];repeat=1 if variant=='clean' else 3;y=np.repeat(np.array([a['target_xy'] for a in sample]),repeat,axis=0);ep=np.repeat(np.array([a['seed'] for a in sample]),repeat);scale=np.array(constants[task]['std']);assert p.shape==y.shape;err=((p-y)/scale)**2;per=np.array([err[ep==e].mean(0) for e in np.unique(ep)]);saved=s['conditions'][key];assert np.allclose(per,saved['per_episode_xy_nmse'],rtol=1e-10,atol=1e-12);assert np.isclose(per.mean(),saved['E'],rtol=1e-10);computed[key]=per
ix=np.random.default_rng(20260922).integers(0,20,(2000,3,20))
for method in ['ridge','mlp']:
 for variant in ['clean','noise0.10','noise0.04']:
  a=np.stack([np.mean([computed[f'{method}/{t}/K4_svae{seed}/{variant}'] for seed in [42,43,44]],axis=0) for t in tasks]);b=np.stack([np.mean([computed[f'{method}/{t}/L12_svae{seed}/{variant}'] for seed in [42,43,44]],axis=0) for t in tasks]);d=a-b;dd=np.stack([d[t,ix[:,t]] for t in range(3)],axis=1).mean(axis=(1,2,3));bb=np.stack([b[t,ix[:,t]] for t in range(3)],axis=1).mean(axis=(1,2,3));saved=s['secondary_K4_SVAE_vs_L12_SVAE'][method][variant];assert np.allclose(np.quantile(100*dd/bb,[.025,.975]),saved['relative_percent95'],rtol=1e-9,atol=1e-10);assert np.isclose(100*d.mean()/b.mean(),saved['E_relative_change_percent'],rtol=1e-10)
print(json.dumps({'passed':True,'prediction_conditions_recomputed':len(computed),'paired_intervals_recomputed':6,'episodes':60,'scope':'secondary endpoint; same cohort as primary, not a new independent replication'},indent=2))
