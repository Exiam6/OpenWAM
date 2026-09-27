"""Recompute published OpenWAM fresh-readout metrics and paired intervals.

Requires only NumPy and the report's predictions/labels/scales/summary files.
This verifies arithmetic; it does not prove raw-image, model or split provenance.
"""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args();r=a.directory
summary=json.loads((r/'confirmation-summary.json').read_text());pred=np.load(r/'confirmation-predictions.npz',allow_pickle=False);data=np.load(r/'confirmation-labels-scales.npz',allow_pickle=False);task=data['task'];episode=data['episode'];ref=data['reference'];condition=data['condition'];computed={};counts={};tasks=summary['primary_paired']['ridge']['clean']['task_order']
for method in ['ridge','mlp']:
    expected=summary['conditions'][method];assert {k[len(method)+1:] for k in pred.files if k.startswith(method+'/')}==set(expected)
    for key,saved in expected.items():
        t,name,variant=key.split('/');ii=np.flatnonzero((task[ref]==t)&(condition==variant));rr=ref[ii];y=data['target'][rr];y=y.astype(float) if method=='ridge' else y;scale=data[method+'/'+t+'/scale'];estimate=pred[method+'/'+key]
        assert estimate.shape==y.shape and np.isfinite(estimate).all();err=((estimate-y)/scale)**2;ids=np.unique(episode[rr]);values=np.array([[err[episode[rr]==i,:6].mean(),err[episode[rr]==i,6:].mean()] for i in ids])
        assert ids.tolist()==saved['episode_ids'];assert np.allclose(values,saved['per_episode_groups'],rtol=1e-7,atol=1e-10);assert np.isclose(values.mean(),saved['E'],rtol=1e-7,atol=1e-10);computed[method+'/'+key]=values.astype(float)
    counts[method]=len(expected)
    n=20;indices=np.random.default_rng(20260922).integers(0,n,(2000,len(tasks),n))
    for variant,saved in summary['primary_paired'][method].items():
        left=np.stack([np.mean([computed[f'{method}/{t}/K4_svae{s}/{variant}'] for s in [42,43,44]],axis=0) for t in tasks]);right=np.stack([np.mean([computed[f'{method}/{t}/L12_svae{s}/{variant}'] for s in [42,43,44]],axis=0) for t in tasks]);diff=left-right;assert diff.shape==(3,20,2)
        dd=np.stack([diff[t,indices[:,t]] for t in range(3)],axis=1).mean(axis=(1,2,3));bb=np.stack([right[t,indices[:,t]] for t in range(3)],axis=1).mean(axis=(1,2,3));ci=np.quantile(100*dd/bb,[.025,.975]);change=100*diff.mean()/right.mean()
        assert np.allclose(ci,saved['relative_percent95'],rtol=1e-7,atol=1e-10);assert np.isclose(change,saved['E_relative_change_percent'],rtol=1e-7)
print(json.dumps({'passed':True,'prediction_conditions_recomputed':counts,'primary_intervals_recomputed':8,'independent_episodes':60,'scope':'numerical audit only; raw provenance requires original artifacts'},indent=2))
