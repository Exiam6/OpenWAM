"""Cached prediction arithmetic parity, no new episodes, fits, or encoder runs."""
import json,os,sys
from pathlib import Path
R=Path(os.environ['WAM_ROOT']);sys.path.insert(0,str(R/'scripts'))
import numpy as np
import rae_noise_control as g
from readout_v2 import scores
O=R/'results/confirmation-20260923';d=np.load(g.L/'confirmation/vectors.npz');w=np.load(g.B/'wan-evaluation-vectors.npz');rows=json.loads((g.L/'confirmation/sample-manifest.json').read_text());task=np.array([x['task'] for x in rows]);frame=np.array([x['frame'] for x in rows]);ep=np.array([x['seed'] for x in rows]);maxdiff=0.;count=0
for kind,folder in [('goal',g.O),('state',R/'results/rae-state-control-20260923')]:
 coeff=np.load(folder/'coefficients.npz');preds=np.load(folder/'predictions.npz')
 for name in g.NAMES:
  data=w if name in g.WN else d;ref=data['reference'];cond=data['condition']
  for t in g.TASKS:
   models=[{k:coeff[f'{name}/{t}/{j}/{k}'] for k in ['xm','xs','ym','ys','w']} for j in [0,1]] if kind=='state' else [{k:coeff[f'{name}/{t}/{k}'] for k in ['xm','xs','ym','ys','w']}]
   for con in g.CONDS:
    ii=np.flatnonzero((task[ref]==t)&(cond==con)&((frame[ref]==0) if kind=='goal' else True));rr=ref[ii];x=data[name][ii].astype(float)
    if kind=='state':x=np.concatenate([x,d['state'][rr].astype(float)],1)
    p=np.concatenate([g.predict(m,x) for m in models],1);q=preds[f'{name}/{t}/{con}'];delta=float(abs(p-q).max());maxdiff=max(maxdiff,delta);assert np.allclose(p,q,rtol=1e-12,atol=1e-12);count+=1
pred=np.array([[1.]*6+[2.]*2,[1.]*6+[2.]*2,[3.]*6+[4.]*2]);metric=scores(pred,np.zeros_like(pred),np.ones(8),np.array([1,1,2]));assert metric['translation']==5 and metric['gripper']==10 and metric['E']==7.5
report={'passed':True,'cached_prediction_comparisons':count,'max_abs':maxdiff,'scene_equal_weight_metric_fixture':True,'new_model_fits':0,'encoder_runs':0,'confirmation_data_used':False};(O/'cpu-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
