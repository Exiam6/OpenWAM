"""Audit saved new predictions and episode metrics without model inference/refits."""
from pathlib import Path
import hashlib,json
import numpy as np
R=Path('/data02/zifanz4/openwam-experiments/results/rae-baseline-20260922');L=R.parent/'layers-20260922'
def main():
 result=json.loads((R/'probe-result.json').read_text());coef=np.load(R/'wan-coefficients.npz');pred=np.load(R/'wan-predictions.npz');test=np.load(R/'wan-evaluation-vectors.npz');old=np.load(L/'confirmation/vectors.npz')
 train=np.load(R/'wan-training-vectors.npz');trainlab=np.load(R/'train-labels.npz');trainrows=json.loads((R/'train-rows.json').read_text());rows=json.loads((L/'confirmation/sample-manifest.json').read_text());refs=test['reference'];conds=test['condition'];ts=np.array([x['task'] for x in rows]);ep=np.array([x['seed'] for x in rows]);ff=np.array([x['frame'] for x in rows]);tts=np.array([x['task'] for x in trainrows]);tee=np.array([x['episode'] for x in trainrows]);tff=np.array([x['frame'] for x in trainrows]);g={ (x['task'],x['seed']):x['target_xy'] for x in json.loads((L/'initial-goal/fresh-labels.json').read_text())};residual=0.;metric_residual=0.;conditions=0
 for key,saved in result['conditions'].items():
  endpoint,name,task,condition=key.split('/');mask=(ts[refs]==task)&(conds==condition)
  if endpoint=='goal':mask &= ff[refs]==0
  ix=np.flatnonzero(mask);rr=refs[ix];x=test[name][ix].astype(float);ee=ep[rr]
  if endpoint=='state':
   x=np.column_stack((x,old['state'][rr].astype(float)));groups=['translation','gripper'];y=old['target'][rr].astype(float);prefix='state/'+name+'/'+task+'/'
  else:groups=[''];y=np.array([g[task,int(v)] for v in ee]);prefix='goal/'+name+'/'+task
  pieces=[];scales=[]
  for group in groups:
   k=prefix+group;xm,xs,ym,ys,w=[coef[k+'/'+n] for n in ['xm','xs','ym','ys','w']]
   # Expanded affine form, separate from the production prediction expression.
   slopes=w/(xs[:,None]*np.sqrt(x.shape[1]));offset=ym-(xm@slopes)*ys
   pieces.append((x@slopes)*ys+offset);scales.extend(ys)
  pp=np.column_stack(pieces);residual=max(residual,float(np.max(np.abs(pp-pred[key]))));assert np.allclose(pp,pred[key],rtol=1e-10,atol=1e-11),key
  error=((pp-y)/np.array(scales))**2;per=[]
  for e in sorted(set(ee.tolist())):
   selected=error[ee==e]
   per.append([selected[:,:6].mean(),selected[:,6:].mean()] if endpoint=='state' else selected.mean(0))
  per=np.array(per);field='per_episode_groups' if endpoint=='state' else 'per_episode_xy_nmse';metric_residual=max(metric_residual,float(np.max(np.abs(per-np.array(saved[field])))));assert np.allclose(per,saved[field],rtol=1e-9,atol=1e-11),key;assert abs(per.mean()-saved['E'])<1e-9
  assert len(saved['episode_ids'])==20 and saved['episode_ids']==sorted(set(ee.tolist()));conditions+=1
 choices=json.loads((R/'wan-selection.json').read_text())
 for key,v in choices.items():
  task=key.split('/')[2];folds=v['folds'];assert sorted(x for f in folds for x in f)==sorted(set(tee[tts==task].tolist()));assert len({x for f in folds for x in f})==35
  assert v['selected']==min(v['all_scores'],key=lambda q:q['score'])['alpha']
 fit=json.loads((R/'probe-fit-complete.json').read_text());assert not fit['heldout_images_accessed'] and fit['ridge_fits']==18
 freeze=json.loads((R/'probe-freeze.json').read_text())
 for f,h in freeze['files'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
 for condition in ['clean','noise0.10','noise0.04']:
  assert int((conds==condition).sum())==(720 if condition=='clean' else 2160)
 out={'passed':True,'saved_prediction_conditions':conditions,'readout_fits':len(choices),'max_abs_expanded_affine_prediction_residual':residual,'max_abs_episode_metric_residual':metric_residual,'train_episode_fold_coverage':True,'frozen_sources_and_references_unchanged':True,'no_model_inference_or_refitting':True,'limitations':'Checks saved artifacts and arithmetic, not independence of exposed old60 or fullRAE/policy benefit.'}
 (R/'probe-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
