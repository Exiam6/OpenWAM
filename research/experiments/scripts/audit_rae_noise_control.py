"""Saved-artifact prediction, normalization, ridge optimality and fold audit."""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path('/data02/zifanz4/openwam-experiments/results/rae-noise-control-20260922');L=R.parent/'layers-20260922';B=R.parent/'rae-baseline-20260922'
def main():
 d=json.loads((R/'result.json').read_text());coef=np.load(R/'coefficients.npz');pred=np.load(R/'predictions.npz');train=np.load(R/'training-vectors.npz');rows=json.loads((R/'clean-rows.json').read_text());tr=train['reference'];tasks=np.array([x['task'] for x in rows]);eps=np.array([x['episode'] for x in rows]);gt={(x['task'],x['episode']):x['target_xy'] for x in json.loads((L/'initial-goal/training-samples.json').read_text())};y=np.array([gt[x['task'],x['episode']] for x in rows],dtype=np.float32).astype(float);selection=json.loads((R/'selection.json').read_text());maxgrad=0.;maxscale=0.
 for key,choice in selection.items():
  name,task=key.split('/');ii=tasks[tr]==task;xx=train[name][ii].astype(float);yy=y[tr[ii]];ep=eps[tr[ii]];xm,xs,ym,ys,w=[coef[key+'/'+k] for k in ['xm','xs','ym','ys','w']]
  for a,b in [(xm,xx.mean(0)),(xs,xx.std(0).clip(1e-6)),(ym,yy.mean(0)),(ys,yy.std(0).clip(1e-6))]:maxscale=max(maxscale,float(abs(a-b).max()));assert np.allclose(a,b,rtol=1e-12,atol=1e-12)
  z=(xx-xm)/xs/np.sqrt(xx.shape[1]);target=(yy-ym)/ys;alpha=choice['effective_alpha'];assert alpha==choice['selected_base_alpha']*7
  grad=z.T@(z@w-target)+alpha*w;den=max(1.,float(np.linalg.norm(z.T@target)));res=float(np.linalg.norm(grad)/den);assert res<1e-8;maxgrad=max(maxgrad,res)
  assert len(xx)==245 and len(np.unique(ep))==35 and all((ep==e).sum()==7 for e in np.unique(ep))
  folds=choice['folds'];assert len(folds)==5 and sorted(e for f in folds for e in f)==sorted(set(ep.tolist()));assert choice['selected_base_alpha']==min(choice['all_scores'],key=lambda x:x['score'])['alpha']
 evd=np.load(L/'confirmation/vectors.npz');evw=np.load(B/'wan-evaluation-vectors.npz');erows=json.loads((L/'confirmation/sample-manifest.json').read_text());et=np.array([x['task'] for x in erows]);ee=np.array([x['seed'] for x in erows]);ff=np.array([x['frame'] for x in erows]);gl={(x['task'],x['seed']):x['target_xy'] for x in json.loads((L/'initial-goal/fresh-labels.json').read_text())};const=json.loads((L/'initial-goal/constants.json').read_text());maxpred=0.;maxmetric=0.
 for key,saved in d['conditions'].items():
  name,task,cond=key.split('/');v=evw if name.startswith('Wan') else evd;ref=v['reference'];ix=np.flatnonzero((et[ref]==task)&(ff[ref]==0)&(v['condition']==cond));rr=ref[ix];x=v[name][ix].astype(float);k=name+'/'+task;xm,xs,ym,ys,w=[coef[k+'/'+n] for n in ['xm','xs','ym','ys','w']];slope=w/(xs[:,None]*np.sqrt(x.shape[1]));pp=(x@slope)*ys+(ym-(xm@slope)*ys);delta=float(abs(pp-pred[key]).max());maxpred=max(maxpred,delta);assert np.allclose(pp,pred[key],atol=1e-11,rtol=1e-10)
  target=np.array([gl[task,int(e)] for e in ee[rr]]);error=((pp-target)/np.array(const[task]['std']))**2;ids=sorted(set(ee[rr].tolist()));per=np.array([error[ee[rr]==e].mean(0) for e in ids]);assert ids==saved['episode_ids'] and len(ids)==20;assert len(rr)==(20 if cond=='clean' else 60);delta=float(abs(per-np.array(saved['per_episode_xy_nmse'])).max());maxmetric=max(maxmetric,delta);assert np.allclose(per,saved['per_episode_xy_nmse'],rtol=1e-9,atol=1e-11)
 for path,h in json.loads((R/'freeze.json').read_text())['files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
 assert not json.loads((R/'fit-complete.json').read_text())['eval_vectors_or_labels_accessed']
 audit={'passed':True,'new_fits':len(selection),'prediction_conditions':len(d['conditions']),'training_scale_max_abs':maxscale,'normalized_ridge_optimality_residual':maxgrad,'expanded_affine_prediction_max_abs':maxpred,'episode_metric_max_abs':maxmetric,'folds_disjoint_by_original_episode':True,'seven_variants_per_training_episode':True,'frozen_sources_and_inputs_unchanged':True,'no_model_inference_or_refits':True,'limitations':'A saved-artifact audit, not new independent samples, full RAE, or policy validation.'};(R/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))
if __name__=='__main__':main()
