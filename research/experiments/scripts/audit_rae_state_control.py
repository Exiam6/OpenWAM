"""Independent arithmetic and provenance checks of saved state-control artifacts."""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path('/data02/zifanz4/openwam-experiments/results/rae-state-control-20260923');L=R.parent/'layers-20260922';B=R.parent/'rae-baseline-20260922';G=R.parent/'rae-noise-control-20260922'
def main():
 d=json.loads((R/'result.json').read_text());coef=np.load(R/'coefficients.npz');pred=np.load(R/'predictions.npz');v=np.load(R/'training-vectors.npz');lab=np.load(B/'train-labels.npz');rows=json.loads((B/'train-rows.json').read_text());ref=v['reference'];tt=np.array([x['task'] for x in rows]);ep=np.array([x['episode'] for x in rows]);choices=json.loads((R/'selection.json').read_text());scale_err=0.;grad_err=0.
 assert len(rows)==1260 and len(ref)==8820 and all((ref==i).sum()==7 for i in range(1260))
 for key,ch in choices.items():
  name,task,group=key.split('/');ii=tt[ref]==task;xx=np.concatenate([v[name][ii],lab['state'][ref[ii]]],1).astype(float);sl=slice(0,6) if group=='0' else slice(6,8);yy=lab['target'][ref[ii],sl].astype(float);xm,xs,ym,ys,w=[coef[key+'/'+n] for n in ['xm','xs','ym','ys','w']]
  for a,b in [(xm,xx.mean(0)),(xs,xx.std(0).clip(1e-6)),(ym,yy.mean(0)),(ys,yy.std(0).clip(1e-6))]:scale_err=max(scale_err,float(abs(a-b).max()));assert np.allclose(a,b,rtol=1e-12,atol=1e-12)
  z=(xx-xm)/xs/np.sqrt(xx.shape[1]);y=(yy-ym)/ys;alpha=ch['effective_alpha'];assert alpha==ch['selected_base_alpha']*7
  grad=z.T@(z@w-y)+alpha*w;relative=float(np.linalg.norm(grad)/max(1.,np.linalg.norm(z.T@y)));assert relative<1e-8,(key,relative);grad_err=max(grad_err,relative)
  e=ep[ref[ii]];assert len(xx)==2940 and len(np.unique(e))==35 and all((e==k).sum()==84 for k in np.unique(e));folds=ch['folds'];assert len(folds)==5 and sorted(k for f in folds for k in f)==sorted(np.unique(e).tolist());assert ch['selected_base_alpha']==min(ch['all_scores'],key=lambda a:a['score'])['alpha']
 gv=np.load(G/'training-vectors.npz');gr=json.loads((G/'clean-rows.json').read_text());lookup={(x['task'],x['episode']):i for i,x in enumerate(gr)};reuse=0
 for i,row in enumerate(rows):
  if row['frame']!=0:continue
  gi=lookup[row['task'],row['episode']];dst=1260+6*i+np.arange(6);src=105+6*gi+np.arange(6)
  for name in [k for k in v.files if k not in ['reference','condition']]:assert np.array_equal(v[name][dst],gv[name][src]),(name,i)
  reuse+=6
 assert reuse==630
 evd=np.load(L/'confirmation/vectors.npz');evw=np.load(B/'wan-evaluation-vectors.npz');er=json.loads((L/'confirmation/sample-manifest.json').read_text());et=np.array([x['task'] for x in er]);ee=np.array([x['seed'] for x in er]);oldcoef=np.load(L/'readout/coefficients.npz');maxpred=0.;maxmetric=0.
 for key,saved in d['conditions'].items():
  name,task,cond=key.split('/');data=evw if name.startswith('Wan') else evd;rr=data['reference'];ii=np.flatnonzero((et[rr]==task)&(data['condition']==cond));ri=rr[ii];x=np.concatenate([data[name][ii],evd['state'][ri]],1).astype(float);pieces=[]
  for group in [0,1]:
   k=name+'/'+task+'/'+str(group);xm,xs,ym,ys,w=[coef[k+'/'+n] for n in ['xm','xs','ym','ys','w']];slopes=w/(xs[:,None]*np.sqrt(len(xm)));pieces.append((x@slopes)*ys+(ym-(xm@slopes)*ys))
  pp=np.column_stack(pieces);maxpred=max(maxpred,float(abs(pp-pred[key]).max()));assert np.allclose(pp,pred[key],atol=1e-10,rtol=1e-9)
  scale=np.concatenate([oldcoef[f'{task}/L12_raw/{group}/ys'] for group in [0,1]]);error=((pp-evd['target'][ri].astype(float))/scale)**2;ids=sorted(set(ee[ri].tolist()));per=np.array([[error[ee[ri]==e,:6].mean(),error[ee[ri]==e,6:].mean()] for e in ids]);assert ids==saved['episode_ids'] and len(ids)==20;assert len(ri)==(240 if cond=='clean' else 720);maxmetric=max(maxmetric,float(abs(per-np.array(saved['per_episode_groups'])).max()));assert np.allclose(per,saved['per_episode_groups'],atol=1e-10,rtol=1e-9)
 for p,h in json.loads((R/'freeze.json').read_text())['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 assert not json.loads((R/'fit-complete.json').read_text())['eval_vectors_or_labels_accessed']
 out={'passed':True,'group_readout_fits':len(choices),'prediction_conditions':len(d['conditions']),'max_training_scale_difference':scale_err,'normalized_normal_equation_residual':grad_err,'expanded_affine_prediction_max_abs':maxpred,'episode_metric_max_abs':maxmetric,'reused_noisy_initial_vectors_verified':reuse,'episode_disjoint_folds':True,'alpha_multiplier_exactly7_not84':True,'frozen_sources_and_references_unchanged':True,'no_model_inference_or_refits':True};(R/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
