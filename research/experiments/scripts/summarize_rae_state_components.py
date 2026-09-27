"""Saved-artifact, all-component summary; no training, inference or selection."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
R=Path('/data02/zifanz4/openwam-experiments/results/rae-state-control-20260923')
L=R.parent/'layers-20260922'
d=json.loads((R/'result.json').read_text()); oldpath=L/'confirmation/summary.json';old=json.loads(oldpath.read_text())['conditions']['ridge'];decl=json.loads((R/'proprio-reference-declaration.json').read_text())
assert hashlib.sha256(oldpath.read_bytes()).hexdigest()==decl['source_sha256']
groups={'Wan48_native_scale':['Wan48_native_scale'],'Wan48_per_token_layernorm':['Wan48_per_token_layernorm'],'DINO_raw768':['L12_raw'],'DINO_PCA48':['L12_pca'],'DINO_SVAE48_seed_mean':['L12_svae42','L12_svae43','L12_svae44']};tasks=d['task_order'];draw=np.random.default_rng(20260922).integers(0,20,(2000,3,20))
def comparison(a,b):
 bs=np.stack([b[t,draw[:,t]] for t in range(3)],1).mean((1,2));ds=np.stack([(a-b)[t,draw[:,t]] for t in range(3)],1).mean((1,2));ci=np.percentile(100*ds/bs,[2.5,97.5],axis=0).T
 return {'E_change_percent':float(100*(a.mean()/b.mean()-1)),'per_task_group_change_percent':(100*(a.mean(1)/b.mean(1)-1)).tolist(),'group_change_percent':(100*(a.mean((0,1))/b.mean((0,1))-1)).tolist(),'descriptive_group_paired_95_percent':ci.tolist()}
out={'computed_at':datetime.datetime.now().astimezone().isoformat(),'scope':'All saved short-state components; exposed60, descriptive intervals; proprio reference declared before results; group intervals added after results without inferential promotion','task_order':tasks,'group_order':['translation','gripper'],'reference_source_sha256':decl['source_sha256'],'new_fits':0,'new_model_inference':0,'conditions':{}}
for cond in ['clean','noise0.04','noise0.10']:
 arrays={}
 for name,members in groups.items():
  arrays[name]=np.array([[d['conditions'][f'{m}/{task}/{cond}']['per_episode_groups'] for task in tasks] for m in members]).mean(0)
  for task in tasks:
   assert d['conditions'][f'{members[0]}/{task}/{cond}']['episode_ids']==old[f'{task}/proprio/{cond}']['episode_ids']
 p=np.array([old[f'{task}/proprio/{cond}']['per_episode_groups'] for task in tasks]);assert np.allclose(p,np.array([old[f'{task}/proprio/clean']['per_episode_groups'] for task in tasks]),atol=1e-14,rtol=0)  # saved reduction roundoff <=4.44e-16; raw scores retained
 out['conditions'][cond]={'proprio_only_E':float(p.mean()),'proprio_only_task_groups':p.mean(1).tolist(),'proprio_only_groups':p.mean((0,1)).tolist(),'routes':{}}
 for name,a in arrays.items():
  out['conditions'][cond]['routes'][name]={'E':float(a.mean()),'task_groups':a.mean(1).tolist(),'groups':a.mean((0,1)).tolist(),'versus_proprio_only':comparison(a,p),'versus_Wan_native':comparison(a,arrays['Wan48_native_scale']),'versus_Wan_LN':comparison(a,arrays['Wan48_per_token_layernorm'])}
(R/'component-summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out['conditions']['noise0.10'],indent=2))
