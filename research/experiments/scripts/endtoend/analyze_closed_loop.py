"""Complete-matrix, paired hierarchical analysis; no model calls or refitting."""
import json
from pathlib import Path
import numpy as np
E=Path('/data02/zifanz4/openwam-experiments/endtoend-20260923')
METHODS=['svae','pca','wan'];SEEDS=[42,43,44];TASKS=['adjust_bottle','handover_block','place_object_basket'];CONDITIONS=['clean','gaussian_sigma_0.04','gaussian_sigma_0.10','head_camera_yaw_5deg']

def paired_ci(difference,draws=20000,seed=42420):
 """d[training_seed,task,condition,scene], all comparisons remain paired."""
 assert difference.shape==(3,3,4,50)
 rng=np.random.default_rng(seed);values=[]
 for start in range(0,draws,250):
  n=min(250,draws-start);si=rng.integers(0,3,(n,3));ep=rng.integers(0,50,(n,1,3,1,50))
  x=difference[si];x=np.take_along_axis(x,ep,axis=-1)
  values.append(x.mean(axis=(1,2,4))*100)
 v=np.concatenate(values);point=difference.mean(axis=(0,1,3))*100
 return {'point_pp':point.tolist(),'condition_95pp':np.quantile(v,[.025,.975],axis=0).T.tolist(),'perturbed_mean_pp':float(point[1:].mean()),'perturbed_mean_95pp':np.quantile(v[:,1:].mean(1),[.025,.975]).tolist(),'draws':draws,'bootstrap_seed':seed,'resampling':'paired training seed + scene within task, same scenes across all training seeds/conditions'}

def main():
 y=np.empty((3,3,3,4,50));sceneids={}
 for ti,task in enumerate(TASKS):
  m=json.loads((E/'scenes'/task/'manifest.json').read_text());assert m['complete'] and len(m['accepted'])==50
  sceneids[task]=[x['seed'] for x in m['accepted']];assert len(set(sceneids[task]))==50
 for mi,method in enumerate(METHODS):
  for si,seed in enumerate(SEEDS):
   train=E/'training'/f'{method}-seed{seed}';assert json.loads((train/'exit.json').read_text())['returncode']==0
   last=json.loads((train/'metrics-rank0.jsonl').read_text().splitlines()[-1]);assert last['global_step']==12000 and last['opt_step']==6000
   for ti,task in enumerate(TASKS):
    for ci,condition in enumerate(CONDITIONS):
     src=E/'evaluation'/f'{method}-seed{seed}'/task/condition/'summary.json';r=json.loads(src.read_text());assert r['complete'] and r['episodes']==50
     assert [x['scene_seed'] for x in r['records']]==sceneids[task]
     assert all(x['status']=='complete' and isinstance(x['success'],bool) for x in r['records'])
     y[mi,si,ti,ci]=[x['success'] for x in r['records']]
 # Compare every training sample ID, augmentation strength and prompt in all paired runs.
 import hashlib
 streams={}
 for seed in SEEDS:
  hashes={m:hashlib.sha256((E/'training'/f'{m}-seed{seed}'/'batch-stream.jsonl').read_bytes()).hexdigest() for m in METHODS}
  assert len(set(hashes.values()))==1,('training streams differ',seed,hashes)
  streams[seed]=hashes
 contrasts={'pca_minus_svae':paired_ci(y[1]-y[0]),'pca_minus_wan':paired_ci(y[1]-y[2])}
 primary=contrasts['pca_minus_svae'];ni=primary['condition_95pp'][0][0]>-5;robust=primary['perturbed_mean_95pp'][0]>0
 rates={m:{c:float(y[mi,:,:,ci].mean()) for ci,c in enumerate(CONDITIONS)} for mi,m in enumerate(METHODS)}
 byseedtask={f'{m}/seed{seed}/{task}/{c}':int(y[mi,si,ti,ci].sum()) for mi,m in enumerate(METHODS) for si,seed in enumerate(SEEDS) for ti,task in enumerate(TASKS) for ci,c in enumerate(CONDITIONS)}
 result={'complete':True,'rollouts':int(y.size),'primary_clean_noninferiority_pass':bool(ni),'primary_perturbed_improvement_pass':bool(robust),'joint_primary_pass':bool(ni and robust),'condition_success_rates':rates,'contrasts':contrasts,'per_seed_task_condition_successes_out_of50':byseedtask,'training_stream_hashes':streams,'scale':'native295M trainable-parameter controlled study; not5B','three_training_seed_limitation':True}
 out=E/'analysis';out.mkdir(exist_ok=True);assert not (out/'closed-loop.json').exists()
 np.savez_compressed(out/'success-matrix.npz',success=y,methods=METHODS,seeds=SEEDS,tasks=TASKS,conditions=CONDITIONS,**{t:sceneids[t] for t in TASKS})
 (out/'closed-loop.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
