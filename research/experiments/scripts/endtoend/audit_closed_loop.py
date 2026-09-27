"""Independent raw-outcome reconstruction and paired-bootstrap arithmetic audit."""
import hashlib,json
from pathlib import Path
import numpy as np
E=Path('/data02/zifanz4/openwam-experiments/endtoend-20260923');methods=['svae','pca','wan'];seeds=[42,43,44];tasks=['adjust_bottle','handover_block','place_object_basket'];conditions=['clean','gaussian_sigma_0.04','gaussian_sigma_0.10','head_camera_yaw_5deg']
report=json.loads((E/'analysis/closed-loop.json').read_text());saved=np.load(E/'analysis/success-matrix.npz');y=np.zeros((3,3,3,4,50),int);hashes={}
for ti,task in enumerate(tasks):
 manifest=json.loads((E/'scenes'/task/'manifest.json').read_text())
 for mi,method in enumerate(methods):
  for si,seed in enumerate(seeds):
   for ci,condition in enumerate(conditions):
    for ei,scene in enumerate(manifest['accepted']):
     base=E/'evaluation'/f'{method}-seed{seed}'/task/condition/f'seed-{scene["seed"]}';p=base/'result.json';raw=p.read_bytes();r=json.loads(raw);hashes[str(p)]=hashlib.sha256(raw).hexdigest()
     assert r['status']=='complete' and isinstance(r['success'],bool) and r['training_seed']==seed and r['route']==method and r['task']==task and r['condition']==condition and r['scene_seed']==scene['seed']
     assert r['instruction']==scene['instruction'] and max(r['initial_render_mae'].values())<=1.
     lines=(base/'trace.jsonl').read_text().splitlines();trace_last=json.loads(lines[-1]);assert trace_last['success']==r['success'] and len(lines)==r['steps']
     if not r['success']:assert r['steps']==r['step_limit']
     y[mi,si,ti,ci,ei]=int(r['success'])
assert np.array_equal(y,saved['success'])
residuals=[]
for mi,name in [(0,'pca_minus_svae'),(2,'pca_minus_wan')]:
 d=y[1]-y[mi];rng=np.random.default_rng(42420);estimates=[]
 # Same prespecified draws, independent explicit indexing/aggregation implementation.
 for block in range(80):
  seed_indices=rng.integers(0,3,(250,3));scene_indices=rng.integers(0,50,(250,1,3,1,50))
  for draw in range(250):
   per_task=[]
   for task in range(3):
    per_seed=[]
    for seed in seed_indices[draw]:per_seed.append(d[seed,task][:,scene_indices[draw,0,task,0]].mean(axis=1))
    per_task.append(np.mean(per_seed,axis=0))
   estimates.append(np.mean(per_task,axis=0)*100)
 v=np.array(estimates);ci=np.percentile(v,[2.5,97.5],axis=0).T;robust=np.percentile(v[:,1:].mean(axis=1),[2.5,97.5]);point=d.mean(axis=(0,1,3))*100
 target=report['contrasts'][name]
 for actual,key in [(ci,'condition_95pp'),(robust,'perturbed_mean_95pp'),(point,'point_pp')]:
  residual=float(np.max(np.abs(actual-np.asarray(target[key]))));residuals.append(residual);assert residual<1e-10,(name,key,residual)
ni=report['contrasts']['pca_minus_svae']['condition_95pp'][0][0]>-5;robust=report['contrasts']['pca_minus_svae']['perturbed_mean_95pp'][0]>0
assert report['joint_primary_pass']==(ni and robust)
result={'passed':True,'raw_episode_results':len(hashes),'raw_traces_checked':len(hashes),'numerical_max_residual':max(residuals),'audit_scope':'raw outcomes and complete traces; independently indexed paired hierarchical bootstrap; fixed primary intersection','input_manifest_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),'report_sha256':hashlib.sha256((E/'analysis/closed-loop.json').read_bytes()).hexdigest()}
out=E/'analysis/audit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
