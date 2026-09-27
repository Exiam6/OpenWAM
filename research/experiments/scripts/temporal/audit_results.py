"""Independent raw-rollout completeness and paired success audit; no model fitting."""
import json,hashlib
from pathlib import Path
import numpy as np
N=Path('/data02/zifanz4/openwam-experiments/temporal-20260926');S=Path('/home/zifanz4/openwam-experiments/studies/temporal-20260926')
def main():
 plan=json.loads((S/'protocol.json').read_text());tasks=plan['evaluation']['tasks'];conditions=plan['evaluation']['conditions'];outcomes=np.zeros((2,3,3,3,20));latencies=[[],[]];raw=0
 for ai,arm in enumerate(['mean','learned']):
  for si,seed in enumerate([42,43,44]):
   root=N/'evaluation'/f'{arm}-seed{seed}';assert json.loads((root/'exit.json').read_text())['returncode']==0
   assert json.loads((root/'complete.json').read_text())['episodes']==180
   for ti,task in enumerate(tasks):
    expected=json.loads((N/'scenes'/task/'manifest.json').read_text())['accepted'];assert len(expected)==20
    for ci,condition in enumerate(conditions):
     group=root/task/condition;summary=json.loads((group/'summary.json').read_text());assert summary['complete'] and summary['episodes']==20
     records=[]
     for ei,scene in enumerate(expected):
      folder=group/f'seed-{scene["seed"]}';row=json.loads((folder/'result.json').read_text());assert row['status']=='complete' and isinstance(row['success'],bool)
      assert (row['scene_seed'],row['training_seed'],row['route'],row['task'],row['condition'])==(scene['seed'],seed,arm,task,condition)
      trace=[json.loads(x) for x in (folder/'trace.jsonl').read_text().splitlines()];assert trace and row['steps']==len(trace)
      assert trace[-1]['success']==row['success'] and all(np.isfinite(x['action']).all() and len(x['action'])==20 for x in trace)
      assert max(row['initial_render_mae'].values())<=1.
      latencies[ai].extend(x['latency_ms'] for x in trace if x['latency_ms'] is not None);outcomes[ai,si,ti,ci,ei]=int(row['success']);raw+=1;records.append(row)
     assert records==summary['records'] and sum(x['success'] for x in records)==summary['successes']
 assert raw==1080
 rng=np.random.default_rng(426);result={'raw_episodes':raw,'complete':True,'conditions':{},'scope':'295M three-task matched continuation; shared sealed60scene subset; not official benchmark or fullRAE'}
 for ci,condition in enumerate(conditions):
  delta=outcomes[1,:,:,ci,:]-outcomes[0,:,:,ci,:];boot=[]
  for _ in range(10000):
   seeds=rng.integers(0,3,3);values=[delta[seeds[:,None],ti,rng.integers(0,20,(1,20))].mean() for ti in range(3)];boot.append(np.mean(values))
  interval=np.quantile(boot,[.025,.975]);result['conditions'][condition]={'mean_success':float(outcomes[0,:,:,ci,:].mean()),'learned_success':float(outcomes[1,:,:,ci,:].mean()),'difference_pp':float(delta.mean()*100),'paired95_ci_pp':(interval*100).tolist(),'per_seed_difference_pp':(delta.mean((1,2))*100).tolist(),'per_task_difference_pp':dict(zip(tasks,(delta.mean((0,2))*100).tolist()))}
 primary=result['conditions']['gaussian_sigma_0.10'];clean=result['conditions']['clean'];result['claim_rule_passed']=bool(primary['paired95_ci_pp'][0]>0 and min(primary['per_seed_difference_pp'])>0 and clean['paired95_ci_pp'][0]>-5)
 result['latency_ms']={arm:{'median':float(np.median(latencies[i])),'p95':float(np.quantile(latencies[i],.95))} for i,arm in enumerate(['mean','learned'])}
 (S/'results-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
