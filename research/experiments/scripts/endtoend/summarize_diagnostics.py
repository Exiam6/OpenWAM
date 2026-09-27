"""All component/phase diagnostics with episode aggregation; descriptive secondary endpoints."""
import json
from pathlib import Path
import numpy as np
from action_diagnostics import errors
E=Path('/data02/zifanz4/openwam-experiments/endtoend-20260923');methods=['svae','pca','wan'];seeds=[42,43,44];tasks=['adjust_bottle','handover_block','place_object_basket'];sigmas=[0.,.04,.10]
result={};byepisode={};maxres=0.
for method in methods:
 for seed in seeds:
  root=E/'action-diagnostics'/f'{method}-seed{seed}';assert json.loads((root/'exit.json').read_text())['returncode']==0
  summary=json.loads((root/'summary.json').read_text());assert summary['complete'] and summary['predictions']==5400
  rows=[json.loads(s) for s in (root/'rows.jsonl').read_text().splitlines()];assert len(rows)==5400
  for row in rows:
   assert row['route']==method and row['training_seed']==seed
   for source,key in [('prediction','prediction_errors'),('state','hold_current_state_errors')]:
    measured=errors(row[source],row['target'])
    for k,v in measured.items():maxres=max(maxres,abs(v-row[key][k]));assert abs(v-row[key][k])<1e-10
  for task in tasks:
   scene_ids=[x['seed'] for x in json.loads((E/'scenes'/task/'manifest.json').read_text())['accepted']]
   for sigma in sigmas:
    for phase in ['all_frames','gripper_closing_proxy','non_closing_proxy']:
     episodes=[]
     for scene in scene_ids:
      items=[r for r in rows if r['task']==task and r['scene_seed']==scene and r['sigma']==sigma];assert len(items)==12
      if phase!='all_frames':items=[r for r in items if r['gripper_closing_proxy']==(phase=='gripper_closing_proxy')]
      means={source:{k:float(np.mean([r[source][k] for r in items])) for k in rows[0][source]} for source in ['prediction_errors','hold_current_state_errors']} if items else None
      episodes.append({'scene_seed':scene,'frames':len(items),'means':means})
     key=f'{method}/seed{seed}/{task}/sigma{sigma}/{phase}';byepisode[key]=episodes;valid=[e for e in episodes if e['frames']]
     result[key]={'episodes_with_observations':len(valid),'episodes_without_observations':50-len(valid),'frames':sum(e['frames'] for e in episodes),'episode_equal_mean':{source:{k:float(np.mean([e['means'][source][k] for e in valid])) for k in rows[0][source]} for source in ['prediction_errors','hold_current_state_errors']} if valid else None}
out=E/'analysis';assert not (out/'action-diagnostics.json').exists()
(out/'action-diagnostics.json').write_text(json.dumps({'complete':True,'predictions':48600,'arithmetic_max_residual':maxres,'secondary_descriptive_only':True,'phase_warning':'closing is kinematic proxy; no ground-truth contact label for saved expert frames','per_seed_task_sigma_phase':result,'episode_aggregates':byepisode},indent=2)+'\n')
print(json.dumps({'complete':True,'predictions':48600,'secondary_descriptive_only':True}))
