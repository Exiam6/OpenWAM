from pathlib import Path
import json,hashlib,datetime
import numpy as np
import torch
V=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018');D=V/'fixed-policy-diagnostic-0825'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
e=json.loads((D/'execution.json').read_text());assert e['status']=='complete' and e['original_evidence_unchanged'];r=json.loads((D/'observations/result.json').read_text());assert r['completed'] and len(r['rows'])==3
plan=json.loads((D/'plan.json').read_text())
for p,h in plan['frozen_files'].items():assert sha(p)==h
assert not (V/'full-training-release.json').exists()
cache=torch.load(V/'vae-cache/cache.pt',map_location='cpu',mmap=True,weights_only=False)
rows=[]
for row in r['rows']:
 assert sha(row['trace'])==row['trace_sha256'];assert sha(row['captured_inputs'])==row['captured_inputs_sha256']
 with np.load(row['trace'],allow_pickle=False) as a:
  assert all(a[k].shape==(32,20) and np.isfinite(a[k]).all() for k in a.files)
  ad=float(np.max(np.abs(a['image']-a['cached'])));rd=float(np.max(np.abs(a['image']-a['repeated'])))
  assert ad==row['action_difference'] and rd==row['repeat_difference']
  previous_exact=None
  if row['task']=='adjust_bottle':
   with np.load(V/'deployment-diagnostic-0807/observations/adjust_bottle-actions.npz',allow_pickle=False) as old:previous_exact=np.array_equal(a['cached'],old['cached']);assert previous_exact
 captures=torch.load(row['captured_inputs'],map_location='cpu',weights_only=True)
 assert set(captures)=={'image','cached','repeated'}
 compared={}
 for label in ['cached','repeated']:
  compared[label]={}
  assert captures['image'].keys()==captures[label].keys()
  for k,val in captures['image'].items():
   other=captures[label][k];equal=torch.equal(val,other) and val.dtype==other.dtype if torch.is_tensor(val) else val==other
   compared[label][k]=equal
  assert all(compared[label].values())
 assert torch.equal(captures['image']['first_frame_latents'],cache['vae'][row['cache_index']:row['cache_index']+1,:,:1])
 rows.append({'task':row['task'],'cache_index':row['cache_index'],'action_difference':ad,'repeat_difference':rd,'meets_original_tolerance':ad<=1e-5 and rd<=1e-5,'first_forward_fields_exact':compared,'image_first_latent_exactly_matches_cache':True,'prior_unmodified_cached_actions_exact':previous_exact,'actions_sha256':row['trace_sha256'],'inputs_sha256':row['captured_inputs_sha256']})
assert all(x['meets_original_tolerance'] for x in rows)
print('execution',json.dumps(e['execution']))
costs={}
for name,path in [('original_preflight',V/'dedicated-preflight-run.json'),('policy_input_diagnostic',V/'deployment-diagnostic-0807/execution.json'),('encoder_context_diagnostic',V/'encoder-precision-diagnostic-0812/execution.json'),('fixed_policy_diagnostic',D/'execution.json')]:
 data=json.loads(path.read_text());costs[name]=data['occupied_seconds'] if name=='original_preflight' else data['execution']['occupied_seconds']
result={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'all_original_tolerances_passed':True,'raw_result_sha256':sha(D/'observations/result.json'),'plan_sha256':sha(D/'plan.json'),'original_failed_preflight_preserved':True,'original_gate_status':'failed','full_training_released':False,'costs':costs,'total_occupied_gpu_seconds':sum(costs.values()),'scope':'Candidate method bound only in an isolated process. Full 10-step image/cache/repeated inference on the first training window of three tasks from the retained step3 checkpoint. No trained endpoint, optimizer, tiny-fit or closed-loop success evidence.'}
(D/'independent-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
