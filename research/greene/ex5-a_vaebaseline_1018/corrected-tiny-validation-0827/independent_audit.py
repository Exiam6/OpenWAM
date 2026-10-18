from pathlib import Path
import hashlib,json,math,datetime
V=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018');D=V/'corrected-tiny-validation-0827'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
e=json.loads((D/'execution.json').read_text());assert e['status']=='complete' and e['original_evidence_unchanged']
p=json.loads((D/'plan.json').read_text())
for path,h in p['frozen_files'].items():assert sha(path)==h,path
r=json.loads((D/'observations/result.json').read_text());assert r['steps']==512 and not r['checkpoint_reused_for_full_training']
a,b=r['before'],r['after'];criteria={};ratios={}
for key in ['flow_action_loss','mean_generated_mse']:
 assert math.isfinite(a[key]) and a[key]>0 and math.isfinite(b[key]) and b[key]>=0
 ratios[key]=b[key]/a[key];criteria[key]=b[key]<=.25*a[key]
assert len(a['rows'])==len(b['rows'])==3
manifest=json.loads((V/'attempt-data-restored/window-manifest.json').read_text());expected=[next(x['cache_index'] for x in manifest['windows'] if x['task']==t and x['split']=='train') for t in ['adjust_bottle','handover_block','place_object_basket']];assert r['training_indices']==expected
for old,new in zip(a['rows'],b['rows']):
 assert (old['task'],old['cache_index'])==(new['task'],new['cache_index'])
 assert math.isfinite(old['generated_action_mse']) and old['generated_action_mse']>0 and math.isfinite(new['generated_action_mse']) and new['generated_action_mse']>=0
 name='task_'+old['task'];ratios[name]=new['generated_action_mse']/old['generated_action_mse'];criteria[name]=ratios[name]<=.5
assert r['passed']==all(criteria.values())
metrics=[json.loads(s) for s in (D/'observations/metrics.jsonl').read_text().splitlines()];assert [x['step'] for x in metrics]==list(range(32,513,32))
assert all(all(math.isfinite(x[k]) and x[k]>=0 for k in ['seconds','loss_action','gradient_norm']) for x in metrics)
tail=metrics[-9:];rates=[(b['seconds']-a['seconds'])/(b['step']-a['step']) for a,b in zip(tail,tail[1:])];assert all(x>0 for x in rates)
projection=max(rates)*20000+900
prior=json.loads((V/'fixed-policy-diagnostic-0825/independent-analysis.json').read_text());costs=dict(prior['costs'],corrected_tiny_validation=e['execution']['occupied_seconds']);total=sum(costs.values());combined=p['prior_noncache_occupied_seconds']+e['execution']['occupied_seconds'];assert combined<1800 and total<73800
assert json.loads((V/'dedicated-preflight-run.json').read_text())['status']=='failed' and not (V/'full-training-release.json').exists()
result={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':all(criteria.values()),'criteria_recomputed':criteria,'after_to_before_ratios':ratios,'training_indices':expected,'steps':512,'training_projection_seconds_per_seed':projection,'training_projection_within_original_10740s_guard':projection<10740,'projection_scope':'Maximum of last eight 32-step intervals times20000 plus900s fixed reserve; operational estimate, not guaranteed runtime.','combined_noncache_occupied_seconds':combined,'combined_noncache_cap':1800,'costs':costs,'total_occupied_gpu_seconds':total,'result_sha256':sha(D/'observations/result.json'),'metrics_sha256':sha(D/'observations/metrics.jsonl'),'original_preflight_status':'failed','full_training_released':False,'scope':'Independent recomputation of all original tiny-fit criteria. These three training windows are not held-out or closed-loop evaluations. Full training requires a separately registered corrected protocol, fresh initialization and current full endpoint capacity/deadline checks.'}
(D/'independent-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
