#!/usr/bin/env python3
"""Audit all frozen pairs before reporting final normalization outcomes."""
import hashlib,json,math,re,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/norm-20260921'
ARMS=['identity','renorm','adapter','adapter_renorm']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def interval(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n;m=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [max(0.,m-h),min(1.,m+h)]
def paired(a,b):
 assert len(a)==len(b)
 for x,y in zip(a,b):
  assert all(x[k]==y[k] for k in ['seed','instruction','initial_head_sha256','initial_state_sha256'])
 wins=sum(y['success'] and not x['success'] for x,y in zip(a,b));losses=sum(x['success'] and not y['success'] for x,y in zip(a,b));n=wins+losses
 p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
 return {'wins':wins,'losses':losses,'ties':len(a)-n,'rate_delta':(wins-losses)/len(a),'two_sided_exact_p':p}
def collect(name,count,role,manifest=None):
 d=OUT/name;assert (d/'exit-code.txt').read_text().strip()=='0'
 s=json.loads((d/'summary.json').read_text());p=json.loads((d/'provenance.json').read_text())
 assert s['episodes']==len(s['records'])==count
 assert s['successes']==sum(r['success'] for r in s['records'])
 assert p['role']==role
 text=re.sub(r'\x1b\[[0-9;]*m','',(d/'client.log').read_text())
 progress=re.findall(r'Success rate:\s*(\d+)/(\d+).*?current seed:\s*(\d+)',text)
 assert len(progress)==count and int(progress[-1][0])==s['successes'] and int(progress[-1][1])==count
 assert [int(x[2]) for x in progress]==[x['seed'] for x in s['records']]
 assert all(r['actions']==r['calls'] and 0<r['actions']<=400 for r in s['records'])
 if manifest:
  assert p['manifest_sha256']==sha(manifest)
  for x,y in zip(s['records'],json.loads(manifest.read_text())['scenes']):
   assert all(x[k]==y[k] for k in ['seed','instruction','initial_head_sha256','initial_state_sha256'])
 native=list((d/'runtime/eval_result').rglob('_result.txt'));assert len(native)==1
 native_rate=float(native[0].read_text().strip().splitlines()[-1]);assert abs(native_rate-s['successes']/100)<1e-12
 videos=list((d/'runtime/eval_result').rglob('episode*.mp4'));assert len(videos)==count
 s.update({'success_rate':s['successes']/count,'wilson_95ci':interval(s['successes'],count),'mean_actions':float(np.mean([r['actions'] for r in s['records']])),'native_result_rate':native_rate,'provenance':p,'videos':[{'name':v.name,'sha256':sha(v),'bytes':v.stat().st_size} for v in sorted(videos)],'client_log_sha256':sha(d/'client.log')})
 return s
def main():
 assert (OUT/'study-exit-code.txt').read_text().strip()=='0'
 freeze=json.loads((OUT/'source-freeze.json').read_text())
 for name,value in {**freeze['runtime_sources'],**freeze['upstream_sources']}.items():assert sha(ROOT/name)==value,name
 assert sha(OUT/'protocol.md')==freeze['protocol_sha256']
 assert sha(ROOT/'results/robustness-20260921/adapter-selected.pt')==freeze['adapter_sha256']
 reference=collect('reference-clean',10,'reference');fresh=OUT/'fresh-scenes.json';hist=OUT/'historical-scene.json'
 assert json.loads(fresh.read_text())['reference_summary_sha256']==sha(OUT/'reference-clean/summary.json')
 primary={};historical={};hooks={}
 for arm in ARMS:
  assert (OUT/f'{arm}-exit-code.txt').read_text().strip()=='0'
  hooks[arm]=json.loads((OUT/f'{arm}-hook.json').read_text());assert hooks[arm]['calls']>0 and hooks[arm]['future_slots_exact']
  assert (hooks[arm]['max_change']==0) if arm=='identity' else (hooks[arm]['max_change']>0)
  historical[arm]=collect(f'{arm}-historical-clean',1,'historical',hist)
  for condition in ['clean','noise']:
   name=f'{arm}-{condition}';primary[name]=collect(name,10,'primary',fresh)
   assert primary[name]['provenance']['noise_sigma']==(0 if condition=='clean' else .20)
 comparisons={}
 for condition in ['clean','noise']:
  for ref,candidate in [('adapter','adapter_renorm'),('identity','renorm'),('identity','adapter'),('identity','adapter_renorm')]:
   comparisons[f'{candidate}_vs_{ref}-{condition}']=paired(primary[f'{ref}-{condition}']['records'],primary[f'{candidate}-{condition}']['records'])
 for arm in ARMS:paired(primary[f'{arm}-clean']['records'],primary[f'{arm}-noise']['records'])
 for arm in ARMS[1:]:paired(historical['identity']['records'],historical[arm]['records'])
 result={'scope':'80 primary paired rollouts + 4 selected historical diagnostics + 10 expert-feasibility reference rollouts','completed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'primary':primary,'paired_comparisons':comparisons,'historical':historical,'reference':reference,'hooks':hooks,'source_freeze':freeze,'all_integrity_checks_passed':True,'fresh_manifest_sha256':sha(fresh),'noise_sigma':.20,'adapter_training_noise_sigma':.10,'policy_diffusion_seed':42,'one_task_only':True,'contact_labels':False,'new_training':False}
 (OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:[v['successes'],v['episodes']] for k,v in primary.items()}))
if __name__=='__main__':main()
