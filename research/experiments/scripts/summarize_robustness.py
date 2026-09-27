#!/usr/bin/env python3
import hashlib,json,math,re,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/robustness-20260921'
NAMES=['baseline-clean','baseline-noise','baseline-front','identity-confirm-clean','identity-confirm-noise','adapter-confirm-clean','adapter-confirm-noise']

def interval(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n;mid=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [max(0.,mid-half),min(1.,mid+half)]

def paired(a,b):
 assert len(a)==len(b)==5
 checks=[]
 for x,y in zip(a,b):
  fields=['seed','instruction','initial_head_sha256','initial_state_sha256']
  assert all(x[k]==y[k] for k in fields),{'reference':x,'candidate':y}
  checks.append({k:x[k] for k in fields})
 wins=sum(y['success'] and not x['success'] for x,y in zip(a,b));losses=sum(x['success'] and not y['success'] for x,y in zip(a,b));n=wins+losses
 pvalue=min(1.,2*sum(math.comb(n,k)for k in range(min(wins,losses)+1))/2**n) if n else 1.
 return {'candidate_wins':wins,'candidate_losses':losses,'same_outcome':5-n,'success_rate_delta':(wins-losses)/5,'two_sided_exact_discordant_pair_p':pvalue,'initial_state_image_instruction_checks':checks}

def main():
 for phase in ['baseline','identity-confirm','adapter-confirm']:
  assert (OUT/f'{phase}-exit-code.txt').read_text().strip()=='0'
 results={}
 for name in NAMES:
  d=OUT/name;s=json.loads((d/'summary.json').read_text());log=(d/'client.log').read_text();text=re.sub(r'\x1b\[[0-9;]*m','',log)
  progresses=re.findall(r'Success rate:\s*(\d+)/(\d+).*?current seed:\s*(\d+)',text)
  assert len(progresses)==5 and int(progresses[-1][0])==s['successes'] and int(progresses[-1][1])==5
  assert [int(row[2])for row in progresses]==[r['seed']for r in s['records']]
  original=list((d/'runtime/eval_result').rglob('_result.txt'));assert len(original)==1
  s['native_result_rate']=float(original[0].read_text().strip().splitlines()[-1])
  assert abs(s['native_result_rate']-s['successes']/100)<1e-12
  s['success_rate']=s['successes']/s['episodes'];s['wilson_95ci']=interval(s['successes'],s['episodes'])
  s['mean_actions_all_episodes']=float(np.mean([r['actions']for r in s['records']]))
  s['environment_exceptions_in_log']=text.count('[eval_policy_wrapper] exception in')
  s['client_log_sha256']=hashlib.sha256((d/'client.log').read_bytes()).hexdigest()
  videos=list((d/'runtime/eval_result').rglob('episode*.mp4'));assert len(videos)==5
  s['videos']=[{'name':v.name,'sha256':hashlib.sha256(v.read_bytes()).hexdigest(),'bytes':v.stat().st_size}for v in sorted(videos)]
  results[name]=s
 comparisons={}
 for ref,candidate in [('baseline-clean','baseline-noise'),('baseline-clean','baseline-front'),('identity-confirm-clean','identity-confirm-noise'),('identity-confirm-clean','adapter-confirm-clean'),('identity-confirm-noise','adapter-confirm-noise'),('adapter-confirm-clean','adapter-confirm-noise')]:
  comparisons[candidate+'_vs_'+ref]=paired(results[ref]['records'],results[candidate]['records'])
 selection=json.loads((OUT/'adapter-selection.json').read_text());unlock=json.loads((OUT/'confirmation-unlock.json').read_text())
 assert unlock['selection_sha256']==hashlib.sha256((OUT/'adapter-selection.json').read_bytes()).hexdigest()
 assert unlock['checkpoint_sha256']==hashlib.sha256((OUT/'adapter-selected.pt').read_bytes()).hexdigest()
 manifest=json.loads((OUT/'fixed-scenes.json').read_text())
 manifest_hash=hashlib.sha256((OUT/'fixed-scenes.json').read_bytes()).hexdigest()
 assert unlock['fixed_scenes_sha256']==manifest_hash
 for name in NAMES[3:]:
  replay=json.loads((OUT/name/'replay-provenance.json').read_text())
  assert replay['manifest_sha256']==manifest_hash
  for row,reference in zip(results[name]['records'],manifest['scenes']):
   assert all(row[key]==reference[key] for key in reference),name
 hook=json.loads((OUT/'adapter-hook.json').read_text());assert hook['calls']>0 and hook['max_change']>0
 hashes=json.loads((OUT/'source-hashes.json').read_text())
 # Protocol is copied into the run; scripts are hashed at the executable source.
 source_audit={}
 legacy_snapshots={'scripts/control_robustness.py','scripts/run-control-robustness.sh'}
 for relative,value in hashes.items():
  path=OUT/'protocol.md' if relative=='ROBUSTNESS_PROTOCOL.md' else ROOT/relative
  if relative in legacy_snapshots:
   # The initial broad filename snapshot included two unused prior-study files.
   # Verify them at their original repository location, not an invented runtime copy.
   path=Path('/home/zifanz4/openwam-experiments')/relative
  assert hashlib.sha256(path.read_bytes()).hexdigest()==value,relative
  source_audit[relative]={'sha256_matches':True,'scope':'unused prior-study repository snapshot' if relative in legacy_snapshots else 'runtime source or protocol'}
 (OUT/'source-audit-final.json').write_text(json.dumps(source_audit,indent=2))
 initial=OUT/'invalid-unpaired-attempt'
 invalid={f.parent.name:json.loads(f.read_text()) for f in initial.glob('*/episodes.json')}
 invalid_refilter={f.parent.name:json.loads(f.read_text()) for f in (OUT/'invalid-refiltering-attempt').glob('*/episodes.json')}
 summary={'scope':'35 paired closed-loop pilot episodes, six train/validation-only adapter fits; one task; not official full benchmark','completed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'conditions':results,'paired_comparisons':comparisons,'adapter_selection':selection,'adapter_hook':hook,'confirmation_unlock':unlock,'protocol_sha256':hashlib.sha256((OUT/'protocol.md').read_bytes()).hexdigest(),'checkpoint_revision':'af1595c8ee54955116abbc0b0ffa17fa48deb691','policy_diffusion_seed':42,'noise_sigma':.10,'denoise_steps':10,'compile_enabled':False,'task':'pick_dual_bottles','mode':'demo_clean','step_limit':400,'planner_fallback':False,'paired_checks_all_passed':True,'source_hashes_all_passed':True,'invalid_initial_attempt':invalid,'invalid_refiltering_attempt':invalid_refilter,'confirmation_scene_manifest':manifest,'confirmation_protocol':'fixed reference-feasible seeds and exact prompts; no repeated per-condition expert filter; amendment recorded before any adapted rollout','contact_claims':False}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:(v['successes'],v['episodes'])for k,v in results.items()},indent=2))
if __name__=='__main__':main()
