"""Freeze a conditional review artifact only; never register or submit jobs."""
import datetime as dt,json,math,pathlib,subprocess,shutil
from prepare import A,B,OUT,read,sha,allocation
from run_full import validate_full_suite
D=OUT/'audit-20261021T1745Z';PAUSE=pathlib.Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')
def guard():assert not any(p.exists() for p in [PAUSE,A/'paused.json',B/'paused.json',OUT/'paused.json'])
def measured_allocation(base,audit,cap):
 rows=allocation(base,cap);known={(g['mode'],g['task']):g for g in audit['groups']}
 for row in rows:
  g=known.get((row['mode'],row['task']))
  if g:
   cpu=sum(v['decode_cpu_seconds'] for v in audit['decoded_videos'] if f"/{g['mode']}-{g['task']}/" in v['path'])
   row['max_seconds']=math.ceil(20*g['seconds']*1.15+600+120+20*cpu)
   row['allocation_basis']='Measured five-episode duration x20 x1.15, plus 600s startup, 120s shutdown guard, 20x measured pilot video decode time'
  else:row['allocation_basis']='Unmeasured task; residual pool proportional to native step limit'
 unknown=[r for r in rows if (r['mode'],r['task']) not in known];pool=math.floor(cap*3600)-sum(r['max_seconds'] for r in rows if r not in unknown)-720*len(unknown)
 assert pool>0
 weight=sum(r['native_step_limit'] for r in unknown)
 for row in unknown:row['max_seconds']=720+int(pool*row['native_step_limit']/weight)
 assert sum(r['max_seconds'] for r in rows)<=cap*3600
 return rows

def main():
 guard();audit=read(D/'combined-audit.json');assert audit['mechanical_checks_passed'] and read(D/'visual-review.json')['sample_review_passed']
 base=read(A/'protocol.json');rows=measured_allocation(base,audit,280);validate_full_suite(dict(rows=rows,planned_episodes=10000))
 old=allocation(base,280);short=[]
 for g in audit['groups']:
  row=next(x for x in old if x['task']==g['task'] and x['mode']==g['mode']);need=20*g['seconds']+720
  if row['max_seconds']<need:short.append(dict(key=row['key'],old_seconds=row['max_seconds'],measured_projection_plus_startup_guard_seconds=need))
 files=[D/'combined-audit.json',D/'visual-review.json',D/'visual-samples.json',A/'protocol.json',A/'pilot/result.json',B/'protocol.json',B/'pilot/result.json']
 source=pathlib.Path(__file__).parent
 source_files=[source/name for name in ['prepare.py','PLAN.md','L4_PLAN_REVIEW.md','run_full.py','classify_log.py','aggregate_full.py','audit_combined.py','prepare_review.py','test_fullgate.py']]
 snapshot=D/'code-v2';guard();snapshot.mkdir(exist_ok=False)
 for f in source_files:shutil.copyfile(f,snapshot/f.name)
 frozen_files=[snapshot/f.name for f in source_files]
 commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
 manifest=dict(source_commit=commit,supersedes=str(D/'conditional-L4-manifest.json'),kind='FROZEN_CONDITIONAL_L4_REVIEW_NOT_REGISTRATION',created=dt.datetime.now(dt.timezone.utc).isoformat(),proposed_name='ex8-c_nativefull_1021',full_registered_for_dispatch=False,scientific_registration_complete=False,authorization_status='USER_COST_REVIEW_REQUIRED',authorized_gpu_hour_ceiling=200,proposed_gpu_hour_ceiling_not_authorized=280,measured_full_gpu_hours=audit['conservative_startup_and_guard_estimate_gpu_hours'],nominal_pilot_extrapolation_gpu_hours=audit['full_cost_estimate_gpu_hours'],measured_full_gpu_hours_note='Field used by draft admission: conservative pilot extrapolation, not a measured full suite',proposed_margin_over_conservative_estimate_gpu_hours=280-audit['conservative_startup_and_guard_estimate_gpu_hours'],reserved_gpu_hours=0,proposed_allocation_gpu_hours=sum(r['max_seconds'] for r in rows)/3600,cost_gate_passed=False,combined_readiness_passed=True,readiness_scope='Mechanics and sampled visual review only; L3 affordability fails current ceiling',pilot_protocol_sha256=sha(A/'protocol.json'),readiness_input_sha256={str(p):sha(p) for p in files},input_sha256={str(p):sha(p) for p in frozen_files},checkpoint=base['checkpoint'],checkpoint_sha256=base['checkpoint_sha256'],official_sync=read(A.parent/'official-sync/latest.json'),compatibility='Same upstream commit as both actual attempts; no merge/rebase/runtime edits',inference=base['inference'],benchmark_versions=base['benchmark_versions'],compatibility_deviations=base['compatibility_deviations'],criterion=base['criterion'],planned_episodes=10000,rows=rows,debug_recording=False,deadline=None,deadline_rule='Registration timestamp + seven calendar days; no registration has occurred',retry_policy='None; preserve claims, incomplete attempts, technical failures; separate registered recovery only',all_pilot_episodes_excluded_from_full_score=True,original_failed_attempt_retained=True,original_partial_episodes_excluded=2,readiness_attempts=[str(A),str(B)],pause_paths=[str(PAUSE),str(A/'paused.json'),str(B/'paused.json'),'/home/zifanz4/openwam-runtime/ex8-c_nativefull_1021/paused.json'],dispatch_policy='Existing dispatcher only; priority gpu01 1/2 any hour; other cards Chicago 01:00-09:30; all idle30min; never preempt',old_280_allocation_underfit_pilot_rows=short,unmeasured_task_mode_rows=94,remaining_steps=['User decision on proposed ceiling; no silent expansion','Fresh official sync, exact final compatibility and input hash checks','Commit scientific preregistration and registry under separate full-stage branch; attach approval evidence','Freeze new runtime with protocol/source/input/claim identities; derive dispatcher jobs and validate_job before atomic submission','Native full runner remains CPU-reviewed only; first full job must be watched at start; no broad launch on unclassified error'],full_endpoint_complete=False,full_paper_reproduced=False)
 guard();out=D/'conditional-L4-manifest-v2.json';assert not out.exists();out.write_text(json.dumps(manifest,indent=2)+'\n');(D/'conditional-L4-manifest-v2.sha256').write_text(sha(out)+'  '+out.name+'\n')
 print(json.dumps({k:manifest[k] for k in ['kind','proposed_allocation_gpu_hours','proposed_margin_over_conservative_estimate_gpu_hours','old_280_allocation_underfit_pilot_rows','cost_gate_passed']},indent=2))
if __name__=='__main__':main()
