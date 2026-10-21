"""Register a separate bounded L4 phase; old conditional review stays unchanged."""
import datetime as dt,hashlib,json,pathlib,shutil,subprocess,sys
R=pathlib.Path('/home/zifanz4/openwam-runtime');ROOT=R/'ex8-d_nativefull_1021';SOURCE=pathlib.Path(__file__).resolve().parent
REVIEW=R/'ex8-c_fullgate_1021/audit-20261021T1745Z/conditional-L4-manifest-v2.json'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 t=p.with_suffix('.pending');t.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def main():
 assert not ROOT.exists()
 decision=pathlib.Path('/home/zifanz4/.local/state/openwam-selfcheck/planning-policy.json');policy=read(decision);assert policy['new_full_stage_ceiling_gpu_hours']==280
 p=read(REVIEW);assert p['combined_readiness_passed'];assert p['measured_full_gpu_hours']<280
 assert len(p['rows'])==100 and sum(x['max_seconds'] for x in p['rows'])<=280*3600
 for name,digest in {**p['readiness_input_sha256'],**p['input_sha256']}.items():assert sha(pathlib.Path(name))==digest,name
 assert not any(pathlib.Path(x).exists() for x in p['pause_paths'])
 sync=read(R/'official-sync/latest.json');assert sync['passed'] and sync['after']==p['official_sync']['after'] and (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(sync['time'])).total_seconds()<3600
 names=['PLAN.md','run_full.py','classify_log.py','aggregate_full.py','register.py','gate.py','test_fullgate.py','prepare.py']
 for name in names:
  subprocess.run(['git','-C',str(SOURCE),'ls-files','--error-unmatch',str(SOURCE/name)],check=True,stdout=subprocess.DEVNULL)
  assert not subprocess.check_output(['git','-C',str(SOURCE),'diff','HEAD','--',str(SOURCE/name)],text=True)
 ROOT.mkdir();(ROOT/'code').mkdir();(ROOT/'runs').mkdir();(ROOT/'evidence').mkdir();frozen_decision=ROOT/'evidence/planning-policy.json';shutil.copyfile(decision,frozen_decision);inputs={}
 for name in names:
  target=ROOT/'code'/name;shutil.copyfile(SOURCE/name,target);inputs[str(target)]=sha(target)
 for k in ['proposed_gpu_hour_ceiling_not_authorized','proposed_name','proposed_margin_over_conservative_estimate_gpu_hours']:p.pop(k,None)
 now=dt.datetime.now(dt.timezone.utc);p.update(name=ROOT.name,kind='FROZEN_NATIVE_FULL_REGISTRATION',source_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),registered=now.isoformat(),deadline=(now+dt.timedelta(days=7)).isoformat(),deadline_rule='Seven days from this phase registration',full_registered_for_dispatch=True,scientific_registration_complete=True,authorization_status='EXISTING_TASK_SCOPE_WITH_RECORDED_AGENT_PLANNING_AMENDMENT',authorized_gpu_hour_ceiling=280,planning_ceiling_source='Agent planning amendment; initial200 was not an explicit user cap',initial_agent_planning_ceiling_gpu_hours=200,planning_policy=str(frozen_decision),planning_policy_sha256=sha(frozen_decision),planning_policy_original_source=str(decision),conditional_review=str(REVIEW),conditional_review_sha256=sha(REVIEW),input_sha256=inputs,official_sync=sync,cost_gate_passed=True,reserved_gpu_hours=sum(x['max_seconds'] for x in p['rows'])/3600,readiness_scope='Audited mechanics, sampled visual review and revised finite cost gate',pause_paths=[str(ROOT/'paused.json')]+[x for x in p['pause_paths'] if 'ex8-c_nativefull' not in x],dispatch_stage='First row only until technical scaling gate',remaining_steps=['Native first-row technical gate','Remaining99 native full rows','Complete independent endpoint verification'])
 p['readiness_input_sha256'][str(frozen_decision)]=sha(frozen_decision)
 write(ROOT/'protocol.json',p)
 sys.path.insert(0,str(R/'nightly-gpu/code-v3'));from policy import validate_job
 jobs=[]
 for index,row in enumerate(p['rows']):
  jid=f'ex8d-full-{index:03d}-{row["task"].replace("_","-")}'
  argv=['/usr/bin/python3',str(ROOT/'code/run_full.py'),'--key',row['key']]
  job=dict(id=jid,status='ready',scientific_identity=f'{ROOT.name}/native-full/{row["key"]}/seed0',max_seconds=row['max_seconds'],authorization_evidence=str(ROOT/'protocol.json'),authorization_sha256=sha(ROOT/'protocol.json'),pause_paths=p['pause_paths'],host_commands={'gpu01':dict(argv=argv,admission_argv=argv+['--check-admission'],cwd=str(ROOT),input_sha256={str(ROOT/'protocol.json'):sha(ROOT/'protocol.json')},outputs_absent=[str(ROOT/'runs'/row['key'])])})
  validate_job(job);assert not (R/'nightly-gpu/claims'/f'{jid}.json').exists();assert not (R/'nightly-gpu/jobs'/f'{jid}.json').exists();jobs.append(job)
 write(ROOT/'registry.json',dict(stage='registered_L4_first_row_gate',jobs=jobs,protocol_sha256=sha(ROOT/'protocol.json')))
 (ROOT/'registry.sha256').write_text(sha(ROOT/'registry.json')+'\n')
 # CPU admission includes all frozen scientific inputs, but not expensive model load.
 subprocess.run(jobs[0]['host_commands']['gpu01']['admission_argv'],check=True,timeout=60)
 write(R/'nightly-gpu/jobs'/f'{jobs[0]["id"]}.json',jobs[0])
 print(json.dumps(dict(registered_rows=100,initially_submitted=1,reserved_gpu_hours=p['reserved_gpu_hours'],protocol_sha256=sha(ROOT/'protocol.json'))))
if __name__=='__main__':main()
