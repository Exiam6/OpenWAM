"""One new immutable registration; never touches existing claims or budgets."""
import datetime as dt,hashlib,json,os,pathlib,shutil,subprocess,sys
S=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/home/zifanz4/openwam-runtime');F=R/S.name;N=R/'nightly-gpu';H=pathlib.Path('/home/zifanz4')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def guard(paths):assert not any(pathlib.Path(p).exists() for p in paths),'Paused'
def write(p,v,paths):
 guard(paths)
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 old=read(R/'ex8-k_glfwprefix_1021/protocol.json');paths=old['pause_paths']+[str(F/'paused.json')];guard(paths)
 assert not F.exists() and not (N/'jobs/ex8l-native-headless-prefix.json').exists()
 official=read(R/'official-sync/latest.json');assert official['passed'] and official['after']=='782e3133f674449e2986754cfc5f7c4e204835a2'
 assert (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(official['time'])).total_seconds()<600
 assert all(read(p)['status'] in ['complete','failed'] for p in (N/'claims').glob('ex8*.json'))
 costref=H/'.local/state/openwam-selfcheck/evidence/20261022T0315Z/cost-after-k.json';cost=read(costref)
 assert all(sha(p)==v for p,v in cost['input_sha256'].items())
 assert cost['cost']['total_reserved_seconds']==1007928
 assert read(S/'CPU_VALIDATION.json')['integration_and_supervisor_tests_per_interpreter']==24
 commit=subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()
 assert not subprocess.check_output(['git','-C',str(S),'status','--porcelain'],text=True).strip()
 guard(paths);F.mkdir();shutil.copytree(S,F/'code');(F/'tmp').mkdir();(F/'cache').mkdir()
 prep=read(S.parent/'native_headless_boundary_prep/PREPARATION.json')
 assert sha(prep['production_artifact'])==prep['production_sha256']
 guard(paths);shutil.copyfile(prep['production_artifact'],F/'code/libheadless.so')
 # Private cache clone; all old attempts stay untouched.
 guard(paths);shutil.copytree(R/'ex8-k_glfwprefix_1021/cache/extensions',F/'cache/extensions')
 dry=[]
 for d in sorted((F/'cache/extensions').iterdir()):
  if (d/'build.ninja').exists():
   q=subprocess.run([str(R/'benchmark-env/bin/ninja'),'-n','-C',str(d)],check=True,text=True,capture_output=True,timeout=4)
   assert 'no work to do' in q.stdout,q.stdout;dry.append(q.stdout)
 inputs={str(p):sha(p) for p in (F/'code').glob('*') if p.is_file()}
 deps=prep['observer_dependency_closure'];assert all(sha(p)==v for p,v in deps.items());inputs.update(deps)
 validation=read(H/'.local/state/openwam-selfcheck/evidence/20261022T0330Z/headless-validation-v3/validation.json')
 for p,v in validation['input_sha256'].items():assert sha(p)==v;inputs[p]=v
 for p,v in old['input_sha256'].items():
  if any(t in p for t in ['/official/','/sapien.libs/','/RoboTwin/envs/_base_task.py']):assert sha(p)==v;inputs[p]=v
 for p in (F/'cache/extensions').rglob('*'):
  if p.is_file():inputs[str(p)]=sha(p)
 inputs[str(costref)]=sha(costref)
 inputs[str(R/'ex8-d_nativefull_1021/protocol.json')]=sha(R/'ex8-d_nativefull_1021/protocol.json')
 for n in ['run_full.py','classify_log.py']:inputs[str(R/'ex8-d_nativefull_1021/code'/n)]=sha(R/'ex8-d_nativefull_1021/code'/n)
 env=dict(old['required_environment']);env.update(TORCH_EXTENSIONS_DIR=str(F/'cache/extensions'),TMPDIR=str(F/'tmp'))
 inv=read(N/'latest.json');mapping={k:v['uuid'] for k,v in inv['hosts']['gpu01'].items()};assert len(mapping)==8
 p=dict(kind='FROZEN_NATIVE_HEADLESS_PREFIX',name=S.name,registered_for_dispatch=True,scientific_registration_complete=True,registered=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=commit,plan_commit='2135f0db',max_seconds=150,probe_alarm_seconds=70,deadline=old['deadline'],pause_paths=paths,input_sha256=inputs,required_environment=env,device_map=mapping,official_sync=official,compatibility='Official unchanged; original native ELF/ABI and scientific settings unchanged; headless prefix diagnostic only; exact native binary build revision unproven.',planned_episodes=0,scored_episodes=0,full_endpoint_complete=False,full_paper_reproduced=False,native_L2_L3_passed=False,full_release_allowed=False,cost=dict(separate_diagnostic_max_seconds=150,existing_caps_changed=False,existing_reserved_seconds=1007928,combined_future_reserved_seconds=1008078,combined_future_reserved_gpu_hours=1008078/3600,increment_above_previous_planning_allocation_seconds=78,full_stage_ex8d_ceiling_gpu_hours_unchanged=280,scene_validation_reserve_seconds_unchanged=9000,basis='Existing autonomous reproduction/diagnosis authorization; agent planning estimate is not user cap; separately reported bounded diagnostic'),cache_dry_runs=dry,retry_policy='None; one new scientific identity; retain every failure',criterion=dict(stable_samples_per_phase=3,required_phases=13,raw_durable_packets=True,native_prewarm_required=True,assigned_PCI=True,no_sampled_peer=True,terminal_scientific_worker_claim_agreement=True),publication='Local only')
 write(F/'protocol.json',p,paths)
 job=dict(id='ex8l-native-headless-prefix',status='ready',scientific_identity=S.name+'/native-headless-prefix/attempt1',max_seconds=150,authorization_evidence=str(F/'protocol.json'),authorization_sha256=sha(F/'protocol.json'),pause_paths=paths,host_commands=dict(gpu01=dict(argv=['/usr/bin/python3',str(F/'code/run_probe.py')],admission_argv=['/usr/bin/python3',str(F/'code/run_probe.py'),'--check-admission'],cwd=str(F),input_sha256={str(F/'protocol.json'):sha(F/'protocol.json')},outputs_absent=[str(F/'result.json'),str(F/'observer-config.json'),str(F/'journal')])) )
 write(F/'dispatcher-job.staged.json',job,paths)
 guard(paths);q=subprocess.run(job['host_commands']['gpu01']['admission_argv'],check=True,text=True,capture_output=True,timeout=10,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
 write(F/'admission-cpu.json',dict(passed=True,stdout=q.stdout,native_executed=False,GPU_seconds=0),paths)
 # Controller reads only ready jobs; atomic publication prevents partial JSON.
 guard(paths);target=N/'jobs'/('ex8l-native-headless-prefix.json');assert not target.exists()
 staged=F/'dispatcher-job.staged.json';raw=staged.read_bytes();temp=N/'jobs/ex8l-native-headless-prefix.pending'
 with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 guard(paths);os.link(temp,target);temp.unlink()
 print(json.dumps(dict(runtime=str(F),job=str(target),source_commit=commit,max_seconds=150,queued=True)))
if __name__=='__main__':main()
