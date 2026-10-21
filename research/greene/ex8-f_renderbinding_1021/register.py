"""Freeze committed probe and prepared dispatcher job. Does not submit."""
import datetime as dt,hashlib,json,math,pathlib,shutil,subprocess,sys
R=pathlib.Path('/home/zifanz4/openwam-runtime');S=pathlib.Path(__file__).resolve().parent;F=R/'ex8-f_renderbinding_1021';N=R/'nightly-gpu'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def reserve(e,failed):
 assert failed['status']=='failed' and failed.get('finished') and 'Another process started using this GPU' in failed['error']
 occupied=failed['occupied_parent_seconds'];assert 0<occupied<1000
 return e['combined_reserved_seconds']+math.ceil(occupied)+180

def main():
 base=read(R/'ex8-d_nativefull_1021/protocol.json');e=read(R/'ex8-e_rowrecovery_1021/protocol.json')
 pauses=list(dict.fromkeys(e['pause_paths']+[str(F/'paused.json')]))
 def guard():assert not any(pathlib.Path(p).exists() for p in pauses)
 def write(path,data):guard();path.write_text(json.dumps(data,indent=2)+'\n')
 guard();assert not F.exists()
 sync=read(R/'official-sync/latest.json');assert sync['passed'] and sync['after']==base['official_sync']['after'] and not sync['changed_files']
 assert (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(sync['time'])).total_seconds()<3600
 failed=N/'runs/ex8d-full-002-blocks-ranking-rgb/result.json';total=reserve(e,read(failed));assert total<=280*3600
 assert base['deadline']==e['deadline'] and dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(base['deadline'])
 names=['PLAN.md','adapter.py','probe_candidate.py','run_probe.py','register.py','test_adapter.py','test_registration.py']
 for name in names:
  subprocess.run(['git','-C',str(S),'ls-files','--error-unmatch',str(S/name)],check=True,stdout=subprocess.DEVNULL)
  assert not subprocess.check_output(['git','-C',str(S),'diff','HEAD','--',str(S/name)],text=True)
 guard();F.mkdir();guard();(F/'code').mkdir()
 inputs={}
 for name in names:
  guard();target=F/'code'/name;shutil.copyfile(S/name,target);inputs[str(target)]=sha(target)
 draft=read(S.parent/'native_render_binding/probe-registration.draft.json')
 for name,digest in draft['input_sha256'].items():
  if '/native_render_binding/' in name:continue
  assert sha(name)==digest,name;inputs[name]=digest
 for path in [R/'ex8-d_nativefull_1021/protocol.json',R/'ex8-e_rowrecovery_1021/protocol.json',failed]:inputs[str(path)]=sha(path)
 p=dict(kind='FROZEN_NATIVE_RENDER_PROBE',name=F.name,registered_for_dispatch=True,scientific_registration_complete=True,registered=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip(),official_sync=sync,compatibility='Same upstream commit; all frozen native runtimes unchanged',deadline=base['deadline'],max_seconds=180,probe_alarm_seconds=90,planned_episodes=0,scored_episodes=0,input_sha256=inputs,pause_paths=pauses,required_environment=draft['required_environment'],cost=dict(conditional_recovery_reserved_seconds=total-180,probe_reserved_seconds=180,total_reserved_seconds=total,total_reserved_gpu_hours=total/3600,headroom_seconds=280*3600-total,unchanged_ceiling_gpu_hours=280),criterion=dict(explicit_and_default_PCI_match=True,minimum_complete_samples=3,assigned_context_observed=True,no_sampled_peer_context=True),publication='Local registry only; no external upload',retry_policy='None; one immutable claim',staged_validation=dict(L0_static=True,L1_CPU=True,L1_GPU=False,L2_L3_native=False,L4_recovery=False),full_endpoint_complete=False,full_paper_reproduced=False)
 write(F/'protocol.json',p)
 argv=['/usr/bin/python3',str(F/'code/run_probe.py')];jid='ex8f-native-render-probe'
 job=dict(id=jid,status='ready',scientific_identity=F.name+'/renderer-allocation/attempt1',max_seconds=180,authorization_evidence=str(F/'protocol.json'),authorization_sha256=sha(F/'protocol.json'),pause_paths=pauses,host_commands={'gpu01':dict(argv=argv,admission_argv=argv+['--check-admission'],cwd=str(F),input_sha256={str(F/'protocol.json'):sha(F/'protocol.json')},outputs_absent=[str(F/'result.json')])})
 sys.path.insert(0,str(N/'code-v3'));from policy import validate_job
 validate_job(job);assert not (N/'jobs'/f'{jid}.json').exists() and not (N/'claims'/f'{jid}.json').exists()
 write(F/'prepared-job.json',job)
 write(F/'registry.json',dict(name=F.name,question='Does explicit renderer priming confine default scenes to assigned GPU?',baseline='ex8-d_nativefull_1021 failed blocks_ranking_rgb',protocol_sha256=sha(F/'protocol.json'),job_sha256=sha(F/'prepared-job.json'),submitted=False,release_conditions=['No active/uncertain native peer claims; admission succeeds','Fresh full inventory; exact idle30min eligibility through existing dispatcher','Re-read all pauses and current official-sync; compare compatibility','Recheck immutable source hashes and <=280GPUh accounting','Avoid simultaneous native peer release; controller has no dependency gate'],next_stage='Native rollout validation separately required; no full recovery dispatch from renderer-only result'))
 print(json.dumps(dict(runtime=str(F),prepared_job=str(F/'prepared-job.json'),submitted=False,total_reserved_gpu_hours=total/3600)))
if __name__=='__main__':main()
