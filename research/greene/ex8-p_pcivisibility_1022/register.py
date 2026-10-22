"""Freeze one bounded, preregistered native constructor diagnostic."""
import datetime as dt,hashlib,json,os,pathlib,shutil,subprocess
S=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/home/zifanz4/openwam-runtime');F=R/S.name;N=R/'nightly-gpu';J='ex8p-native-pci-visibility';OLD=R/'ex8-m_hostidentity_1022';ABI=R/'ex8-j_instancesplit_1021'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def guard(paths):assert not any(pathlib.Path(x).exists() for x in paths),'Paused'
def write(p,v,paths):
 guard(paths)
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 old=read(OLD/'protocol.json');paths=old['pause_paths']+[str(F/'paused.json')];guard(paths)
 assert not F.exists() and not (N/'jobs'/(J+'.json')).exists() and not (N/'claims'/(J+'.json')).exists()
 assert read(N/'claims/ex8m-native-host-identity.json')['status']=='complete'
 assert read(OLD/'result.json')['status']=='supervisor_sequence_passed'
 assert all(sha(p)==v for p,v in old['input_sha256'].items()),'Old input drift'
 auditpath=pathlib.Path('/home/zifanz4/openwam-json-receipt-audit/research/greene/ex8-m_hostidentity_1022/verify_chain.py')
 audited=json.loads(subprocess.check_output(['/usr/bin/python3',str(auditpath)],text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')))
 assert audited['GPU_diagnostic_validated'],audited
 abi=read(ABI/'protocol.json');assert all(sha(p)==v for p,v in abi['input_sha256'].items()),'ABI baseline drift'
 assert all(read(p)['status'] in ['complete','failed'] for p in (N/'claims').glob('ex8*.json'))
 official=read(R/'official-sync/latest.json');assert official['passed'] and official['after']=='782e3133f674449e2986754cfc5f7c4e204835a2'
 assert 0<=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(official['time'])).total_seconds()<600
 validation=read(S/'CPU_VALIDATION.json');assert validation['passed'] and validation['native_executed'] is False
 commit=subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()
 assert not subprocess.check_output(['git','-C',str(S),'status','--porcelain'],text=True).strip()
 guard(paths);F.mkdir();shutil.copytree(S,F/'code');guard(paths);(F/'tmp').mkdir();(F/'cache').mkdir()
 for name,source in [('libheadless.so',OLD),('libinstance_boundary.so',ABI)]:
  guard(paths);shutil.copyfile(source/'code'/name,F/'code'/name)
 guard(paths);shutil.copytree(OLD/'cache/extensions',F/'cache/extensions')
 dry=[]
 for d in sorted((F/'cache/extensions').iterdir()):
  if (d/'build.ninja').exists():
   q=subprocess.run([str(R/'benchmark-env/bin/ninja'),'-n','-C',str(d)],check=True,text=True,capture_output=True,timeout=4)
   assert 'no work to do' in q.stdout;dry.append(q.stdout)
 inputs={str(p):v for p,v in old['input_sha256'].items() if not str(p).startswith(str(OLD)+'/')}
 # Preserve exact installed native ABI and already tested helper provenance.
 inputs.update(abi['input_sha256'])
 for p in list((F/'code').glob('*'))+list((F/'cache/extensions').rglob('*')):
  if p.is_file():inputs[str(p)]=sha(p)
 for p in [OLD/'protocol.json',OLD/'result.json',ABI/'protocol.json',ABI/'result.json',N/'policy.json',auditpath,auditpath.with_name('verify_raw.py'),N/'claims/ex8m-native-host-identity.json',N/'runs/ex8m-native-host-identity/result.json']:
  inputs[str(p)]=sha(p)
 for p in [R/'ex8-n_instanceaudit_1022/protocol.json',R/'ex8-n_instanceaudit_1022/result.json',R/'ex8-n_instanceaudit_1022/journal/0009.json',N/'claims/ex8n-native-instance-audit.json',N/'runs/ex8n-native-instance-audit/result.json']:
  inputs[str(p)]=sha(p)
 for p in [R/'ex8-o_observerrecovery_1022/protocol.json',R/'ex8-o_observerrecovery_1022/result.json',R/'ex8-o_observerrecovery_1022/journal/0047.json',N/'claims/ex8o-native-observer-recovery.json',N/'runs/ex8o-native-observer-recovery/result.json',pathlib.Path('/usr/lib/x86_64-linux-gnu/libVkLayer_MESA_device_select.so'),pathlib.Path('/usr/share/vulkan/implicit_layer.d/VkLayer_MESA_device_select.json')]:
  inputs[str(p)]=sha(p)
 protocol=dict(old);protocol.update(kind='FROZEN_NATIVE_INSTANCE_AUDIT',name=S.name,source_commit=commit,plan_commit=commit,registered=dt.datetime.now(dt.timezone.utc).isoformat(),pause_paths=paths,input_sha256=inputs,official_sync=official,max_seconds=180,probe_alarm_seconds=95,baseline_attempt=str(OLD),baseline_acceptance=audited,cache_dry_runs=dry,compatibility='Official782e3133 unchanged; unchanged native prewarm/planner/inference; same native and observation sequence as ex8-o; single process-local DRI_PRIME PCI visibility intervention via installed Mesa layer. Scientific compatibility beyond prefix unvalidated. Exact installed native source revision unproven.')
 protocol['baseline_failed_attempt']=str(R/'ex8-o_observerrecovery_1022')
 protocol['observer_timing']=dict(inner_seconds=8,outer_seconds=9,phase_seconds=30,previous_inner_seconds=3,previous_outer_seconds=4,previous_phase_seconds=14)
 protocol['intervention']='Only process-local DRI_PRIME PCI visibility; same native/observer sequence as ex8-o'
 protocol['device_visibility']=dict(variable='DRI_PRIME',rule='pci-DDDD_BB_DD_F! from assigned UUID PCI only',layer='/usr/share/vulkan/implicit_layer.d/VkLayer_MESA_device_select.json',reference='https://docs.mesa3d.org/envvars.html#dri-prime',native_rendering_settings_unchanged=True,requires_new_scene_validation=True)
 protocol['retry_policy']='One separately frozen PCI-visibility attempt; no in-attempt retry'
 protocol['criterion']=dict(stable_samples_per_phase=3,required_phases=15,raw_durable_packets=True,native_prewarm_required=True,assigned_PCI=True,no_sampled_peer=True,terminal_scientific_worker_claim_agreement=True,native_constructor_return=0)
 protocol['staged_validation']=dict(L0_static=True,L1_CPU=True,L1_GPU=False,L2_L3_native=False,L4_recovery=False)
 env=dict(old['required_environment']);env.update(TORCH_EXTENSIONS_DIR=str(F/'cache/extensions'),TMPDIR=str(F/'tmp'));protocol['required_environment']=env
 protocol['cost']=dict(separate_diagnostic_max_seconds=180,separate_diagnostic_max_gpu_hours=180/3600,previous_registered_reserved_seconds=old['cost']['combined_future_reserved_seconds'],combined_reserved_upper_seconds=old['cost']['combined_future_reserved_seconds']+180,combined_reserved_upper_gpu_hours=(old['cost']['combined_future_reserved_seconds']+180)/3600,full_stage_ex8d_ceiling_gpu_hours_unchanged=280,scene_validation_reserve_seconds_unchanged=9000,existing_caps_changed=False,basis='Separate bounded diagnostic under existing reproduction authorization; conservative cap sum, not measured consumption; no agent planning estimate treated as user cap')
 write(F/'protocol.json',protocol,paths)
 job=dict(id=J,status='ready',scientific_identity=S.name+'/native-instance-boundary/attempt1',max_seconds=180,authorization_evidence=str(F/'protocol.json'),authorization_sha256=sha(F/'protocol.json'),pause_paths=paths,host_commands=dict(gpu01=dict(argv=['/usr/bin/python3',str(F/'code/run_probe.py')],admission_argv=['/usr/bin/python3',str(F/'code/run_probe.py'),'--check-admission'],cwd=str(F),input_sha256={str(F/'protocol.json'):sha(F/'protocol.json')},outputs_absent=[str(F/'result.json'),str(F/'observer-config.json'),str(F/'journal')])))
 write(F/'dispatcher-job.staged.json',job,paths)
 guard(paths);q=subprocess.run(job['host_commands']['gpu01']['admission_argv'],check=True,text=True,capture_output=True,timeout=10,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
 write(F/'admission-cpu.json',dict(passed=True,stdout=q.stdout,native_executed=False,GPU_seconds=0),paths)
 guard(paths);target=N/'jobs'/(J+'.json');temp=N/'jobs'/(J+'.pending');assert not target.exists()
 with temp.open('xb') as f:f.write((F/'dispatcher-job.staged.json').read_bytes());f.flush();os.fsync(f.fileno())
 guard(paths);os.link(temp,target);guard(paths);temp.unlink()
 print(json.dumps(dict(runtime=str(F),job=str(target),source_commit=commit,queued=True,max_seconds=180)))
if __name__=='__main__':main()
