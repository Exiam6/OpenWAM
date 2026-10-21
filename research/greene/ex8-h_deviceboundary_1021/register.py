"""Freeze committed ex8-h; never run a GPU or submit a job here."""
import datetime as dt,hashlib,json,math,pathlib,shutil,subprocess,sys
from prewarm import native_environment
R=pathlib.Path('/home/zifanz4/openwam-runtime');N=R/'nightly-gpu';S=pathlib.Path(__file__).resolve().parent;F=R/'ex8-h_deviceboundary_1021'
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def accounting(terminal, held, recovery=18749, new=300):
 assert terminal and len(held)==97
 assert all(x['status'] in ['complete','failed'] and x.get('finished') and math.isfinite(x['occupied_parent_seconds']) and x['occupied_parent_seconds']>0 for x in terminal)
 used=sum(math.ceil(x['occupied_parent_seconds']) for x in terminal)
 caps=sum(x['max_seconds'] for x in held)
 total=used+caps+recovery+new+9000
 assert total<=280*3600,'Registered280GPUh ceiling exceeded'
 return dict(terminal_rounded_seconds=used,held_caps_seconds=caps,conditional_blocks_recovery_seconds=recovery,new_probe_cap_seconds=new,conditional_native_validation_seconds=9000,total_reserved_seconds=total,total_reserved_gpu_hours=total/3600,headroom_seconds=280*3600-total)
def main():
 old=read(R/'ex8-g_prewarmaudit_1021/protocol.json');base=read(R/'ex8-d_nativefull_1021/protocol.json')
 pauses=list(dict.fromkeys(old['pause_paths']+[str(F/'paused.json')]))
 def guard():assert not any(pathlib.Path(x).exists() for x in pauses)
 def write(path,value):
  guard()
  with path.open('x') as handle:handle.write(json.dumps(value,indent=2)+'\n')
 guard();assert not F.exists()
 sync=read(R/'official-sync/latest.json')
 assert sync['passed'] and not sync['changed_files'] and sync['after']==base['official_sync']['after']
 assert (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(sync['time'])).total_seconds()<3600
 assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(base['deadline'])
 assert read(R/'ex8-g_prewarmaudit_1021/result.json')['status']=='failed'
 assert read(N/'runs/ex8g-native-prewarm-audit/result.json')['status']=='failed'
 claims=[read(p) for p in (N/'claims').glob('ex8*.json')]
 assert all(x['status'] in ['complete','failed'] for x in claims),'Active/uncertain native claim'
 assert not [p for p in (N/'jobs').glob('ex8*.json') if not (N/'claims'/p.name).exists()],'Other queued native work'
 manifest=read(N/'held-for-render-validation-20261021/manifest.json');held=[]
 for entry in manifest['entries']:
  path=pathlib.Path(entry['held']);assert sha(path)==entry['sha256'];assert not pathlib.Path(entry['original']).exists()
  held.append(read(path))
 terminal_paths=sorted(p for p in (N/'runs').glob('ex8*/result.json') if p.parent.name.startswith(('ex8d-','ex8e-','ex8f-','ex8g-','ex8h-')))
 terminal=[read(p) for p in terminal_paths];cost=accounting(terminal,held)
 names=sorted(p.name for p in S.iterdir() if p.suffix in ['.py','.md','.json','.cpp','.h'])
 assert not subprocess.check_output(['git','-C',str(S),'status','--porcelain','--',str(S)],text=True),'Commit plan and code before freeze'
 guard();F.mkdir();guard();(F/'code').mkdir()
 inputs={}
 for name in names:
  guard();target=F/'code'/name;shutil.copyfile(S/name,target);inputs[str(target)]=sha(target)
 # Keep exact native/package evidence without importing any GPU library.
 for name,digest in old['input_sha256'].items():
  if '/ex8-g_prewarmaudit_1021/code/' not in name:
   assert sha(name)==digest,name;inputs[name]=digest
 extra=[R/'benchmark-env/lib/python3.10/site-packages/sapien.libs/libsapien.so',R/'benchmark-env/lib/python3.10/site-packages/sapien.libs/libsvulkan2.so',R/'benchmark-env/lib/python3.10/site-packages/sapien/include/sapien/device.h',R/'benchmark-env/lib/python3.10/site-packages/sapien/__init__.pyi',R/'ex8-g_prewarmaudit_1021/protocol.json',R/'ex8-g_prewarmaudit_1021/result.json',N/'held-for-render-validation-20261021/manifest.json',R/'benchmark-env/bin/ninja',R/'cuda-toolkit/bin/nvcc']+terminal_paths
 for path in extra:inputs[str(path)]=sha(path)
 env=dict(old['required_environment'],**native_environment(R,F))
 p=dict(kind='FROZEN_NATIVE_DEVICE_BOUNDARY',name=F.name,registered_for_dispatch=True,scientific_registration_complete=True,registered=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip(),official_sync=sync,compatibility='Unchanged official782e3133; installed Device overload split from RenderSystem; native binaries and all prior runtimes unchanged',deadline=base['deadline'],max_seconds=300,probe_alarm_seconds=210,planned_episodes=0,scored_episodes=0,input_sha256=inputs,pause_paths=pauses,required_environment=env,cost=cost,criterion=dict(native_prewarm_success=True,explicit_and_default_PCI_match=True,minimum_complete_samples_per_phase=3,assigned_context_observed=True,no_sampled_peer_context=True),retry_policy='None; unique immutable claim',publication='Local registry only',staged_validation=dict(L0_static=True,L1_CPU=True,L1_GPU=False,L2_L3_native=False,L4_recovery=False),full_endpoint_complete=False,full_paper_reproduced=False)
 write(F/'protocol.json',p)
 argv=['/usr/bin/python3',str(F/'code/run_probe.py')];jid='ex8h-native-device-boundary'
 job=dict(id=jid,status='ready',scientific_identity=F.name+'/native-device-boundary/attempt1',max_seconds=300,authorization_evidence=str(F/'protocol.json'),authorization_sha256=sha(F/'protocol.json'),pause_paths=pauses,host_commands={'gpu01':dict(argv=argv,admission_argv=argv+['--check-admission'],cwd=str(F),input_sha256={str(F/'protocol.json'):sha(F/'protocol.json')},outputs_absent=[str(F/'result.json')])})
 sys.path.insert(0,str(N/'code-v3'));from policy import validate_job
 validate_job(job);assert not (N/'jobs'/f'{jid}.json').exists() and not (N/'claims'/f'{jid}.json').exists()
 write(F/'prepared-job.json',job)
 write(F/'registry.json',dict(name=F.name,baseline='ex8-g_prewarmaudit_1021 failed',setup='Native prewarm then Device(cuda:0), then RenderSystem(device=Device)',purpose='Distinguish discovery from render-system construction as earliest observed peer-context boundary',protocol_sha256=sha(F/'protocol.json'),job_sha256=sha(F/'prepared-job.json'),submitted=False,full_release_allowed=False))
 print(json.dumps(dict(runtime=str(F),cost=cost,prepared=True,submitted=False)))
if __name__=='__main__':main()
