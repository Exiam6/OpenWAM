"""Freeze committed diagnostic, without GPU dispatch or library loading."""
import datetime as dt,hashlib,json,math,pathlib,shutil,subprocess,sys
from prewarm import native_environment
R=pathlib.Path('/home/zifanz4/openwam-runtime');N=R/'nightly-gpu';J=R/'ex8-j_instancesplit_1021';F=R/'ex8-k_glfwprefix_1021';S=pathlib.Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 old=read(J/'protocol.json');base=read(R/'ex8-d_nativefull_1021/protocol.json');pauses=old['pause_paths']+[str(F/'paused.json')]
 def guard():assert not any(pathlib.Path(x).exists() for x in pauses)
 def write(p,v):
  guard()
  with p.open('x') as h:h.write(json.dumps(v,indent=2)+'\n')
 guard();assert not F.exists();assert not subprocess.check_output(['git','-C',str(S),'status','--porcelain','--',str(S)],text=True)
 sync=read(R/'official-sync/latest.json');assert sync['passed'] and not sync['changed_files'] and sync['after']==base['official_sync']['after']
 assert 0<=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(sync['time'])).total_seconds()<3600
 assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(base['deadline'])
 for p in [J/'result.json',N/'runs/ex8j-native-instance-split/result.json',N/'claims/ex8j-native-instance-split.json']:assert read(p)['status']=='failed'
 assert all(read(p)['status'] in ['complete','failed'] for p in (N/'claims').glob('ex8*.json'))
 assert not [p for p in (N/'jobs').glob('ex8*.json') if not (N/'claims'/p.name).exists()]
 manifest=read(N/'held-for-render-validation-20261021/manifest.json');held=[]
 for e in manifest['entries']:
  p=pathlib.Path(e['held']);assert sha(p)==e['sha256'] and not pathlib.Path(e['original']).exists();held.append(read(p))
 assert len(held)==97
 paths=sorted(p for p in (N/'runs').glob('ex8*/result.json') if p.parent.name.startswith(('ex8d-','ex8e-','ex8f-','ex8g-','ex8h-','ex8i-','ex8j-')))
 terminal=[read(p) for p in paths]
 assert all(x['status'] in ['complete','failed'] and x.get('finished') and math.isfinite(x['occupied_parent_seconds']) and x['occupied_parent_seconds']>0 for x in terminal)
 used=sum(math.ceil(x['occupied_parent_seconds']) for x in terminal);caps=sum(x['max_seconds'] for x in held);total=used+caps+18749+100+9000
 assert total<=280*3600
 cost=dict(terminal_rounded_seconds=used,held_caps_seconds=caps,conditional_blocks_recovery_seconds=18749,new_probe_cap_seconds=100,conditional_native_validation_seconds=9000,total_reserved_seconds=total,total_reserved_gpu_hours=total/3600,headroom_seconds=280*3600-total)
 guard();F.mkdir();guard();(F/'code').mkdir();inputs={}
 for src in sorted(S.iterdir()):
  if src.suffix in ['.py','.md','.json','.cpp','.so']:
   guard();dst=F/'code'/src.name;shutil.copyfile(src,dst);inputs[str(dst)]=sha(dst)
 for name,digest in old['input_sha256'].items():
  if '/ex8-j_instancesplit_1021/' not in name:
   assert sha(name)==digest,name;inputs[name]=digest
 for p in paths+[J/'protocol.json',J/'result.json',N/'claims/ex8j-native-instance-split.json',N/'jobs/ex8j-native-instance-split.json']:
  inputs[str(p)]=sha(p)
 cache=read(S/'cache-provenance.json');guard();shutil.copytree(pathlib.Path(cache['copy']),F/'cache/extensions',copy_function=shutil.copy2)
 assert not list((F/'cache/extensions').rglob('lock'))
 for relative,digest in cache['files'].items():
  p=F/'cache/extensions'/relative;assert sha(p)==digest;inputs[str(p)]=digest
 checks=[]
 for directory in sorted((F/'cache/extensions').iterdir()):
  c=subprocess.run([str(R/'benchmark-env/bin/ninja'),'-n','-C',str(directory)],check=True,text=True,capture_output=True);assert 'no work to do' in c.stdout;checks.append(dict(directory=str(directory),stdout=c.stdout))
 env=dict(old['required_environment'],**native_environment(R,F))
 p=dict(kind='FROZEN_NATIVE_GLFW_PREFIX',name=F.name,registered_for_dispatch=True,scientific_registration_complete=True,registered=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip(),official_sync=sync,compatibility='Official782e3133 unchanged. Direct unchanged first two native Instance GLFW calls; checkpoint delay/prefix exposure diagnostic only. Native library unchanged; exact source build revision unproven. No scene or policy settings change.',deadline=base['deadline'],max_seconds=100,probe_alarm_seconds=75,planned_episodes=0,scored_episodes=0,input_sha256=inputs,pause_paths=pauses,required_environment=env,cost=cost,cache_reuse=dict(provenance=cache,dry_runs=checks,GPU_validated=False,prewarm_omitted=False),criterion=dict(native_prewarm_success=True,glfw_init=1,glfw_vulkan_supported=1,minimum_complete_samples_per_phase=3,assigned_context_observed=True,no_sampled_peer_context=True),retry_policy='None; unique immutable claim',publication='Local registry only',staged_validation=dict(L0_static=True,L1_CPU=True,L1_GPU=False,L2_L3_native=False,L4_recovery=False),full_endpoint_complete=False,full_paper_reproduced=False)
 write(F/'protocol.json',p);argv=['/usr/bin/python3',str(F/'code/run_probe.py')];jid='ex8k-native-glfw-prefix'
 job=dict(id=jid,status='ready',scientific_identity=F.name+'/native-glfw-prefix/attempt1',max_seconds=100,authorization_evidence=str(F/'protocol.json'),authorization_sha256=sha(F/'protocol.json'),pause_paths=pauses,host_commands={'gpu01':dict(argv=argv,admission_argv=argv+['--check-admission'],cwd=str(F),input_sha256={str(F/'protocol.json'):sha(F/'protocol.json')},outputs_absent=[str(F/'result.json')])})
 sys.path.insert(0,str(N/'code-v3'));from policy import validate_job
 validate_job(job);assert not (N/'jobs'/f'{jid}.json').exists() and not (N/'claims'/f'{jid}.json').exists()
 write(F/'prepared-job.json',job);write(F/'registry.json',dict(name=F.name,baseline='ex8-j_instancesplit_1021 failed',setup='Native prewarm then unchanged glfwInit / glfwVulkanSupported',purpose='Distinguish GLFW prefix from later native Instance operations',protocol_sha256=sha(F/'protocol.json'),job_sha256=sha(F/'prepared-job.json'),submitted=False,full_release_allowed=False))
 print(json.dumps(dict(runtime=str(F),cost=cost,prepared=True,submitted=False)))
if __name__=='__main__':main()
