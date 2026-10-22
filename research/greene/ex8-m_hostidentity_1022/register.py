"""Freeze one separately preregistered alias-corrected attempt; no old mutations."""
import datetime as dt,hashlib,json,os,pathlib,shutil,subprocess
S=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/home/zifanz4/openwam-runtime');F=R/S.name;N=R/'nightly-gpu';J='ex8m-native-host-identity';OLD=R/'ex8-l_headlessprefix_1022'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def guard(paths):assert not any(pathlib.Path(x).exists() for x in paths),'Paused'
def write(p,v,paths):
 guard(paths)
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 old=read(OLD/'protocol.json');paths=old['pause_paths']+[str(F/'paused.json')];guard(paths)
 assert not F.exists() and not (N/'jobs'/(J+'.json')).exists() and not (N/'claims'/(J+'.json')).exists()
 assert read(N/'claims/ex8l-native-headless-prefix.json')['status']=='failed'
 assert read(OLD/'result.json')['status']=='failed' and not (OLD/'observer-config.json').exists()
 assert all(sha(p)==v for p,v in old['input_sha256'].items()),'Old inputs drifted'
 assert all(read(p)['status'] in ['complete','failed'] for p in (N/'claims').glob('ex8*.json'))
 official=read(R/'official-sync/latest.json');assert official['passed'] and official['after']=='782e3133f674449e2986754cfc5f7c4e204835a2'
 assert (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(official['time'])).total_seconds()<600
 policy=read(N/'policy.json');hostnames={'gpu01':policy['hosts']['gpu01']['hostname']}
 import socket
 from host_identity import checked_host
 checked_host(socket.gethostname(),'gpu01',hostnames)
 assert read(S/'CPU_VALIDATION.json')['passed'] and read(S/'CPU_VALIDATION.json')['tests_per_interpreter']==34
 commit=subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()
 assert not subprocess.check_output(['git','-C',str(S),'status','--porcelain'],text=True).strip()
 guard(paths);F.mkdir();shutil.copytree(S,F/'code');(F/'tmp').mkdir();(F/'cache').mkdir()
 guard(paths);shutil.copyfile(OLD/'code/libheadless.so',F/'code/libheadless.so')
 guard(paths);shutil.copytree(OLD/'cache/extensions',F/'cache/extensions')
 dry=[]
 for d in sorted((F/'cache/extensions').iterdir()):
  if (d/'build.ninja').exists():
   q=subprocess.run([str(R/'benchmark-env/bin/ninja'),'-n','-C',str(d)],check=True,text=True,capture_output=True,timeout=4)
   assert 'no work to do' in q.stdout;dry.append(q.stdout)
 inputs={str(p):v for p,v in old['input_sha256'].items() if not str(p).startswith(str(OLD)+'/')}
 for p in list((F/'code').glob('*'))+list((F/'cache/extensions').rglob('*')):
  if p.is_file():inputs[str(p)]=sha(p)
 for p in [OLD/'protocol.json',OLD/'result.json',N/'policy.json',N/'claims/ex8l-native-headless-prefix.json',N/'runs/ex8l-native-headless-prefix/result.json']:
  inputs[str(p)]=sha(p)
 p=dict(old);p.update(name=S.name,source_commit=commit,plan_commit='6ec2611b',registered=dt.datetime.now(dt.timezone.utc).isoformat(),pause_paths=paths,input_sha256=inputs,hostnames=hostnames,official_sync=official,compatibility='Official unchanged782e3133; same original native helper/ELF/planner/inference and observer closure; only frozen alias/FQDN identity validation corrected, diagnostic tracebacks preserved.',baseline_failed_attempt=str(OLD),cache_dry_runs=dry)
 env=dict(old['required_environment']);env.update(TORCH_EXTENSIONS_DIR=str(F/'cache/extensions'),TMPDIR=str(F/'tmp'));p['required_environment']=env
 p['cost']=dict(separate_diagnostic_max_seconds=150,existing_reserved_seconds=1008078,combined_future_reserved_seconds=1008228,combined_future_reserved_gpu_hours=1008228/3600,all_including_pilots_gpu_hours=(1008228+3073)/3600,full_stage_ex8d_ceiling_gpu_hours_unchanged=280,scene_validation_reserve_seconds_unchanged=9000,existing_caps_changed=False,basis='Separate bounded150s diagnostic scope under existing reproduction authorization; no user cap inferred from agent planning allocation')
 assert p['max_seconds']==150 and p['probe_alarm_seconds']==70 and p['planned_episodes']==0
 write(F/'protocol.json',p,paths)
 job=dict(id=J,status='ready',scientific_identity=S.name+'/native-headless-prefix/attempt1',max_seconds=150,authorization_evidence=str(F/'protocol.json'),authorization_sha256=sha(F/'protocol.json'),pause_paths=paths,host_commands=dict(gpu01=dict(argv=['/usr/bin/python3',str(F/'code/run_probe.py')],admission_argv=['/usr/bin/python3',str(F/'code/run_probe.py'),'--check-admission'],cwd=str(F),input_sha256={str(F/'protocol.json'):sha(F/'protocol.json')},outputs_absent=[str(F/'result.json'),str(F/'observer-config.json'),str(F/'journal')])))
 write(F/'dispatcher-job.staged.json',job,paths)
 guard(paths);q=subprocess.run(job['host_commands']['gpu01']['admission_argv'],check=True,text=True,capture_output=True,timeout=10,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
 write(F/'admission-cpu.json',dict(passed=True,stdout=q.stdout,native_executed=False,GPU_seconds=0),paths)
 guard(paths);target=N/'jobs'/(J+'.json');temp=N/'jobs'/(J+'.pending');assert not target.exists()
 with temp.open('xb') as f:f.write((F/'dispatcher-job.staged.json').read_bytes());f.flush();os.fsync(f.fileno())
 guard(paths);os.link(temp,target);temp.unlink()
 print(json.dumps(dict(runtime=str(F),job=str(target),source_commit=commit,queued=True,max_seconds=150)))
if __name__=='__main__':main()
