"""Read-only evidence from live native processes; never dump full environments."""
import argparse, datetime as dt, hashlib, json, os, pathlib, sys
R=pathlib.Path('/home/zifanz4/openwam-runtime')
S=pathlib.Path('/home/zifanz4/.local/state/openwam-selfcheck')
sys.path.insert(0,str(R/'ex8-d_nativefull_1021/code'))
from run_full import classify_full, score

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def unpaused():
 assert not (S/'paused.json').exists(), 'Global pause'
 for name in ['ex8-d_nativefull_1021','ex8-e_rowrecovery_1021']:
  assert not any(pathlib.Path(p).exists() for p in read(R/name/'protocol.json')['pause_paths']), 'Stage pause'

def validate_environment(env, uuid, output, group, worker_group):
 assert env.get('CUDA_VISIBLE_DEVICES')==uuid, 'Wrong GPU UUID'
 assert group==worker_group, 'Process outside supervising cgroup'
 assert env.get('TMPDIR')==str(output/'tmp'), 'Wrong temporary directory'
 assert env.get('ROBOTWIN_SKIP_VERSION_CHECK')!='1', 'Version check bypassed'
 for key in ['ROBOTWIN_TEST_NUM','ROBOTWIN_ENABLE_PLANNER_FALLBACK','ROBOTWIN_LABTASKER_OPERATION']:
  assert not env.get(key), key
 return {k:env[k] for k in ['CUDA_VISIBLE_DEVICES','TMPDIR','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','ROBOTWIN_RUNTIME_ROOT','POLICY_CONFIG_PATH'] if k in env}

def process(pid):
 p=pathlib.Path('/proc')/str(pid)
 assert p.stat().st_uid==os.getuid(), 'Only inspect our processes'
 args=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
 env=dict(x.decode().split('=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x)
 return args,env,(p/'cgroup').read_text()

def audit():
 unpaused(); inventory=read(R/'nightly-gpu/latest.json')
 assert (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(inventory['time'])).total_seconds()<30
 checks=[]
 for claim in inventory['claims']:
  if claim['status']!='running' or not claim['id'].startswith('ex8d-full-'): continue
  assert claim['host']=='gpu01', 'This witness reads local /proc only'
  spec=read(R/'nightly-gpu/jobs'/(claim['id']+'.json'))
  command=spec['host_commands']['gpu01']['argv']; key=command[command.index('--key')+1]
  root=pathlib.Path(spec['authorization_evidence']).parent; out=root/'runs'/key
  state=read(out/'result.json'); commands=read(out/'commands.json')
  worker=read(R/'nightly-gpu/runs'/claim['id']/'result.json')
  worker_group=process(worker['pid'])[2]
  gpu=inventory['hosts']['gpu01'][str(claim['gpu_index'])]
  assert gpu['uuid']==claim['gpu_uuid']==state['gpu_uuid']
  matching=[]
  for proc in gpu['processes']:
   if proc.get('user')!=pathlib.Path.home().name: continue
   if proc['name'] not in [str(R/'policy-env/bin/python'),str(R/'benchmark-env/bin/python')]: continue
   argv,env,group=process(proc['pid'])
   selected=validate_environment(env,claim['gpu_uuid'],out,group,worker_group)
   role='policy' if argv[0]==str(R/'policy-env/bin/python') else 'benchmark'
   if role=='policy': assert argv==commands['server'], 'Server invocation drift'
   else:
    assert argv[1]==str(R/'ex8-a_officialrepro_1021/official/benchmarks/robotwin/eval_policy_wrapper.py')
    for flag,value in [('--task_name',state['task']),('--task_config',state['mode']),('--seed','0')]:
     assert argv[argv.index(flag)+1]==value, flag
   matching.append(dict(pid=proc['pid'],role=role,argv=argv,environment=selected,cgroup=group.strip()))
  assert sorted(x['role'] for x in matching)==['benchmark','policy'], 'Both native processes required'
  server=(out/'server.log').read_text(errors='replace'); client=(out/'client.log').read_text(errors='replace')
  markers=['steps=10 denoise_mode=sync','DiT velocity cache enabled (threshold=0.990)','Enabled lazy Wan block compile','Enabled dual self-attn MoT compile',"camera_layout=['head_camera', 'left_camera', 'right_camera']",'action 80D -> raw 20D']
  assert all(m in server for m in markers), 'Missing native inference confirmation'
  classified=classify_full(client,state['task']); assert not classified['fatal'], classified
  episodes=score(client); assert episodes[:len(state['episodes'])]==state['episodes']
  checks.append(dict(job_id=claim['id'],key=key,row_status=state['status'],observed_episodes=len(episodes),server_markers=markers,classifier=classified,processes=matching,client_log_sha256=sha(out/'client.log'),server_log_sha256=sha(out/'server.log'),log_digest_note='Live logs can grow after this read; digests are snapshot evidence only.'))
 for check in checks:
  for proc in check['processes']:
   proc['gpu_contexts']=[dict(host=host,index=int(index),type=context['type'],memory_mib=context['memory_mib']) for host,devices in inventory['hosts'].items() for index,gpu in devices.items() for context in gpu['processes'] if host=='gpu01' and context['pid']==proc['pid']]
 return dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),kind='CPU_ONLY_LIVE_NATIVE_EXECUTION_WITNESS',inventory_time=inventory['time'],checks=checks,new_gpu_hours=0,full_endpoint_complete=False,full_paper_reproduced=False,source_sha256=sha(pathlib.Path(__file__)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
 assert a.output.resolve().is_relative_to((S/'evidence').resolve())
 result=audit();unpaused();tmp=a.output.with_suffix('.pending');tmp.write_text(json.dumps(result,indent=2)+'\n');unpaused();tmp.replace(a.output)
 print(json.dumps(dict(checked_jobs=len(result['checks']),observed_episodes={x['key']:x['observed_episodes'] for x in result['checks']})))
