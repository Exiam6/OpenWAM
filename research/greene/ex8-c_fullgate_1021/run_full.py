"""DRAFT ONLY: needs separate frozen L4 registration, audited readiness and budget approval."""
from classify_log import classify
import argparse,datetime as dt,hashlib,json,os,pathlib,re,signal,socket,subprocess,time,shutil
R=pathlib.Path('/home/zifanz4/openwam-runtime');ROOT=R/'ex8-a_officialrepro_1021';FULL=R/'ex8-c_nativefull_1021'
def read(p):return json.loads(p.read_text())
def write(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def stop(child):
 if child is None:return
 try:os.killpg(child.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:child.wait(timeout=12)
 except subprocess.TimeoutExpired:
  try:os.killpg(child.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  child.wait(timeout=10)
def native_environment_guard():
 assert not os.environ.get('ROBOTWIN_LABTASKER_OPERATION'),'Native full run cannot inherit manifest/batched evaluation mode'
 assert os.environ.get('ROBOTWIN_SKIP_VERSION_CHECK')!='1','Native version check must remain enabled'

def validate_full_suite(p):
 base=read(ROOT/'protocol.json')
 expected={(m,t) for m in base['full_modes'] for t in base['full_tasks']}
 rows=p['rows'];assert len(rows)==100 and len({r['key'] for r in rows})==100
 assert {(r['mode'],r['task']) for r in rows}==expected
 assert all(r['episodes']==100 and r['seed']==0 and r['key']==f"{r['mode']}-{r['task']}" for r in rows)
 assert p['planned_episodes']==10000

def admit(key):
 native_environment_guard()
 p=read(FULL/'protocol.json');validate_full_suite(p)
 assert p.get('full_registered_for_dispatch') is True,'Conditional draft is not dispatch authorization'
 row=next(x for x in p['rows'] if x['key']==key)
 assert not (FULL/'runs'/key).exists();assert not (FULL/'paused.json').exists()
 assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(p['deadline'])
 assert shutil.disk_usage(ROOT).free>80*(1<<30)
 assert sha(ROOT/'protocol.json')==p['pilot_protocol_sha256']
 assert p['combined_readiness_passed'] and p['measured_full_gpu_hours']<=p['authorized_gpu_hour_ceiling']
 assert sum(x['max_seconds'] for x in p['rows'])<=3600*p['authorized_gpu_hour_ceiling']
 for name,digest in p['readiness_input_sha256'].items():assert sha(pathlib.Path(name))==digest,name
 assert not pathlib.Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json').exists()
 assert not (ROOT/'paused.json').exists()
 assert not (R/'ex8-b_candidatefilter_1021/paused.json').exists()
 for name,digest in p['input_sha256'].items():assert sha(pathlib.Path(name))==digest,name
 for name,digest in read(ROOT/'protocol.json')['input_sha256'].items():assert sha(pathlib.Path(name))==digest,name
 return p,row
ERRORS=['CUDA/Curobo prewarm failed:','planner fallback engaged:','error occurs !','Traceback (most recent call last):']
def classify_full(text, task, final=False):
 # The sole known caught expert error is permitted only for its actual task.
 if task=='place_object_basket':return classify(text,final=final)
 return dict(rejected_candidates=0,fatal=[x for x in ERRORS if x in text],pending=False)

def score(text):
 text=re.sub(r'\x1b\[[0-9;]*m','',text)
 matches=re.findall(r'Success rate:\s*(\d+)/(\d+).*?current seed:\s*(\d+)',text);rows=[];previous=0
 for s,n,seed in matches:
  s,n,seed=map(int,[s,n,seed]);assert n==len(rows)+1 and s-previous in [0,1];assert not rows or seed>rows[-1]['scene_seed']
  rows.append(dict(episode_index=n-1,scene_seed=seed,success=bool(s-previous)));previous=s
 return rows

def main(key):
 p,row=admit(key);base=read(ROOT/'protocol.json');out=FULL/'runs'/key;out.mkdir(parents=True)
 started=time.monotonic();server=client=None;state=dict(status='preflight',task=row['task'],mode=row['mode'],started=dt.datetime.now(dt.timezone.utc).isoformat(),protocol_sha256=sha(FULL/'protocol.json'),episodes=[],full_paper_reproduced=False,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],host=os.environ['OPENWAM_GPU_HOST'])
 def save():write(out/'result.json',dict(state,occupied_seconds=time.monotonic()-started))
 def guard():
  for pause in [FULL/'paused.json',ROOT/'paused.json',R/'ex8-b_candidatefilter_1021/paused.json',pathlib.Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')]:assert not pause.exists(),'Paused; retain partial attempt'
  assert time.monotonic()-started<row['max_seconds']-120,'Row allocation cap reached; incomplete is not policy failure'
  assert dt.datetime.now(dt.timezone.utc)<min(dt.datetime.fromisoformat(p['deadline']),dt.datetime.fromisoformat(os.environ['OPENWAM_ALLOCATION_END']))-dt.timedelta(seconds=120)
  assert shutil.disk_usage(ROOT).free>60*(1<<30)
  if server is not None:assert server.poll() is None,'Native server exited'
 def terminated(*_):raise RuntimeError('Resource interruption; retain partial attempt')
 signal.signal(signal.SIGTERM,terminated);signal.signal(signal.SIGINT,terminated);save()
 try:
  assert sha(pathlib.Path(base['checkpoint']))==base['checkpoint_sha256']
  env=dict(os.environ);env.update(PYTHONPATH=str(ROOT/'official'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',MAX_JOBS='2',TORCHINDUCTOR_COMPILE_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',CUDA_HOME=str(R/'cuda-toolkit'),TORCH_CUDA_ARCH_LIST='12.0',CPATH=str(R/'cuda-toolkit/targets/x86_64-linux/include'),TMPDIR=str(out/'tmp'),TORCH_EXTENSIONS_DIR=str(out/'cache/extensions'),TRITON_CACHE_DIR=str(out/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(out/'cache/inductor'),PATH=f'{ROOT}/bin:{R}/policy-env/bin:{R}/cuda-toolkit/bin:/usr/bin:/bin')
  for f in ['tmp','cache/triton','cache/inductor','cache/extensions']:(out/f).mkdir(parents=True,exist_ok=True)
  for k in ['ROBOTWIN_ENABLE_PLANNER_FALLBACK','ROBOT_CONFIG_ROOT','EXPECTED_RENDER_PCI','ROBOTWIN_TEST_NUM']:env.pop(k,None)
  with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
  server_argv=[str(R/'policy-env/bin/python'),'-u',str(ROOT/'official/scripts/deploy.py'),'--ckpt-dir',str(pathlib.Path(base['checkpoint']).parent),'--host','127.0.0.1','--port',str(port)]
  clientenv=dict(env,ROBOTWIN_PATH=str(ROOT/'RoboTwin'),ROBOTWIN_PYTHON=str(R/'benchmark-env/bin/python'),ROBOTWIN_RUNTIME_ROOT=str(out/'simulator'),POLICY_CONFIG_PATH=str(ROOT/'official/benchmarks/robotwin/policy_config.yml'),PYTHONPATH=f'{ROOT}/official:{ROOT}/RoboTwin/envs/curobo/src',PATH=f'{ROOT}/bin:{R}/benchmark-env/bin:{R}/cuda-toolkit/bin:/usr/bin:/bin',LD_LIBRARY_PATH=str(R/'benchmark-env/lib/python3.10/site-packages/sapien/oidn_library'))
  argv=['/bin/bash',str(ROOT/'official/benchmarks/robotwin/single_eval.sh'),row['task'],row['mode'],'ex8a-native-full',env['CUDA_VISIBLE_DEVICES'],str(port),'127.0.0.1']
  write(out/'commands.json',dict(server=server_argv,client=argv,env_overrides={k:v for k,v in clientenv.items() if k not in os.environ or os.environ[k]!=v},episode_cap_override=False,passive_debug=False))
  state['status']='loading_native_server';save()
  with (out/'server.log').open('x') as log:server=subprocess.Popen(server_argv,cwd=ROOT/'official',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  tick=time.monotonic()
  while 'WebSocket server started' not in (out/'server.log').read_text(errors='replace'):
   guard();assert time.monotonic()-tick<1800;time.sleep(2)
  state.update(status='native_evaluation',server_ready=dt.datetime.now(dt.timezone.utc).isoformat());save()
  with (out/'client.log').open('x') as log:client=subprocess.Popen(argv,cwd=ROOT/'official',env=clientenv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while client.poll() is None:
   guard();text=(out/'client.log').read_text(errors='replace');assert not classify_full(text,row['task'])['fatal'],'Native technical error; preserve log'
   state['episodes']=score(text);save();time.sleep(3)
  assert client.returncode==0,'Native client failed';client=None;text=(out/'client.log').read_text(errors='replace')
  classified=classify_full(text,row['task'],final=True);assert not classified['fatal']
  state['native_rejected_candidate_exceptions']=classified['rejected_candidates']
  rows=score(text);assert len(rows)==100
  videos=list((out/'simulator/eval_result').rglob('episode*.mp4'));assert len(videos)==100
  assert {f.name for f in videos}=={f'episode{i}.mp4' for i in range(100)}
  for f in videos:
   guard();subprocess.run([str(ROOT/'bin/ffmpeg'),'-v','error','-i',str(f),'-f','null','-'],check=True,timeout=60,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
  scores=list((out/'simulator/eval_result').rglob('_result.txt'));assert len(scores)==1
  native_rate=float(scores[0].read_text().strip().splitlines()[-1]);successes=sum(x['success'] for x in rows);assert abs(native_rate-successes/100)<1e-8
  state.update(status='complete',episodes=rows,successes=successes,scored_episodes=100,success_rate_pct=successes,native_result_crosscheck=True,video_full_decode_passed=100,client_log_sha256=sha(out/'client.log'),video_sha256={str(f):sha(f) for f in videos})
 except BaseException as exc:
  state.update(status='failed',error=repr(exc));raise
 finally:
  stop(client);stop(server);state['finished']=dt.datetime.now(dt.timezone.utc).isoformat();save()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--check-admission',action='store_true');a=p.parse_args()
 if a.check_admission:admit(a.key);print('admitted')
 else:main(a.key)
