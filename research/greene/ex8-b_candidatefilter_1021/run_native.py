"""Bounded missing-group recovery. Only the outer error classifier changes."""
from classify_log import classify
import argparse,datetime as dt,hashlib,json,math,os,pathlib,re,signal,socket,subprocess,time,shutil
R=pathlib.Path('/home/zifanz4/openwam-runtime');ROOT=R/'ex8-b_candidatefilter_1021'
def read(p):return json.loads(p.read_text())
def write(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def admit():
 p=read(ROOT/'protocol.json');assert not (ROOT/'pilot').exists();assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(p['deadline']);assert shutil.disk_usage(ROOT).free>80*(1<<30)
 for name,digest in p['input_sha256'].items():assert sha(pathlib.Path(name))==digest,name
 assert (ROOT/'RoboTwin/assets/background_texture/unseen').is_dir()
 return p
def stop(child):
 if child is None:return
 try:os.killpg(child.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:child.wait(timeout=12)
 except subprocess.TimeoutExpired:
  try:os.killpg(child.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  child.wait(timeout=10)
def finite(v):
 if isinstance(v,list):return all(finite(x) for x in v)
 return isinstance(v,(int,float)) and math.isfinite(v)
def main():
 p=admit();out=ROOT/'pilot';out.mkdir();started=time.monotonic();server=client=None
 state=dict(status='preflight',started=dt.datetime.now(dt.timezone.utc).isoformat(),protocol_sha256=sha(ROOT/'protocol.json'),groups=[],full_paper_reproduced=False)
 def save():write(out/'result.json',dict(state,occupied_seconds=time.monotonic()-started))
 def guard():
  assert time.monotonic()-started<p['max_seconds']-120,'Pilot budget exhausted'
  assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(os.environ['OPENWAM_ALLOCATION_END'])-dt.timedelta(seconds=120)
  assert shutil.disk_usage(ROOT).free>60*(1<<30)
  if server is not None:assert server.poll() is None,'Native server exited'
 def terminated(*_):raise RuntimeError('Resource interruption; preserve partial evidence')
 signal.signal(signal.SIGTERM,terminated);signal.signal(signal.SIGINT,terminated);save()
 try:
  assert sha(pathlib.Path(p['checkpoint']))==p['checkpoint_sha256']
  env=dict(os.environ);env.update(PYTHONPATH=str(ROOT/'official'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',MAX_JOBS='2',TORCHINDUCTOR_COMPILE_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',CUDA_HOME=str(R/'cuda-toolkit'),TORCH_CUDA_ARCH_LIST='12.0',CPATH=str(R/'cuda-toolkit/targets/x86_64-linux/include'),TMPDIR=str(ROOT/'tmp'),TORCH_EXTENSIONS_DIR=str(ROOT/'cache/extensions'),TRITON_CACHE_DIR=str(ROOT/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(ROOT/'cache/inductor'),PATH=f'{ROOT}/bin:{R}/policy-env/bin:{R}/cuda-toolkit/bin:/usr/bin:/bin')
  for f in ['tmp','cache/extensions','cache/triton','cache/inductor']:(ROOT/f).mkdir(parents=True,exist_ok=True)
  for key in ['ROBOTWIN_ENABLE_PLANNER_FALLBACK','ROBOT_CONFIG_ROOT','EXPECTED_RENDER_PCI','ROBOTWIN_TEST_NUM']:env.pop(key,None)
  with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
  server_argv=[str(R/'policy-env/bin/python'),'-u',str(ROOT/'official/scripts/deploy.py'),'--ckpt-dir',str(pathlib.Path(p['checkpoint']).parent),'--host','127.0.0.1','--port',str(port)]
  write(out/'commands.json',dict(server=server_argv,port=port));state['status']='loading_native_server';save()
  with (out/'server.log').open('x') as log:server=subprocess.Popen(server_argv,cwd=ROOT/'official',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  tick=time.monotonic()
  while 'WebSocket server started' not in (out/'server.log').read_text(errors='replace'):
   guard();assert time.monotonic()-tick<1800,'Native server readiness timeout';time.sleep(2)
  state['server_ready']=dt.datetime.now(dt.timezone.utc).isoformat();save()
  # Debug is the official client's passive recording option. Scientific settings stay native.
  template=(ROOT/'official/benchmarks/robotwin/policy_config.yml').read_text()
  for mode in ['demo_randomized']:
   for task in p['pilot_tasks']:
    guard();dest=out/f'{mode}-{task}';dest.mkdir();debug=dest/'policy-inputs';config=dest/'policy.yml';config.write_text(template.replace('debug: false','debug: true').replace('debug_dir: "./debug_images"','debug_dir: '+json.dumps(str(debug))))
    clientenv=dict(env,ROBOTWIN_PATH=str(ROOT/'RoboTwin'),ROBOTWIN_PYTHON=str(R/'benchmark-env/bin/python'),ROBOTWIN_RUNTIME_ROOT=str(dest/'simulator'),ROBOTWIN_TEST_NUM=str(p['pilot_episodes_per_group']),POLICY_CONFIG_PATH=str(config),PYTHONPATH=f'{ROOT}/official:{ROOT}/RoboTwin/envs/curobo/src',PATH=f'{ROOT}/bin:{R}/benchmark-env/bin:{R}/cuda-toolkit/bin:/usr/bin:/bin',LD_LIBRARY_PATH=str(R/'benchmark-env/lib/python3.10/site-packages/sapien/oidn_library'))
    argv=['/bin/bash',str(ROOT/'official/benchmarks/robotwin/single_eval.sh'),task,mode,'ex8a-native-pilot',env['CUDA_VISIBLE_DEVICES'],str(port),'127.0.0.1'];write(dest/'command.json',dict(argv=argv,overrides=dict(debug=True,episodes=p['pilot_episodes_per_group'],fallback=False)))
    state.update(status='native_evaluation',active_group=f'{mode}/{task}');save();tick=time.monotonic()
    with (dest/'client.log').open('x') as log:client=subprocess.Popen(argv,cwd=ROOT/'official',env=clientenv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    while client.poll() is None:
     guard();text=(dest/'client.log').read_text(errors='replace');assert not classify(text)['fatal'],'Unrecognized native technical error; inspect preserved client log';time.sleep(3)
    assert client.returncode==0,'Native client failed';assert not classify((dest/'client.log').read_text(errors='replace'),final=True)['fatal'];client=None;text=re.sub(r'\x1b\[[0-9;]*m','',(dest/'client.log').read_text(errors='replace'));matches=re.findall(r'Success rate:\s*(\d+)/(\d+).*?current seed:\s*(\d+)',text);assert matches and int(matches[-1][1])==p['pilot_episodes_per_group'],matches
    assert [int(x[1]) for x in matches]==list(range(1,p['pilot_episodes_per_group']+1))
    metas=list(debug.glob('ep*/step_*/meta.json'));assert metas
    for f in metas:
     m=read(f);assert len(m['state'])==20 and finite(m['state']) and finite(m['action']),str(f)
     assert all((f.parent/(camera+'.png')).is_file() for camera in ['head','left','right'])
    videos=list((dest/'simulator/eval_result').rglob('episode*.mp4'));assert len(videos)==p['pilot_episodes_per_group']
    for f in videos:subprocess.run([str(ROOT/'bin/ffmpeg'),'-v','error','-i',str(f),'-f','null','-'],check=True,timeout=60,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    group=dict(task=task,mode=mode,episodes=int(matches[-1][1]),successes=int(matches[-1][0]),accepted_scene_seeds=[int(x[2]) for x in matches],seconds=time.monotonic()-tick,debug_records=len(metas),videos=[str(x) for x in videos],status='complete',upstream_capped_result_denominator_ignored=True)
    group['native_rejected_candidate_exceptions']=classify(text,final=True)['rejected_candidates'];state['groups'].append(group);save()
   if mode=='demo_clean':assert sum(g['successes'] for g in state['groups'])>=1,'No fixed clean pilot success; diagnose before scale-up'
  seconds=sum(g['seconds'] for g in state['groups']);episodes=sum(g['episodes'] for g in state['groups']);state.update(status='pilot_complete',readiness_passed=True,full_paper_reproduced=False,measured_seconds_per_episode=seconds/episodes,rough_full_gpu_hours_excluding_repeated_startup=seconds/episodes*10000/3600,full_dispatch_requires_separate_manifest=True)
 except BaseException as exc:
  state.update(status='failed',readiness_passed=False,error=repr(exc));raise
 finally:
  stop(client);stop(server);state['finished']=dt.datetime.now(dt.timezone.utc).isoformat();save()
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--check-admission',action='store_true');args=parser.parse_args()
 if args.check_admission:admit();print('admitted')
 else:main()
