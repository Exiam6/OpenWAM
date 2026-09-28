import datetime,fcntl,hashlib,json,os,shutil,signal,socket,subprocess,time,traceback
from resume_state import inventory
from pathlib import Path
R=Path('/home/zifanz4/openwam-runtime');S=R/'temporal-new-20260928/resume-v1';B=S/'scripts';RUN=S/'runs'/datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f');RUN.mkdir(parents=True);O=RUN/'evaluation';M=R/'migration-20260927'
UUID='GPU-a4c607bb-f8f8-b999-19ea-0eb65290bb10';PORT=19565
plan=json.loads((S/'launch-plan.json').read_text());protocol=json.loads((S/'protocol.json').read_text());deadline=datetime.datetime.fromisoformat(plan['deadline'])
def write(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def now():return datetime.datetime.now().astimezone().isoformat()
def save_status():
 status['run_dir']=str(RUN);write(RUN/'status.json',status);write(S/'status.json',status)
def guard():
 assert datetime.datetime.now().astimezone()<deadline,'Original absolute deadline reached'
 assert shutil.disk_usage(R).free>32*2**30,'Disk floor'
 for p in [S/'paused.json',M/'paused.json',M/'study/paused.json',Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')]:assert not p.exists(),str(p)
 ecc=subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).strip();assert int(ecc)==0,ecc
 if seed_started is not None:assert time.monotonic()-seed_started<64800,'Per-seed 18h limit'
def stop(p):
 if p is not None and p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def cancelled(*_):raise RuntimeError('Experiment interrupted; no automatic retry')
def child(cmd,env,log,limit):
 global simulator
 with log.open('x') as f:simulator=subprocess.Popen(cmd,env=env,cwd='/home/zifanz4/OpenWAM',stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 status['simulator_pid']=simulator.pid;save_status();started=time.monotonic()
 try:
  while simulator.poll() is None:
   guard();assert server.poll() is None,'Policy server exited';assert time.monotonic()-started<limit,'Child wall limit';time.sleep(5)
  assert simulator.returncode==0,('Child failed; no automatic retry',str(log),simulator.returncode)
 finally:stop(simulator);simulator=None
O.mkdir(exist_ok=False);server=None;simulator=None;seed_started=None
status={'started':now(),'pid':os.getpid(),'gpu_uuid':UUID,'phase':'admission','completed_rollouts':0,'expected_rollouts':1080,'complete':False}
signal.signal(signal.SIGTERM,cancelled);signal.signal(signal.SIGINT,cancelled)
lock=(R/'gpu-5.lock').open('a');save_status()
try:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 guard()
 for i in range(3):
  row=subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
  assert row[0]==UUID and int(row[1])<100 and int(row[2])==0,row
  assert UUID not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
  if i<2:time.sleep(10)
 manifest=json.loads((S/'code-manifest.json').read_text())
 for p,sha in manifest.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha,p
 cohort=json.loads((S/'fresh-cohort-integrity.json').read_text());assert cohort['passed']
 for p,sha in cohort['files_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha,p
 audit=json.loads((S/'training-audit.json').read_text());assert audit['passed']
 assert json.loads((M/'required-transfer-verified.json').read_text())['all_sha256_verified']
 pci=subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=pci.bus_id','--format=csv,noheader'],text=True).strip()
 sock=socket.socket();sock.bind(('127.0.0.1',PORT));sock.close()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,EXPECTED_RENDER_PCI=pci,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',MAX_JOBS='2',CUDA_HOME=str(R/'cuda-toolkit'),CPATH=str(R/'cuda-toolkit/targets/x86_64-linux/include'),TORCH_CUDA_ARCH_LIST='12.0',TMPDIR=str(R/'tmp'),TORCH_EXTENSIONS_DIR=str(R/'cache/torch_extensions-sm120'),TRITON_CACHE_DIR=str(R/'cache/triton-sm120'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',ROBOTWIN_PATH=str(R/'assets-source/benchmarks/RoboTwin'),ROBOTWIN_ENABLE_PLANNER_FALLBACK='1',WAM_ROOT=str(R/'assets-source'))
 env['LD_LIBRARY_PATH']=str(R/'benchmark-env/lib/python3.10/site-packages/sapien/oidn_library')
 env['PATH']=str(R/'benchmark-env/bin')+':'+str(R/'cuda-toolkit/bin')+':/usr/bin:/bin';env.pop('PYTHONPATH',None)
 for arm,seed in plan['order']:
  counts,pending=inventory();assert not counts['duplicates'],counts['duplicates']
  if not any(v for k,v in pending.items() if k[:2]==(arm,seed)):continue
  seed_started=time.monotonic();guard();out=O/f'{arm}-seed{seed}';out.mkdir();env['ROBOTWIN_RUNTIME_ROOT']=str(out/'runtime')
  status.update(arm=arm,seed=seed,phase='model_loading');save_status()
  cp=Path(audit['seeds'][str(seed)][arm]['checkpoint'].replace('/data02/zifanz4/openwam-experiments',str(R/'assets-source')))
  h=hashlib.sha256()
  with cp.open('rb') as f:
   for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
  assert h.hexdigest()==audit['seeds'][str(seed)][arm]['checkpoint_sha256'],'Checkpoint audit mismatch'
  with (out/'server.log').open('x') as log:server=subprocess.Popen([str(R/'policy-env/bin/python'),'-u',str(B/'serve_policy.py'),'--checkpoint',str(cp),'--port',str(PORT),'--log',str(out/'episode-rng.jsonl')],env=dict(env,PYTHONPATH='/home/zifanz4/OpenWAM'),cwd='/home/zifanz4/OpenWAM',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  status['server_pid']=server.pid;save_status();t=time.monotonic()
  while 'ENDTOEND_SERVER_READY' not in (out/'server.log').read_text(errors='replace'):
   guard();assert server.poll() is None,'Model load failed';assert time.monotonic()-t<600;time.sleep(5)
  dev=out/'development';dev.mkdir();status['phase']='development_gate';save_status()
  child([str(R/'benchmark-env/bin/python'),'-u',str(B/'simulation_preflight.py'),'--port',str(PORT),'--output',str(dev)],env,dev/'run.log',1800)
  result=json.loads((dev/'result.json').read_text());assert result['passed'] and result['tasks'][0]['full_development_rollout']['completed']
  write(S/f'simulation-preflight-{arm}-{seed}.json',{'passed':True,'time':now(),'checkpoint':str(cp),'development_result':str(dev/'result.json'),'code_sha256':manifest})
  for task in protocol['evaluation']['tasks']:
   for condition in protocol['evaluation']['conditions']:
    guard();counts,pending=inventory();selected=pending[(arm,seed,task,condition)]
    if not selected:continue
    dest=out/task/condition;dest.mkdir(parents=True)
    status.update(phase='heldout_evaluation',task=task,condition=condition);save_status()
    try:
     child([str(R/'benchmark-env/bin/python'),'-u',str(B/'evaluate_policy.py'),'--scene-seeds',json.dumps(selected),'--task',task,'--route',arm,'--seed',str(seed),'--condition',condition,'--port',str(PORT),'--output',str(dest)],env,dest/'run.log',12600)
    except AssertionError as exc:
     guard();assert server.poll() is None
     if not (isinstance(exc.args[0],tuple) and exc.args[0][0]=='Child failed; no automatic retry'):raise
     assert list(dest.glob('seed-*')),'Failure before any scene attempt; stop for diagnosis'
     write(dest/'group-failure.json',{'error':repr(exc),'time':now(),'action':'preserve attempted cells and advance queue'})
    counts,_=inventory();status.update(completed_rollouts=counts['completed'],technical_failures=counts['failures_or_interrupted']);save_status()
  write(out/'pass-finished.json',{'scope':'only pending cells attempted; see global inventory for completeness','time':now()});stop(server);server=None
 counts,_=inventory();status.update(complete=counts['completed']==1080,phase='pass_finished',inventory=counts)
except BaseException as e:
 status.update(error=repr(e),phase='stopped');traceback.print_exc()
finally:
 stop(simulator);stop(server);status['finished']=now();save_status();lock.close();print(json.dumps(status),flush=True)
raise SystemExit(0 if status['phase']=='pass_finished' else 1)
