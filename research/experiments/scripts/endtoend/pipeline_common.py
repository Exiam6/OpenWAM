"""Explicit resources and bounded child ownership for this finite validation pipeline."""
import datetime,fcntl,hashlib,json,os,signal,socket,subprocess,time
from pathlib import Path
R=Path('/data02/zifanz4/openwam-experiments');E=R/'endtoend-20260923';P=Path('/home/zifanz4/openwam-experiments/scripts/endtoend');S=Path('/home/zifanz4/openwam-experiments/studies/endtoend-20260923')
def now():return datetime.datetime.now().astimezone()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def guard():
 for p in [S/'paused.json',Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')]:assert not p.exists(),p
 assert now()<datetime.datetime.fromisoformat(json.loads((E/'protocol.json').read_text())['bounds']['review_horizon'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frozen(files):
 for p,want in files.items():assert sha(p)==want,p

def reserve(index):
 guard();host=socket.gethostname().split('.')[0];lock=(R/f'gpu-{host}-{index}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 row=[x.strip() for x in subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).strip().split(',')]
 uuid,pci,mem,util,ecc=row;assert int(mem)<100 and int(util)==0 and int(ecc)==0,row
 assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
 return lock,row

def environments(row,out):
 uuid,pci,*_=row
 common=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,EXPECTED_RENDER_PCI=pci,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MAX_JOBS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',TORCH_EXTENSIONS_DIR=str(R/'cache/torch_extensions'))
 policy=dict(common,PYTHONPATH=f'{R}/results/layers-20260922/native-preflight/deps:{E}/OpenWAM',TRITON_CACHE_DIR=str(out/'triton'))
 sim=dict(common,WAM_ROOT=str(R),ROBOTWIN_PATH=str(R/'benchmarks/RoboTwin'),ROBOTWIN_RUNTIME_ROOT=str(out/'runtime'),ROBOTWIN_ENABLE_PLANNER_FALLBACK='1');sim.pop('PYTHONPATH',None)
 return policy,sim

def terminate_owned(proc):
 if proc.poll() is not None:return
 os.killpg(proc.pid,signal.SIGTERM)
 try:proc.wait(timeout=20)
 except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=20)

def run_child(cmd,env,log,timeout):
 guard();start=time.monotonic()
 with Path(log).open('x') as f:
  proc=subprocess.Popen(cmd,env=env,cwd=E/'OpenWAM',stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   while proc.poll() is None:
    guard()
    if time.monotonic()-start>=timeout:raise TimeoutError(f'Owned child timeout {timeout}s')
    time.sleep(2)
   return proc.returncode
  finally:terminate_owned(proc)

def start_server(checkpoint,port,out,env):
 # Never connect to, replace, or terminate an unrelated server.
 probe=socket.socket();probe.bind(('127.0.0.1',port));probe.close()
 f=(out/'server.log').open('x');cmd=[str(R/'policy-env/bin/python'),'-u',str(P/'serve_policy.py'),'--checkpoint',str(checkpoint),'--port',str(port),'--log',str(out/'episode-rng.jsonl')]
 proc=subprocess.Popen(cmd,env=env,cwd=E/'OpenWAM',stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 write(out/'owned-server.json',{'pid':proc.pid,'command':cmd,'time':now().isoformat()})
 try:
  deadline=time.monotonic()+600
  while True:
   guard();assert proc.poll() is None,'Native policy server failed during startup'
   if 'ENDTOEND_SERVER_READY' in (out/'server.log').read_text(errors='replace'):break
   assert time.monotonic()<deadline,'Native policy server startup exceeded600s'
   time.sleep(2)
 except BaseException:terminate_owned(proc);f.close();raise
 return proc,f
