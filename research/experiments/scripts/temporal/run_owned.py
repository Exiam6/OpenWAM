"""Finite job with checked GPU ownership, frozen code and terminal status."""
import datetime,fcntl,hashlib,json,os,shutil,signal,socket,subprocess,sys,time
from pathlib import Path
R=Path('/data02/zifanz4/openwam-experiments');N=R/'temporal-20260926';P=Path('/home/zifanz4/openwam-experiments');S=P/'studies/temporal-20260926';SCRIPT=P/'scripts/temporal'
def now():return datetime.datetime.now().astimezone()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def guard():
 assert now()<datetime.datetime.fromisoformat(json.loads((S/'protocol.json').read_text())['deadline'])
 for p in [S/'paused.json',Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')]:assert not p.exists(),p

def main(path):
 job=json.loads(Path(path).read_text());out=Path(job['run_dir']);out.mkdir(exist_ok=True,parents=True);assert not (out/'runner-started.json').exists();write(out/'runner-started.json',{'time':now().isoformat(),'pid':os.getpid(),'host':socket.gethostname(),'job':job})
 proc=None;lock=None;code=1;error=None;started=time.monotonic()
 try:
  guard();assert socket.gethostname().split('.')[0]==job['host'];frozen=json.loads(Path(job['manifest']).read_text())
  for file,expected in frozen.items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==expected,file
  lock=(R/f'gpu-{job["host"]}-{job["gpu_index"]}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for observation in range(3):
   guard();row=subprocess.check_output(['nvidia-smi','-i',str(job['gpu_index']),'--query-gpu=uuid,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).strip().split(', ')
   assert row[0]==job['gpu_uuid'] and int(row[1])<100 and int(row[2])==0 and int(row[3])==0,row
   assert job['gpu_uuid'] not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
   if observation<2:time.sleep(10)
  assert shutil.disk_usage(N).free>32*2**30
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=job['gpu_uuid'],PYTHONPATH=f'{R}/results/layers-20260922/native-preflight/deps:{N}/OpenWAM',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',MAX_JOBS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',OPENWAM_RUN_DIR=str(out),TORCH_EXTENSIONS_DIR=str(R/'cache/torch_extensions'),TRITON_CACHE_DIR=str(out/'triton'))
  with (out/'run.log').open('x') as log:
   proc=subprocess.Popen(job['command'],env=env,cwd=N/'OpenWAM',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   write(out/'child.json',{'time':now().isoformat(),'pid':proc.pid,'command':job['command'],'gpu_uuid':job['gpu_uuid']});begin=time.monotonic()
   while proc.poll() is None:
    guard();assert time.monotonic()-begin<job['timeout_seconds'],'Owned job timeout';time.sleep(5)
   code=proc.returncode
 except BaseException as exc:error=repr(exc)
 finally:
  if proc is not None and proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=20)
  write(out/'exit.json',{'time':now().isoformat(),'returncode':code,'error':error,'seconds':time.monotonic()-started,'child_returncode':proc.returncode if proc is not None else None})
  if lock is not None:lock.close()
 return code
if __name__=='__main__':sys.exit(main(sys.argv[1]))
