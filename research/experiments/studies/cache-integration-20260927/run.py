import datetime,fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path
S=Path(__file__).parent;plan=json.loads((S/'protocol.json').read_text());O=Path(plan['output']);R=O.parent
assert not (O/'started.json').exists()
assert not Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json').exists()
assert datetime.datetime.now().astimezone()<datetime.datetime.fromisoformat(plan['deadline'])
assert hashlib.sha256((S/'check_inference.py').read_bytes()).hexdigest()==plan['check_script_sha256']
lock=(R/'gpu-cm001-5.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
row=subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=uuid,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).strip().split(', ')
assert int(row[1])<100 and int(row[2])==0 and int(row[3])==0,row
assert row[0] not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
assert __import__('shutil').disk_usage(R).free>32*2**30
started={'time':datetime.datetime.now().astimezone().isoformat(),'pid':os.getpid(),'gpu':row,'protocol_sha256':hashlib.sha256((S/'protocol.json').read_bytes()).hexdigest()};(O/'started.json').write_text(json.dumps(started,indent=2)+'\n')
env=dict(os.environ,CUDA_VISIBLE_DEVICES=row[0],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(R/'results/layers-20260922/native-preflight/deps')+':'+plan['source'],TORCH_EXTENSIONS_DIR=str(R/'cache/torch_extensions'),TRITON_CACHE_DIR=str(O/'triton'),TMPDIR=str(O))
child=None;rc=1;err=None;begin=time.monotonic()
try:
 with (O/'run.log').open('x') as f:
  child=subprocess.Popen([str(R/'policy-env/bin/python'),'-u',str(S/'check_inference.py')],cwd=plan['source'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  while child.poll() is None:
   assert not Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json').exists()
   assert datetime.datetime.now().astimezone()<datetime.datetime.fromisoformat(plan['deadline'])
   assert time.monotonic()-begin<plan['timeout_seconds']
   time.sleep(2)
  rc=child.returncode
except BaseException as e:err=repr(e)
finally:
 if child and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 (O/'exit.json').write_text(json.dumps({'time':datetime.datetime.now().astimezone().isoformat(),'returncode':rc,'error':err,'seconds':time.monotonic()-begin},indent=2)+'\n')
raise SystemExit(rc)
