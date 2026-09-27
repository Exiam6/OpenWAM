"""Bounded, GPU-locked experiment worker; no scheduler or recursive launching."""
import datetime, fcntl, hashlib, json, os, socket, subprocess, sys
from pathlib import Path
R=Path('/data02/zifanz4/openwam-experiments');E=R/'endtoend-20260923'
jpath=Path(sys.argv[1]);job=json.loads(jpath.read_text());out=Path(job['run_dir'])
assert out.is_relative_to(E) and not (out/'exit.json').exists()
assert socket.gethostname().split('.')[0]==job['host']
assert 0<job['timeout_seconds']<=86400
for p in [Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json'),Path('/home/zifanz4/openwam-experiments/studies/endtoend-20260923/paused.json')]:assert not p.exists(),p
for p,want in job['code_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==want,p
out.mkdir(parents=True,exist_ok=True)
def write(name,d):
 p=out/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def now():return datetime.datetime.now().astimezone().isoformat()
code=1
try:
 lock=open(R/f'gpu-{job["host"]}-{job["gpu_index"]}.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 u=job['gpu_uuid'];row=subprocess.check_output(['nvidia-smi','-i',str(job['gpu_index']),'--query-gpu=uuid,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).strip().split(',')
 assert row[0].strip()==u and int(row[1])<100 and int(row[2])==0 and int(row[3])==0,row
 assert u not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
 import shutil
 assert shutil.disk_usage(E).free>40*2**30,'less than40GiBfree'
 write('start.json',{'time':now(),'pid':os.getpid(),'job':job,'gpu_check':row})
 env=os.environ.copy();deps=R/'results/layers-20260922/native-preflight/deps'
 env.update(CUDA_VISIBLE_DEVICES=u,PYTHONPATH=f'{deps}:{E}/OpenWAM',OPENWAM_RUN_DIR=str(out),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MAX_JOBS='4',WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',TORCH_EXTENSIONS_DIR=str(E/'torch_extensions'),TRITON_CACHE_DIR=str(E/'triton'/job['name']),PATH=f'{deps}/bin:'+env['PATH'])
 cmd=['timeout','--signal=TERM','--kill-after=30s',str(job['timeout_seconds']),str(R/'policy-env/bin/python'),'-m','torch.distributed.run','--standalone','--nproc_per_node=1',job.get('entrypoint','/home/zifanz4/openwam-experiments/scripts/endtoend/train_native.py'),'--config-path',str(out),'--config-name','config']
 if job.get('kind')=='deploy-check':cmd=['timeout','--signal=TERM','--kill-after=30s',str(job['timeout_seconds']),str(R/'policy-env/bin/python'),'/home/zifanz4/openwam-experiments/scripts/endtoend/check_native_deploy.py','--route',job['route']]
 with (out/'run.log').open('x') as f:code=subprocess.run(cmd,env=env,cwd=E/'OpenWAM',stdout=f,stderr=subprocess.STDOUT).returncode
except Exception as e:
 write('launcher-error.json',{'time':now(),'error':repr(e)})
 raise
finally:write('exit.json',{'time':now(),'returncode':code})
sys.exit(code)
