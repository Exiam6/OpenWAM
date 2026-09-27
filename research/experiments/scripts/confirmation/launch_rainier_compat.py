"""Repair pre-interpreter OS mismatch; preserve prior failures and frozen science."""
import datetime,fcntl,json,os,subprocess
from pathlib import Path
from worker import R,O,guards,now,write,sha

def main():
 guards();assert os.uname().nodename=='rainier';config=json.loads((O/'runtime-compatibility.json').read_text());assert sha(Path(__file__))==config['launcher_sha256']
 bundle=R/'assets/runtime-compat-rainier-20260923';manifest=json.loads((bundle/'manifest.json').read_text());assert sha(R/'env/bin/python')==manifest['python_binary_sha256']
 for name,value in manifest['files'].items():assert sha(bundle/'glibc'/name)==value['sha256']
 index=4;uuid='GPU-38557357-3f58-c864-2f53-8b14087b5e8c';lock=(R/'results/gpu-rainier-4.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 q=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=uuid,name,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total,driver_version','--format=csv,noheader,nounits'],text=True).strip();u,name,mem,util,ecc,driver=[x.strip() for x in q.split(',')]
 assert u==uuid and name=='NVIDIA A40' and int(mem)<100 and int(util)==0 and ecc=='0',q
 assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
 d=O/'features';assert json.loads((d/'exit.json').read_text())['exit_code']==1
 assert not (d/'started.json').exists() and not (d/'hardware-parity.json').exists() and not list(d.glob('*/seed-*')) and not (O/'result.json').exists()
 assert 'GLIBC_2.35' in (d/'run.log').read_text();assert not (O/'feature-attempt-rainier-startup').exists()
 d.rename(O/'feature-attempt-rainier-startup');d.mkdir();(d/'launch-claim').mkdir()
 deadline=datetime.datetime.fromisoformat(json.loads((O/'window.json').read_text())['deadline']);seconds=int((deadline-now()).total_seconds())-20;assert seconds>1800
 write(d/'launch.json',{'started_at':now().isoformat(),'host':'rainier','gpu_query':q,'timeout_seconds':seconds,'deadline':deadline.isoformat(),'runtime_manifest_sha256':sha(bundle/'manifest.json'),'reason':'Prior process failed in dynamic loader before Python; identical interpreter/package/code/weight bytes, private glibc2.35 compatibility, unchanged numerical parity required','failed_attempts_preserved':['feature-attempt-l40s','feature-attempt-rainier-startup']})
 env=dict(os.environ,WAM_ROOT=str(R),LAYER_RUN=str(R/'results/layers-20260922'),PYTHONHOME=str(R/'assets/python310-runtime'),PYTHONPATH=str(R/'env/lib/python3.10/site-packages'),CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 command=[str(bundle/'glibc/ld-linux-x86-64.so.2'),'--library-path',str(bundle/'glibc'),str(R/'env/bin/python'),'-u',str(R/'scripts/confirmation/features.py')]
 with (d/'run.log').open('x') as log:run=subprocess.run(['timeout','--signal=TERM','--kill-after=15s',str(seconds)]+command,env=env,stdout=log,stderr=subprocess.STDOUT)
 write(d/'exit.json',{'finished_at':now().isoformat(),'exit_code':run.returncode});return run.returncode
if __name__=='__main__':
 try:code=main()
 except Exception:
  import traceback
  write(O/'runtime-compatibility-error.json',{'time':now().isoformat(),'traceback':traceback.format_exc()});code=1
 raise SystemExit(code)
