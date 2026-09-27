"""One bounded job on an idle L40S; never stops another user's process."""
import datetime,fcntl,json,os,subprocess,traceback
from pathlib import Path
ROOT=Path('/vol13/zifanz4/openwam-experiments');O=ROOT/'results/goalaux-20260922'
def write(name,x): (O/name).write_text(json.dumps(x,indent=2)+'\n')
def main():
 p=json.loads((O/'matched-protocol.json').read_text());now=datetime.datetime.now().astimezone();deadline=datetime.datetime.fromisoformat(p['deadline']);assert now<deadline
 for q in [ROOT/'paused.json',ROOT/'results/layers-20260922/paused.json',O/'paused.json',Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')]:assert not q.exists(),q
 lock=(ROOT/'gpu-cm002-5.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 u=json.loads((O/'gpu-preflight/relocation-v2.json').read_text())['gpu_uuid'];assert not (O/'training-started.json').exists()
 s=subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).strip();mem,util,ecc=map(int,s.split(','));assert mem<100 and util==0 and ecc==0,s
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert u not in apps
 (O/'training-launch-claim').mkdir();seconds=min(1800,int((deadline-now).total_seconds()));assert seconds>0
 write('training-launch.json',{'started_at':now.isoformat(),'fixed_study_deadline':p['deadline'],'wall_seconds_cap':seconds,'host':'cm002','gpu_uuid':u,'prelaunch_gpu_state':s,'cpu_threads':4,'models':6,'steps_each':2000,'policy_rollouts':0})
 env=os.environ.copy();env.update(WAM_ROOT=str(ROOT),CUDA_VISIBLE_DEVICES=u,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 with (O/'training.log').open('w') as f:
  run=subprocess.run(['timeout','--signal=TERM','--kill-after=20s',str(seconds),str(ROOT/'env/bin/python'),'-u',str(ROOT/'scripts/goalaux/train_matched.py')],env=env,stdout=f,stderr=subprocess.STDOUT)
 return run.returncode
if __name__=='__main__':
 try:code=main()
 except Exception:
  code=1;write('training-launch-error.json',{'time':datetime.datetime.now().astimezone().isoformat(),'traceback':traceback.format_exc()});traceback.print_exc()
 (O/'training-exit-code.txt').write_text(str(code)+'\n');raise SystemExit(code)
