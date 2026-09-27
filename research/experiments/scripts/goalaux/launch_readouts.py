import datetime,fcntl,json,os,subprocess,traceback
from pathlib import Path
R=Path('/vol13/zifanz4/openwam-experiments');B=R/'results/goalaux-20260922';O=B/'readouts'
def write(name,obj):(O/name).write_text(json.dumps(obj,indent=2)+'\n')
def main():
 p=json.loads((O/'protocol.json').read_text());now=datetime.datetime.now().astimezone();deadline=datetime.datetime.fromisoformat(p['deadline']);assert now<deadline
 for q in [B/'paused.json',R/'paused.json',Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')]:assert not q.exists(),q
 lock=(R/'gpu-cm002-5.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);u=json.loads((B/'gpu-preflight/relocation-v2.json').read_text())['gpu_uuid']
 s=subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).strip();a,b,c=map(int,s.split(','));assert a<100 and b==0 and c==0,s
 assert u not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
 (O/'launch-claim').mkdir();seconds=min(900,int((deadline-now).total_seconds()));assert seconds>0
 write('launch.json',{'started_at':now.isoformat(),'deadline':p['deadline'],'wall_cap':seconds,'gpu_uuid':u,'host':'cm002','resource_state':s,'cpu_threads':4,'fresh_scoring':False})
 env=os.environ.copy();env.update(WAM_ROOT=str(R),CUDA_VISIBLE_DEVICES=u,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 with (O/'run.log').open('w') as f:
  run=subprocess.run(['timeout','--signal=TERM','--kill-after=20s',str(seconds),str(R/'env/bin/python'),'-u',str(R/'scripts/goalaux/fit_readouts.py')],env=env,stdout=f,stderr=subprocess.STDOUT)
 return run.returncode
if __name__=='__main__':
 try:code=main()
 except Exception:code=1;write('launch-error.json',{'time':datetime.datetime.now().astimezone().isoformat(),'traceback':traceback.format_exc()})
 (O/'exit-code.txt').write_text(str(code)+'\n');raise SystemExit(code)
