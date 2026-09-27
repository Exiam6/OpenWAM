import datetime,fcntl,json,os,subprocess,traceback
from pathlib import Path
R=Path('/vol13/zifanz4/openwam-experiments');B=R/'results/goalaux-20260922';O=B/'native-preflight'
def write(n,d):(O/n).write_text(json.dumps(d,indent=2)+'\n')
def main():
 p=json.loads((O/'protocol.json').read_text());now=datetime.datetime.now().astimezone();deadline=datetime.datetime.fromisoformat(p['deadline']);assert now<deadline
 for q in [B/'paused.json',R/'paused.json',Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')]:assert not q.exists(),q
 lock=(R/'gpu-cm002-5.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);u=p['gpu_uuid']
 s=subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).strip();a,b,c=map(int,s.split(','));assert a<100 and b==0 and c==0,s
 assert u not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
 (O/'launch-claim').mkdir();seconds=min(600,int((deadline-now).total_seconds()));assert seconds>0
 write('launch.json',{'time':now.isoformat(),'wall_cap':seconds,'gpu_uuid':u,'resource_state':s,'cpu_threads':4})
 env=os.environ.copy();env.update(WAM_ROOT=str(R),CUDA_VISIBLE_DEVICES=u,EXPECTED_RENDER_PCI=p['pci'],ROBOTWIN_PATH=str(R/'goalaux-native/RoboTwin'),ROBOTWIN_RUNTIME_ROOT=str(O/'runtime'),ROBOTWIN_ENABLE_PLANNER_FALLBACK='1',TORCH_EXTENSIONS_DIR=str(R/'goalaux-native/torch_extensions'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
 with (O/'run.log').open('w') as f:run=subprocess.run(['timeout','--signal=TERM','--kill-after=20s',str(seconds),str(R/'goalaux-native/benchmark-env/bin/python'),'-u',str(R/'scripts/goalaux/native_preflight.py')],env=env,stdout=f,stderr=subprocess.STDOUT)
 return run.returncode
if __name__=='__main__':
 try:code=main()
 except Exception:code=1;write('failure.json',{'time':datetime.datetime.now().astimezone().isoformat(),'traceback':traceback.format_exc()})
 (O/'exit-code.txt').write_text(str(code)+'\n');raise SystemExit(code)
