"""One bounded native expert cohort; original study deadline never reset."""
import datetime,fcntl,json,os,subprocess,traceback
from pathlib import Path
R=Path('/vol13/zifanz4/openwam-experiments');B=R/'results/goalaux-20260922';O=B/'fresh-v4'
def write(n,d):(O/n).write_text(json.dumps(d,indent=2)+'\n')
def main():
 p=json.loads((B/'matched-protocol.json').read_text());now=datetime.datetime.now().astimezone();study_deadline=datetime.datetime.fromisoformat(p['deadline']);assert now<study_deadline
 for q in [B/'paused.json',R/'paused.json',R/'results/layers-20260922/paused.json',Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')]:assert not q.exists(),q
 assert json.loads((B/'native-preflight/result.json').read_text())['passed'];assert json.loads((B/'readouts/audit.json').read_text())['passed']
 lock=(R/'gpu-cm002-5.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu=json.loads((B/'native-preflight/protocol.json').read_text());u=gpu['gpu_uuid']
 s=subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).strip();a,b,c=map(int,s.split(','));assert a<100 and b==0 and c==0,s
 assert u not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
 (O/'launch-claim').mkdir();assert not (O/'launch.json').exists();deadline=min(datetime.datetime.fromisoformat(json.loads((B/'fresh/launch.json').read_text())['collection_deadline']),study_deadline);seconds=int((deadline-now).total_seconds());assert seconds>0
 write('launch.json',{'started_at':now.isoformat(),'collection_deadline':deadline.isoformat(),'study_deadline':p['deadline'],'wall_cap_seconds':seconds,'gpu_uuid':u,'resource_state':s,'cpu_threads':4,'expert_only':True,'max_seeds_per_task':40,'scenes_per_task':20,'watchdog_cleanup_grace_seconds':30})
 env=os.environ.copy();env.update(WAM_ROOT=str(R),CUDA_VISIBLE_DEVICES=u,EXPECTED_RENDER_PCI=gpu['pci'],ROBOTWIN_PATH=str(R/'goalaux-native/RoboTwin'),ROBOTWIN_ENABLE_PLANNER_FALLBACK='1',TORCH_EXTENSIONS_DIR=str(R/'goalaux-native/torch_extensions'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
 with (O/'run.log').open('w') as f:
  run=subprocess.run(['timeout','--signal=TERM','--kill-after=10s',str(seconds+30),str(R/'goalaux-native/benchmark-env/bin/python'),'-u',str(R/'scripts/goalaux/collect_confirmation_v4.py')],env=env,stdout=f,stderr=subprocess.STDOUT)
 return run.returncode
if __name__=='__main__':
 try:code=main()
 except Exception:code=1;write('launch-error.json',{'time':datetime.datetime.now().astimezone().isoformat(),'traceback':traceback.format_exc()})
 (O/'exit-code.txt').write_text(str(code)+'\n');raise SystemExit(code)
