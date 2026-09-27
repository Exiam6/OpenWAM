"""Single bounded dependency job; waits without GPU, then gates/extracts/scores."""
import datetime,fcntl,json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/vol13/zifanz4/openwam-experiments');B=R/'results/goalaux-20260922';O=B/'confirmation-v2';F=B/'fresh-v3'
def now():return datetime.datetime.now().astimezone()
def write(n,d):(O/n).write_text(json.dumps(d,indent=2)+'\n')
def unpaused():
 for p in [B/'paused.json',R/'paused.json',R/'results/layers-20260922/paused.json',Path('/home/zifanz4/.local/state/openwam-selfcheck/paused.json')]:assert not p.exists(),p

def main():
 unpaused();plan=json.loads((O/'freeze.json').read_text());end=datetime.datetime.fromisoformat(plan['deadline']);collection_end=datetime.datetime.fromisoformat(json.loads((F/'launch.json').read_text())['collection_deadline']);assert now()<end;(O/'launch-claim').mkdir();write('pending.json',{'time':now().isoformat(),'waiting_for':'full60scene collector exit0','gpu_allocated':False,'study_deadline':end.isoformat()})
 while not (F/'exit-code.txt').exists():
  unpaused();assert now()<min(end,collection_end+datetime.timedelta(seconds=45)),'Original collection window exhausted';time.sleep(15)
 assert (F/'exit-code.txt').read_text().strip()=='0','Collection failed; do not run confirmation';assert json.loads((F/'manifest.json').read_text())['complete'],'Incomplete fixed cohort; no scoring'
 env=os.environ.copy();env.update(WAM_ROOT=str(R),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 python=str(R/'env/bin/python');gpu=json.loads((B/'native-preflight/protocol.json').read_text());u=gpu['gpu_uuid']
 def run(stage,script,args,limit,stage_end):
  unpaused();seconds=min(limit,int((stage_end-now()).total_seconds()));assert seconds>0
  write('progress.json',{'stage':stage,'started_at':now().isoformat(),'time_limit_seconds':seconds,'stage_deadline':stage_end.isoformat()})
  with (O/(stage+'.log')).open('w') as log:r=subprocess.run(['timeout','--signal=TERM','--kill-after=10s',str(seconds),python,'-u',str(R/'scripts/goalaux'/script),*args],env=env,stdout=log,stderr=subprocess.STDOUT)
  (O/(stage+'-exit-code.txt')).write_text(str(r.returncode)+'\n');assert r.returncode==0,f'{stage} failed:{r.returncode}'
 run('identity','confirmation_gates_v2.py',['identity'],180,end)
 lock=(R/'gpu-cm002-5.lock').open('a+');wait_end=min(end,now()+datetime.timedelta(minutes=30))
 while True:
  unpaused();assert now()<wait_end,'No idle device within fixed resource-wait cap'
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:time.sleep(15);continue
  s=subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).strip();a,b,c=map(int,s.split(','));assert c==0,'ECC error; no allocation'
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
  if a<100 and b==0 and u not in apps:break
  fcntl.flock(lock,fcntl.LOCK_UN);time.sleep(15)
 env['CUDA_VISIBLE_DEVICES']=u;stage_end=min(end,now()+datetime.timedelta(hours=1));write('launch.json',{'time':now().isoformat(),'deadline':stage_end.isoformat(),'study_deadline':end.isoformat(),'gpu_uuid':u,'resource_state':s,'gpu_cap':1,'cpu_threads':4,'no_refits':True})
 run('parity','confirmation_gates_v2.py',['parity'],600,stage_end)
 run('extract','score_confirmation_v2.py',['extract'],2400,stage_end)
 run('score','score_confirmation_v2.py',['score'],300,stage_end)
 env['CUDA_VISIBLE_DEVICES']='';run('audit','audit_confirmation_v2.py',[],300,stage_end)
 write('complete.json',{'time':now().isoformat(),'audit_passed':json.loads((O/'audit.json').read_text())['passed'],'original_study_deadline':end.isoformat()});return 0
if __name__=='__main__':
 try:code=main()
 except Exception:code=1;write('failure.json',{'time':now().isoformat(),'traceback':traceback.format_exc(),'completed_stages_not_replayed':True})
 (O/'exit-code.txt').write_text(str(code)+'\n');raise SystemExit(code)
