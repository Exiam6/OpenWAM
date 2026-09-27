import datetime,json,os,subprocess,sys,time
from pathlib import Path
from run_owned import guard,write
S=Path(__file__).resolve().parent.parent;B=S/'scripts';plan=json.loads((S/'recovery-plan.json').read_text());O=Path(plan['output'])
def main(arm):
 out=O/'lanes'/arm;out.mkdir(parents=True,exist_ok=False);write(out/'started.json',{'time':datetime.datetime.now().astimezone().isoformat(),'pid':os.getpid()});start=time.monotonic();error=None;os.environ["OPENWAM_RECOVERY_LANE_DEADLINE"]=str(time.time()+plan["max_lane_seconds"])
 try:
  for seed in [42,43,44]:
   guard();assert time.monotonic()-start<plan['max_lane_seconds']
   rc=subprocess.call([sys.executable,str(B/'run_owned.py'),str(S/f'eval-{arm}-seed{seed}-job.json')]);assert rc==0,('evaluation failed; no automatic replay',arm,seed)
  write(out/'complete.json',{'episodes':540})
  if arm=='mean':
   other=O/'lanes/learned'
   while not (other/'complete.json').exists():
    guard()
    if (other/'exit.json').exists():assert json.loads((other/'exit.json').read_text())['returncode']==0
    time.sleep(20)
   rc=subprocess.call(['/data02/zifanz4/openwam-experiments/policy-env/bin/python',str(B/'audit_results.py')]);assert rc==0

 except BaseException as exc:error=repr(exc)
 write(out/'exit.json',{'time':datetime.datetime.now().astimezone().isoformat(),'returncode':1 if error else 0,'error':error})
if __name__=='__main__':main(sys.argv[1])
