"""Two finite lanes: finish matched training, audit, then paired closed-loop eval."""
import datetime,json,subprocess,sys,time,fcntl,hashlib
from pathlib import Path
from run_owned import guard,write
P=Path('/home/zifanz4/openwam-experiments');S=P/'studies/temporal-20260926';N=Path('/data02/zifanz4/openwam-experiments/temporal-20260926');SCRIPT=P/'scripts/temporal'
def read(p):return json.loads(p.read_text())
def main(arm):
 out=N/'lanes'/arm;out.mkdir(parents=True,exist_ok=False);write(out/'started.json',{'time':datetime.datetime.now().astimezone().isoformat(),'arm':arm});error=None
 try:
  frozen=read(S/'evaluation-code-manifest.json')
  for file,expected in frozen.items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==expected,file
  # Seed42 is already running; never launch it again.
  for seed in [42,43,44]:
   guard();run=N/'training'/f'{arm}-seed{seed}';job=S/f'{arm}-seed{seed}-job.json'
   if seed!=42:
    assert not (run/'runner-started.json').exists();rc=subprocess.call([sys.executable,str(SCRIPT/'run_owned.py'),str(job)]);assert rc==0
   while not (run/'exit.json').exists():guard();time.sleep(20)
   assert read(run/'exit.json')['returncode']==0,(arm,seed)
  write(out/'training-complete.json',{'time':datetime.datetime.now().astimezone().isoformat()})
  # All six complete before any test outcome or parameter choice.
  while not all((N/'lanes'/a/'training-complete.json').exists() for a in ['mean','learned']):
   guard()
   for a in ['mean','learned']:
    if (N/'lanes'/a/'exit.json').exists():assert read(N/'lanes'/a/'exit.json')['returncode']==0
   time.sleep(20)
  if arm=='mean':
   from audit_training import audit
   audit()
  while not (S/'training-audit.json').exists():
   guard();other=N/'lanes/mean/exit.json'
   if other.exists():assert read(other)['returncode']==0
   time.sleep(20)
  assert read(S/'training-audit.json')['passed']
  for seed in [42,43,44]:
   guard();job=S/f'eval-{arm}-seed{seed}-job.json';assert not (Path(read(job)['run_dir'])/'runner-started.json').exists()
   rc=subprocess.call([sys.executable,str(SCRIPT/'run_owned.py'),str(job)]);assert rc==0
  write(out/'evaluation-complete.json',{'time':datetime.datetime.now().astimezone().isoformat(),'episodes':540})
  if arm=='mean':
   while not (N/'lanes/learned/evaluation-complete.json').exists():
    guard();other=N/'lanes/learned/exit.json'
    if other.exists():assert read(other)['returncode']==0
    time.sleep(20)
   rc=subprocess.call(['/data02/zifanz4/openwam-experiments/policy-env/bin/python',str(SCRIPT/'audit_results.py')]);assert rc==0
 except BaseException as e:error=repr(e)
 write(out/'exit.json',{'time':datetime.datetime.now().astimezone().isoformat(),'returncode':1 if error else 0,'error':error})
 return 1 if error else 0
if __name__=='__main__':sys.exit(main(sys.argv[1]))
