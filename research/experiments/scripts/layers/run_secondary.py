"""Bounded once-only CPU follow-up to the unchanged primary confirmation."""
import datetime,hashlib,json,os,subprocess,time
from pathlib import Path
R=Path(os.environ['LAYER_RUN']);root=Path(os.environ['WAM_ROOT']);(R/'secondary-launch-claim').mkdir();deadline=datetime.datetime.fromisoformat(json.loads((R/'launch.json').read_text())['deadline']);freeze=json.loads((R/'secondary-freeze.json').read_text());status={};overall=1
try:
 for p,h in freeze['source_sha256'].items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,p
 while not (R/'confirmation-exit-code.txt').exists():
  if datetime.datetime.now().astimezone()>=deadline:raise TimeoutError('Original study deadline reached')
  time.sleep(15)
 assert (R/'confirmation-exit-code.txt').read_text().strip()=='0','Primary audit not successful; do not score secondary endpoints'
 end=min(time.monotonic()+600,time.monotonic()+(deadline-datetime.datetime.now().astimezone()).total_seconds())
 for name in ['describe_confirmation.py','score_wrist.py','score_initial_goal.py']:
  with (R/(name[:-3]+'.log')).open('w') as out:
   try:result=subprocess.run([str(root/'policy-env/bin/python'),'-u',str(root/'scripts/layers'/name)],stdout=out,stderr=subprocess.STDOUT,timeout=max(1,end-time.monotonic()));code=result.returncode
   except subprocess.TimeoutExpired:code=124
  status[name]=code;(R/(name[:-3]+'-exit-code.txt')).write_text(str(code)+'\n');print(name,code,flush=True)
 overall=0 if all(v==0 for v in status.values()) else 1
finally:
 (R/'secondary-exit-code.txt').write_text(str(overall)+'\n');(R/'secondary-status.json').write_text(json.dumps({'completed_at':datetime.datetime.now().astimezone().isoformat(),'stages':status,'exit_code':overall},indent=2)+'\n')
raise SystemExit(overall)
