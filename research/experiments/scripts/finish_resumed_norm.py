"""Wait for the untouched final arm, then run the frozen complete audit."""
import json,subprocess,time
from pathlib import Path
ROOT=Path('/data02/zifanz4/openwam-experiments');REPO=Path('/home/zifanz4/openwam-experiments');OUT=ROOT/'results/norm-20260921'
def status(row):
 p=OUT/'postprocess-status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2));tmp.replace(p)
status({'status':'waiting_for_resumed_final_arm','started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'original_interruption':'interruption-20260921T1627/postprocess-status.json'})
try:
 deadline=time.monotonic()+4*3600
 while not (OUT/'resume-exit-code.txt').exists():
  if time.monotonic()>deadline:raise TimeoutError('Final-arm completion did not arrive within four hours')
  time.sleep(10)
 assert (OUT/'resume-exit-code.txt').read_text().strip()=='0','Resumed final arm failed; preserve logs'
 assert (OUT/'study-exit-code.txt').read_text().strip()=='0'
 subprocess.run([str(ROOT/'policy-env/bin/python'),str(REPO/'scripts/finish_norm_study.py')],check=True)
except Exception as e:
 status({'status':'failed','error':str(e),'time':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
 raise
