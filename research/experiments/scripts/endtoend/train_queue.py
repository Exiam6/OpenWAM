"""Finite predeclared job queue, not a recurring timer."""
import json,subprocess,sys,datetime
from pathlib import Path
E=Path('/data02/zifanz4/openwam-experiments/endtoend-20260923');index=int(sys.argv[1]);assert index in range(1,7)
route=['svae','pca','wan'][(index-1)%3];seeds=[42,44] if index<=3 else [43]
log=E/'training'/f'queue-gpu{index}.json'
for seed in seeds:
 job=E/'training'/f'{route}-seed{seed}'/'job.json'
 assert not job.with_name('exit.json').exists(),'No automatic restart of attempted jobs'
 log.write_text(json.dumps({'time':datetime.datetime.now().astimezone().isoformat(),'status':'running','current':str(job),'ordered_seeds':seeds}))
 code=subprocess.run(['python3','/home/zifanz4/openwam-experiments/scripts/endtoend/run_job.py',str(job)]).returncode
 if code:
  log.write_text(json.dumps({'time':datetime.datetime.now().astimezone().isoformat(),'status':'failed','job':str(job),'returncode':code}));sys.exit(code)
log.write_text(json.dumps({'time':datetime.datetime.now().astimezone().isoformat(),'status':'complete','ordered_seeds':seeds}))
