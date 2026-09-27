"""Bounded CPU-only postprocessing; never launches more policy jobs or publishes."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
REPO=Path('/home/zifanz4/openwam-experiments');ROOT=Path('/data02/zifanz4/openwam-experiments');OUT=ROOT/'results/norm-20260921'
STATUS=OUT/'postprocess-status.json'
def write(obj):
 tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(STATUS)
def run(argv,**kwargs):subprocess.run(argv,check=True,**kwargs)
write({'status':'waiting_for_rollouts','started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
try:
 deadline=time.monotonic()+10*3600
 while not (OUT/'study-exit-code.txt').exists():
  if time.monotonic()>deadline:raise TimeoutError('Study completion did not arrive within ten hours')
  time.sleep(10)
 code=(OUT/'study-exit-code.txt').read_text().strip()
 if code!='0':raise RuntimeError('GPU study exited '+code+'; preserve all partial results and inspect launcher/client/server logs')
 freeze=json.loads((OUT/'analysis-freeze.json').read_text())
 assert hashlib.sha256((ROOT/freeze['script']).read_bytes()).hexdigest()==freeze['sha256']
 write({'status':'auditing'})
 run([str(ROOT/'policy-env/bin/python'),str(ROOT/'scripts/summarize_norm.py')],cwd=ROOT)
 # Isolated output avoids editing a Site checkout behind an active editor.
 site=OUT/'final-report/dist'
 if site.exists():raise FileExistsError(site)
 shutil.copytree('/home/zifanz4/research-reports/dist',site)
 env={**os.environ,'NORM_SITE_DIR':str(site)}
 run([str(ROOT/'env/bin/python'),str(REPO/'build_norm_report.py')],cwd=REPO,env=env)
 write({'status':'complete_local_report','completed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'summary':str(OUT/'summary.json'),'report':str(site/'norm.html'),'publication':'Final static report built locally; public Site remains an explicitly labeled snapshot until next native deployment.'})
except Exception as e:
 write({'status':'failed','error':str(e),'time':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
 raise
