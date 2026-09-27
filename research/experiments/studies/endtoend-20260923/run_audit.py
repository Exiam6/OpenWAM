from pathlib import Path
import datetime,os,subprocess,json
s=Path(__file__).parent
with (s/'audit.log').open('x') as log:
 r=subprocess.run(['timeout','--signal=TERM','--kill-after=20s','1800','/data02/zifanz4/openwam-experiments/policy-env/bin/python','-u',str(s/'audit_confirmation.py')],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
(s/'audit-exit.json').write_text(json.dumps({'finished_at':datetime.datetime.now().astimezone().isoformat(),'exit_code':r.returncode},indent=2)+'\n')
