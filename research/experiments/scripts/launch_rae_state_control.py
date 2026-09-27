import datetime,fcntl,hashlib,json,os,subprocess
from pathlib import Path
r=Path('/data02/zifanz4/openwam-experiments');b=r/'results/rae-state-control-20260923';p=json.loads((b/'protocol.json').read_text());deadline=datetime.datetime.fromisoformat(p['deadline']);assert datetime.datetime.now().astimezone()<deadline
assert not (b/'launch.json').exists(),'No automatic rerun'
for d in [b,r/'results/layers-20260922',Path('/home/zifanz4/openwam-experiments/studies/layers-20260922'),Path('/home/zifanz4/openwam-experiments/studies/rae-state-control-20260923'),Path('/home/zifanz4/.local/state/openwam-selfcheck')]:assert not (d/'paused.json').exists(),d
freeze=json.loads((b/'freeze.json').read_text());assert hashlib.sha256((b/'protocol.json').read_bytes()).hexdigest()==freeze['protocol_sha256']
for name,h in freeze['files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
asset=r/'assets/wan22-vae-baseline/Wan2.2_VAE.pth';h=hashlib.sha256()
with asset.open('rb') as f:
 for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
assert h.hexdigest()=='20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36'
(b/'asset-check.json').write_text(json.dumps({'Wan_sha256':h.hexdigest(),'checked_at':datetime.datetime.now().astimezone().isoformat()},indent=2)+'\n')
uuid='GPU-d13929da-086d-1be5-6414-dbf6aeded08e';assert os.uname().nodename=='glacier';lock=(r/'results/gpu-glacier-7.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
query=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=uuid,name,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).strip();fields=[x.strip() for x in query.split(',')];assert fields[0]==uuid and int(fields[2])<100 and int(fields[3])==0 and int(fields[4])==0,query
assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
seconds=min(p['max_wall_seconds'],int((deadline-datetime.datetime.now().astimezone()).total_seconds())-5);assert seconds>30
info={'started_at':datetime.datetime.now().astimezone().isoformat(),'host':os.uname().nodename,'gpu':query,'timeout_seconds':seconds,'deadline':p['deadline'],'protocol_sha256':freeze['protocol_sha256']};(b/'launch.json').write_text(json.dumps(info,indent=2)+'\n')
env=dict(os.environ,WAM_ROOT=str(r),LAYER_RUN=str(r/'results/layers-20260922'),PYTHONHOME=str(r/'assets/python310-runtime'),PYTHONPATH=str(r/'env/lib/python3.10/site-packages'),CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
with (b/'run.log').open('x') as log:result=subprocess.run(['timeout','--signal=TERM','--kill-after=20s',str(seconds)+'s',str(r/'env/bin/python'),'-u',str(r/'scripts/rae_state_control.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
end={'finished_at':datetime.datetime.now().astimezone().isoformat(),'exit_code':result.returncode};(b/'exit.json').write_text(json.dumps(end,indent=2)+'\n');print(json.dumps(dict(info,**end)))
