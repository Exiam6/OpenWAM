#!/usr/bin/env bash
set -euo pipefail
cd /data02/zifanz4/openwam-experiments
export WAM_ROOT="$PWD" LAYER_RUN="$PWD/results/layers-20260922"
export CUDA_VISIBLE_DEVICES=GPU-025b7fc9-8f25-5467-5c28-8e59884ab6d3
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Atomic directory prevents a second launcher. Never overwrite an earlier exit.
mkdir "$LAYER_RUN/confirmation-launch-claim"
trap 'printf "%s\n" "$?" > "$LAYER_RUN/confirmation-exit-code.txt"' EXIT
python3 - <<'PY'
import datetime,hashlib,json,os,time
from pathlib import Path
r=Path(os.environ['LAYER_RUN']);freeze=json.loads((r/'confirmation-freeze.json').read_text())
for name,h in freeze['source_sha256'].items():assert hashlib.sha256((Path(os.environ['WAM_ROOT'])/name).read_bytes()).hexdigest()==h,name
assert json.loads((r/'confirmation-parity.json').read_text())['passed']
# Waiting consumes no GPU slot; includes a short grace for collector exit flush.
deadline=datetime.datetime.fromisoformat('2026-09-22T15:26:23.153134-05:00')
while not (r/'fresh/continuation-exit-code.txt').exists():
 if datetime.datetime.now().astimezone()>=deadline:raise TimeoutError('Fixed fresh collection deadline elapsed; no scoring incomplete set')
 time.sleep(15)
assert (r/'fresh/continuation-exit-code.txt').read_text().strip()=='0','Fresh collector failed: preserve partial data, do not score'
PY
policy-env/bin/python -u scripts/layers/confirmation_gates.py identity
exec 9>gpu-cm001-2.lock
flock -w 60 9
python3 - <<'PY'
import subprocess,os,json,datetime,time
from pathlib import Path
u=os.environ['CUDA_VISIBLE_DEVICES']
for attempt in range(10):
 assert u not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
 a,b,c=map(int,subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).split(','));assert a<100 and c==0,(a,b,c)
 if b==0:break
 time.sleep(2)
else:raise RuntimeError('GPU utilization did not settle within20seconds')
p=Path(os.environ['LAYER_RUN']);deadline=datetime.datetime.fromisoformat(json.loads((p/'launch.json').read_text())['deadline']);assert datetime.datetime.now().astimezone()<deadline
assert not (p/'confirmation/summary.json').exists()
(p/'confirmation-started.json').write_text(json.dumps({'started_at':datetime.datetime.now().astimezone().isoformat(),'gpu_uuid':u,'max_seconds':7200},indent=2)+'\n')
PY
remaining=$(python3 -c 'import datetime,json,os; from pathlib import Path; d=datetime.datetime.fromisoformat(json.loads((Path(os.environ["LAYER_RUN"])/"launch.json").read_text())["deadline"]); print(max(1,min(7200,int((d-datetime.datetime.now().astimezone()).total_seconds()))))')
timeout --signal=TERM --kill-after=20s "$remaining" policy-env/bin/python -u scripts/layers/confirm_fresh.py
policy-env/bin/python -u scripts/layers/audit_confirmation.py
