#!/usr/bin/env bash
set -euo pipefail
cd /vol13/zifanz4/openwam-experiments
export WAM_ROOT="$PWD" LAYER_RUN="$PWD/results/layers-20260922"
export CUDA_VISIBLE_DEVICES=GPU-0c486005-5f03-f4da-5f3a-e67def845364
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec 9>gpu-cm004-0.lock
flock -w 900 9
python3 - <<'PY'
import subprocess,os,json,datetime
from pathlib import Path
u=os.environ['CUDA_VISIBLE_DEVICES'];s=subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True)
a,b,c=map(int,s.split(','));assert a<100 and b==0 and c==0,s
assert u not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
p=Path(os.environ['LAYER_RUN']);deadline=datetime.datetime.fromisoformat(json.loads((p/'launch.json').read_text())['deadline']);assert datetime.datetime.now().astimezone()<deadline
assert (p/'compression-exit-code.txt').read_text().strip()=='0','Compression must have completed successfully'
assert not (p/'readout-exit-code.txt').exists()
PY
trap 'printf "%s\n" "$?" > "$LAYER_RUN/readout-exit-code.txt"' EXIT
remaining=$(python3 -c 'import datetime,json,os; from pathlib import Path; d=datetime.datetime.fromisoformat(json.loads((Path(os.environ["LAYER_RUN"])/"launch.json").read_text())["deadline"]); print(max(1,min(7200,int((d-datetime.datetime.now().astimezone()).total_seconds()))))')
timeout --signal=TERM --kill-after=20s "$remaining" bash -c '
  set -euo pipefail
  env/bin/python -u scripts/layers/readout.py check
  env/bin/python -u scripts/layers/readout.py vectors
  env/bin/python -u scripts/layers/readout.py probes
'
