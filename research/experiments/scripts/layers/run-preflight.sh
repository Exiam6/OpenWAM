#!/usr/bin/env bash
set -euo pipefail
cd /vol13/zifanz4/openwam-experiments
export WAM_ROOT="$PWD" LAYER_RUN="$PWD/results/layers-20260922"
export CUDA_VISIBLE_DEVICES=GPU-0c486005-5f03-f4da-5f3a-e67def845364
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$LAYER_RUN"
exec 9>gpu-cm004-0.lock
flock -n 9
python3 - <<'PY'
import subprocess,os
u=os.environ['CUDA_VISIBLE_DEVICES']
s=subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True)
a,b,c=map(int,s.split(','));assert a<100 and b==0 and c==0,s
assert u not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
PY
trap 'printf "%s\n" "$?" > "$LAYER_RUN/preflight-exit-code.txt"' EXIT
timeout --signal=TERM --kill-after=20s 15m env/bin/python -u scripts/layers/preflight.py
