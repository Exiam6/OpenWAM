#!/usr/bin/env bash
set -euo pipefail
cd /data02/zifanz4/openwam-experiments
export CUDA_VISIBLE_DEVICES=GPU-c83b7d25-753e-2eae-f13b-8f9c3df9fa4b
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME="$PWD/cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec 9>gpu-cm001-0.lock
flock -n 9
python3 - <<'PY'
import os,subprocess
u=os.environ['CUDA_VISIBLE_DEVICES']
a,b,c=map(int,subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).split(','))
assert a<100 and b==0 and c==0,(a,b,c)
assert u not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
PY
finish() { printf '%s\n' "$?" > results/robustness-20260921/adapter-exit-code.txt; }
trap finish EXIT
timeout --signal=TERM --kill-after=20s 35m policy-env/bin/python -u scripts/train_restoration.py
