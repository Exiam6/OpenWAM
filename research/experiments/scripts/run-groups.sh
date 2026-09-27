#!/usr/bin/env bash
set -euo pipefail
cd /data02/zifanz4/openwam-experiments
export CUDA_VISIBLE_DEVICES=GPU-c83b7d25-753e-2eae-f13b-8f9c3df9fa4b
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
export HF_HOME=/data02/zifanz4/openwam-experiments/cache/huggingface
exec 9>/data02/zifanz4/openwam-experiments/gpu-cm001-0.lock
flock -n 9
python3 - <<'PY'
import os, subprocess
uid = os.environ['CUDA_VISIBLE_DEVICES']
r = subprocess.check_output(['nvidia-smi','-i',uid,'--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True)
a,b,c = [int(v.strip()) for v in r.split(',')]
assert a < 100 and b == 0 and c == 0, r
assert uid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
PY
mkdir -p results/groups-20260921
trap 'run_exit=$?; printf "%s\n" "$run_exit" > results/groups-20260921/exit-code.txt' EXIT
env/bin/python -u scripts/group_study.py
