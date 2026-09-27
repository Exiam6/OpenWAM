#!/usr/bin/env bash
set -euo pipefail
cd /data02/zifanz4/openwam-experiments
export CUDA_VISIBLE_DEVICES=GPU-67b700c2-eb7a-93aa-28dd-57e8b2711926
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export ROBOTWIN_PATH="$PWD/benchmarks/RoboTwin"
export TORCH_EXTENSIONS_DIR="$PWD/cache/torch_extensions"
export ROBOTWIN_ENABLE_PLANNER_FALLBACK=1
mkdir -p results/layers-20260922/fresh
exec 9>gpu-cm001-1.lock
flock -w 1200 9
python3 - <<'PY'
import subprocess,os
u=os.environ['CUDA_VISIBLE_DEVICES'];s=subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True)
a,b,c=map(int,s.split(','));assert a<100 and b==0 and c==0,s
assert u not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
PY
trap 'printf "%s\n" "$?" > results/layers-20260922/fresh/continuation-exit-code.txt' EXIT
timeout --signal=TERM --kill-after=20s 2h benchmark-env/bin/python -u scripts/layers/collect_fresh_v2.py
