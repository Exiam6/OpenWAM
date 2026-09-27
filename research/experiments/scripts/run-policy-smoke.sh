#!/usr/bin/env bash
set -euo pipefail
cd /data02/zifanz4/openwam-experiments
export CUDA_VISIBLE_DEVICES=GPU-025b7fc9-8f25-5467-5c28-8e59884ab6d3
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
export HF_HOME=/data02/zifanz4/openwam-experiments/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR=/data02/zifanz4/openwam-experiments/cache/torchinductor
exec 9>/data02/zifanz4/openwam-experiments/gpu-cm001-2.lock
flock -n 9
python3 - <<'PY'
import os,subprocess
u=os.environ['CUDA_VISIBLE_DEVICES']
a,b,c=map(int,subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).split(','))
assert a<100 and b==0 and c==0
assert u not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
PY
mkdir -p results/policy-smoke-20260921
trap 'code=$?; printf "%s\n" "$code" > results/policy-smoke-20260921/exit-code.txt' EXIT
policy-env/bin/python -u scripts/policy_smoke.py
