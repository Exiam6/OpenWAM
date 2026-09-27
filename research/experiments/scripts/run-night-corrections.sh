#!/usr/bin/env bash
set -euo pipefail
cd /vol13/zifanz4/openwam-experiments
export CUDA_VISIBLE_DEVICES=GPU-d5dc6850-099b-e2da-8b07-a438cb854fdd
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
export HF_HOME=/vol13/zifanz4/openwam-experiments/cache/huggingface
exec 9>/vol13/zifanz4/openwam-experiments/gpu-cm005-0.lock
flock -n 9
python3 - <<'PY'
import os,subprocess
uid=os.environ['CUDA_VISIBLE_DEVICES']
s=subprocess.check_output(['nvidia-smi','-i',uid,'--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True)
a,b,c=[int(v.strip()) for v in s.split(',')]
assert a<100 and b==0 and c==0,s
assert uid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
PY
env/bin/python -u scripts/correct_corruptions.py
MPLCONFIGDIR=/vol13/zifanz4/openwam-experiments/cache/matplotlib env/bin/python -u scripts/summarize_night.py
