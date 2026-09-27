#!/usr/bin/env bash
set -euo pipefail
cd /vol13/zifanz4/openwam-experiments
export CUDA_VISIBLE_DEVICES=GPU-56da2cc3-8cdf-dcc4-f3ac-903b764b5c29
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
export HF_HOME=/vol13/zifanz4/openwam-experiments/cache/huggingface
export MPLCONFIGDIR=/vol13/zifanz4/openwam-experiments/cache/matplotlib
exec 9>/vol13/zifanz4/openwam-experiments/gpu-cm006-0.lock
flock -n 9
python3 - <<'PY'
import subprocess,os
uid=os.environ['CUDA_VISIBLE_DEVICES']
r=subprocess.check_output(['nvidia-smi','-i',uid,'--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True)
a,b,c=[int(v.strip())for v in r.split(',')]
assert a<100 and b==0 and c==0,r
p=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
assert uid not in p
PY
for pilot_seed in 42 43 44; do
 env/bin/python -u scripts/pilot.py --stage fit --steps 2000 --train-seed "$pilot_seed" --run-tag "s${pilot_seed}-2000" > "logs/pilot-s${pilot_seed}-2000.log" 2>&1
 printf '0\n' > "logs/pilot-s${pilot_seed}-2000.exit"
done
