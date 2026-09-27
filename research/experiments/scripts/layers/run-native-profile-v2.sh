#!/usr/bin/env bash
set -euo pipefail
cd /data02/zifanz4/openwam-experiments
run="$PWD/results/layers-20260922/native-preflight/trial2"
deps="$PWD/results/layers-20260922/native-preflight/deps"
export PYTHONPATH="$deps:$PWD/OpenWAM"
export CUDA_VISIBLE_DEVICES=GPU-025b7fc9-8f25-5467-5c28-8e59884ab6d3
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MAX_JOBS=6
export PATH="$deps/bin:$PATH"
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_EXTENSIONS_DIR="$run/torch_extensions" TRITON_CACHE_DIR=/tmp/zifanz4-layers-triton
export TOKENIZERS_PARALLELISM=false
exec 9>gpu-cm001-2.lock
flock -n 9
python3 - <<'PY'
import os,subprocess
for u in os.environ['CUDA_VISIBLE_DEVICES'].split(','):
 s=subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True)
 a,b,c=map(int,s.split(','));assert a<100 and b==0 and c==0,s
 assert u not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
PY
trap 'printf "%s\n" "$?" > "$run/profile-exit-code.txt"' EXIT
timeout --signal=TERM --kill-after=30s 20m policy-env/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 scripts/layers/native_profile_v2.py --config-path "$run" --config-name profile
