#!/usr/bin/env bash
set -euo pipefail
cd /data02/zifanz4/openwam-experiments
export CUDA_VISIBLE_DEVICES=GPU-025b7fc9-8f25-5467-5c28-8e59884ab6d3
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME=/data02/zifanz4/openwam-experiments/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR=/data02/zifanz4/openwam-experiments/cache/torchinductor
export TORCH_EXTENSIONS_DIR=/data02/zifanz4/openwam-experiments/cache/torch_extensions
export ROBOTWIN_PATH=/data02/zifanz4/openwam-experiments/benchmarks/RoboTwin
export ROBOTWIN_PYTHON=/data02/zifanz4/openwam-experiments/benchmark-env/bin/python
export ROBOTWIN_TEST_NUM=5
export ROBOTWIN_RUNTIME_ROOT=/data02/zifanz4/openwam-experiments/results/closed-smoke-20260921/runtime
exec 9>/data02/zifanz4/openwam-experiments/gpu-cm001-2.lock
flock -n 9
python3 - <<'PY'
import os,subprocess,socket
u=os.environ['CUDA_VISIBLE_DEVICES']
a,b,c=map(int,subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).split(','))
assert a<100 and b==0 and c==0
assert u not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
s=socket.socket();s.bind(('127.0.0.1',18848));s.close()
PY
mkdir -p results/closed-smoke-20260921
policy-env/bin/python -u scripts/serve_policy_smoke.py > logs/closed-smoke-policy.log 2>&1 &
smoke_server_pid=$!
printf '%s\n' "$smoke_server_pid" > results/closed-smoke-20260921/server-pid.txt
cleanup() {
    code=$?
    printf '%s\n' "$code" > results/closed-smoke-20260921/exit-code.txt
    # Only the background policy process created immediately above is stopped.
    kill -TERM "$smoke_server_pid" 2>/dev/null || true
    wait "$smoke_server_pid" 2>/dev/null || true
}
trap cleanup EXIT
# The official client performs a bounded server health wait during initialization.
timeout --signal=TERM --kill-after=20s 30m bash OpenWAM/benchmarks/robotwin/single_eval.sh pick_dual_bottles demo_clean published_dino_smoke "$CUDA_VISIBLE_DEVICES" 18848 127.0.0.1 > logs/closed-smoke-client.log 2>&1
