#!/usr/bin/env bash
set -euo pipefail
cd /data02/zifanz4/openwam-experiments
phase="${1:?baseline, identity-confirm, or adapter-confirm}"
case "$phase" in
 baseline) conditions=(clean noise front); seed=1; server_args=() ;;
 identity-confirm) conditions=(clean noise); seed=2; server_args=() ;;
 adapter-confirm) conditions=(clean noise); seed=2; server_args=(--adapter results/robustness-20260921/adapter-selected.pt) ;;
 *) exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=GPU-025b7fc9-8f25-5467-5c28-8e59884ab6d3
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME="$PWD/cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor" TORCH_EXTENSIONS_DIR="$PWD/cache/torch_extensions"
export ROBOTWIN_PATH="$PWD/benchmarks/RoboTwin" ROBOTWIN_TEST_NUM=5
export __EGL_VENDOR_LIBRARY_FILENAMES="$PWD/benchmark-env/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
export PYTHONPATH="$ROBOTWIN_PATH:$PWD/OpenWAM/benchmarks/robotwin"
export MPLCONFIGDIR="$PWD/cache/matplotlib"
root="$PWD/results/robustness-20260921"
exec 9>gpu-cm001-2.lock
flock -n 9
python3 - <<'PY'
import os,subprocess,socket
u=os.environ['CUDA_VISIBLE_DEVICES']
a,b,c=map(int,subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).split(','))
assert a<100 and b==0 and c==0,(a,b,c)
assert u not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
s=socket.socket();s.bind(('127.0.0.1',18848));s.close()
PY
test ! -e "$root/$phase-exit-code.txt"
policy-env/bin/python -u scripts/serve_robustness.py "${server_args[@]}" > "$root/$phase-server.log" 2>&1 &
robust_server_pid=$!
cleanup() {
 code=$?
 printf '%s\n' "$code" > "$root/$phase-exit-code.txt"
 kill -TERM "$robust_server_pid" 2>/dev/null || true
 wait "$robust_server_pid" 2>/dev/null || true
}
trap cleanup EXIT
for condition in "${conditions[@]}"; do
 export ROBUST_CONDITION="$condition" ROBUST_RUN_DIR="$root/$phase-$condition"
 export ROBOTWIN_RUNTIME_ROOT="$ROBUST_RUN_DIR/runtime"
 mkdir -p "$ROBUST_RUN_DIR"
 timeout --signal=TERM --kill-after=20s 45m benchmark-env/bin/python -u scripts/robustness_client.py \
  --config "$PWD/OpenWAM/benchmarks/robotwin/policy_config.yml" --overrides \
  --task_name pick_dual_bottles --task_config demo_clean --ckpt_setting "$phase-$condition" \
  --seed "$seed" --policy_name openwam2robotwin_interface --host 127.0.0.1 --port 18848 \
  > "$ROBUST_RUN_DIR/client.log" 2>&1
done
