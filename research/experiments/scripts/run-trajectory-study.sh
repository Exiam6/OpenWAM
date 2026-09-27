#!/usr/bin/env bash
set -euo pipefail
cd /data02/zifanz4/openwam-experiments
trajectory_root="$PWD/results/repeatability-20260921/closedloop"
test ! -e "$trajectory_root/exit-code.txt"
export CUDA_VISIBLE_DEVICES=GPU-025b7fc9-8f25-5467-5c28-8e59884ab6d3
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME="$PWD/cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor" TORCH_EXTENSIONS_DIR="$PWD/cache/torch_extensions"
export ROBOTWIN_PATH="$PWD/benchmarks/RoboTwin"
export __EGL_VENDOR_LIBRARY_FILENAMES="$PWD/benchmark-env/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
export PYTHONPATH="$ROBOTWIN_PATH:$PWD/OpenWAM/benchmarks/robotwin"
export MPLCONFIGDIR="$PWD/cache/matplotlib"
exec 9>gpu-cm001-2.lock
flock -n 9
trajectory_server_pid=''
cleanup() {
 code=$?
 printf '%s\n' "$code" > "$trajectory_root/exit-code.txt"
 if [ -n "$trajectory_server_pid" ]; then
  kill -TERM "$trajectory_server_pid" 2>/dev/null || true
  wait "$trajectory_server_pid" 2>/dev/null || true
 fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
policy-env/bin/python scripts/check_trajectory_trace.py > "$trajectory_root/launch-cpu-checks.log" 2>&1
preflight() {
 TRAJECTORY_ROUND="$1" python3 - <<'PY'
import os,subprocess,socket,time,json,shutil
from pathlib import Path
u=os.environ['CUDA_VISIBLE_DEVICES'];samples=[]
assert shutil.disk_usage('.').free>=4*1024**3
for i in range(3):
 a,b,c=map(int,subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).split(','))
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
 row={'memory_mib':a,'utilization':b,'ecc':c,'compute_process_present':u in apps};samples.append(row)
 Path('results/repeatability-20260921/closedloop',f"round{os.environ['TRAJECTORY_ROUND']}-preflight.json").write_text(json.dumps(samples,indent=2))
 assert a<100 and b==0 and c==0 and u not in apps,row
 if i<2:time.sleep(5)
s=socket.socket();s.bind(('127.0.0.1',18848));s.close()
PY
}
for trajectory_round in 0 1; do
 preflight "$trajectory_round"
 policy-env/bin/python -u scripts/serve_trajectory.py --round "$trajectory_round" > "$trajectory_root/round$trajectory_round-server.log" 2>&1 &
 trajectory_server_pid=$!
 for attempt in $(seq 1 360); do
  if rg -q 'TRAJECTORY_SERVER_READY' "$trajectory_root/round$trajectory_round-server.log"; then break; fi
  kill -0 "$trajectory_server_pid"
  sleep 1
 done
 rg -q 'TRAJECTORY_SERVER_READY' "$trajectory_root/round$trajectory_round-server.log"
 for condition in clean noise; do
  export ROBUST_CONDITION="$condition" NORM_SIGMA=0 NORM_ROLE=trajectory_diagnostic NORM_ARM=identity
  if [ "$condition" = noise ]; then export NORM_SIGMA=.20; fi
  export NORM_SCENES="$PWD/results/repeatability-20260921/scenes.json" ROBOTWIN_TEST_NUM=3
  export ROBUST_RUN_DIR="$trajectory_root/r$trajectory_round-$condition"
  export ROBOTWIN_RUNTIME_ROOT="$ROBUST_RUN_DIR/runtime"
  mkdir -p "$ROBUST_RUN_DIR"
  benchmark-env/bin/python -u scripts/trajectory_client.py --config "$PWD/OpenWAM/benchmarks/robotwin/policy_config.yml" --overrides --task_name pick_dual_bottles --task_config demo_clean --ckpt_setting "r$trajectory_round-$condition" --seed 3 --policy_name openwam2robotwin_interface --host 127.0.0.1 --port 18848 > "$ROBUST_RUN_DIR/client.log" 2>&1
  printf '0\n' > "$ROBUST_RUN_DIR/exit-code.txt"
  printf 'TRAJECTORY_CONDITION_COMPLETE r%s-%s\n' "$trajectory_round" "$condition"
 done
 kill -TERM "$trajectory_server_pid"
 wait "$trajectory_server_pid" || true
 trajectory_server_pid=''
 sleep 5
done
policy-env/bin/python scripts/summarize_trajectories.py > "$trajectory_root/analysis.log" 2>&1
