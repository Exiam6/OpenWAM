#!/usr/bin/env bash
set -euo pipefail
cd /data02/zifanz4/openwam-experiments
root="$PWD/results/norm-20260921"
test ! -e "$root/resume-exit-code.txt"
test ! -e "$root/adapter_renorm-historical-clean/client.log"
test ! -e "$root/adapter_renorm-clean/client.log"
test ! -e "$root/adapter_renorm-noise/client.log"
for completed in identity renorm adapter; do test "$(cat "$root/$completed-exit-code.txt")" = 0; done
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
norm_server_pid=''
cleanup() {
 code=$?
 printf '%s\n' "$code" > "$root/resume-exit-code.txt"
 if [ -n "$norm_server_pid" ]; then
  kill -TERM "$norm_server_pid" 2>/dev/null || true
  wait "$norm_server_pid" 2>/dev/null || true
 fi
}
trap cleanup EXIT
preflight() {
 policy-env/bin/python scripts/norm_resume_preflight.py
}
run_condition() {
 local name="$1" condition="$2" sigma="$3" role="$4" scenes="$5" count="$6"
 export ROBUST_CONDITION="$condition" NORM_SIGMA="$sigma" NORM_ROLE="$role" NORM_SCENES="$scenes" ROBOTWIN_TEST_NUM="$count"
 export ROBUST_RUN_DIR="$root/$name" ROBOTWIN_RUNTIME_ROOT="$root/$name/runtime"
 test ! -e "$ROBUST_RUN_DIR/client.log"
 mkdir -p "$ROBUST_RUN_DIR"
 timeout --signal=TERM --kill-after=20s 60m benchmark-env/bin/python -u scripts/norm_client.py \
  --config "$PWD/OpenWAM/benchmarks/robotwin/policy_config.yml" --overrides \
  --task_name pick_dual_bottles --task_config demo_clean --ckpt_setting "$name" \
  --seed 3 --policy_name openwam2robotwin_interface --host 127.0.0.1 --port 18848 \
  > "$ROBUST_RUN_DIR/client.log" 2>&1
 printf '0\n' > "$ROBUST_RUN_DIR/exit-code.txt"
 printf 'CONDITION_COMPLETE %s\n' "$name"
}
for arm in adapter_renorm; do
 export NORM_ARM="$arm"
 preflight
 policy-env/bin/python -u scripts/serve_norm.py --arm "$arm" > "$root/$arm-server.log" 2>&1 &
 norm_server_pid=$!
 for attempt in $(seq 1 300); do
  if rg -q 'NORM_SERVER_READY' "$root/$arm-server.log"; then break; fi
  kill -0 "$norm_server_pid"
  sleep 1
 done
 rg -q 'NORM_SERVER_READY' "$root/$arm-server.log"
 if [ "$arm" = identity ]; then
  run_condition reference-clean clean 0 reference '' 10
  policy-env/bin/python scripts/norm_freeze_manifest.py
 fi
 run_condition "$arm-historical-clean" clean 0 historical "$root/historical-scene.json" 1
 run_condition "$arm-clean" clean 0 primary "$root/fresh-scenes.json" 10
 run_condition "$arm-noise" noise .20 primary "$root/fresh-scenes.json" 10
 kill -TERM "$norm_server_pid"
 wait "$norm_server_pid" || true
 norm_server_pid=''
 printf '0\n' > "$root/$arm-exit-code.txt"
done
python3 - <<'DONE'
from pathlib import Path
import json,time
p=Path('results/norm-20260921')
for arm in ['identity','renorm','adapter','adapter_renorm']:
 assert (p/f'{arm}-exit-code.txt').read_text().strip()=='0'
 for condition in ['historical-clean','clean','noise']:
  assert (p/f'{arm}-{condition}/exit-code.txt').read_text().strip()=='0'
(p/'completion-after-resume.json').write_text(json.dumps({'completed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'original_interruption':'interruption-20260921T1627/study-exit-code.txt','all_four_arm_exit_codes_zero':True,'no_completed_episode_rerun':True},indent=2))
tmp=p/'study-exit-code.tmp';tmp.write_text('0\n');tmp.replace(p/'study-exit-code.txt')
DONE
