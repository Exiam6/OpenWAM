#!/usr/bin/env bash
set -euo pipefail
cd /data02/zifanz4/openwam-experiments
repeat_root="$PWD/results/repeatability-20260921"
test ! -e "$repeat_root/exit-code.txt"
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
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'code=$?; printf "%s\n" "$code" > "$repeat_root/exit-code.txt"' EXIT
preflight() {
 REPEAT_STAGE="$1" python3 - <<'PY'
import os,subprocess,time,json
from pathlib import Path
u=os.environ['CUDA_VISIBLE_DEVICES'];samples=[]
for i in range(3):
 a,b,c=map(int,subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).split(','))
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
 row={'memory_mib':a,'utilization':b,'ecc':c,'compute_process_present':u in apps};samples.append(row)
 p=Path('results/repeatability-20260921')/(os.environ['REPEAT_STAGE']+'-preflight.json');p.write_text(json.dumps(samples,indent=2))
 assert a<100 and b==0 and c==0 and u not in apps, row
 if i<2:time.sleep(5)
PY
}
policy-env/bin/python scripts/check_repeatability.py > "$repeat_root/launch-cpu-checks.log" 2>&1
test -f "$repeat_root/capture-summary.json"
for repeat_process in A B; do
 sleep 5
 preflight "process-$repeat_process"
 policy-env/bin/python -u scripts/repeat_action_inference.py --process "$repeat_process" > "$repeat_root/process-$repeat_process.log" 2>&1
 printf 'REPEAT_PROCESS_COMPLETE %s\n' "$repeat_process"
done
policy-env/bin/python scripts/summarize_repeatability.py > "$repeat_root/analysis.log" 2>&1
