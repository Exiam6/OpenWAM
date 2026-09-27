#!/usr/bin/env bash
set -euo pipefail
cd /data02/zifanz4/openwam-experiments
attribution_root="$PWD/results/attribution-20260921"
test ! -e "$attribution_root/exit-code.txt"
export CUDA_VISIBLE_DEVICES=GPU-025b7fc9-8f25-5467-5c28-8e59884ab6d3
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME="$PWD/cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor" TORCH_EXTENSIONS_DIR="$PWD/cache/torch_extensions"
export ROBOTWIN_PATH="$PWD/benchmarks/RoboTwin"
export __EGL_VENDOR_LIBRARY_FILENAMES="$PWD/benchmark-env/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
export PYTHONPATH="$ROBOTWIN_PATH:$PWD/OpenWAM/benchmarks/robotwin" MPLCONFIGDIR="$PWD/cache/matplotlib"
exec 9>gpu-cm001-2.lock
flock -n 9
trap 'printf "%s\n" "$?" > "$attribution_root/exit-code.txt"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
benchmark-env/bin/python scripts/check_attribution_trace.py > "$attribution_root/launch-cpu-checks.log" 2>&1
for probe_round in 0 1; do
 PROBE_ROUND="$probe_round" python3 - <<'PY'
import json,os,subprocess,time
from pathlib import Path
u=os.environ['CUDA_VISIBLE_DEVICES'];rows=[]
for i in range(3):
 a,b,c=map(int,subprocess.check_output(['nvidia-smi','-i',u,'--query-gpu=memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).split(','))
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
 row={'memory_mib':a,'utilization':b,'ecc':c,'compute_process_present':u in apps};rows.append(row)
 Path('results/attribution-20260921',f"round{os.environ['PROBE_ROUND']}-preflight.json").write_text(json.dumps(rows,indent=2))
 assert a<100 and b==0 and c==0 and u not in apps,row
 if i<2:time.sleep(5)
PY
 export ROBUST_CONDITION=clean NORM_SIGMA=0 NORM_ROLE=two_command_attribution NORM_ARM=identity
 export NORM_SCENES="$attribution_root/scenes.json" ROBOTWIN_TEST_NUM=3
 export ROBUST_RUN_DIR="$attribution_root/round$probe_round" ROBOTWIN_RUNTIME_ROOT="$attribution_root/round$probe_round/runtime"
 mkdir -p "$ROBUST_RUN_DIR"
 benchmark-env/bin/python -u scripts/attribution_client.py --config "$PWD/OpenWAM/benchmarks/robotwin/policy_config.yml" --overrides --task_name pick_dual_bottles --task_config demo_clean --ckpt_setting "attribution$probe_round" --seed 3 --policy_name openwam2robotwin_interface > "$ROBUST_RUN_DIR/client.log" 2>&1
 printf '0\n' > "$ROBUST_RUN_DIR/exit-code.txt"
 printf 'ATTRIBUTION_ROUND_COMPLETE %s\n' "$probe_round"
 sleep 5
done
benchmark-env/bin/python scripts/summarize_attribution.py > "$attribution_root/analysis.log" 2>&1
