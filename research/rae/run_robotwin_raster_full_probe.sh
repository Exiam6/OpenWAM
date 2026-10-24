#!/usr/bin/env bash
# Full-length exploratory rollout with CPU Vulkan raster rendering.
set -euo pipefail

variant="${1:?usage: run_robotwin_raster_full_probe.sh vae|rae [task]}"
task="${2:-beat_block_hammer}"
if [[ "$variant" != vae && "$variant" != rae ]]; then
  echo "variant must be vae or rae" >&2
  exit 2
fi
if [[ ! "$task" =~ ^[a-z_]+$ ]]; then
  echo "invalid task: $task" >&2
  exit 2
fi

repo=/root/zifan/OpenWAM-rae
robotwin=/root/zifan/RoboTwin-official-runtime
root=/root/zifan/openwam-runs/robotwin_rt_eval_20261024/raster_full_probe_mesa2625
policy_python=/root/zifan/openwam-envs/policy/bin/python
robotwin_python=/root/zifan/openwam-envs/benchmark-conda/bin/python
lavapipe=/root/zifan/openwam-envs/lavapipe-rt-2625
port=8849
if [[ "$variant" == vae ]]; then
  ckpt=/root/zifan/openwam-runs/wam_pilot_20261023/vae/2026-10-24_03-16-34
else
  ckpt=/root/zifan/openwam-runs/wam_pilot_20261023/rae/2026-10-23_20-20-35
fi
out="$root/$variant/$task"
mkdir -p "$out"
if [[ -e "$out/status.json" || -e "$out/episode.log" ]]; then
  echo "probe already has an output: $out" >&2
  exit 1
fi

server_pid=
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"$policy_python" "$repo/scripts/deploy.py" \
  --ckpt-dir "$ckpt" --device cuda:4 --host 127.0.0.1 \
  --port "$port" --compile-enabled false > "$out/server.log" 2>&1 &
server_pid=$!
ready=0
for _ in {1..300}; do
  if grep -q 'WebSocket server started' "$out/server.log"; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done
if (( ready == 0 )); then
  echo "policy server did not start" >&2
  exit 1
fi

started=$(date -u +%FT%TZ)
set +e
env \
  VK_ICD_FILENAMES="$lavapipe/share/vulkan/icd.d/lvp_icd.x86_64.json" \
  LD_LIBRARY_PATH="$lavapipe/lib" \
  LP_NUM_THREADS=16 OMP_NUM_THREADS=4 PYTHONFAULTHANDLER=1 \
  ROBOTWIN_SOFTWARE_RASTER=1 ROBOTWIN_TEST_NUM=1 \
  ROBOTWIN_PATH="$robotwin" ROBOTWIN_PYTHON="$robotwin_python" \
  ROBOTWIN_RUNTIME_ROOT="$out" \
  timeout 1500s bash "$repo/benchmarks/robotwin/single_eval.sh" \
  "$task" demo_clean "pilot_${variant}_raster_mesa2625_full" 5 "$port" \
  > "$out/episode.log" 2>&1
code=$?
set -e
ended=$(date -u +%FT%TZ)
"$policy_python" - "$out/status.json" "$variant" "$task" "$code" "$started" "$ended" <<'PY'
import json
import sys
from pathlib import Path

path, variant, task, code, started, ended = sys.argv[1:]
Path(path).write_text(json.dumps({
    "variant": variant,
    "task": task,
    "exit_code": int(code),
    "started_utc": started,
    "ended_utc": ended,
    "renderer": "Mesa 26.2.5 Lavapipe CPU raster; exploratory, not native RoboTwin RT",
    "scope": "One full-length demo_clean episode; no early step cap",
}, indent=2) + "\n")
PY
exit "$code"
