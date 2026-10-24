#!/usr/bin/env bash
# Paired exploratory RoboTwin rollout on an A100 host using CPU Lavapipe RT.
set -uo pipefail

repo=/root/zifan/OpenWAM-rae
robotwin=/root/zifan/RoboTwin-official-runtime
run_root=/root/zifan/openwam-runs/robotwin_rt_eval_20261024/paired_8spp
policy_python=/root/zifan/openwam-envs/policy/bin/python
robotwin_python=/root/zifan/openwam-envs/benchmark-conda/bin/python
lavapipe=/root/zifan/openwam-envs/lavapipe-rt
checkpoint_root=/root/zifan/openwam-runs/wam_pilot_20261023
port=8848

tasks=(
  beat_block_hammer hanging_mug open_microwave place_bread_skillet
  place_burger_fries place_dual_shoes shake_bottle_horizontally
  stack_blocks_three stack_bowls_three stack_bowls_two
)

mkdir -p "$run_root"
printf 'variant\ttask\texit_code\tstarted_utc\tended_utc\n' > "$run_root/status.tsv"
server_pid=
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for variant in vae rae; do
  if [[ "$variant" == vae ]]; then
    ckpt="$checkpoint_root/vae/2026-10-24_03-16-34"
  else
    ckpt="$checkpoint_root/rae/2026-10-23_20-20-35"
  fi
  mkdir -p "$run_root/$variant"
  "$policy_python" "$repo/scripts/deploy.py" \
    --ckpt-dir "$ckpt" --device cuda:0 --host 127.0.0.1 \
    --port "$port" --compile-enabled false \
    > "$run_root/$variant/server.log" 2>&1 &
  server_pid=$!
  ready=0
  for _ in {1..300}; do
    if grep -q 'WebSocket server started' "$run_root/$variant/server.log"; then
      ready=1
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if (( ready == 0 )); then
    printf '%s\tSERVER_START_FAILED\n' "$variant" >> "$run_root/errors.tsv"
    cleanup
    server_pid=
    continue
  fi

  for task in "${tasks[@]}"; do
    started=$(date -u +%FT%TZ)
    env \
      VK_ICD_FILENAMES="$lavapipe/share/vulkan/icd.d/lvp_icd.x86_64.json" \
      LD_LIBRARY_PATH="$lavapipe/lib" \
      LP_NUM_THREADS=16 OMP_NUM_THREADS=4 \
      ROBOTWIN_SOFTWARE_RT=1 ROBOTWIN_SOFTWARE_RT_SPP=8 \
      ROBOTWIN_SOFTWARE_RT_PATH_DEPTH=4 ROBOTWIN_TEST_NUM=1 \
      ROBOTWIN_PATH="$robotwin" ROBOTWIN_PYTHON="$robotwin_python" \
      ROBOTWIN_RUNTIME_ROOT="$run_root/$variant/$task" \
      timeout 7200s bash "$repo/benchmarks/robotwin/single_eval.sh" \
      "$task" demo_clean "pilot_${variant}_8spp" 1 \
      > "$run_root/$variant/$task.log" 2>&1
    code=$?
    ended=$(date -u +%FT%TZ)
    printf '%s\t%s\t%s\t%s\t%s\n' "$variant" "$task" "$code" "$started" "$ended" \
      >> "$run_root/status.tsv"
    if ! kill -0 "$server_pid" 2>/dev/null; then
      printf '%s\tSERVER_DIED_AFTER_%s\n' "$variant" "$task" >> "$run_root/errors.tsv"
      break
    fi
  done

  cleanup
  server_pid=
done
