#!/usr/bin/env bash
# Released RoboTwin-Full sanity sweep using Mesa 26.2.5 CPU RT and OIDN.
set -uo pipefail

repo=/root/zifan/OpenWAM-rae
robotwin=/root/zifan/RoboTwin-official-runtime
run_root=/root/zifan/openwam-runs/robotwin_rt_eval_20261024/official_1spp_cpu_oidn_mesa2625
policy_python=/root/zifan/openwam-envs/policy/bin/python
robotwin_python=/root/zifan/openwam-envs/benchmark-conda/bin/python
lavapipe=/root/zifan/openwam-envs/lavapipe-rt-2625
oidn=/root/zifan/openwam-envs/oidn-cpu
ckpt=/root/zifan/OpenWAM-official-runtime/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full
port=8848

tasks=(
  beat_block_hammer hanging_mug open_microwave place_bread_skillet
  place_burger_fries place_dual_shoes shake_bottle_horizontally
  stack_blocks_three stack_bowls_three stack_bowls_two
)

mkdir -p "$run_root"
printf 'variant\ttask\texit_code\tattempts\tstarted_utc\tended_utc\n' > "$run_root/status.tsv"
server_pid=
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for variant in official; do
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
    code=1
    for attempt in 1 2; do
      env \
        VK_ICD_FILENAMES="$lavapipe/share/vulkan/icd.d/lvp_icd.x86_64.json" \
        LD_LIBRARY_PATH="$lavapipe/lib:$oidn/lib" \
        LP_NUM_THREADS=16 OMP_NUM_THREADS=4 PYTHONFAULTHANDLER=1 \
        ROBOTWIN_SOFTWARE_RT=1 ROBOTWIN_SOFTWARE_RT_SPP=1 \
        ROBOTWIN_SOFTWARE_RT_PATH_DEPTH=2 \
        ROBOTWIN_SOFTWARE_RT_OIDN_CPU=1 \
        ROBOTWIN_SOFTWARE_RT_OIDN_LIBRARY="$oidn/lib/libOpenImageDenoise.so" \
        ROBOTWIN_TEST_NUM=1 \
        ROBOTWIN_PATH="$robotwin" ROBOTWIN_PYTHON="$robotwin_python" \
        ROBOTWIN_RUNTIME_ROOT="$run_root/$variant/$task" \
        timeout 7200s bash "$repo/benchmarks/robotwin/single_eval.sh" \
        "$task" demo_clean "${variant}_1spp_cpu_oidn_mesa2625" 1 \
        > "$run_root/$variant/$task.attempt${attempt}.log" 2>&1
      code=$?
      ln -sfn "$task.attempt${attempt}.log" "$run_root/$variant/$task.log"
      if (( code == 0 )); then
        break
      fi
      sleep 2
    done
    ended=$(date -u +%FT%TZ)
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$variant" "$task" "$code" "$attempt" "$started" "$ended" \
      >> "$run_root/status.tsv"
    if ! kill -0 "$server_pid" 2>/dev/null; then
      printf '%s\tSERVER_DIED_AFTER_%s\n' "$variant" "$task" >> "$run_root/errors.tsv"
      break
    fi
  done

  cleanup
  server_pid=
done
