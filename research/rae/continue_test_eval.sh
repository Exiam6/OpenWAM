#!/usr/bin/env bash
# Finish the frozen test-window pair after the native-VAE test run exits.
set -euo pipefail

vae_pid="${1:?pass the live VAE test evaluator PID}"
repo=/root/zifan/OpenWAM-rae
runs=/root/zifan/openwam-runs/wam_pilot_20261023
python=/root/zifan/openwam-envs/policy/bin/python

while kill -0 "$vae_pid" 2>/dev/null; do
  sleep 20
done

vae="$runs/vae/test_teacher_forced.json"
rae="$runs/rae/test_teacher_forced.json"
comparison="$runs/paired_test_comparison.json"
if [[ ! -s "$vae" ]]; then
  echo "VAE test report missing; see $runs/vae/test_teacher_forced.log" >&2
  exit 1
fi

cd "$repo"
env CUDA_VISIBLE_DEVICES=4 \
  CUDA_HOME=/root/zifan/openwam-envs/benchmark-conda \
  PYTHONPATH="$repo" \
  "$python" research/rae/eval_teacher_forced.py \
  --checkpoint "$runs/rae/2026-10-23_20-20-35" \
  --output "$rae" --split test --per-task 10 --noise-repeats 4 \
  > "$runs/rae/test_teacher_forced.log" 2>&1

"$python" research/rae/compare_teacher_forced.py \
  --vae "$vae" --rae "$rae" --output "$comparison" \
  > "$runs/paired_test_comparison.log" 2>&1
