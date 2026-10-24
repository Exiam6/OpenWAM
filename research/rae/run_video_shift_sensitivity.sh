#!/usr/bin/env bash
# Fixed-checkpoint dev-set diagnostic: only video noise shift changes.
set -euo pipefail

repo=/root/zifan/OpenWAM-rae
pilot=/root/zifan/openwam-runs/wam_pilot_20261023
out="$pilot/video_shift_sensitivity"
python=/root/zifan/openwam-envs/policy/bin/python
if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
  echo "Commit diagnostic code before evaluating" >&2
  exit 1
fi
if [[ ! -f "$out/registration.json" ]]; then
  echo "Missing registered shift experiment: $out" >&2
  exit 1
fi

export PYTHONPATH="$repo"
export CUDA_HOME=/root/zifan/openwam-envs/benchmark-conda
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=4
cd "$repo"
for shift in 1 10; do
  for variant in vae rae; do
    if [[ "$variant" == vae ]]; then
      ckpt="$pilot/vae/2026-10-24_03-16-34"
    else
      ckpt="$pilot/rae/2026-10-23_20-20-35"
    fi
    "$python" research/rae/eval_teacher_forced.py \
      --checkpoint "$ckpt" --output "$out/${variant}_shift${shift}_dev.json" \
      --split dev --per-task 10 --noise-repeats 4 --seed 20261023 \
      --video-shift "$shift" > "$out/${variant}_shift${shift}_dev.log" 2>&1
  done
  "$python" research/rae/compare_teacher_forced.py \
    --vae "$out/vae_shift${shift}_dev.json" \
    --rae "$out/rae_shift${shift}_dev.json" \
    --output "$out/paired_shift${shift}_dev.json" \
    > "$out/paired_shift${shift}_dev.log" 2>&1
done

"$python" - "$out" "$pilot/paired_dev_comparison.json" "$(git rev-parse HEAD)" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
baseline = json.loads(Path(sys.argv[2]).read_text())
source_commit = sys.argv[3]
rows = [(5, baseline)]
for shift in (1, 10):
    report = json.loads((out / f"paired_shift{shift}_dev.json").read_text())
    if report["evaluation_video_shift"] != shift or report["trained_video_shift"] != [5.0, 5.0]:
        raise ValueError(f"Shift metadata mismatch for {shift}")
    for key in ("split_manifest_sha256", "action_stats_sha256", "evaluation_noise_seed",
                "noise_repeats_per_window", "windows"):
        if report[key] != baseline[key]:
            raise ValueError(f"Changed comparison setting: {key}")
    rows.append((shift, report))
rows.sort()
lines = [
    "# Post-training video-noise shift sensitivity",
    "",
    "Both checkpoints were trained at video shift 5. This diagnostic only changes",
    "the video scheduler during teacher-forced dev evaluation. The 100 windows",
    "and four fixed noise draws per window are identical across shifts.",
    "Lower action loss is better; this is not a retraining or task-success result.",
    "",
    "| Evaluation shift | VAE action loss | RAE action loss | RAE − VAE |",
    "|---:|---:|---:|---:|",
]
for shift, report in rows:
    lines.append(
        f"| {shift} | {report['vae_action_mean_across_tasks']:.7f} | "
        f"{report['rae_action_mean_across_tasks']:.7f} | {report['rae_minus_vae']:+.7f} |"
    )
lines.extend(["", f"Source commit: `{source_commit}`.", ""])
summary = out / "SUMMARY.md"
if summary.exists():
    raise FileExistsError(summary)
summary.write_text("\n".join(lines))
print(summary)
PY
