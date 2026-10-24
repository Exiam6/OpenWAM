#!/usr/bin/env bash
# Run the released-checkpoint software-RT sanity sweep after the matched pair.
set -euo pipefail

paired_pid="${1:?pass the live paired-evaluation controller PID}"
root=/root/zifan/openwam-runs/robotwin_rt_eval_20261024
paired="$root/paired_1spp_cpu_oidn_mesa2625"

while kill -0 "$paired_pid" 2>/dev/null; do
  sleep 30
done

python - "$paired/status.tsv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if len(rows) != 20:
    raise SystemExit(f"paired sweep incomplete: {len(rows)}/20 rows")
failures = sum(int(row["exit_code"]) != 0 for row in rows)
print(
    f"paired sweep ended with {failures} terminal technical failures; "
    "starting released-checkpoint sanity sweep",
    flush=True,
)
PY

python /root/zifan/OpenWAM-rae/research/rae/summarize_robotwin_software_rt.py \
  "$paired" --output "$paired/summary.json" > /dev/null
bash /root/zifan/OpenWAM-rae/research/rae/run_robotwin_official_cpu_oidn_mesa2625.sh
