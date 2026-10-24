"""Summarize the paired one-episode software-RT RoboTwin sweep."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


TASKS = (
    "beat_block_hammer",
    "hanging_mug",
    "open_microwave",
    "place_bread_skillet",
    "place_burger_fries",
    "place_dual_shoes",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_bowls_three",
    "stack_bowls_two",
)


def summarize(root: Path) -> dict:
    status_file = root / "status.tsv"
    with status_file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    status = {(r["variant"], r["task"]): r for r in rows}
    results = []
    for variant in ("vae", "rae"):
        for task in TASKS:
            entry = {"variant": variant, "task": task}
            row = status.get((variant, task))
            if row is None:
                entry["state"] = "pending"
            else:
                entry.update(
                    exit_code=int(row["exit_code"]),
                    started_utc=row["started_utc"],
                    ended_utc=row["ended_utc"],
                )
                result_files = list((root / variant / task / "eval_result").rglob("_result.txt"))
                if entry["exit_code"] != 0 or len(result_files) != 1:
                    entry["state"] = "error"
                    entry["result_files"] = len(result_files)
                else:
                    value = float(result_files[0].read_text(encoding="utf-8").splitlines()[-1])
                    if value not in (0.0, 1.0):
                        raise ValueError(f"Expected one-episode binary rate, got {value}: {result_files[0]}")
                    entry["state"] = "complete"
                    entry["success"] = bool(value)
                    entry["result_file"] = str(result_files[0])
                    log_file = root / variant / f"{task}.log"
                    log = log_file.read_text(encoding="utf-8", errors="replace")
                    seeds = re.findall(r"current seed:\s*\x1b\[[0-9;]*m?(\d+)", log)
                    if not seeds:
                        seeds = re.findall(r"current seed:.*?(\d{5,})", log)
                    entry["accepted_seed"] = int(seeds[-1]) if seeds else None
            results.append(entry)

    pairs = []
    for task in TASKS:
        vae = next(x for x in results if x["variant"] == "vae" and x["task"] == task)
        rae = next(x for x in results if x["variant"] == "rae" and x["task"] == task)
        if vae["state"] == rae["state"] == "complete":
            pairs.append(
                {
                    "task": task,
                    "vae_success": vae["success"],
                    "rae_success": rae["success"],
                    "rae_minus_vae": int(rae["success"]) - int(vae["success"]),
                    "same_accepted_seed": vae["accepted_seed"] == rae["accepted_seed"],
                }
            )
    return {
        "scope": "exploratory; one demo_clean episode per task and checkpoint",
        "renderer": "Mesa Lavapipe CPU RT; 8 spp; path depth 4; OIDN disabled",
        "not_official_benchmark": True,
        "tasks_planned": len(TASKS),
        "pairs_complete": len(pairs),
        "episodes": results,
        "pairs": pairs,
        "vae_successes_in_pairs": sum(x["vae_success"] for x in pairs),
        "rae_successes_in_pairs": sum(x["rae_success"] for x in pairs),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.root)
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
