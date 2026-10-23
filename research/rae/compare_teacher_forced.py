#!/usr/bin/env python3
"""Compare paired held-out action losses from the two registered WAM pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare(vae: dict, rae: dict) -> dict:
    for key in ("split_manifest_sha256", "action_stats_sha256", "seed", "windows_per_task", "noise_repeats"):
        if vae[key] != rae[key]:
            raise ValueError(f"Evaluation setting mismatch: {key}")
    vae_rows = vae["windows"]
    rae_rows = rae["windows"]
    if len(vae_rows) != len(rae_rows):
        raise ValueError("Different number of windows")
    paired = []
    for a, b in zip(vae_rows, rae_rows, strict=True):
        if (a["task"], a["window_index"]) != (b["task"], b["window_index"]):
            raise ValueError("Window order or identity mismatch")
        if len(a["losses"]) != len(b["losses"]):
            raise ValueError("Different repeat counts")
        a_loss = float(np.mean([item["action"] for item in a["losses"]]))
        b_loss = float(np.mean([item["action"] for item in b["losses"]]))
        paired.append({
            "task": a["task"], "window_index": a["window_index"],
            "vae_action": a_loss, "rae_action": b_loss,
            "rae_minus_vae": b_loss - a_loss,
        })
    tasks = sorted({item["task"] for item in paired})
    per_task = {}
    for task in tasks:
        rows = [item for item in paired if item["task"] == task]
        per_task[task] = {
            "windows": len(rows),
            "vae_action_mean": float(np.mean([item["vae_action"] for item in rows])),
            "rae_action_mean": float(np.mean([item["rae_action"] for item in rows])),
            "rae_minus_vae": float(np.mean([item["rae_minus_vae"] for item in rows])),
        }
    vae_mean = float(np.mean([item["vae_action_mean"] for item in per_task.values()]))
    rae_mean = float(np.mean([item["rae_action_mean"] for item in per_task.values()]))
    if not np.isclose(vae_mean, vae["action_mean_across_tasks"], rtol=1e-7):
        raise ValueError("VAE report aggregate mismatch")
    if not np.isclose(rae_mean, rae["action_mean_across_tasks"], rtol=1e-7):
        raise ValueError("RAE report aggregate mismatch")
    return {
        "vae_checkpoint_sha256": vae["checkpoint_sha256"],
        "rae_checkpoint_sha256": rae["checkpoint_sha256"],
        "split_manifest_sha256": vae["split_manifest_sha256"],
        "action_stats_sha256": vae["action_stats_sha256"],
        "windows": len(paired),
        "noise_repeats_per_window": vae["noise_repeats"],
        "vae_action_mean_across_tasks": vae_mean,
        "rae_action_mean_across_tasks": rae_mean,
        "rae_minus_vae": rae_mean - vae_mean,
        "lower_is_better": True,
        "per_task": per_task,
        "paired_windows": paired,
        "scope": "Descriptive, single-seed teacher-forced action loss; no closed-loop success or causal improvement claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--rae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(json.loads(args.vae.read_text()), json.loads(args.rae.read_text()))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "vae_action_mean_across_tasks", "rae_action_mean_across_tasks", "rae_minus_vae"
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
