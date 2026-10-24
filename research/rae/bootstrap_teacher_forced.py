#!/usr/bin/env python3
"""Paired hierarchical bootstrap for task-balanced action-loss comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap(report: dict, *, draws: int, seed: int) -> dict:
    tasks = sorted(report["per_task"])
    rows_by_task = {
        task: [row for row in report["paired_windows"] if row["task"] == task]
        for task in tasks
    }
    if any(not rows for rows in rows_by_task.values()):
        raise ValueError("Each task needs at least one paired window")
    rng = np.random.default_rng(seed)
    deltas = []
    relative = []
    for _ in range(draws):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        vae = []
        rae = []
        for task in sampled_tasks:
            rows = rows_by_task[task]
            for index in rng.integers(0, len(rows), size=len(rows)):
                vae.append(rows[index]["vae_action"])
                rae.append(rows[index]["rae_action"])
        vae_mean = float(np.mean(vae))
        rae_mean = float(np.mean(rae))
        delta = rae_mean - vae_mean
        deltas.append(delta)
        relative.append(delta / vae_mean)
    return {
        "method": (
            f"hierarchical paired bootstrap: resample {len(tasks)} tasks, then "
            "paired windows within each sampled task"
        ),
        "draws": draws,
        "seed": seed,
        "delta_95pct_ci": [float(x) for x in np.quantile(deltas, [0.025, 0.975])],
        "relative_delta_95pct_ci": [float(x) for x in np.quantile(relative, [0.025, 0.975])],
        "fraction_of_draws_with_rae_loss_higher": float(np.mean(np.array(deltas) > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    if args.draws <= 0:
        parser.error("draws must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    report = bootstrap(json.loads(args.comparison.read_text()), draws=args.draws, seed=args.seed)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
