#!/usr/bin/env python3
"""Aggregate matched teacher-forced VAE/RAE losses across training seeds.

This remains an offline proxy. Bootstrap intervals describe this selected set
of tasks, windows, and seeds; they are not closed-loop success intervals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_pair(value: str) -> tuple[int, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.isdigit() or not path:
        raise argparse.ArgumentTypeError("--pair must be SEED=/path/to/paired_comparison.json")
    return int(label), Path(path)


def aggregate(pairs: list[tuple[int, dict]], *, draws: int, bootstrap_seed: int) -> dict:
    if not pairs or draws <= 0:
        raise ValueError("Need at least one pair and positive bootstrap draws")
    seeds = [seed for seed, _ in pairs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must be distinct")
    reference = pairs[0][1]
    tasks = sorted(reference["per_task"])
    if not tasks:
        raise ValueError("No tasks in comparison")
    required_equal = (
        "split", "split_manifest_sha256", "action_stats_sha256",
        "evaluation_noise_seed", "noise_repeats_per_window", "windows",
    )
    rows_by_seed: list[dict[str, dict[int, dict]]] = []
    per_seed = []
    for training_seed, report in pairs:
        for key in required_equal:
            if report.get(key) != reference.get(key):
                raise ValueError(f"Mismatched {key} for training seed {training_seed}")
        if sorted(report["per_task"]) != tasks:
            raise ValueError(f"Task list mismatch for training seed {training_seed}")
        lookup: dict[str, dict[int, dict]] = {task: {} for task in tasks}
        for row in report["paired_windows"]:
            task, index = row["task"], row["window_index"]
            if task not in lookup or index in lookup[task]:
                raise ValueError(f"Unexpected or duplicate window {task}/{index}")
            lookup[task][index] = row
        rows_by_seed.append(lookup)
        per_seed.append({
            "training_seed": training_seed,
            "vae_checkpoint_sha256": report["vae_checkpoint_sha256"],
            "rae_checkpoint_sha256": report["rae_checkpoint_sha256"],
            "vae_action_loss": report["vae_action_mean_across_tasks"],
            "rae_action_loss": report["rae_action_mean_across_tasks"],
            "rae_minus_vae": report["rae_minus_vae"],
        })
    if len({row["vae_checkpoint_sha256"] for row in per_seed}) != len(pairs):
        raise ValueError("VAE checkpoint reused across training seeds")
    if len({row["rae_checkpoint_sha256"] for row in per_seed}) != len(pairs):
        raise ValueError("RAE checkpoint reused across training seeds")

    window_ids = {task: sorted(rows_by_seed[0][task]) for task in tasks}
    if any(not indices for indices in window_ids.values()):
        raise ValueError("Every task needs paired windows")
    for lookup in rows_by_seed[1:]:
        if any(sorted(lookup[task]) != window_ids[task] for task in tasks):
            raise ValueError("Window identities differ across training seeds")
    window_counts = {len(indices) for indices in window_ids.values()}
    if len(window_counts) != 1:
        raise ValueError("Task-balanced bootstrap requires equal window counts")
    windows_per_task = window_counts.pop()
    values = np.asarray([
        [
            [[lookup[task][index]["vae_action"], lookup[task][index]["rae_action"]]
             for index in window_ids[task]]
            for task in tasks
        ]
        for lookup in rows_by_seed
    ], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite action loss")
    actual_vae = float(values[..., 0].mean())
    actual_rae = float(values[..., 1].mean())
    if not np.isclose(actual_vae, np.mean([row["vae_action_loss"] for row in per_seed]), rtol=1e-7):
        raise ValueError("VAE per-window and per-seed aggregates disagree")
    if not np.isclose(actual_rae, np.mean([row["rae_action_loss"] for row in per_seed]), rtol=1e-7):
        raise ValueError("RAE per-window and per-seed aggregates disagree")

    rng = np.random.default_rng(bootstrap_seed)
    sampled_delta = np.empty(draws, dtype=np.float64)
    sampled_relative = np.empty(draws, dtype=np.float64)
    num_seeds, num_tasks = values.shape[:2]
    for draw in range(draws):
        chosen_seeds = rng.integers(num_seeds, size=num_seeds)
        chosen_tasks = rng.integers(num_tasks, size=num_tasks)
        chosen_windows = rng.integers(windows_per_task, size=(num_tasks, windows_per_task))
        sample = values[
            chosen_seeds[:, None, None],
            chosen_tasks[None, :, None],
            chosen_windows[None, :, :],
        ]
        vae, rae = sample.mean(axis=(0, 1, 2))
        sampled_delta[draw] = rae - vae
        sampled_relative[draw] = (rae - vae) / vae
    return {
        "scope": "Teacher-forced action loss only; no closed-loop success inference",
        "split": reference["split"],
        "training_seeds": seeds,
        "tasks": tasks,
        "windows_per_task": windows_per_task,
        "noise_repeats_per_window": reference["noise_repeats_per_window"],
        "evaluation_noise_seed": reference["evaluation_noise_seed"],
        "split_manifest_sha256": reference["split_manifest_sha256"],
        "action_stats_sha256": reference["action_stats_sha256"],
        "per_seed": per_seed,
        "vae_action_loss_mean": actual_vae,
        "rae_action_loss_mean": actual_rae,
        "rae_minus_vae": actual_rae - actual_vae,
        "relative_change": (actual_rae - actual_vae) / actual_vae,
        "bootstrap": {
            "method": "resample training seeds, tasks, then paired windows with replacement",
            "draws": draws,
            "seed": bootstrap_seed,
            "delta_95pct_ci": np.quantile(sampled_delta, [0.025, 0.975]).tolist(),
            "relative_delta_95pct_ci": np.quantile(sampled_relative, [0.025, 0.975]).tolist(),
            "fraction_of_draws_with_rae_loss_higher": float((sampled_delta > 0).mean()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=parse_pair, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261024)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    reports = [(seed, json.loads(path.read_text())) for seed, path in args.pair]
    result = aggregate(reports, draws=args.draws, bootstrap_seed=args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "training_seeds", "vae_action_loss_mean", "rae_action_loss_mean", "rae_minus_vae"
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
