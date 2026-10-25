#!/usr/bin/env python3
"""Audit native success for the seed-42 VAE and two S-VAE48 ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.rae.summarize_native_paired import TASKS, labtasker_rows

VARIANTS = ("vae", "reconstruction_svae48", "action_svae48")
PAIRS = (("reconstruction_svae48", "vae"),
         ("action_svae48", "vae"),
         ("action_svae48", "reconstruction_svae48"))


def summarize_rows(rows: list[dict], *, tasks: tuple[str, ...] = TASKS,
                   episodes_per_task: int = 100, draws: int = 20_000,
                   rng_seed: int = 20261025,
                   variants: tuple[str, ...] = VARIANTS,
                   pairs: tuple[tuple[str, str], ...] = PAIRS,
                   scope: str = "Native RTX/Vulkan RT manifest-paired single-seed S-VAE48 ablations") -> dict:
    if not tasks or episodes_per_task <= 0 or draws <= 0:
        raise ValueError("Tasks, episodes, and bootstrap draws must be positive")
    if len(variants) < 2 or len(set(variants)) != len(variants) or any(
        treatment not in variants or control not in variants or treatment == control
        for treatment, control in pairs
    ):
        raise ValueError("Invalid ablation variants or comparisons")
    expected = {(42, variant, task, episode)
                for variant in variants for task in tasks
                for episode in range(episodes_per_task)}
    indexed: dict[tuple, dict] = {}
    for row in rows:
        key = (row["training_seed"], row["variant"], row["task"], row["episode"])
        if key in indexed:
            raise ValueError(f"Duplicate native ablation result: {key}")
        if (type(row["scene_seed"]) is not int or row["scene_seed"] < 100000
                or type(row["success"]) is not bool
                or not isinstance(row["manifest_hash"], str)
                or not row["manifest_hash"].startswith("sha256:")):
            raise ValueError(f"Invalid native ablation result: {key}")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError(f"Native ablation coverage mismatch: "
                         f"{len(expected - set(indexed))} missing, {len(set(indexed) - expected)} extra")

    values = np.empty((len(tasks), episodes_per_task, len(variants)), dtype=np.int8)
    manifest_hashes: dict[str, str] = {}
    per_task = []
    for ti, task in enumerate(tasks):
        task_scene_seeds: set[int] = set()
        for episode in range(episodes_per_task):
            triplet = [indexed[(42, variant, task, episode)] for variant in variants]
            identities = {(row["scene_seed"], row["manifest_hash"]) for row in triplet}
            if len(identities) != 1:
                raise ValueError(f"Scene or manifest mismatch: {task}/{episode}")
            manifest_hash = triplet[0]["manifest_hash"]
            previous = manifest_hashes.setdefault(task, manifest_hash)
            if previous != manifest_hash:
                raise ValueError(f"Manifest changes within task: {task}")
            scene_seed = triplet[0]["scene_seed"]
            if scene_seed in task_scene_seeds:
                raise ValueError(f"Repeated scene seed within task: {task}/{episode}")
            task_scene_seeds.add(scene_seed)
            values[ti, episode] = [row["success"] for row in triplet]
        per_task.append({"task": task, "episodes_per_variant": episodes_per_task,
                         "successes": {variant: int(values[ti, :, vi].sum())
                                       for vi, variant in enumerate(variants)}})

    rng = np.random.default_rng(rng_seed)
    comparisons = []
    for treatment, control in pairs:
        treatment_index, control_index = variants.index(treatment), variants.index(control)
        difference = values[:, :, treatment_index].astype(np.int16) - values[:, :, control_index]
        bootstrap = np.empty(draws, dtype=np.float64)
        for start in range(0, draws, 256):
            count = min(256, draws - start)
            task_indices = rng.integers(len(tasks), size=(count, len(tasks), 1))
            episode_indices = rng.integers(episodes_per_task,
                                           size=(count, len(tasks), episodes_per_task))
            bootstrap[start:start + count] = difference[task_indices, episode_indices].mean(axis=(1, 2))
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        comparisons.append({
            "treatment": treatment, "control": control,
            "difference": float(difference.mean()),
            "treatment_only": int((difference == 1).sum()),
            "control_only": int((difference == -1).sum()),
            "both_success": int(((values[:, :, treatment_index] == 1)
                                  & (values[:, :, control_index] == 1)).sum()),
            "both_failure": int(((values[:, :, treatment_index] == 0)
                                  & (values[:, :, control_index] == 0)).sum()),
            "bootstrap_95pct_ci": [float(low), float(high)],
        })
    return {
        "scope": scope,
        "mode": "demo_clean", "instruction_type": "unseen", "training_seed": 42,
        "tasks": list(tasks), "variants": list(variants),
        "episodes_per_task_per_variant": episodes_per_task,
        "episodes_per_variant": len(tasks) * episodes_per_task,
        "registered_rows": len(rows), "manifest_hashes": manifest_hashes,
        "success_rates": {variant: float(values[:, :, vi].mean())
                          for vi, variant in enumerate(variants)},
        "successes": {variant: int(values[:, :, vi].sum())
                      for vi, variant in enumerate(variants)},
        "comparisons": comparisons, "per_task": per_task,
        "bootstrap": {"draws": draws, "rng_seed": rng_seed,
                      "resampling": "task, then paired episode; one training seed"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    registry = json.loads(args.registry.read_text())
    rows = labtasker_rows(registry, tasks=TASKS, episodes=args.episodes,
                          seeds=(42,), variants=VARIANTS)
    summary = summarize_rows(rows, episodes_per_task=args.episodes,
                             draws=args.bootstrap_draws)
    summary["submissions"] = registry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"success_rates": summary["success_rates"],
                      "comparisons": summary["comparisons"]}, sort_keys=True))


if __name__ == "__main__":
    main()
