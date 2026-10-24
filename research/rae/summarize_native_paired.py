#!/usr/bin/env python3
"""Audit and summarize three-seed, manifest-paired native RoboTwin results.

The CLI reads authoritative Labtasker Task results on the RTX evaluation host.
The pure `summarize_rows` function accepts the same normalized episode rows so
its pairing and bootstrap can be checked without a simulator or Labtasker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.utils.eval_manifest import load_manifest

TASKS = (
    "beat_block_hammer", "hanging_mug", "open_microwave", "place_bread_skillet",
    "place_burger_fries", "place_dual_shoes", "shake_bottle_horizontally",
    "stack_blocks_three", "stack_bowls_three", "stack_bowls_two",
)
SEEDS = (42, 43, 44)
VARIANTS = ("vae", "rae")


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def summarize_rows(
    rows: list[dict], *, tasks: tuple[str, ...] = TASKS,
    episodes_per_task: int = 100, draws: int = 20_000, rng_seed: int = 20261024,
) -> dict:
    if not tasks or episodes_per_task <= 0 or draws <= 0:
        raise ValueError("Tasks, episodes, and bootstrap draws must be positive")
    expected = {(seed, variant, task, episode)
                for seed in SEEDS for variant in VARIANTS for task in tasks
                for episode in range(episodes_per_task)}
    actual = set()
    indexed = {}
    for row in rows:
        key = (row["training_seed"], row["variant"], row["task"], row["episode"])
        if key in actual:
            raise ValueError(f"Duplicate native episode result: {key}")
        actual.add(key)
        if type(row["scene_seed"]) is not int or type(row["success"]) is not bool:
            raise ValueError(f"Invalid scene seed or success value: {key}")
        if not isinstance(row["manifest_hash"], str) or not row["manifest_hash"].startswith("sha256:"):
            raise ValueError(f"Invalid manifest hash: {key}")
        indexed[key] = row
    if actual != expected:
        missing, extra = expected - actual, actual - expected
        raise ValueError(f"Native episode coverage mismatch: {len(missing)} missing, {len(extra)} extra")

    tensor = np.empty((len(SEEDS), len(tasks), episodes_per_task, 2), dtype=np.int8)
    manifests = {}
    discordant = {"rae_only": 0, "vae_only": 0, "both_success": 0, "both_failure": 0}
    per_seed_task = []
    for si, seed in enumerate(SEEDS):
        for ti, task in enumerate(tasks):
            for episode in range(episodes_per_task):
                a = indexed[(seed, "vae", task, episode)]
                b = indexed[(seed, "rae", task, episode)]
                if (a["scene_seed"], a["manifest_hash"]) != (b["scene_seed"], b["manifest_hash"]):
                    raise ValueError(f"VAE/RAE scene or manifest mismatch: {seed}/{task}/{episode}")
                prior = manifests.setdefault(task, a["manifest_hash"])
                if prior != a["manifest_hash"]:
                    raise ValueError(f"Manifest differs across training seeds: {task}")
                vae, rae = a["success"], b["success"]
                tensor[si, ti, episode] = (vae, rae)
                label = (
                    "both_success" if vae and rae else "vae_only" if vae else
                    "rae_only" if rae else "both_failure"
                )
                discordant[label] += 1
            per_seed_task.append({
                "training_seed": seed,
                "task": task,
                "episodes": episodes_per_task,
                "vae_successes": int(tensor[si, ti, :, 0].sum()),
                "rae_successes": int(tensor[si, ti, :, 1].sum()),
                "rae_minus_vae_rate": float((tensor[si, ti, :, 1] - tensor[si, ti, :, 0]).mean()),
            })

    difference = tensor[..., 1].astype(np.int16) - tensor[..., 0].astype(np.int16)
    rng = np.random.default_rng(rng_seed)
    bootstrap = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 256):
        count = min(256, draws - start)
        seed_indices = rng.integers(len(SEEDS), size=(count, len(SEEDS), 1, 1))
        task_indices = rng.integers(len(tasks), size=(count, len(SEEDS), len(tasks), 1))
        episode_indices = rng.integers(
            episodes_per_task, size=(count, len(SEEDS), len(tasks), episodes_per_task)
        )
        bootstrap[start:start + count] = difference[
            seed_indices, task_indices, episode_indices
        ].mean(axis=(1, 2, 3))
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    per_task = []
    for ti, task in enumerate(tasks):
        per_task.append({
            "task": task,
            "vae_success_rate": float(tensor[:, ti, :, 0].mean()),
            "rae_success_rate": float(tensor[:, ti, :, 1].mean()),
            "rae_minus_vae": float(difference[:, ti, :].mean()),
            "episodes_per_variant": len(SEEDS) * episodes_per_task,
        })
    return {
        "scope": "Native RTX/Vulkan RT manifest-paired RoboTwin success; 3 matched training seeds",
        "mode": "demo_clean",
        "instruction_type": "unseen",
        "training_seeds": list(SEEDS),
        "tasks": list(tasks),
        "episodes_per_task_per_seed": episodes_per_task,
        "episodes_per_variant": len(SEEDS) * len(tasks) * episodes_per_task,
        "manifest_hashes": manifests,
        "vae_success_rate": float(tensor[..., 0].mean()),
        "rae_success_rate": float(tensor[..., 1].mean()),
        "rae_minus_vae": float(difference.mean()),
        "discordant_pairs": discordant,
        "bootstrap": {
            "draws": draws,
            "rng_seed": rng_seed,
            "resampling": "training seed, then task, then paired episode",
            "delta_95pct_ci": [float(low), float(high)],
        },
        "per_task": per_task,
        "per_seed_task": per_seed_task,
    }


def labtasker_rows(registry: dict, *, tasks: tuple[str, ...], episodes: int) -> list[dict]:
    # Import only on the RTX host: this cluster's active training environment
    # intentionally does not install Labtasker while eight-GPU jobs are live.
    import sys

    robotwin_dir = Path(__file__).resolve().parents[2] / "benchmarks/robotwin"
    sys.path.insert(0, str(robotwin_dir))
    import labtasker_runtime as rt

    rows = []
    with rt.project_context(), rt.create_client(queue=None, auto_start_local_server=False) as client:
        for seed in SEEDS:
            for variant in VARIANTS:
                spec = registry[str(seed)][variant]
                checkpoint = Path(spec["checkpoint"]).expanduser().resolve()
                if file_sha256(checkpoint) != spec["checkpoint_sha256"]:
                    raise ValueError(f"Checkpoint SHA256 mismatch: {seed}/{variant}")
                batch_tasks = rt.list_robotwin_submission_tasks(client, spec["submission_id"], "run_eval")
                if not batch_tasks:
                    raise ValueError(f"Empty native eval submission: {seed}/{variant}")
                rt.validate_submission_coverage(batch_tasks)
                for batch in batch_tasks:
                    if batch.status != "succeeded":
                        raise ValueError(f"Incomplete native eval Task: {batch.id} ({batch.status})")
                    args = batch.args
                    if (
                        args.get("task") not in tasks or args.get("mode") != "demo_clean"
                        or args.get("instruction_type") != "unseen"
                        or args.get("total_episodes") != episodes
                        or Path(args["model"]["checkpoint"]).resolve() != checkpoint
                    ):
                        raise ValueError(f"Native Task setting/checkpoint mismatch: {batch.id}")
                    result = rt.validated_task_result(batch)
                    if result is None:
                        raise ValueError(f"Invalid native Task result: {batch.id}")
                    manifest = load_manifest(
                        Path(args["manifest_path"]), expected_manifest_hash=args["manifest_hash"]
                    )
                    for episode in result["episodes"]:
                        index = episode["episode"]
                        entry = manifest["entries"][index]
                        if (
                            episode["seed"] != entry["seed"]
                            or episode["instruction"] != entry["instructions"]["unseen"]
                        ):
                            raise ValueError(f"Native result/manifest mismatch: {batch.id}/{index}")
                        rows.append({
                            "training_seed": seed, "variant": variant,
                            "task": args["task"], "episode": index,
                            "scene_seed": episode["seed"],
                            "success": episode["success"],
                            "manifest_hash": args["manifest_hash"],
                        })
    return rows


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
    rows = labtasker_rows(registry, tasks=TASKS, episodes=args.episodes)
    result = summarize_rows(rows, episodes_per_task=args.episodes, draws=args.bootstrap_draws)
    result["submissions"] = registry
    result["registered_rows"] = len(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in (
        "episodes_per_variant", "vae_success_rate", "rae_success_rate", "rae_minus_vae"
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
