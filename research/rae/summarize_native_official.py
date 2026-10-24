#!/usr/bin/env python3
"""Strictly audit the released checkpoint on both full RoboTwin modes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.utils.eval_manifest import load_manifest

MODES = ("demo_clean", "demo_randomized")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_rows(rows: list[dict], *, tasks: tuple[str, ...], episodes: int = 100) -> dict:
    expected = {(mode, task, index) for mode in MODES for task in tasks for index in range(episodes)}
    actual = set()
    manifests = {}
    successes = {}
    for row in rows:
        key = (row["mode"], row["task"], row["episode"])
        if key in actual:
            raise ValueError(f"Duplicate official episode: {key}")
        actual.add(key)
        if type(row["success"]) is not bool or type(row["scene_seed"]) is not int or row["scene_seed"] < 100000:
            raise ValueError(f"Invalid official episode outcome/seed: {key}")
        manifest_key = key[:2]
        old_hash = manifests.setdefault(manifest_key, row["manifest_hash"])
        if old_hash != row["manifest_hash"]:
            raise ValueError(f"Manifest changed within official task: {manifest_key}")
        successes[manifest_key] = successes.get(manifest_key, 0) + int(row["success"])
    if actual != expected:
        raise ValueError(f"Official coverage mismatch: {len(expected - actual)} missing, {len(actual - expected)} extra")
    per_task = [
        {"mode": mode, "task": task, "successes": successes[(mode, task)],
         "episodes": episodes, "success_rate": successes[(mode, task)] / episodes,
         "manifest_hash": manifests[(mode, task)]}
        for mode in MODES for task in tasks
    ]
    by_mode = [
        {"mode": mode, "tasks": len(tasks), "episodes": len(tasks) * episodes,
         "successes": sum(successes[(mode, task)] for task in tasks),
         "task_balanced_success_rate": sum(successes[(mode, task)] / episodes for task in tasks) / len(tasks)}
        for mode in MODES
    ]
    return {"scope": "Released OpenWAM RoboTwin-Full checkpoint, native RTX/Vulkan RT",
            "instruction_type": "unseen", "episodes_per_task": episodes,
            "total_episodes": len(expected), "by_mode": by_mode, "per_task": per_task}


def labtasker_rows(registry: dict, *, episodes: int) -> tuple[list[dict], tuple[str, ...]]:
    import sys

    robotwin_dir = Path(__file__).resolve().parents[2] / "benchmarks/robotwin"
    sys.path.insert(0, str(robotwin_dir))
    import labtasker_runtime as rt

    tasks = tuple(rt.ROBOTWIN_TASKS)
    if len(tasks) != 50 or len(set(tasks)) != 50:
        raise ValueError("Expected exactly 50 distinct RoboTwin tasks")
    if set(registry) != set(MODES):
        raise ValueError("Registry must contain exactly demo_clean and demo_randomized")
    checkpoint_paths = {Path(registry[mode]["checkpoint"]).expanduser().resolve() for mode in MODES}
    checkpoint_hashes = {registry[mode]["checkpoint_sha256"] for mode in MODES}
    if len(checkpoint_paths) != 1 or len(checkpoint_hashes) != 1:
        raise ValueError("Both official modes must use the same checkpoint")
    checkpoint = checkpoint_paths.pop()
    if sha256_file(checkpoint) != checkpoint_hashes.pop():
        raise ValueError("Released checkpoint SHA256 mismatch")

    rows = []
    seen_submissions = set()
    with rt.project_context(), rt.create_client(queue=None, auto_start_local_server=False) as client:
        for mode in MODES:
            submission_id = registry[mode]["submission_id"]
            if submission_id in seen_submissions:
                raise ValueError("Official modes must have separate submissions")
            seen_submissions.add(submission_id)
            batches = rt.list_robotwin_submission_tasks(client, submission_id, "run_eval")
            if not batches:
                raise ValueError(f"Empty official submission: {mode}")
            rt.validate_submission_coverage(batches)
            for batch in batches:
                if batch.status != "succeeded":
                    raise ValueError(f"Incomplete official Task: {batch.id} ({batch.status})")
                args = batch.args
                if (args.get("mode") != mode or args.get("task") not in tasks
                        or args.get("instruction_type") != "unseen"
                        or args.get("total_episodes") != episodes
                        or Path(args["model"]["checkpoint"]).resolve() != checkpoint):
                    raise ValueError(f"Official Task settings/checkpoint mismatch: {batch.id}")
                result = rt.validated_task_result(batch)
                if result is None:
                    raise ValueError(f"Invalid official Task result: {batch.id}")
                manifest = load_manifest(Path(args["manifest_path"]), expected_manifest_hash=args["manifest_hash"])
                rt.validate_manifest(manifest, args, minimum_entries=episodes)
                for episode in result["episodes"]:
                    index = episode["episode"]
                    entry = manifest["entries"][index]
                    if (episode["seed"] != entry["seed"]
                            or episode["instruction"] != entry["instructions"]["unseen"]):
                        raise ValueError(f"Official result/manifest mismatch: {batch.id}/{index}")
                    rows.append({"mode": mode, "task": args["task"], "episode": index,
                                 "scene_seed": episode["seed"], "success": episode["success"],
                                 "manifest_hash": args["manifest_hash"]})
    return rows, tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.episodes <= 0:
        raise ValueError("Episode count must be positive")
    registry = json.loads(args.registry.read_text())
    rows, tasks = labtasker_rows(registry, episodes=args.episodes)
    report = summarize_rows(rows, tasks=tasks, episodes=args.episodes)
    report["submissions"] = registry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report["by_mode"], sort_keys=True))


if __name__ == "__main__":
    main()
