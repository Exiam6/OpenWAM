"""Prepare a local, non-executable RAE factorial draft from frozen snapshots.

This command only reads JSON/YAML/text and writes a new local output directory.
It never imports model code, connects to a host, or starts an experiment.
"""

import argparse
import difflib
import hashlib
import json
import statistics
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import yaml

BASES = {"raw768_native_shift16": "raw768_native", "raw768_wide_shift16": "raw768_wide"}
TASKS = ["adjust_bottle", "handover_block", "place_object_basket"]
SEEDS = [42, 43, 44]


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        rows = []
        for key in sorted(a.keys() | b.keys()):
            name = f"{path}.{key}".lstrip(".")
            if key not in a or key not in b:
                rows.append(name)
            else:
                rows.extend(differences(a[key], b[key], name))
        return rows
    return [] if a == b else [path]


def prepare(snapshot, output, proposed_runtime):
    sources = json.loads((snapshot / "sources.json").read_text())
    original = snapshot / "originals"
    for name, info in sources["files"].items():
        if hashlib.sha256((original / name).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"source checksum mismatch: {name}")
    if str(proposed_runtime) == sources["root"]:
        raise ValueError("proposed output must not be the frozen source runtime")
    output.mkdir(parents=True, exist_ok=False)
    checks = {}
    for arm, base in BASES.items():
        config = yaml.safe_load((original / "templates" / base / "config.yaml").read_text())
        candidate = deepcopy(config)
        candidate["model"]["video_backbone"]["shift_video"] = 16.4317
        candidate["training"]["output_path"] = str(proposed_runtime / "runs")
        candidate["project"]["output_dir"] = str(proposed_runtime / "hydra")
        diff = differences(config, candidate)
        expected = ["model.video_backbone.shift_video", "project.output_dir", "training.output_path"]
        if diff != expected:
            raise ValueError(f"unexpected configuration changes: {diff}")
        destination = output / "templates" / arm / "config.yaml"
        destination.parent.mkdir(parents=True)
        destination.write_text(yaml.safe_dump(candidate, sort_keys=False))
        if yaml.safe_load(destination.read_text()) != candidate:
            raise ValueError("YAML round trip changed configuration")
        checks[arm] = {
            "base": base,
            "changed_fields": diff,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "shift_action": candidate["model"]["action_backbone"]["shift_action"],
        }

    trainer = (original / "train.py").read_text()
    old = "if a.variant == 'raw768_wide':"
    new = "if getattr(model.video_backbone.dit, 'rae_head', None) is not None:"
    if trainer.count(old) != 1:
        raise ValueError("wide-head preflight check no longer matches frozen trainer")
    patched = trainer.replace(old, new)
    compile(patched, "train.py", "exec")
    # A reviewable patch only: the frozen runtime is never edited.
    (output / "wide-preflight.patch").write_text(
        "".join(
            difflib.unified_diff(
                trainer.splitlines(keepends=True),
                patched.splitlines(keepends=True),
                fromfile="a/train.py",
                tofile="b/train.py",
            )
        )
    )

    training = []
    for arm, base in BASES.items():
        for seed in SEEDS:
            result = json.loads((original / "runs" / f"{base}-seed{seed}" / "result.json").read_text())
            if not result["complete"] or not result["strict_reload"] or result["steps"] != 20000:
                raise ValueError("historical training run is incomplete")
            training.append(
                {
                    "variant": arm,
                    "seed": seed,
                    "proxy_source": f"{base}-seed{seed}",
                    "proxy_seconds": result["seconds"],
                    "proxy_peak_gib": result["peak_gib"],
                }
            )
    ledger = json.loads((original / "evaluation-ledger.json").read_text())["records"]
    by_cell = defaultdict(list)
    for row in ledger:
        if row["status"] == "complete":
            by_cell[(row["route"], row["task"])].append(row["seconds"])
    eval_cells = []
    arms = ["svae48_native", "raw768_native", "raw768_wide", *BASES]
    for arm in arms:
        for task in TASKS:
            base = BASES.get(arm, arm)
            vals = by_cell[(base, task)]
            if not vals:
                raise ValueError(f"missing historical evaluation durations: {base}/{task}")
            eval_cells.append(
                {
                    "variant": arm,
                    "task": task,
                    "proxy_base": base,
                    "historical_valid_attempts": len(vals),
                    "historical_mean_seconds": statistics.mean(vals),
                    "historical_max_seconds": max(vals),
                    "planned_attempts": 60,
                }
            )
    train_hours = sum(r["proxy_seconds"] for r in training) / 3600
    eval_hours = sum(r["historical_mean_seconds"] * r["planned_attempts"] for r in eval_cells) / 3600
    total = train_hours + eval_hours
    timing = {
        "training_gpu_hours_proxy": train_hours,
        "evaluation_gpu_hours_proxy": eval_hours,
        "total_gpu_hours_proxy": total,
        "two_gpu_ideal_wall_hours": total / 2,
        "two_gpu_wall_hours_with_25pct_allowance": total * 1.25 / 2,
        "training_runs": training,
        "evaluation_cells": eval_cells,
        "limitations": [
            "Shift16 timings are inferred from matching shift5 architectures, not measured.",
            "Reported run/episode seconds omit some process startup, staging, scene preparation, "
            "v2 diagnostics and model-load costs; GPU queue availability is unknown.",
            "Two-GPU wall time assumes perfect balancing and exclusive continuous availability.",
            "The 25 percent allowance is a planning assumption, not a measured bound.",
        ],
    }
    (output / "timing-estimate.json").write_text(json.dumps(timing, indent=2) + "\n")
    manifest = {
        "status": "draft_not_runnable",
        "experiment_started": False,
        "proposed_runtime": str(proposed_runtime),
        "source_runtime": sources["root"],
        "primary_candidate": "raw768_wide_shift16",
        "primary_control": "raw768_native",
        "seeds": SEEDS,
        "steps": 20000,
        "new_training_runs": training,
        "candidate_config_checks": checks,
        "closed_loop": {
            "tasks": TASKS,
            "arms": arms,
            "scenes_per_task": 20,
            "total_attempts": 900,
            "new_scene_manifest": None,
        },
        "device_allocation": None,
        "deadline": None,
        "remaining": [
            "User authorization to resume experiments and a resource/time budget.",
            "Freeze new independent study, full template sidecars, inputs and code hashes.",
            "Apply and GPU-test the wide-preflight patch in the new study only.",
            "Strict reload, exact initialization and paired-noise smoke checks for both new arms.",
            "Rehash reused checkpoints before fresh closed-loop evaluation.",
            "Freeze a disjoint fresh scene manifest and add planner/tick instrumentation.",
            "Execute the controlled experiment and measure paired closed-loop success.",
        ],
    }
    (output / "proposal.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in timing.items() if not isinstance(value, list)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposed-runtime", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.snapshot, args.output, args.proposed_runtime)


if __name__ == "__main__":
    main()
