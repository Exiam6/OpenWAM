"""Score only the prespecified complete factorial; retain missing/failed attempts."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def summarize(plan, scenes, records, draws=10000):
    arms = plan["evaluation_variants"]
    seeds = plan["seeds"]
    tasks = plan["tasks"]
    expected = {(v, s, t, int(scene["seed"])) for v in arms for s in seeds for t in tasks for scene in scenes[t]}
    indexed = {}
    for row in records:
        key = (row["route"], row["training_seed"], row["task"], row["scene_seed"])
        if key not in expected or key in indexed:
            raise ValueError(f"unexpected or duplicate result: {key}")
        if row["status"] not in ["complete", "infrastructure_failure"]:
            raise ValueError("result is not terminal")
        indexed[key] = row
    counts = {
        v: {
            "successes": sum(
                bool(r.get("success")) and r["status"] == "complete" for k, r in indexed.items() if k[0] == v
            ),
            "attempted": sum(k[0] == v for k in indexed),
            "valid": sum(k[0] == v and r["status"] == "complete" for k, r in indexed.items()),
            "planned": sum(k[0] == v for k in expected),
        }
        for v in arms
    }
    output = {
        "complete": len(indexed) == len(expected),
        "attempted": len(indexed),
        "planned": len(expected),
        "counts": counts,
        "verified_success_rate_improvement": False,
    }
    if not output["complete"]:
        output["reason"] = "Incomplete cohort; no improvement decision."
        return output
    candidate = "raw768_wide_shift16"
    baseline = "raw768_native"

    def success(arm, seed, task, scene):
        r = indexed[(arm, seed, task, int(scene["seed"]))]
        return float(r["status"] == "complete" and bool(r.get("success")))

    diffs = [
        np.array([[success(candidate, s, t, c) - success(baseline, s, t, c) for s in seeds] for c in scenes[t]])
        for t in tasks
    ]
    point = float(np.mean([values.mean() for values in diffs]))
    rng = np.random.default_rng(20261010)
    samples = np.zeros(draws)
    for values in diffs:
        ids = rng.integers(0, len(values), size=(draws, len(values)))
        samples += values[ids].mean(axis=(1, 2)) / len(tasks)
    interval = np.quantile(samples, [0.025, 0.975]).tolist()
    output["primary"] = {
        "candidate": candidate,
        "baseline": baseline,
        "task_balanced_success_difference": point,
        "scene_cluster_paired_95pct_interval": interval,
        "bootstrap_draws": draws,
        "bootstrap_seed": 20261010,
        "by_training_seed": {str(s): float(np.mean([v[:, i].mean() for v in diffs])) for i, s in enumerate(seeds)},
        "technical_failures_count_as_unsuccessful": True,
        "training_seed_generalization_not_established": True,
    }
    output["verified_success_rate_improvement"] = point >= 0.05 and interval[0] > 0
    paired_sensitivity = []
    for task in tasks:
        valid = [
            scene
            for scene in scenes[task]
            if all(indexed[(v, s, task, int(scene["seed"]))]["status"] == "complete" for v in arms for s in seeds)
        ]
        paired_sensitivity.append(
            {
                "task": task,
                "shared_valid_scenes": len(valid),
                "difference": float(
                    np.mean(
                        [success(candidate, s, task, c) - success(baseline, s, task, c) for c in valid for s in seeds]
                    )
                )
                if valid
                else None,
            }
        )
    output["shared_valid_scene_sensitivity"] = paired_sensitivity
    return output


def main():
    plan = json.loads((ROOT / "protocol.json").read_text())
    scenes = {t: json.loads((ROOT / "scenes" / t / "manifest.json").read_text())["accepted"] for t in plan["tasks"]}
    records = []
    hashes = {}
    for path in (ROOT / "evaluation").glob("fresh/*/*/clean/seed-*/result.json"):
        records.append(json.loads(path.read_text()))
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = summarize(plan, scenes, records)
    result["result_sha256"] = hashes
    result["scope"] = (
        "Prespecified three-task fresh-scene comparison of five local research policies; not the official benchmark."
    )
    tmp = ROOT / f"success-summary.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(ROOT / "success-summary.json")
    print(json.dumps({k: v for k, v in result.items() if k != "result_sha256"}, indent=2))


if __name__ == "__main__":
    main()
