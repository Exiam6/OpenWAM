"""Score only the prespecified complete factorial; retain missing/failed attempts.

2026-10-17 revision (before any closed-loop result existed): every contrast is a
single-variable paired difference, per TOP_PRIORITY/EXPERIMENT_PROTOCOL.md rule 12.
The two-variable diagonal registered on 2026-10-10 is kept, labelled confounded.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

CONTRASTS = [
    # id, candidate, baseline, variable changed, role
    ("H1_shift16", "raw768_native_shift16", "raw768_native", "video shift 5 -> 16.4317 (native head)", "primary"),
    ("H2_widehead", "raw768_wide", "raw768_native", "native head -> RAE wide head (shift 5)", "primary"),
    ("H3a_widehead_at_shift16", "raw768_wide_shift16", "raw768_native_shift16", "native head -> wide head (shift 16.4317)", "primary"),
    ("H3b_shift16_at_widehead", "raw768_wide_shift16", "raw768_wide", "video shift 5 -> 16.4317 (wide head)", "primary"),
    ("C_combination_two_variables", "raw768_wide_shift16", "raw768_native", "wide head AND shift 16.4317 together", "confounded"),
    ("S_representation", "svae48_native", "raw768_native", "768-dim raw latent -> 48-dim S-VAE latent", "secondary"),
]


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
            "by_task": {
                t: {
                    "successes": sum(
                        bool(r.get("success")) and r["status"] == "complete"
                        for k, r in indexed.items()
                        if k[0] == v and k[2] == t
                    ),
                    "attempted": sum(k[0] == v and k[2] == t for k in indexed),
                }
                for t in tasks
            },
            "by_seed": {
                str(s): {
                    "successes": sum(
                        bool(r.get("success")) and r["status"] == "complete"
                        for k, r in indexed.items()
                        if k[0] == v and k[1] == s
                    ),
                    "attempted": sum(k[0] == v and k[1] == s for k in indexed),
                }
                for s in seeds
            },
        }
        for v in arms
    }
    output = {
        "complete": len(indexed) == len(expected),
        "attempted": len(indexed),
        "planned": len(expected),
        "counts": counts,
        "verified_success_rate_improvement": False,
        "contrasts": {},
    }
    if not output["complete"]:
        output["reason"] = "Incomplete cohort; no improvement decision."
        return output

    def success(arm, seed, task, scene):
        r = indexed[(arm, seed, task, int(scene["seed"]))]
        return float(r["status"] == "complete" and bool(r.get("success")))

    for cid, candidate, baseline, variable, role in CONTRASTS:
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
        paired_sensitivity = []
        for ti, task in enumerate(tasks):
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
                        np.mean([success(candidate, s, task, c) - success(baseline, s, task, c) for c in valid for s in seeds])
                    )
                    if valid
                    else None,
                    "difference_all_scenes": float(diffs[ti].mean()),
                }
            )
        output["contrasts"][cid] = {
            "candidate": candidate,
            "baseline": baseline,
            "variable": variable,
            "role": role,
            "task_balanced_success_difference": point,
            "scene_cluster_paired_95pct_interval": interval,
            "bootstrap_draws": draws,
            "bootstrap_seed": 20261010,
            "by_training_seed": {str(s): float(np.mean([v[:, i].mean() for v in diffs])) for i, s in enumerate(seeds)},
            "by_task": paired_sensitivity,
            "technical_failures_count_as_unsuccessful": True,
            "training_seed_generalization_not_established": True,
            "supported": bool(point >= 0.05 and interval[0] > 0),
            "refuted": bool(point <= -0.05 and interval[1] < 0),
        }
    # Backward-compatible field for the original registered diagonal, now labelled confounded.
    output["primary"] = dict(output["contrasts"]["C_combination_two_variables"], note="two variables; see H1/H2/H3a/H3b")
    output["verified_success_rate_improvement"] = any(
        c["supported"] for k, c in output["contrasts"].items() if c["role"] == "primary"
    )
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
