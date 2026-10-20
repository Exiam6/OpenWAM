"""Independent outcome, pairing, capacity and sensitivity audit. CPU only."""
import datetime as dt
import hashlib
import html
import itertools
import json
import math
from pathlib import Path
import struct

import numpy as np

R = Path("/home/zifanz4/openwam-runtime")
OUT = R / "ex6-b_contributionaudit_1020"
TASKS = ["adjust_bottle", "handover_block", "place_object_basket"]
SEEDS = [42, 43, 44]
CONTRASTS = {
    "raw_minus_vae": ("raw768_native", "vae48_native", 921018),
    "compact_minus_vae": ("svae48_native", "vae48_native", 921018),
    "raw_minus_compact": ("raw768_native", "svae48_native", 921017),
    "wide_minus_native": ("raw768_wide", "raw768_native", 921017),
    "shift_with_native": ("raw768_native_shift16", "raw768_native", 921017),
    "shift_with_wide": ("raw768_wide_shift16", "raw768_wide", 921017),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_pair(rows):
    for key in ["scene_seed", "training_seed", "condition", "initial_state",
                "evaluation_instruction", "action_seed", "step_limit"]:
        assert len({json.dumps(r[key], sort_keys=True) for r in rows}) == 1, key
    assert all(r["status"] == "complete" and type(r["success"]) is bool for r in rows)
    assert all(len(r["initial_render_mae"]) == 3 and max(r["initial_render_mae"].values()) <= 1
               for r in rows)


def bootstrap(d, rng_seed, resample_tasks=False):
    """Preserve pairing and crossed seed/scene structure, with optional task sampling."""
    rng = np.random.default_rng(rng_seed)
    seeds = rng.integers(0, 3, (10000, 3))
    scenes = rng.integers(0, 20, (10000, 3, 20))
    values = d[seeds[:, :, None, None], np.arange(3)[None, None, :, None], scenes[:, None, :, :]]
    if resample_tasks:
        tasks = rng.integers(0, 3, (10000, 3))
        values = np.take_along_axis(values, tasks[:, None, :, None], axis=2)
    return np.quantile(values.mean(axis=(1, 2, 3)), [.025, .975]).tolist()


def sensitivity(left, right, rng_seed):
    d = np.asarray(left, float)-np.asarray(right, float)
    assert d.shape == (3, 3, 20)
    per_seed = d.mean(axis=(1, 2))
    signed = np.asarray([np.mean(per_seed*np.asarray(signs))
                         for signs in itertools.product([-1, 1], repeat=3)])
    observed = float(d.mean())
    return {
        "difference": observed,
        "registered_fixed_task_bootstrap_ci95": bootstrap(d, rng_seed),
        "exploratory_task_resampled_ci95": bootstrap(d, 921020, True),
        "per_task": dict(zip(TASKS, d.mean(axis=(0, 2)).tolist())),
        "per_seed": dict(zip(map(str, SEEDS), per_seed.tolist())),
        "leave_one_task_out": {task: float(np.delete(d, i, axis=1).mean()) for i, task in enumerate(TASKS)},
        "leave_one_seed_out": {str(seed): float(np.delete(d, i, axis=0).mean()) for i, seed in enumerate(SEEDS)},
        "gains": int((d > 0).sum()), "losses": int((d < 0).sum()),
        "seed_cluster_signflip_two_sided_p": float((np.abs(signed) >= abs(observed)-1e-12).mean()),
        "seed_cluster_signflip_positive_one_sided_p": float((signed >= observed-1e-12).mean()),
        "limitations": [
            "Sensitivity analyses added after observing outcomes; they do not replace the registered primary test.",
            "Task sampling treats the three development tasks as exchangeable, an unverified assumption.",
            "Seed sign flips assume independent exchangeable seed-pair effects under a symmetric null. Only eight sign assignments exist.",
            "Neither analysis establishes generalization to new benchmarks or a population of training seeds.",
        ],
    }


def header_shapes(path):
    with Path(path).open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        assert size < 100 << 20
        header = json.loads(f.read(size))
    return {name: row["shape"] for name, row in header.items() if name != "__metadata__"}


def capacity():
    frozen = ("video_backbone.text_encoder.", "video_backbone.reason1.", "video_backbone.vae.",
              "video_backbone.image_encoder.", "video_backbone.video_encoder.")
    records = {}
    for route in ["raw768_native", "svae48_native", "raw768_wide", "vae48_native"]:
        base = R / ("ex5-a_vaebaseline_1018" if route == "vae48_native" else "rae-factorial-20261010")
        result_path = base / "runs" / f"{route}-seed42/result.json"
        result = json.loads(result_path.read_text())
        checkpoint = Path(result["checkpoint"])
        shapes = header_shapes(checkpoint)
        excluded = frozen + (("video_backbone.dit.head.",) if route == "raw768_wide" else ())
        # configure_rae_head() explicitly freezes the bypassed native output head.
        count = sum(math.prod(shape) for name, shape in shapes.items() if not name.startswith(excluded))
        init = json.loads((base / "runs" / f"{route}-seed42/initialization.json").read_text())
        assert count == init["trainable_parameters"], (route, count, init["trainable_parameters"])
        records[route] = {"trainable_parameters": count, "checkpoint": str(checkpoint),
                          "parameter_count_matches_training_record": True,
                          "header_only_read": True, "checkpoint_bytes": checkpoint.stat().st_size,
                          "video_input_channels": shapes["video_backbone.dit.patch_embedding.weight"][1],
                          "excluded_frozen_prefixes": list(excluded)}
    return records


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "audit.json").exists(), "Do not overwrite a completed audit"
    sources = [R / "ex4-g_calibratedfactorial_1017/success-summary.json",
               R / "ex5-a_vaebaseline_1018/success-summary.json"]
    rows = {}
    for source in sources:
        summary = json.loads(source.read_text())
        assert summary["complete"]
        for stored in summary["rows"]:
            identity = tuple(stored["identity"])
            if identity in rows:
                assert rows[identity]["result_sha256"] == stored["result_sha256"]
                continue
            path = Path(stored["result_path"])
            assert sha(path) == stored["result_sha256"]
            actual = json.loads(path.read_text())
            assert actual == stored["result"]
            assert (actual["route"], actual["training_seed"], actual["task"], actual["scene_seed"]) == identity
            trace_path = path.with_name("trace.jsonl")
            assert sha(trace_path) == stored["trace_sha256"]
            trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
            assert len(trace) == actual["steps"] and [t["step"] for t in trace] == list(range(actual["steps"]))
            assert bool(trace[-1]["success"]) == actual["success"]
            assert all(len(t["action"]) == 20 and np.isfinite(t["action"]).all() for t in trace)
            assert all(t["physics_ticks_this_action"] >= 0 for t in trace)
            calls = [call for t in trace for call in t["planner_calls"]]
            failures = sum(c.get("status") != "Success" for c in calls)
            # Match recorded counts; inspect exact status labels if the runtime differs.
            assert len(calls) == stored["planner_calls"]
            rows[identity] = dict(stored, observed_planner_status_counts={
                status: sum(c.get("status") == status for c in calls)
                for status in sorted({str(c.get("status")) for c in calls})},
                prefix_replan_ms=[t["latency_ms"] for t in trace if t["step"] in [8, 16, 24, 32]],
                trace_steps=len(trace))
    assert len(rows) == 1080
    arms = sorted({identity[0] for identity in rows})
    assert len(arms) == 6
    arrays = {}
    for arm in arms:
        data = np.zeros((3, 3, 20))
        for i, seed in enumerate(SEEDS):
            for j, task in enumerate(TASKS):
                selected = sorted((key, value) for key, value in rows.items()
                                  if key[:3] == (arm, seed, task))
                assert len(selected) == 20
                for k, (key, row) in enumerate(selected):
                    cohort = [rows[(other, seed, task, key[3])]["result"] for other in arms]
                    validate_pair(cohort)
                    data[i, j, k] = row["result"]["success"]
        arrays[arm] = data
    comparisons = {name: sensitivity(arrays[a], arrays[b], rng) for name, (a, b, rng) in CONTRASTS.items()}
    v = json.loads(sources[1].read_text())
    primary = comparisons["raw_minus_vae"]
    original = v["comparisons"]["primary_raw_minus_vae"]
    assert np.isclose(primary["difference"], original["difference"])
    assert np.allclose(primary["registered_fixed_task_bootstrap_ci95"], original["ci95"])
    models = capacity()
    timing = {}
    for arm in arms:
        timing[arm] = {}
        for task in TASKS:
            episodes = [v for k, v in rows.items() if k[0] == arm and k[2] == task]
            values = [float(np.median(e["prefix_replan_ms"])) for e in episodes
                      if len(e["prefix_replan_ms"]) == 4]
            timing[arm][task] = {"episodes_with_all_four_early_replans": len(values),
                                "median_episode_replan_ms": float(np.median(values)) if values else None}
    record = {
        "time": dt.datetime.now(dt.timezone.utc).isoformat(), "passed": True,
        "unique_corrected_episodes": len(rows), "source_sha256": {str(p): sha(p) for p in sources},
        "counts": {arm: {"successes": int(data.sum()), "episodes": int(data.size),
                          "per_task": dict(zip(TASKS, data.sum(axis=(0, 2)).astype(int).tolist()))}
                   for arm, data in arrays.items()},
        "comparisons": comparisons, "capacity": models, "observational_latency": timing,
        "latency_limitation": "Historical asynchronous jobs, not a controlled speed benchmark. Each episode contributes the median of steps8/16/24/32; missing prefixes are explicitly counted. No compute-efficiency claim.",
        "attribution": {
            "supported_scope": "Raw DINO encoder/latent route improves known development-cohort success vs Wan VAE under matched data, minibatches, core initialization and steps.",
            "task_concentration": "All gains occur in adjust_bottle; removing that task eliminates the effect.",
            "unidentified": ["Encoder pretraining versus representation objective", "Latent dimension versus dimension-specific projection capacity",
                             "Equal-FLOP or equal-wall-time benefit", "Broad task/seed generalization"],
            "not_demonstrated": ["Raw versus compact superiority", "Wide-head gain", "Shift16 gain"],
            "wide_head_location": "Video DiT RAE output head; the action backbone is unchanged. Historical prose calling this a wide action head is imprecise.",
            "engineering_separation": "Native planner calibration and frozen-VAE precision fixes are implementation corrections, not evidence of a novel representation effect.",
            "novelty": "Reproducing a known RAE design is not itself a new method; currently the contribution is a controlled empirical finding and validated implementation fixes."
        },
    }
    (OUT / "audit.json").write_text(json.dumps(record, indent=2) + "\n")
    audit_rows = [{"identity": list(k), "result_path": v["result_path"], "result_sha256": v["result_sha256"],
                   "trace_sha256": v["trace_sha256"], "trace_steps": v["trace_steps"],
                   "planner_status_counts": v["observed_planner_status_counts"]} for k, v in sorted(rows.items())]
    (OUT / "verified-episodes.json").write_text(json.dumps(audit_rows, indent=2) + "\n")
    table = "".join(f"<tr><td>{html.escape(name)}</td><td>{100*x['difference']:+.2f}</td>"
                    f"<td>{html.escape(str([round(100*z,2) for z in x['registered_fixed_task_bootstrap_ci95']]))}</td>"
                    f"<td>{html.escape(str([round(100*z,2) for z in x['exploratory_task_resampled_ci95']]))}</td></tr>"
                    for name, x in comparisons.items())
    (OUT / "index.html").write_text("<!doctype html><meta charset=utf-8><title>RAE contribution audit</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:40px auto}td,th{padding:10px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style>"
        "<h1>RAE contribution audit</h1><p>1080 unique completed episodes independently re-read and paired. "
        "Registered inference and post-hoc sensitivity are separate.</p>"
        "<table><tr><th>Contrast</th><th>Difference (pp)</th><th>Registered fixed-task 95% interval (pp)</th>"
        "<th>Exploratory task-resampled interval (pp)</th></tr>"+table+"</table>"
        "<p>All observed representation gains come from adjust_bottle. Three training seeds and three known tasks "
        "do not establish broad generalization. Encoder pretraining, latent dimension and projection size change together.</p>"
        '<p><a href="audit.json">Full results and limitations</a> · <a href="verified-episodes.json">Episode audit</a></p>')
    print(json.dumps({"passed": True, "episodes": len(rows), "primary": primary,
                      "capacity": models, "report": str(OUT / "index.html")}, indent=2))


if __name__ == "__main__":
    main()
