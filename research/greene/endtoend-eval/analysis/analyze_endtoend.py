#!/usr/bin/env python3
"""Audit + frozen analysis of the Greene endtoend re-evaluation (protocol.json, array 18893528).

Read-only over the run dirs. Writes
  research/greene/records/endtoend-audit.json     (per-cell validity checks)
  research/greene/records/endtoend-analysis.json  (frozen comparisons, bootstrap CIs, per-seed effects)
and prints a markdown summary.

Analysis is exactly protocol.json["analysis"]: task-equal-weight success-rate difference per condition;
scene x seed crossed paired bootstrap (10000 resamples, seed 426); per-seed effects; claim only if all
1620 valid outcomes are present, the 95% CI lower bound > 0 and all 3 seed effects share its sign.

    python3 analyze_endtoend.py --array 18893528
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import subprocess

import numpy as np

REPO = "/scratch/zz4330/OpenWAM"
RUNS = "/scratch/zz4330/openwam-runtime/runs"
SCENES = "/scratch/zz4330/openwam-runtime/endtoend-root/endtoend-20260923/scenes"
PROTOCOL = "research/greene/endtoend-eval/protocol.json"
AUDIT = "research/greene/records/endtoend-audit.json"
ANALYSIS = "research/greene/records/endtoend-analysis.json"
ROUTES, SEEDS = ["wan", "svae", "pca"], [42, 43, 44]
MAE_MAX = 1.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    with open(path) as fh:
        return json.load(fh)


def sacct(array):
    out = subprocess.run(["sacct", "-n", "-X", "-P", "-j", array, "-o",
                          "JobIDRaw,JobID,State,Start,End,ExitCode,NodeList,Partition"],
                         capture_output=True, text=True, check=True).stdout
    tasks = {}
    for line in out.splitlines():
        raw, jid, state, start, end, code, node, part = line.split("|")
        if "_" in jid and "[" not in jid:
            tasks[int(jid.split("_")[1])] = dict(job=raw, state=state.split()[0], start=start, end=end,
                                                  exit_code=code, node=node, partition=part)
    return tasks


def audit_cell(i, cell, job, protocol, deadline):
    route, seed, task, cond = cell
    problems = []
    check = lambda ok, msg: ok or problems.append(msg)
    run = os.path.join(RUNS, "endtoend-eval-%s-seed%d-%s" % (route, seed, job["job"]))
    check(job["state"] == "COMPLETED", "slurm state %s" % job["state"])
    check(datetime.datetime.fromisoformat(job["start"]).astimezone() < deadline, "started after deadline")
    check(load(os.path.join(run, "exit.json"))["returncode"] == 0, "exit.json returncode != 0")

    rec = load(os.path.join(run, "run-record.json"))
    check(rec["checkpoint_sha256"] == protocol["checkpoints"]["%s-seed%d" % (route, seed)]["checkpoint_sha256"],
          "checkpoint sha256 mismatch")
    # run_cell hashes the 8 runtime scripts; each must be one of the 13 frozen files, byte-identical
    check(len(rec["code_sha256"]) == 8 and all(protocol["code_sha256"].get(p) == h
                                               for p, h in rec["code_sha256"].items()),
          "runtime script sha256 differs from frozen protocol")
    check(rec["protocol"] == protocol, "run-record protocol differs from frozen protocol.json")
    check([list(c) for c in rec["cells"]] == [[task, cond]], "cells %s" % rec["cells"])
    check(load(os.path.join(run, "cohort-integrity.json"))["passed"] is True, "cohort integrity not passed")
    check(load(os.path.join(run, "simulation-preflight-%s-%d.json" % (route, seed)))["passed"] is True,
          "deploy gate not passed")

    manifest = load(os.path.join(SCENES, task, "manifest.json"))
    expected = [int(x["seed"]) for x in manifest["accepted"]][:protocol["evaluation"]["scenes_per_task"]]
    s = load(os.path.join(run, task, cond, "summary.json"))
    check(s["complete"] is True and s["all_selected_attempted"] is True, "summary not complete")
    recs = s["records"]
    check([r["scene_seed"] for r in recs] == expected, "scene seeds differ from first-20 accepted")
    files = sorted(glob.glob(os.path.join(run, task, cond, "seed-*", "result.json")))
    check(sorted(load(f)["scene_seed"] for f in files) == sorted(expected), "per-episode result.json set differs")
    mae = 0.0
    for r in recs:
        check((r["route"], r["training_seed"], r["task"], r["condition"]) == (route, seed, task, cond),
              "record identity %s" % r["scene_seed"])
        check(r["status"] == "complete", "episode %s status %s" % (r["scene_seed"], r["status"]))
        want = int(np.random.SeedSequence([r["scene_seed"], seed, 811]).generate_state(1, dtype=np.uint32)[0])
        check(r["action_seed"] == want, "action seed %s" % r["scene_seed"])
        check(r["steps"] <= r["step_limit"], "steps > limit %s" % r["scene_seed"])
        mae = max([mae] + list(r["initial_render_mae"].values()))
    check(mae <= MAE_MAX, "render MAE %.3f > %.1f" % (mae, MAE_MAX))

    cache = 0
    with open(os.path.join(run, "episode-rng.jsonl")) as fh:
        for line in fh:
            d = json.loads(line)
            cache = max(cache, d.get("cache_entries_after_prediction", 0))
            check(d.get("cache_capacity", 128) == 128, "cache capacity != 128")
    check(cache < 128, "prompt cache reached capacity (eviction possible)")
    with open(os.path.join(run, "server.log"), errors="replace") as fh:
        check("Traceback" not in fh.read(), "Traceback in server.log")

    return {
        "index": i, "route": route, "seed": seed, "task": task, "condition": cond,
        "slurm_job": job["job"], "node": job["node"], "partition": job["partition"],
        "start": job["start"], "end": job["end"], "run_dir": run,
        "episodes": len(recs), "successes": sum(r["success"] for r in recs),
        "technical_failures": s["technical_failures"] + sum(r["status"] != "complete" for r in recs),
        "max_initial_render_mae": mae, "max_prompt_cache_entries": cache,
        "valid": not problems, "problems": problems,
    }, {r["scene_seed"]: bool(r["success"]) for r in recs if r["status"] == "complete"}


def effect(Y, a, b, seeds_idx=None, scenes_idx=None):
    """Task-equal-weight success-rate difference a - b. Y[task][route] is a (seed, scene) 0/1 array."""
    d = []
    for t in sorted(Y):
        D = Y[t][a] - Y[t][b]
        if seeds_idx is not None:
            D = D[seeds_idx]
        if scenes_idx is not None:
            D = D[:, scenes_idx[t]]
        d.append(D.mean())
    return float(np.mean(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", required=True)
    a = ap.parse_args()
    protocol = load(os.path.join(REPO, PROTOCOL))
    assert protocol["frozen"] is True
    deadline = datetime.datetime.fromisoformat(protocol["deadline"])

    # frozen code still byte-identical
    code_now = {p: sha256(p) for p in protocol["code_sha256"]}
    code_ok = code_now == protocol["code_sha256"]

    jobs = sacct(a.array)
    cells, outcomes = [], {}
    for i, cell in enumerate(protocol["cells"]):
        if i not in jobs:
            cells.append({"index": i, "cell": cell, "valid": False, "problems": ["no slurm task"]})
            continue
        c, o = audit_cell(i, cell, jobs[i], protocol, deadline)
        cells.append(c)
        outcomes[tuple(cell)] = o

    n_valid = sum(c.get("episodes", 0) - c.get("technical_failures", 0) for c in cells if c["valid"])
    tech = sum(c.get("technical_failures", 0) for c in cells)
    all_valid = code_ok and all(c["valid"] for c in cells) and n_valid == protocol["evaluation"]["episodes"]
    audit = {
        "updated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "slurm_array": a.array, "protocol": PROTOCOL, "deadline": protocol["deadline"],
        "frozen_code_unchanged": code_ok,
        "cells_total": len(cells), "cells_valid": sum(c["valid"] for c in cells),
        "episodes_valid": n_valid, "technical_failures": tech,
        "partitions": sorted({c.get("partition") for c in cells if c.get("partition")}),
        "first_start": min(c["start"] for c in cells if "start" in c),
        "last_end": max(c["end"] for c in cells if "end" in c),
        "all_1620_valid": all_valid,
        "checks": ["slurm COMPLETED + exit 0", "job start before deadline", "checkpoint sha256 == frozen",
                   "runtime script sha256 (8) == frozen", "run-record protocol == protocol.json", "cohort integrity passed",
                   "deploy gate passed", "scene seeds == first 20 accepted", "20 per-episode result.json",
                   "record identity", "action seed == SeedSequence([scene, seed, 811])", "steps <= native limit",
                   "initial render MAE <= 1", "prompt cache never full (no eviction)", "no Traceback in server.log"],
        "cells": cells,
    }
    with open(os.path.join(REPO, AUDIT), "w") as fh:
        json.dump(audit, fh, indent=2)
        fh.write("\n")

    # ---- frozen analysis ----
    tasks, conds = protocol["evaluation"]["tasks"], protocol["evaluation"]["conditions"]
    scenes = {t: [int(x["seed"]) for x in load(os.path.join(SCENES, t, "manifest.json"))["accepted"]][:20]
              for t in tasks}
    comparisons = [("svae", "wan", "primary"), ("pca", "wan", "primary"), ("pca", "svae", "secondary")]
    B = 10000
    result = {"condition": {}}
    rates = {}
    for cond in conds:
        Y = {t: {r: np.array([[float(outcomes[(r, s, t, cond)][sc]) for sc in scenes[t]] for s in SEEDS])
                 for r in ROUTES} for t in tasks}
        rates[cond] = {r: {"task_equal": float(np.mean([Y[t][r].mean() for t in tasks])),
                           "per_task": {t: float(Y[t][r].mean()) for t in tasks},
                           "per_seed": {str(s): float(np.mean([Y[t][r][k].mean() for t in tasks]))
                                        for k, s in enumerate(SEEDS)},
                           "successes": int(sum(Y[t][r].sum() for t in tasks))} for r in ROUTES}
        rng = np.random.default_rng(426)      # same resample stream for every comparison in a condition
        draws = [(rng.integers(0, 3, 3), {t: rng.integers(0, 20, 20) for t in tasks}) for _ in range(B)]
        out = {}
        for x, y, kind in comparisons:
            est = effect(Y, x, y)
            boot = np.array([effect(Y, x, y, si, sc) for si, sc in draws])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            per_seed = {str(s): effect(Y, x, y, np.array([k])) for k, s in enumerate(SEEDS)}
            same_sign = all(np.sign(v) == np.sign(est) and v != 0 for v in per_seed.values())
            if all_valid and lo > 0 and same_sign:
                verdict = "improvement (claim rule met)"
            elif all_valid and hi < 0 and same_sign:
                verdict = "negative (CI upper < 0, same sign in all seeds)"
            else:
                verdict = "inconclusive / limited"
            out["%s-%s" % (x, y)] = {"kind": kind, "estimate": est, "ci95": [float(lo), float(hi)],
                                     "per_seed": per_seed, "per_task": {t: float((Y[t][x] - Y[t][y]).mean())
                                                                        for t in tasks},
                                     "seeds_same_sign": bool(same_sign), "verdict": verdict}
        result["condition"][cond] = out
    analysis = {
        "updated_at": audit["updated_at"], "protocol": PROTOCOL, "audit": AUDIT,
        "all_1620_valid": all_valid, "spec": protocol["analysis"],
        "bootstrap": {"resamples": B, "seed": 426, "unit": "crossed: 3 training seeds with replacement x "
                      "20 scenes with replacement per task; same draws for all routes (paired)",
                      "ci": "percentile 2.5/97.5"},
        "success_rates": rates, **result,
    }
    with open(os.path.join(REPO, ANALYSIS), "w") as fh:
        json.dump(analysis, fh, indent=2)
        fh.write("\n")

    print("audit: %d/%d cells valid, %d valid episodes, %d technical, frozen code unchanged=%s, all valid=%s"
          % (audit["cells_valid"], len(cells), n_valid, tech, code_ok, all_valid))
    for c in cells:
        if not c["valid"]:
            print("  INVALID", c["index"], c.get("problems"))
    for cond in conds:
        print("\n##", cond, " ".join("%s=%.3f" % (r, rates[cond][r]["task_equal"]) for r in ROUTES))
        for k, v in result["condition"][cond].items():
            print("  %-9s %+.3f [%+.3f, %+.3f] seeds %s  -> %s"
                  % (k, v["estimate"], v["ci95"][0], v["ci95"][1],
                     " ".join("%+.3f" % e for e in v["per_seed"].values()), v["verdict"]))


if __name__ == "__main__":
    main()
