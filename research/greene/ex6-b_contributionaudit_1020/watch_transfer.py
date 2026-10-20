"""Bounded CPU-only readout for the registered transfer pilot; never launches jobs."""
import argparse
import datetime as dt
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess
import time

R = Path("/home/zifanz4/openwam-runtime")
STUDY = R / "ex6-a_tasktransfer_1020"
OUT = R / "ex6-b_contributionaudit_1020"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def validate_episode(path, task, arm, selected):
    row = read(path)
    assert row["task"] == task and row["route"] == arm and row["training_seed"] == 42
    assert row["scene_seed"] in selected and row["condition"] == "clean"
    if row["status"] != "complete":
        return {"status": "technical_failure", "source": str(path), "error": row.get("error")}
    assert type(row["success"]) is bool
    trace = [json.loads(line) for line in path.with_name("trace.jsonl").read_text().splitlines()]
    assert len(trace) == row["steps"] and trace
    assert [t["step"] for t in trace] == list(range(row["steps"]))
    assert bool(trace[-1]["success"]) == row["success"]
    assert all(len(t["action"]) == 20 and all(math.isfinite(v) for v in t["action"]) for t in trace)
    assert len(row["initial_render_mae"]) == 3 and max(row["initial_render_mae"].values()) <= 1
    return {"status": "verified", "success": row["success"], "source": str(path), "sha256": sha(path),
            "trace_sha256": sha(path.with_name("trace.jsonl")), "scene_seed": row["scene_seed"],
            "pairing": [row[k] for k in ["initial_state", "evaluation_instruction", "action_seed", "step_limit"]]}


def snapshot(protocol_hash):
    assert sha(STUDY / "protocol.json") == protocol_hash
    p = read(STUDY / "protocol.json")
    tasks = {}
    for task in p["tasks"]:
        job_id = "ex6a-" + task.replace("_", "-")
        worker = read(R / "nightly-gpu/runs" / job_id / "result.json")
        claim = read(R / "nightly-gpu/claims" / f"{job_id}.json")
        if claim and claim.get("host") == "gpu02":
            worker = claim.get("worker_result", {"status": claim["status"]})
        parent = read(STUDY / "tasks" / task / "result.json")
        cohort = read(STUDY / "tasks" / task / "code/scenes" / task / "manifest.json")
        selected = {r["seed"] for r in cohort["accepted"]} if cohort else set()
        state = {"worker_status": worker["status"] if worker else "pending",
                 "task_status": parent["status"] if parent else "not_started",
                 "expert_scenes": len(selected), "arms": {}, "complete_pairs": 0}
        if worker and worker.get("error"):
            state["worker_error"] = worker["error"]
            state["resource_interruption"] = "Another process started using this GPU" in worker["error"]
        for arm in p["arms"]:
            name = arm["name"]
            rows = []
            for path in sorted((STUDY / "tasks" / task / name / "episodes").glob("seed-*/result.json")):
                try:
                    rows.append(validate_episode(path, task, name, selected))
                except Exception as exc:
                    rows.append({"status": "audit_failure", "source": str(path), "error": repr(exc)})
            verified = [r for r in rows if r["status"] == "verified"]
            state["arms"][name] = {"planned": 5, "verified": len(verified),
                                  "successes": sum(r["success"] for r in verified),
                                  "missing_or_unverified": 5-len(verified), "rows": rows}
        for scene in selected:
            pair = [next((r for r in a["rows"] if r.get("scene_seed") == scene and r["status"] == "verified"), None)
                    for a in state["arms"].values()]
            if all(pair):
                assert len({json.dumps(r["pairing"]) for r in pair}) == 1
                state["complete_pairs"] += 1
        tasks[task] = state
    now = dt.datetime.now(dt.timezone.utc)
    verified = sum(a["verified"] for task in tasks.values() for a in task["arms"].values())
    complete = verified == 140 and all(t["complete_pairs"] == 5 for t in tasks.values())
    terminal = all(t["worker_status"] in ["complete", "failed"] for t in tasks.values())
    expired = now >= dt.datetime.fromisoformat(p["deadline"])
    return {"time": now.isoformat(), "protocol_sha256": protocol_hash,
            "planned_episodes": 140, "verified_episodes": verified, "complete": complete,
            "all_jobs_terminal": terminal, "expired": expired, "tasks": tasks,
            "interpretation": p["interpretation"],
            "conclusion": "Exploratory transfer pilot only; no multi-seed superiority claim." if complete
            else "Incomplete coverage. Missing or interrupted episodes are not policy failures; no overall superiority claim."}


def sync_remote():
    """Copy only tasks actually assigned to gpu02, never a local-owned attempt."""
    p = read(STUDY / "protocol.json")
    errors = []
    for task in p["tasks"]:
        job_id = "ex6a-" + task.replace("_", "-")
        claim = read(R / "nightly-gpu/claims" / f"{job_id}.json")
        if not claim or claim.get("host") != "gpu02":
            continue
        try:
            (STUDY / "tasks").mkdir(exist_ok=True)
            source = f"gpu02:{STUDY}/tasks/{task}"
            run = subprocess.run(["rsync", "-a", "--partial-dir=.rsync-partial",
                                  "--exclude=simulator-runtime/", "--exclude=__pycache__/",
                                  "--exclude=*.log", source, str(STUDY / "tasks") + "/"],
                                 text=True, capture_output=True, timeout=30)
            if run.returncode:
                errors.append({"task": task, "error": run.stderr[-1000:]})
        except Exception as exc:
            errors.append({"task": task, "error": repr(exc)})
    return errors


def publish(result):
    write(OUT / "transfer-status.json", result)
    rows = []
    for task, state in result["tasks"].items():
        arms = " · ".join(f"{html.escape(name)} {a['successes']}/{a['verified']} verified, {a['missing_or_unverified']} missing"
                          for name, a in state["arms"].items())
        rows.append(f"<tr><td>{html.escape(task)}</td><td>{html.escape(state['worker_status'])}</td><td>{arms}</td></tr>")
    body = ("<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=60>"
            "<title>New-task transfer audit</title><style>body{font:16px system-ui;margin:40px}td{padding:12px;border-bottom:1px solid #ddd}</style>"
            "<h1>New-task transfer audit</h1><p>"+html.escape(result["time"])+"</p><p>"
            +html.escape(result["conclusion"])+"</p><p>"
            +html.escape(result["interpretation"])+"</p><table>"+"".join(rows)+"</table>"
            '<p><a href="transfer-status.json">Full evidence</a></p>')
    temporary = OUT / "transfer-status.html.tmp"
    temporary.write_text(body)
    temporary.replace(OUT / "transfer-status.html")


def main(protocol_hash, once):
    while True:
        try:
            errors = sync_remote()
            result = snapshot(protocol_hash)
            if result["all_jobs_terminal"] and not errors:
                # One final copy after observing terminal workers, so a copy
                # started during a worker's last write cannot hide its outcome.
                errors = sync_remote()
                result = snapshot(protocol_hash)
            result["artifact_sync_errors"] = errors
            publish(result)
        except Exception as exc:
            write(OUT / "watch-error.json", {"time": dt.datetime.now(dt.timezone.utc).isoformat(),
                                            "error": repr(exc)})
            raise
        if once or (result["all_jobs_terminal"] and not errors) or result["expired"]:
            return
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    main(args.protocol_sha256, args.once)
