"""Run a bounded official-checkpoint diagnostic on an isolated native queue.

By default print a plan without touching the simulator, task server, or GPU.
Execution reuses sealed production scenes and has its own submission/route.
Probe results never count as completion of the full registered benchmark.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import signal
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.robotwin import labtasker_runtime as rt
from benchmarks.utils.eval_manifest import load_manifest
from research.rae.run_native_eval import OFFICIAL_SHA256, load_checkpoints, stop_workers, write_json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS = ("stack_blocks_three", "open_microwave", "shake_bottle_horizontally")


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--robotwin-python", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--execute", action="store_true", help="Launch only on explicitly authorized resources")
    args = parser.parse_args(argv)
    args.tasks = args.tasks.split(",")
    if (not args.tasks or len(set(args.tasks)) != len(args.tasks)
            or not set(args.tasks) <= set(rt.task_names())):
        parser.error("--tasks must be distinct RoboTwin task names")
    if args.gpu < 0 or not 1 <= args.episodes <= 100 or min(args.poll_seconds, args.timeout_seconds) < 1:
        parser.error("invalid GPU, episode count, polling interval, or timeout")
    for name in ("checkpoints", "manifest_dir", "run_dir", "robotwin_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    # Resolving a venv Python symlink can escape that environment.
    args.orchestrator_python = Path(sys.executable).absolute()
    args.policy_python = args.policy_python.expanduser().absolute()
    args.robotwin_python = args.robotwin_python.expanduser().absolute()
    return args


def build_plan(args, registry):
    official = registry["official"]
    if official.get("checkpoint_sha256") != OFFICIAL_SHA256:
        raise ValueError("Official checkpoint digest does not match the registered release")
    filename = "checkpoint_step_118655.safetensors"
    if official.get("weights_file", filename) != filename:
        raise ValueError("Official checkpoint step mismatch")
    checkpoint = Path(official["directory"]).expanduser().resolve() / filename
    if args.run_dir == args.manifest_dir.parent or (args.run_dir / "controller-state.json").exists():
        raise ValueError("Diagnostic output must be separate from the production run directory")
    definition = {"checkpoint": str(checkpoint), "checkpoint_sha256": OFFICIAL_SHA256,
                  "tasks": args.tasks, "episodes_per_task": args.episodes,
                  "manifest_dir": str(args.manifest_dir), "run_dir": str(args.run_dir),
                  "mode": "demo_clean", "instruction_type": "unseen"}
    identity = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()[:16]
    submission = f"official-probe-{identity}"
    route = f"openwam-robotwin-{submission}"
    submit = [str(args.orchestrator_python), str(ROOT / "benchmarks/robotwin/labtasker_submit.py"),
              "--auto-start-local-server", "--operation", "run_eval", "--mode", "demo_clean",
              "--tasks", ",".join(args.tasks), "--episodes", str(args.episodes),
              "--episode-batch-size", "5", "--instruction-type", "unseen",
              "--manifest-dir", str(args.manifest_dir), "--route", route,
              "--submission-id", submission, "--policy-python", str(args.policy_python),
              "--", "--ckpt-dir", str(checkpoint.parent), "--ckpt-name", filename]
    worker = [str(args.orchestrator_python), str(ROOT / "benchmarks/robotwin/labtasker_worker.py"),
              "--operation", "run_eval", "--route", route, "--gpu", str(args.gpu),
              "--robotwin-path", str(args.robotwin_root), "--robotwin-python", str(args.robotwin_python),
              "--server-python", str(args.policy_python), "--idle-timeout", str(args.timeout_seconds),
              "--output-dir", str(args.run_dir / "worker-output")]
    return {**definition, "scope": "Bounded diagnostic only; not a full benchmark result",
            "submission_id": submission, "route": route,
            "expected_batches": len(args.tasks) * math.ceil(args.episodes / 5),
            "expected_episodes": len(args.tasks) * args.episodes,
            "submit_command": submit, "worker_command": worker}


def summarize_batches(batches, plan):
    """Reject missing coverage, wrong models, and unsealed scene/instruction drift."""
    rt.validate_submission_coverage(batches)
    if len(batches) != plan["expected_batches"] or any(t.status != "succeeded" for t in batches):
        raise ValueError("Official diagnostic is incomplete")
    rows = {}
    for batch in batches:
        args = batch.args
        if (args.get("task") not in plan["tasks"] or args.get("mode") != plan["mode"]
                or args.get("instruction_type") != plan["instruction_type"]
                or args.get("total_episodes") != plan["episodes_per_task"]
                or Path(args["model"]["checkpoint"]).resolve() != Path(plan["checkpoint"])):
            raise ValueError(f"Diagnostic Task settings/model mismatch: {batch.id}")
        result = rt.validated_task_result(batch)
        if result is None:
            raise ValueError(f"Invalid accepted diagnostic result: {batch.id}")
        manifest = load_manifest(Path(args["manifest_path"]), expected_manifest_hash=args["manifest_hash"])
        rt.validate_manifest(manifest, args, minimum_entries=plan["episodes_per_task"])
        for row in result["episodes"]:
            entry = manifest["entries"][row["episode"]]
            if (row["seed"] != entry["seed"]
                    or row.get("instruction") != entry["instructions"][plan["instruction_type"]]):
                raise ValueError(f"Diagnostic result/manifest mismatch: {batch.id}")
            key = (args["task"], row["episode"])
            if key in rows:
                raise ValueError(f"Duplicate accepted diagnostic episode: {key}")
            rows[key] = {**row, "task": args["task"], "manifest_hash": args["manifest_hash"]}
    expected = {(task, episode) for task in plan["tasks"] for episode in range(plan["episodes_per_task"])}
    if set(rows) != expected:
        raise ValueError("Official diagnostic episode coverage mismatch")
    per_task = []
    for task in plan["tasks"]:
        successes = sum(row["success"] for row in rows.values() if row["task"] == task)
        per_task.append({"task": task, "episodes": plan["episodes_per_task"], "successes": successes,
                         "success_rate": successes / plan["episodes_per_task"]})
    successes = sum(row["success"] for row in rows.values())
    return {"scope": plan["scope"], "submission_id": plan["submission_id"], "route": plan["route"],
            "checkpoint_sha256": plan["checkpoint_sha256"], "whole_benchmark_complete": False,
            "episodes": [rows[key] for key in sorted(rows)], "per_task": per_task,
            "successes": successes, "episode_count": len(rows), "success_rate": successes / len(rows),
            "all_zero_success": successes == 0}


def execute(args, plan, registry):
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = args.run_dir / "probe-plan.json"
        if saved.exists() and json.loads(saved.read_text()) != plan:
            raise ValueError("Existing diagnostic plan changed; use its original settings")
        write_json(saved, plan)
        official_registry = args.run_dir / "official-registry.json"
        write_json(official_registry, {"official": registry["official"]})
        load_checkpoints(official_registry, verify_bytes=True)
        # Probe the selected device, not merely another RTX on a mixed host.
        name = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=name",
                                        "--format=csv,noheader"], text=True)
        if "RTX" not in name.upper() and "A6000" not in name.upper():
            raise ValueError(f"Selected GPU lacks the required native RTX device: {name.strip()}")
        subprocess.run([str(args.orchestrator_python), str(ROOT / "research/rae/preflight_native_host.py"),
                        "--robotwin-root", str(args.robotwin_root),
                        "--robotwin-python", str(args.robotwin_python),
                        "--policy-python", str(args.policy_python),
                        "--output", str(args.run_dir / "native-preflight.json")], cwd=ROOT, check=True)
        subprocess.run(plan["submit_command"], cwd=ROOT, check=True)
        processes = []
        deadline = time.monotonic() + args.timeout_seconds
        try:
            with (args.run_dir / "worker.log").open("a") as log:
                processes.append(subprocess.Popen(plan["worker_command"], cwd=ROOT, stdout=log,
                                                   stderr=subprocess.STDOUT, start_new_session=True))
            with rt.project_context(), rt.create_client(queue=None) as client:
                while True:
                    batches = rt.list_robotwin_submission_tasks(client, plan["submission_id"], "run_eval")
                    counts = rt.count_task_statuses(batches)
                    print(json.dumps({"submission_id": plan["submission_id"], "counts": counts}), flush=True)
                    if counts["failed"]:
                        raise RuntimeError("Official diagnostic has failed infrastructure Tasks; inspect its worker.log")
                    if counts["succeeded"] == plan["expected_batches"]:
                        summary = summarize_batches(batches, plan)
                        write_json(args.run_dir / "summary.json", summary)
                        return 2 if summary["all_zero_success"] else 0
                    if any(p.poll() is not None for p in processes):
                        raise RuntimeError("Official diagnostic Worker exited before completion")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Official diagnostic time budget exhausted; accepted Task results are preserved")
                    time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
        finally:
            stop_workers(processes)


def main(argv=None):
    args = parse(argv)
    registry = json.loads(args.checkpoints.read_text())
    plan = build_plan(args, registry)
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        return execute(args, plan, registry)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
