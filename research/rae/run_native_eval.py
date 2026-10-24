"""Resume the registered RoboTwin native evaluation on one RTX host.

This runner owns submission order and local Workers. Labtasker owns task retries
and durable task results. Stop on a failed batch, repair the cause, requeue the
failed Task, then rerun this command with the same state directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from benchmarks.robotwin import labtasker_runtime as rt

ROOT = Path(__file__).resolve().parents[2]
SELECTED = (
    "beat_block_hammer,hanging_mug,open_microwave,place_bread_skillet,"
    "place_burger_fries,place_dual_shoes,shake_bottle_horizontally,"
    "stack_blocks_three,stack_bowls_three,stack_bowls_two"
)
MATCHED = tuple(f"seed{seed}_{kind}" for seed in (42, 43, 44) for kind in ("vae", "rae"))
ABLATIONS = ("seed42_reconstruction_svae48", "seed42_action_svae48")
MODELS = MATCHED + ABLATIONS + ("official",)
OFFICIAL_SHA256 = "d07ff6f8cfb627ebf47313e65af38773fabcb3861d52fdc7c3bc5e7bee33f6fe"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_checkpoints(path: Path, *, verify_bytes: bool) -> dict[str, dict[str, str]]:
    raw = json.loads(path.read_text())
    if not raw or not set(raw) <= set(MODELS):
        raise ValueError(f"checkpoint registry may contain only {list(MODELS)}")
    models = {}
    for name in MODELS:
        if name not in raw:
            continue
        item = raw[name]
        directory = Path(item["directory"]).expanduser().resolve(strict=True)
        sha = item["checkpoint_sha256"]
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError(f"invalid checkpoint SHA256: {name}")
        expected_file = "checkpoint_step_118655.safetensors" if name == "official" else "checkpoint_step_1000.safetensors"
        if item.get("weights_file", expected_file) != expected_file:
            raise ValueError(f"wrong registered checkpoint step: {name}")
        if name == "official" and sha != OFFICIAL_SHA256:
            raise ValueError("released official checkpoint SHA256 mismatch")
        weight = directory / expected_file
        if not weight.is_file() or not (directory / "config.yaml").is_file():
            raise FileNotFoundError(f"incomplete checkpoint directory: {name}: {directory}")
        if not (directory / "normalization_stats.npy").is_file() or not (directory / "tokenizer").is_dir():
            raise FileNotFoundError(f"missing checkpoint sidecar: {name}: {directory}")
        if (name.endswith("rae") or "svae48" in name) and not (directory / "dinov2/config.json").is_file():
            raise FileNotFoundError(f"missing DINOv2 sidecar: {name}: {directory}")
        if verify_bytes and digest(weight) != sha:
            raise ValueError(f"checkpoint SHA256 mismatch: {name}: {weight}")
        models[name] = {"directory": str(directory), "checkpoint": str(weight), "checkpoint_sha256": sha}
    return models


def plan(models: dict | None = None) -> list[tuple[str, str, str]]:
    available = set(MODELS if models is None else models)
    stages = [("manifest_clean", "build_manifest", "demo_clean"),
              ("manifest_randomized", "build_manifest", "demo_randomized")]
    stages += [(name, "run_eval", "demo_clean") for name in MATCHED + ABLATIONS if name in available]
    if "official" in available:
        stages += [("official_clean", "run_eval", "demo_clean"),
                   ("official_randomized", "run_eval", "demo_randomized")]
    return stages


def reconcile_state(state: dict | None, models: dict) -> dict:
    """Preserve submitted IDs while adding stages for newly trained checkpoints."""
    if state is None:
        state = {"models": {}, "submission_ids": {}, "completed": []}
    for name, previous in state["models"].items():
        if models.get(name) != previous:
            raise ValueError(f"registered checkpoint changed since native submission: {name}")
    state["models"] = models
    for name, _, _ in plan(models):
        state["submission_ids"].setdefault(name, f"native-{uuid.uuid4().hex[:12]}")
    return state


def stage_args(stage: str, operation: str, mode: str, args: argparse.Namespace, models: dict) -> list[str]:
    manifest = args.run_dir / "manifests"
    command = [str(args.orchestrator_python), "benchmarks/robotwin/labtasker_submit.py",
               "--auto-start-local-server", "--operation", operation, "--mode", mode,
               "--tasks", "all" if stage.startswith("manifest_") or stage.startswith("official_") else SELECTED,
               "--episodes", "100", "--manifest-dir", str(manifest),
               "--submission-id", args.submission_ids[stage]]
    if operation == "run_eval":
        name = "official" if stage.startswith("official_") else stage
        command += ["--episode-batch-size", "5", "--instruction-type", "unseen",
                    "--policy-python", str(args.policy_python), "--",
                    "--ckpt-dir", models[name]["directory"], "--ckpt-name", Path(models[name]["checkpoint"]).name]
    return command


def workers(operation: str, args: argparse.Namespace) -> list[subprocess.Popen]:
    processes = []
    (args.run_dir / "logs").mkdir(parents=True, exist_ok=True)
    for gpu in args.gpus:
        command = [str(args.orchestrator_python), "benchmarks/robotwin/labtasker_worker.py",
                   "--operation", operation, "--gpu", str(gpu),
                   "--robotwin-path", str(args.robotwin_root),
                   "--robotwin-python", str(args.robotwin_python),
                   "--idle-timeout", "86400", "--output-dir", str(args.run_dir / "worker-output" / operation)]
        if operation == "run_eval":
            command += ["--server-python", str(args.policy_python)]
        log = (args.run_dir / "logs" / f"{operation}_gpu{gpu}.log").open("a")
        processes.append(subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True, env=os.environ.copy()))
        log.close()
    return processes


def stop_workers(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def submission_counts(stage: str, operation: str, args: argparse.Namespace) -> tuple[int, dict[str, int]]:
    submission_id = args.submission_ids[stage]
    expected = (50 if stage.startswith("manifest_") else
                1000 if stage.startswith("official_") else 200)
    with rt.project_context(), rt.create_client(queue=None) as client:
        tasks = rt.list_robotwin_submission_tasks(client, submission_id, operation)
    if len(tasks) != expected:
        raise RuntimeError(f"{stage}: incomplete submission {len(tasks)}/{expected}; rerun to fill missing Tasks")
    rt.validate_submission_coverage(tasks)
    return expected, rt.count_task_statuses(tasks)


def wait_submission(stage: str, operation: str, args: argparse.Namespace, processes: list[subprocess.Popen]) -> None:
    previous = None
    while True:
        expected, counts = submission_counts(stage, operation, args)
        current = (counts["succeeded"], counts["running"], counts["pending"], counts["failed"])
        if current != previous:
            print(f"{stage}: succeeded/running/pending/failed = {current}", flush=True)
            previous = current
        if counts["failed"]:
            raise RuntimeError(f"{stage}: {counts['failed']} exhausted or cancelled Tasks; inspect and requeue")
        if counts["succeeded"] == expected:
            return
        if any(process.poll() is not None for process in processes):
            raise RuntimeError(f"{stage}: Worker exited before submission completed")
        time.sleep(args.poll_seconds)


def write_registries(args: argparse.Namespace, models: dict) -> None:
    submission = args.submission_ids
    paired = {str(seed): {kind: {**models[f"seed{seed}_{kind}"], "submission_id": submission[f"seed{seed}_{kind}"]}
                          for kind in ("vae", "rae")} for seed in (42, 43, 44)}
    ablation = {"42": {"vae": paired["42"]["vae"],
                       "reconstruction_svae48": {**models[ABLATIONS[0]], "submission_id": submission[ABLATIONS[0]]},
                       "action_svae48": {**models[ABLATIONS[1]], "submission_id": submission[ABLATIONS[1]]}}}
    official = {mode: {**models["official"], "submission_id": submission[f"official_{mode.split('_')[1]}"]}
                for mode in ("demo_clean", "demo_randomized")}
    for name, value in (("paired", paired), ("ablation", ablation), ("official", official)):
        write_json(args.run_dir / f"{name}-submissions.json", value)


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True, help="Host-local registry of available checkpoint directories and SHA256 values")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--robotwin-python", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, required=True)
    parser.add_argument("--gpus", default="0", help="Comma-separated RTX GPU indices; start with GPU 0 for the native smoke")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--skip-checkpoint-hash", action="store_true", help="Planning only; never use for final native runs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without submitting or launching Workers")
    args = parser.parse_args(argv)
    # Keep the venv launcher path: resolving its symlink can escape the venv.
    args.orchestrator_python = Path(sys.executable).absolute()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.robotwin_root = args.robotwin_root.expanduser().resolve()
    args.robotwin_python = args.robotwin_python.expanduser().absolute()
    args.policy_python = args.policy_python.expanduser().absolute()
    args.gpus = tuple(int(value) for value in args.gpus.split(","))
    if not args.gpus or len(args.gpus) != len(set(args.gpus)) or min(args.gpus) < 0 or args.poll_seconds < 1:
        parser.error("invalid GPU list or polling interval")
    if args.skip_checkpoint_hash and not args.dry_run:
        parser.error("--skip-checkpoint-hash is permitted only with --dry-run")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    models = load_checkpoints(args.checkpoints, verify_bytes=not args.skip_checkpoint_hash)
    state_path = args.run_dir / "controller-state.json"
    state = reconcile_state(json.loads(state_path.read_text()) if state_path.exists() else None, models)
    if not args.dry_run:
        write_json(state_path, state)
    args.submission_ids = state["submission_ids"]
    if args.dry_run:
        for name, operation, mode in plan(models):
            print(name, " ".join(stage_args(name, operation, mode, args, models)))
        return 0
    preflight = [str(args.orchestrator_python), "research/rae/preflight_native_host.py",
                 "--robotwin-root", str(args.robotwin_root), "--robotwin-python", str(args.robotwin_python),
                 "--policy-python", str(args.policy_python), "--output", str(args.run_dir / "native-preflight.json")]
    subprocess.run(preflight, cwd=ROOT, check=True)
    active_operation = None
    active_workers: list[subprocess.Popen] = []
    try:
        for name, operation, mode in plan(models):
            if name in state["completed"]:
                expected, counts = submission_counts(name, operation, args)
                if counts["succeeded"] != expected:
                    raise RuntimeError(f"{name}: recorded complete but Labtasker has {counts['succeeded']}/{expected} succeeded")
                continue
            if operation != active_operation:
                stop_workers(active_workers)
                active_workers = []
                active_operation = operation
            command = stage_args(name, operation, mode, args, models)
            subprocess.run(command, cwd=ROOT, check=True)
            if not active_workers:
                active_workers = workers(operation, args)
            wait_submission(name, operation, args, active_workers)
            state["completed"].append(name)
            write_json(state_path, state)
        if set(models) != set(MODELS):
            print(f"Pending checkpoints: {sorted(set(MODELS) - set(models))}", flush=True)
            return 0
        write_registries(args, models)
        for name, module in (("paired", "summarize_native_paired"),
                             ("ablation", "summarize_native_ablation"),
                             ("official", "summarize_native_official")):
            subprocess.run([str(args.orchestrator_python), "-m", f"research.rae.{module}",
                            "--registry", str(args.run_dir / f"{name}-submissions.json"),
                            "--output", str(args.run_dir / f"native-{name}-summary.json")], cwd=ROOT, check=True)
    finally:
        stop_workers(active_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
