#!/usr/bin/env python3
"""Train and evaluate the DINO/S-VAE 48-channel width ablation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from research.rae.run_additional_seeds import digest, ensure_eval_stopped, run_logged, verify_final

REPO = Path("/root/zifan/OpenWAM-rae")
PYTHON = Path("/root/zifan/openwam-envs/policy/bin/python")
RUNS = Path("/root/zifan/openwam-runs/wam_svae48_seed42")
WEIGHTS = Path("/data/zifan-openwam-rae/wam_svae48_seed42")
GRAFT = Path("/dev/shm/openwam-svae48-seed42-graft")
SVAE_STATE = Path("/root/zifan/openwam-runs/svae48_ablation_20261024_state.json")
VAE_BASELINE = Path("/root/zifan/openwam-runs/wam_pilot_20261023/vae")


def record(stage: str, **details: object) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    target = RUNS / "state.json"
    payload = {"stage": stage, "updated_unix": time.time(), **details}
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    print(json.dumps(payload, sort_keys=True), flush=True)


def wait_for(pid: int, marker: str) -> None:
    while True:
        proc = Path(f"/proc/{pid}")
        try:
            state = (proc / "stat").read_text().split(") ", 1)[1].split()[0]
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except FileNotFoundError:
            return
        if state == "Z":
            return
        if marker not in command:
            raise RuntimeError(f"PID {pid} is no longer {marker}")
        time.sleep(30)


def verify_svae() -> tuple[Path, str]:
    status = json.loads(SVAE_STATE.read_text())
    if status["stage"] != "complete":
        raise RuntimeError(f"S-VAE reducer training did not complete: {status}")
    checkpoint = Path(status["checkpoint"])
    actual = digest(checkpoint)
    if actual != status["checkpoint_sha256"]:
        raise ValueError("S-VAE reducer changed since stage-one verification")
    return checkpoint, actual


def build_graft(svae_path: Path, svae_hash: str, env: dict[str, str]) -> str:
    if not (GRAFT / "graft_manifest.json").is_file():
        if shutil.disk_usage("/dev/shm").free < 27_000_000_000:
            raise RuntimeError("Insufficient tmpfs for width-matched graft")
        record("graft_started", svae=str(svae_path))
        run_logged(
            [str(PYTHON), "research/rae/build_grafted_checkpoint.py", "--output", str(GRAFT),
             "--seed", "42", "--svae-path", str(svae_path)],
            RUNS / "graft.log", env=env,
        )
    manifest = json.loads((GRAFT / "graft_manifest.json").read_text())
    if manifest.get("seed") != 42 or manifest.get("svae_sha256") != svae_hash:
        raise ValueError("Width-matched graft source mismatch")
    weights = GRAFT / "checkpoint_step_0.safetensors"
    actual = digest(weights)
    if actual != manifest["output_weights_sha256"]:
        raise ValueError("Width-matched graft weight hash mismatch")
    record("graft_verified", sha256=actual, svae_sha256=svae_hash)
    return actual


def train_and_evaluate(svae_path: Path, svae_hash: str) -> None:
    ensure_eval_stopped()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise RuntimeError("Commit the ablation code before an eight-GPU training run")
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(WEIGHTS).free < 30_000_000_000:
        raise RuntimeError("Insufficient NFS space for ablation checkpoint")
    env = os.environ.copy()
    env["PATH"] = f"{PYTHON.parent}:{env['PATH']}"
    env["PYTHONPATH"] = str(REPO)
    env["CUDA_HOME"] = "/root/zifan/openwam-envs/benchmark-conda"
    env["WAM_PILOT_SEED"] = "42"
    env["WAM_PILOT_GRAFT"] = str(GRAFT)
    env["WAM_PILOT_RAE_SVAE_PATH"] = str(svae_path)
    env["WAM_PILOT_RUN_ROOT"] = str(WEIGHTS)
    env["MASTER_PORT"] = "29688"

    graft_hash = build_graft(svae_path, svae_hash, env)
    train_log = RUNS / "rae_train.log"
    record("wam_training_started", output=str(WEIGHTS / "rae"))
    run_logged(["bash", "research/rae/launch_wam_pilot.sh", "rae"], train_log, env=env)
    checkpoint, checkpoint_hash = verify_final(WEIGHTS, "rae", train_log)
    record("wam_checkpoint_verified", checkpoint=str(checkpoint), sha256=checkpoint_hash)
    graft_weights = GRAFT / "checkpoint_step_0.safetensors"
    if digest(graft_weights) != graft_hash:
        raise ValueError("Graft changed during WAM training; refusing cleanup")
    graft_weights.unlink()
    record("graft_weight_released", sha256=graft_hash)

    eval_env = env.copy()
    eval_env["CUDA_VISIBLE_DEVICES"] = "4"
    comparisons = {}
    for split in ("dev", "test"):
        report = RUNS / f"rae_{split}_teacher_forced.json"
        run_logged(
            [str(PYTHON), "research/rae/eval_teacher_forced.py", "--checkpoint", str(checkpoint),
             "--output", str(report), "--split", split,
             "--per-task", "10", "--noise-repeats", "4"],
            RUNS / f"rae_{split}_teacher_forced.log", env=eval_env,
        )
        comparison = RUNS / f"paired_{split}_comparison.json"
        run_logged(
            [str(PYTHON), "research/rae/compare_teacher_forced.py", "--vae",
             str(VAE_BASELINE / f"{split}_teacher_forced.json"), "--rae", str(report),
             "--output", str(comparison)],
            RUNS / f"paired_{split}_comparison.log", env=env,
        )
        result = json.loads(comparison.read_text())
        comparisons[split] = {
            "report": str(comparison),
            "vae_action_loss": result["vae_action_mean_across_tasks"],
            "svae48_action_loss": result["rae_action_mean_across_tasks"],
            "svae48_minus_vae": result["rae_minus_vae"],
        }
        record(f"{split}_evaluated", **comparisons[split])
    record("complete", checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
           svae_sha256=svae_hash, comparisons=comparisons,
           scope="Single-seed width-matched offline action loss; no task-success conclusion")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svae-pid", type=int, required=True)
    parser.add_argument("--multiseed-pid", type=int, required=True)
    args = parser.parse_args()
    try:
        record("waiting_for_svae", pid=args.svae_pid)
        wait_for(args.svae_pid, "research/rae/continue_svae48.py")
        svae_path, svae_hash = verify_svae()
        record("waiting_for_multiseed", pid=args.multiseed_pid, svae_sha256=svae_hash)
        wait_for(args.multiseed_pid, "research/rae/run_additional_seeds.py")
        train_and_evaluate(svae_path, svae_hash)
    except BaseException as exc:
        record("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
