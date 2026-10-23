#!/usr/bin/env python3
"""Continue the registered RAE/VAE pilot after the already-running RAE job.

The input PID is the torchrun parent from the RAE launch. This process writes
an auditable state file and exits on any missing or invalid artifact.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from safetensors import safe_open

ROOT = Path("/root/zifan")
REPO = ROOT / "OpenWAM-rae"
RUNS = ROOT / "openwam-runs/wam_pilot_20261023"
GRAFT = ROOT / "openwam-runs/rae_foundation_graft_calibrated_seed42/checkpoint_step_0.safetensors"
GRAFT_SHA256 = "79633cc4754699387686302e3139e8721f28855cea492026cf20d965a3b5814f"
PYTHON = ROOT / "openwam-envs/policy/bin/python"
STATE = RUNS / "continuation_state.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record(stage: str, **details: object) -> None:
    payload = {"stage": stage, "updated_unix": time.time(), **details}
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(STATE)
    print(json.dumps(payload, sort_keys=True), flush=True)


def pid_running(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0]
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except FileNotFoundError:
        return False
    return state != "Z" and b"torchrun" in command and b"wam_pilot_20261023/rae" in command


def wait_for_rae(pid: int) -> None:
    record("waiting_for_rae", pid=pid)
    while pid_running(pid):
        time.sleep(30)


def verified_checkpoint(variant: str) -> tuple[Path, str]:
    candidates = sorted((RUNS / variant).glob("*/checkpoint_step_1000.safetensors"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one final {variant} step-1000 checkpoint; found {candidates}")
    weights = candidates[0]
    log = RUNS / variant / "train.log"
    if f"[checkpoint] Saved final: {weights}" not in log.read_text(errors="replace"):
        raise RuntimeError(f"Final-save marker missing from {log}")
    if not (weights.parent / "config.yaml").is_file():
        raise RuntimeError(f"Missing config beside {weights}")
    with safe_open(weights, framework="pt", device="cpu") as stream:
        tensor_count = len(stream.keys())
    if tensor_count < 1000 or weights.stat().st_size < 20_000_000_000:
        raise RuntimeError(f"Invalid checkpoint shape/size: {weights}, {tensor_count}")
    sha = digest(weights)
    record(f"{variant}_checkpoint_verified", checkpoint=str(weights), sha256=sha, tensors=tensor_count)
    return weights.parent, sha


def release_reproducible_graft() -> None:
    manifest = GRAFT.parent / "graft_manifest.json"
    if not manifest.is_file() or digest(GRAFT) != GRAFT_SHA256:
        raise RuntimeError("Graft manifest or registered checksum changed; refusing cleanup")
    GRAFT.unlink()
    record("graft_weights_released", released=str(GRAFT))


def run_logged(command: list[str], log: Path, *, env: dict[str, str] | None = None) -> None:
    if log.exists():
        raise FileExistsError(log)
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}; see {log}")


def evaluate(variant: str, checkpoint: Path) -> None:
    output = RUNS / variant / "dev_teacher_forced.json"
    if output.exists():
        raise FileExistsError(output)
    log = RUNS / variant / "dev_teacher_forced.log"
    env = os.environ.copy()
    env["PATH"] = f"{PYTHON.parent}:{env['PATH']}"
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}"
    env["CUDA_HOME"] = str(ROOT / "openwam-envs/benchmark-conda")
    run_logged(
        [
            str(PYTHON), "research/rae/eval_teacher_forced.py",
            "--checkpoint", str(checkpoint), "--output", str(output),
            "--per-task", "10", "--noise-repeats", "4",
        ],
        log,
        env=env,
    )
    report = json.loads(output.read_text())
    record(f"{variant}_dev_evaluated", report=str(output), action_mean=report["action_mean_across_tasks"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rae-pid", type=int, required=True)
    args = parser.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    with (RUNS / ".continuation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            wait_for_rae(args.rae_pid)
            rae_checkpoint, rae_sha = verified_checkpoint("rae")
            release_reproducible_graft()
            record("vae_training_started", rae_checkpoint_sha256=rae_sha)
            run_logged(["bash", "research/rae/launch_wam_pilot.sh", "vae"], RUNS / "vae/train.log")
            vae_checkpoint, vae_sha = verified_checkpoint("vae")
            record("evaluating", rae_checkpoint_sha256=rae_sha, vae_checkpoint_sha256=vae_sha)
            evaluate("rae", rae_checkpoint)
            evaluate("vae", vae_checkpoint)
            comparison = RUNS / "paired_dev_comparison.json"
            run_logged(
                [
                    str(PYTHON), "research/rae/compare_teacher_forced.py",
                    "--vae", str(RUNS / "vae/dev_teacher_forced.json"),
                    "--rae", str(RUNS / "rae/dev_teacher_forced.json"),
                    "--output", str(comparison),
                ],
                RUNS / "paired_dev_comparison.log",
            )
            record("complete", rae_checkpoint_sha256=rae_sha,
                   vae_checkpoint_sha256=vae_sha, comparison=str(comparison))
        except BaseException as exc:
            record("failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
