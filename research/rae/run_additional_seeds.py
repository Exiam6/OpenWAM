#!/usr/bin/env python3
"""Run matched WAM seeds after the exploratory RoboTwin GPU sweep exits.

Final weights live in /dev/shm because the persistent volume has less than one
checkpoint of free space. Reports, hashes, and logs live on persistent storage.
Do not treat a lost tmpfs checkpoint as a completed reproducible artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from safetensors import safe_open

REPO = Path("/root/zifan/OpenWAM-rae")
PYTHON = Path("/root/zifan/openwam-envs/policy/bin/python")
RUNS = Path("/root/zifan/openwam-runs/wam_multiseed_20261024")
FEATURE_STATS = Path("/root/zifan/openwam-runs/rae_feature_stats_stride16.pt")
MIN_FREE_BYTES = 27_000_000_000


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def record(seed: int, stage: str, **details: object) -> None:
    target = RUNS / f"seed{seed}" / "state.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": seed, "stage": stage, "updated_unix": time.time(), **details}
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    print(json.dumps(payload, sort_keys=True), flush=True)


def process_state(pid: int) -> tuple[str, str] | None:
    proc = Path(f"/proc/{pid}")
    try:
        state = (proc / "stat").read_text().split(") ", 1)[1].split()[0]
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except FileNotFoundError:
        return None
    return state, command


def wait_for_official_controller(pid: int | None) -> None:
    if pid is None:
        return
    while (status := process_state(pid)) is not None and status[0] != "Z":
        if "continue_official_after_paired_mesa2625.sh" not in status[1]:
            raise RuntimeError(f"PID {pid} is no longer the registered official controller")
        print(f"Waiting for official CPU-RT evaluation PID {pid}", flush=True)
        time.sleep(30)


def ensure_eval_stopped() -> None:
    listing = subprocess.check_output(["ps", "-eo", "args="], text=True)
    for line in listing.splitlines():
        if any(marker in line for marker in (
            "benchmarks/robotwin/single_eval.sh",
            "benchmarks/robotwin/eval_policy_wrapper.py",
            "OpenWAM-rae/scripts/deploy.py",
        )):
            raise RuntimeError(f"RoboTwin evaluation still owns GPUs: {line}")


def require_tmpfs_space(stage: str) -> None:
    free = shutil.disk_usage("/dev/shm").free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"Only {free} tmpfs bytes free before {stage}; need {MIN_FREE_BYTES}")


def run_logged(command: list[str], log: Path, *, env: dict[str, str]) -> None:
    if log.exists():
        raise FileExistsError(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Exit {result.returncode}: {' '.join(command)}; see {log}")


def verify_graft(graft: Path, seed: int) -> tuple[Path, str]:
    manifest = json.loads((graft / "graft_manifest.json").read_text())
    if manifest["seed"] != seed:
        raise ValueError(f"Graft seed mismatch in {graft}")
    weights = graft / "checkpoint_step_0.safetensors"
    actual = digest(weights)
    if actual != manifest["output_weights_sha256"]:
        raise ValueError(f"Graft weight hash mismatch in {graft}")
    return weights, actual


def verify_final(weights_root: Path, variant: str, log: Path) -> tuple[Path, str]:
    candidates = list((weights_root / variant).glob("*/checkpoint_step_1000.safetensors"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one {variant} final checkpoint, found {candidates}")
    weights = candidates[0]
    if f"[checkpoint] Saved final: {weights}" not in log.read_text(errors="replace"):
        raise ValueError(f"Final-save marker missing in {log}")
    if not (weights.parent / "config.yaml").is_file():
        raise FileNotFoundError(weights.parent / "config.yaml")
    with safe_open(weights, framework="pt", device="cpu") as stream:
        count = len(stream.keys())
    if count < 1000 or weights.stat().st_size < 20_000_000_000:
        raise ValueError(f"Incomplete {variant} checkpoint: {weights} ({count} tensors)")
    return weights.parent, digest(weights)


def evaluate_pair(seed: int, checkpoints: dict[str, Path], env: dict[str, str]) -> None:
    reports = RUNS / f"seed{seed}"
    for split in ("dev", "test"):
        for variant in ("rae", "vae"):
            output = reports / f"{variant}_{split}_teacher_forced.json"
            run_logged(
                [str(PYTHON), "research/rae/eval_teacher_forced.py", "--checkpoint",
                 str(checkpoints[variant]), "--output", str(output), "--split", split,
                 "--per-task", "10", "--noise-repeats", "4"],
                reports / f"{variant}_{split}_teacher_forced.log", env=env,
            )
        comparison = reports / f"paired_{split}_comparison.json"
        run_logged(
            [str(PYTHON), "research/rae/compare_teacher_forced.py", "--vae",
             str(reports / f"vae_{split}_teacher_forced.json"), "--rae",
             str(reports / f"rae_{split}_teacher_forced.json"), "--output", str(comparison)],
            reports / f"paired_{split}_comparison.log", env=env,
        )
        summary = json.loads(comparison.read_text())
        record(seed, f"{split}_evaluated", comparison=str(comparison),
               vae_action=summary["vae_action_mean_across_tasks"],
               rae_action=summary["rae_action_mean_across_tasks"])


def run_seed(seed: int) -> None:
    reports = RUNS / f"seed{seed}"
    reports.mkdir(parents=True, exist_ok=True)
    weights_root = Path(f"/dev/shm/openwam-seed{seed}-wam")
    graft = Path(f"/dev/shm/openwam-seed{seed}-graft")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["PATH"] = f"{PYTHON.parent}:{env['PATH']}"
    env["CUDA_HOME"] = "/root/zifan/openwam-envs/benchmark-conda"
    env["WAM_PILOT_SEED"] = str(seed)
    env["WAM_PILOT_GRAFT"] = str(graft)
    env["WAM_PILOT_RUN_ROOT"] = str(weights_root)
    env["MASTER_PORT"] = str(29600 + seed)

    if not (graft / "graft_manifest.json").is_file():
        require_tmpfs_space(f"seed{seed} graft")
        record(seed, "graft_started", graft=str(graft))
        run_logged(
            [str(PYTHON), "research/rae/build_grafted_checkpoint.py", "--output",
             str(graft), "--seed", str(seed), "--feature-stats", str(FEATURE_STATS)],
            reports / "graft.log", env=env,
        )
    graft_weights, graft_hash = verify_graft(graft, seed)
    record(seed, "graft_verified", sha256=graft_hash)

    checkpoints: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for variant in ("rae", "vae"):
        require_tmpfs_space(f"seed{seed} {variant} training")
        log = reports / f"{variant}_train.log"
        record(seed, f"{variant}_training_started", output=str(weights_root / variant))
        run_logged(["bash", "research/rae/launch_wam_pilot.sh", variant], log, env=env)
        checkpoints[variant], hashes[variant] = verify_final(weights_root, variant, log)
        record(seed, f"{variant}_checkpoint_verified", checkpoint=str(checkpoints[variant]),
               sha256=hashes[variant])
        if variant == "rae":
            # The graft can be rebuilt byte-for-byte from its fixed donor and seed.
            if digest(graft_weights) != graft_hash:
                raise ValueError("Graft changed during RAE training; refusing cleanup")
            graft_weights.unlink()
            record(seed, "graft_weight_released", sha256=graft_hash)

    evaluate_pair(seed, checkpoints, env)
    record(seed, "complete", checkpoints={name: str(path) for name, path in checkpoints.items()},
           checkpoint_hashes=hashes, storage="/dev/shm is ephemeral")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-pid", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44])
    args = parser.parse_args()
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be distinct non-negative integers")
    wait_for_official_controller(args.after_pid)
    ensure_eval_stopped()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise RuntimeError("Commit research code before a GPU training run")
    for seed in args.seeds:
        try:
            run_seed(seed)
        except BaseException as exc:
            record(seed, "failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
