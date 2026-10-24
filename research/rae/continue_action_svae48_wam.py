#!/usr/bin/env python3
"""Run action-informed S-VAE and matched WAM after the width ablation.

The official width-matched reducer and its WAM evaluation finish first. This
controller then uses the same registered DINO feature archive, WAM source
commit, seed, data split, training budget, and offline evaluation settings.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch

from research.rae.continue_svae48_wam import wait_for
from research.rae.run_additional_seeds import digest, ensure_eval_stopped, run_logged, verify_final

ISOLATED_REPO = Path("/root/zifan/OpenWAM-action-bottleneck")
MAIN_REPO = Path("/root/zifan/OpenWAM-rae")
EXPECTED_MAIN_COMMIT = "c414e853d0c42cbf1915a9de7823c777034160ab"
PYTHON = Path("/root/zifan/openwam-envs/policy/bin/python")
FEATURES = Path("/data/zifan-openwam-rae/svae_features/dinov2_20261024_181345")
ACTIONS = Path("/data/zifan-openwam-rae/action_bottleneck/action_targets_8192.pt")
ACTION_SHA = "541c09dc4c932a4eb187be34bec88f1b07caa167097221df320e370720098b0c"
REDUCER = Path("/data/zifan-openwam-rae/action_bottleneck/action_svae48_seed42.pt")
RUNS = Path("/root/zifan/openwam-runs/wam_action_svae48_seed42")
WEIGHTS = Path("/data/zifan-openwam-rae/wam_action_svae48_seed42")
GRAFT = Path("/dev/shm/openwam-action-svae48-seed42-graft")
WIDTH_RUNS = Path("/root/zifan/openwam-runs/wam_svae48_seed42")
VAE_REPORTS = Path("/root/zifan/openwam-runs/wam_pilot_20261023/vae")


def record(stage: str, **details: object) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    target = RUNS / "state.json"
    data = {"stage": stage, "updated_unix": time.time(), **details}
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temp.replace(target)
    print(json.dumps(data, sort_keys=True), flush=True)


def verify_sources() -> None:
    if digest(ACTIONS) != ACTION_SHA:
        raise ValueError("Registered action target archive changed")
    if not (FEATURES / "stats.pt").is_file():
        raise FileNotFoundError(FEATURES / "stats.pt")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=MAIN_REPO, text=True).strip() != EXPECTED_MAIN_COMMIT:
        raise ValueError("Matched WAM source commit changed")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=MAIN_REPO).strip():
        raise RuntimeError("Matched WAM source tree is dirty")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ISOLATED_REPO).strip():
        raise RuntimeError("Action-informed stage-one source tree is dirty")


def train_reducer() -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ISOLATED_REPO)
    env["PATH"] = f"{PYTHON.parent}:{env['PATH']}"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OMP_NUM_THREADS"] = "4"
    record("action_reducer_training_started", output=str(REDUCER))
    run_logged(
        [str(PYTHON), str(ISOLATED_REPO / "research/rae/train_action_informed_svae.py"),
         "--features", str(FEATURES), "--actions", str(ACTIONS),
         "--output", str(REDUCER), "--steps", "1000", "--batch-size", "8",
         "--num-workers", "2", "--action-weight", "1.0", "--seed", "42", "--device", "cuda:0"],
        RUNS / "action_svae_train.log", env=env,
    )
    checkpoint = torch.load(REDUCER, map_location="cpu", weights_only=True)
    if checkpoint.get("stage_one_method") != "offline_action_informed_svae":
        raise ValueError("Unexpected reducer method")
    if checkpoint.get("steps") != 1000 or checkpoint.get("seed") != 42:
        raise ValueError("Wrong reducer steps or seed")
    if checkpoint.get("action_targets_sha256") != ACTION_SHA:
        raise ValueError("Reducer used the wrong action target archive")
    if (checkpoint["model_config"]["input_dim"], checkpoint["model_config"]["latent_dim"]) != (768, 48):
        raise ValueError("Wrong reducer channel dimensions")
    actual = digest(REDUCER)
    record("action_reducer_verified", checkpoint=str(REDUCER), sha256=actual)
    return actual


def build_graft(reducer_sha: str, env: dict[str, str]) -> str:
    if shutil.disk_usage("/dev/shm").free < 27_000_000_000:
        raise RuntimeError("Insufficient tmpfs for action-informed graft")
    record("graft_started", reducer=str(REDUCER))
    run_logged(
        [str(PYTHON), "research/rae/build_grafted_checkpoint.py", "--output", str(GRAFT),
         "--seed", "42", "--svae-path", str(REDUCER)],
        RUNS / "graft.log", env=env,
    )
    manifest = json.loads((GRAFT / "graft_manifest.json").read_text())
    if manifest.get("seed") != 42 or manifest.get("svae_sha256") != reducer_sha:
        raise ValueError("Action-informed graft source mismatch")
    weights = GRAFT / "checkpoint_step_0.safetensors"
    actual = digest(weights)
    if actual != manifest["output_weights_sha256"]:
        raise ValueError("Action-informed graft weight hash mismatch")
    record("graft_verified", sha256=actual)
    return actual


def train_and_evaluate(reducer_sha: str) -> None:
    ensure_eval_stopped()
    if shutil.disk_usage(WEIGHTS.parent).free < 30_000_000_000:
        raise RuntimeError("Insufficient NFS space for action-informed WAM checkpoint")
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(MAIN_REPO)
    env["PATH"] = f"{PYTHON.parent}:{env['PATH']}"
    env["CUDA_HOME"] = "/root/zifan/openwam-envs/benchmark-conda"
    env["WAM_PILOT_SEED"] = "42"
    env["WAM_PILOT_GRAFT"] = str(GRAFT)
    env["WAM_PILOT_RAE_SVAE_PATH"] = str(REDUCER)
    env["WAM_PILOT_RUN_ROOT"] = str(WEIGHTS)
    env["MASTER_PORT"] = "29689"
    graft_sha = build_graft(reducer_sha, env)

    train_log = RUNS / "rae_train.log"
    record("wam_training_started", output=str(WEIGHTS / "rae"))
    run_logged(["bash", "research/rae/launch_wam_pilot.sh", "rae"], train_log, env=env)
    checkpoint, checkpoint_sha = verify_final(WEIGHTS, "rae", train_log)
    record("wam_checkpoint_verified", checkpoint=str(checkpoint), sha256=checkpoint_sha)
    graft_weights = GRAFT / "checkpoint_step_0.safetensors"
    if digest(graft_weights) != graft_sha:
        raise ValueError("Graft changed during WAM training")
    graft_weights.unlink()
    record("graft_weight_released", sha256=graft_sha)

    eval_env = env.copy()
    eval_env["CUDA_VISIBLE_DEVICES"] = "4"
    results = {}
    for split in ("dev", "test"):
        output = RUNS / f"rae_{split}_teacher_forced.json"
        run_logged(
            [str(PYTHON), "research/rae/eval_teacher_forced.py", "--checkpoint", str(checkpoint),
             "--output", str(output), "--split", split, "--per-task", "10", "--noise-repeats", "4"],
            RUNS / f"rae_{split}_teacher_forced.log", env=eval_env,
        )
        results[split] = {}
        for control, baseline in (
            ("native_vae", VAE_REPORTS / f"{split}_teacher_forced.json"),
            ("reconstruction_svae", WIDTH_RUNS / f"rae_{split}_teacher_forced.json"),
        ):
            comparison = RUNS / f"paired_{split}_vs_{control}.json"
            run_logged(
                [str(PYTHON), "research/rae/compare_teacher_forced.py", "--vae", str(baseline),
                 "--rae", str(output), "--output", str(comparison)],
                RUNS / f"paired_{split}_vs_{control}.log", env=env,
            )
            report = json.loads(comparison.read_text())
            results[split][control] = {
                "baseline": report["vae_action_mean_across_tasks"],
                "action_informed": report["rae_action_mean_across_tasks"],
                "difference": report["rae_minus_vae"],
                "comparison": str(comparison),
            }
        record(f"{split}_evaluated", comparisons=results[split])

    lines = [
        "# Action-informed DINO/S-VAE WAM ablation", "",
        "Seed 42, 1,000 reducer steps and 1,000 WAM steps. The reducer was fitted",
        "on 8,192 train-only DINO clips with feature reconstruction plus a masked",
        "action prediction loss (weight 1.0). The action head is discarded for WAM.",
        "All values below are paired, teacher-forced action losses; lower is better.", "",
        "| Split | Control | Control loss | Action-informed loss | Difference |",
        "|---|---|---:|---:|---:|",
    ]
    for split in ("dev", "test"):
        for control in ("native_vae", "reconstruction_svae"):
            row = results[split][control]
            lines.append(
                f"| {split} | {control} | {row['baseline']:.7f} | "
                f"{row['action_informed']:.7f} | {row['difference']:+.7f} |"
            )
    lines.extend([
        "", f"Reducer SHA256: `{reducer_sha}`.",
        f"WAM checkpoint SHA256: `{checkpoint_sha}`.", "",
        "The stage-one action objective is an offline auxiliary loss. This is",
        "not online ReWAM gradient routing and does not establish task success.", "",
    ])
    result_file = RUNS / "RESULTS.md"
    if result_file.exists():
        raise FileExistsError(result_file)
    result_file.write_text("\n".join(lines))
    record("complete", reducer_sha256=reducer_sha, checkpoint=str(checkpoint),
           checkpoint_sha256=checkpoint_sha, comparisons=results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width-pid", type=int, required=True)
    args = parser.parse_args()
    try:
        record("waiting_for_width", pid=args.width_pid)
        wait_for(args.width_pid, "research/rae/continue_svae48_wam.py")
        width_state = json.loads((WIDTH_RUNS / "state.json").read_text())
        if width_state.get("stage") != "complete":
            raise RuntimeError(f"Width ablation did not complete: {width_state}")
        verify_sources()
        reducer_sha = train_reducer()
        train_and_evaluate(reducer_sha)
    except BaseException as exc:
        record("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
