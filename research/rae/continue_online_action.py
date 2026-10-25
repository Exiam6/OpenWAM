#!/usr/bin/env python3
"""Run a guarded online-action smoke, 1000-step WAM, and paired offline eval.

This controller waits for the already registered main/action report controller
so it never preempts the matched seed and S-VAE experiments. Native closed-loop
evaluation remains a separate RT-host gate.
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

import torch
from safetensors import safe_open

from research.rae.continue_svae48_wam import wait_for

REPO = Path("/root/zifan/OpenWAM-online-action")
PYTHON = Path("/root/zifan/openwam-envs/policy/bin/python")
GRAFT = Path("/data/zifan-openwam-rae/online_action_adapter_20261025/graft_seed42")
GRAFT_SHA256 = "8521d59e54b50bea0b4a43385b8ddf2384921624ee4a29f24a28dedfdf18d8e4"
SMOKE = Path("/dev/shm/openwam-online-action-smoke-20261025")
WEIGHTS = Path("/data/zifan-openwam-rae/online_action_adapter_20261025/wam_seed42")
RUNS = Path("/root/zifan/openwam-runs/wam_online_action_seed42")
PRIOR_REPORT_STATE = Path("/root/zifan/openwam-runs/openwam_rae_extended_results_state.json")
PILOT = Path("/root/zifan/openwam-runs/wam_pilot_20261023")
ADAPTER_UP = "video_backbone.video_encoder._online_action_adapter.up.weight"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def record(stage: str, **details: object) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    target = RUNS / "state.json"
    payload = {"stage": stage, "updated_unix": time.time(), **details}
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    print(json.dumps(payload, sort_keys=True), flush=True)


def run_logged(command: list[str], log: Path, env: dict[str, str]) -> None:
    if log.exists():
        raise FileExistsError(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        outcome = subprocess.run(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if outcome.returncode:
        raise RuntimeError(f"Command exited {outcome.returncode}; see {log}")


def verify_sources() -> str:
    prior = json.loads(PRIOR_REPORT_STATE.read_text())
    if prior.get("stage") != "complete":
        raise RuntimeError(f"Existing experiment queue did not finish: {prior}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise RuntimeError("Online-action source worktree is dirty")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    manifest = json.loads((GRAFT / "graft_manifest.json").read_text())
    if (manifest.get("seed"), manifest.get("online_action_adapter_hidden_dim")) != (42, 128):
        raise ValueError("Wrong online-action graft configuration")
    if manifest.get("output_weights_sha256") != GRAFT_SHA256:
        raise ValueError("Registered graft SHA mismatch")
    if digest(GRAFT / "checkpoint_step_0.safetensors") != GRAFT_SHA256:
        raise ValueError("Online-action graft changed")
    listing = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    )
    active = sorted({line.strip() for line in listing.splitlines() if line.strip().isdigit()})
    if active:
        raise RuntimeError(f"GPUs remain occupied by compute processes: {active}")
    return commit


def final_checkpoint(root: Path, steps: int, log: Path) -> Path:
    candidates = list(root.glob(f"*/checkpoint_step_{steps}.safetensors"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one step-{steps} checkpoint under {root}, got {candidates}")
    weights = candidates[0]
    if f"[checkpoint] Saved final: {weights}" not in log.read_text(errors="replace"):
        raise ValueError("Training log lacks final-save marker")
    if weights.stat().st_size < 20_000_000_000 or not (weights.parent / "config.yaml").is_file():
        raise ValueError("Final checkpoint or config is incomplete")
    with safe_open(weights, framework="pt", device="cpu") as stream:
        if len(stream.keys()) < 2000 or ADAPTER_UP not in stream.keys():
            raise ValueError("Final checkpoint lacks the online adapter")
    return weights


def run_training(steps: int, root: Path, log: Path, env: dict[str, str]) -> Path:
    if shutil.disk_usage(root.parent).free < 35_000_000_000:
        raise RuntimeError(f"Insufficient storage for {steps}-step checkpoint at {root.parent}")
    record(f"training_{steps}_started", output=str(root))
    run_logged(["bash", "research/rae/launch_online_action.sh", str(steps), str(root)], log, env)
    return final_checkpoint(root, steps, log)


def verify_smoke(weights: Path) -> dict[str, object]:
    with safe_open(weights, framework="pt", device="cpu") as stream:
        up = stream.get_tensor(ADAPTER_UP).float()
    changed = int(torch.count_nonzero(up).item())
    if changed == 0 or not torch.isfinite(up).all():
        raise ValueError("Two-step DeepSpeed run did not update the adapter output projection")
    return {"sha256": digest(weights), "up_nonzero_elements": changed,
            "up_norm": float(torch.linalg.vector_norm(up).item())}


def evaluate(checkpoint: Path, checkpoint_sha: str, env: dict[str, str]) -> dict:
    eval_env = env.copy()
    eval_env["CUDA_VISIBLE_DEVICES"] = "4"
    results = {}
    for split in ("dev", "test"):
        output = RUNS / f"online_{split}_teacher_forced.json"
        run_logged(
            [str(PYTHON), "research/rae/eval_teacher_forced.py", "--checkpoint", str(checkpoint.parent),
             "--output", str(output), "--split", split, "--per-task", "10", "--noise-repeats", "4"],
            RUNS / f"online_{split}_teacher_forced.log", eval_env,
        )
        report = json.loads(output.read_text())
        if report.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError(f"{split} eval used a different checkpoint")
        results[split] = {}
        for control, baseline in (
            ("native_vae", PILOT / "vae" / f"{split}_teacher_forced.json"),
            ("raw_dino", PILOT / "rae" / f"{split}_teacher_forced.json"),
        ):
            comparison = RUNS / f"paired_{split}_vs_{control}.json"
            run_logged(
                [str(PYTHON), "research/rae/compare_teacher_forced.py", "--vae", str(baseline),
                 "--rae", str(output), "--output", str(comparison)],
                RUNS / f"paired_{split}_vs_{control}.log", env,
            )
            pair = json.loads(comparison.read_text())
            if pair.get("rae_checkpoint_sha256") != checkpoint_sha:
                raise ValueError("Paired report used a different online checkpoint")
            results[split][control] = {
                "baseline_loss": pair["vae_action_mean_across_tasks"],
                "online_loss": pair["rae_action_mean_across_tasks"],
                "difference": pair["rae_minus_vae"],
                "comparison": str(comparison),
            }
        record(f"{split}_evaluated", comparisons=results[split])
    return results


def write_results(commit: str, checkpoint_sha: str, smoke: dict, comparisons: dict) -> Path:
    target = RUNS / "RESULTS.md"
    if target.exists():
        raise FileExistsError(target)
    lines = ["# Online action-grounded DINO ablation", "",
             "Seed 42, 1,000 matched WAM steps, causal width-preserving adapter.",
             "Only action loss updates the adapter through the current latent.",
             "Values are paired teacher-forced action losses (lower is better), not task success.", "",
             "| Split | Control | Control loss | Online loss | Online minus control |",
             "|---|---|---:|---:|---:|"]
    for split in ("dev", "test"):
        for control in ("native_vae", "raw_dino"):
            row = comparisons[split][control]
            lines.append(f"| {split} | {control} | {row['baseline_loss']:.7f} | "
                         f"{row['online_loss']:.7f} | {row['difference']:+.7f} |")
    lines += ["", f"Source commit: `{commit}`.", f"Graft SHA256: `{GRAFT_SHA256}`.",
              f"Two-step smoke SHA256: `{smoke['sha256']}`; adapter output projection changed "
              f"in {smoke['up_nonzero_elements']} elements.",
              f"Final WAM checkpoint SHA256: `{checkpoint_sha}`.", "",
              "This is an OpenWAM action-gradient ablation, not a full ReWAM TRB reproduction.",
              "Closed-loop RoboTwin success requires native Vulkan RT evaluation.", ""]
    temporary = target.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-pid", type=int, required=True)
    args = parser.parse_args()
    try:
        record("waiting_for_registered_queue", pid=args.after_pid)
        wait_for(args.after_pid, "research/rae/continue_extended_results.py")
        commit = verify_sources()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO)
        env["PATH"] = f"{PYTHON.parent}:{env['PATH']}"
        smoke_weights = run_training(2, SMOKE, RUNS / "smoke_train.log", env)
        smoke = verify_smoke(smoke_weights)
        record("smoke_verified", checkpoint=str(smoke_weights), **smoke)
        # The smoke is an implementation check, not a result checkpoint. Keep
        # its hash/norm evidence but release tmpfs for the full run.
        smoke_weights.unlink()
        record("smoke_weights_released", **smoke)
        final_weights = run_training(1000, WEIGHTS, RUNS / "train.log", env)
        final_sha = digest(final_weights)
        record("checkpoint_verified", checkpoint=str(final_weights), sha256=final_sha)
        comparisons = evaluate(final_weights, final_sha, env)
        results = write_results(commit, final_sha, smoke, comparisons)
        record("complete", output=str(results), checkpoint=str(final_weights),
               checkpoint_sha256=final_sha, source_commit=commit, comparisons=comparisons)
    except BaseException as exc:
        record("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
