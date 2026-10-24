#!/usr/bin/env python3
"""Train the frozen 48-channel DINO reducer after feature collection."""

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

from scripts.svae_train.train_svae import _ShardedFeatureDataset

REPO = Path("/root/zifan/OpenWAM-rae")
PYTHON = Path("/root/zifan/openwam-envs/policy/bin/python")
FEATURE_ROOT = Path("/data/zifan-openwam-rae/svae_features")
TRAIN_ROOT = Path("/data/zifan-openwam-rae/svae48_runs")
STATE = Path("/root/zifan/openwam-runs/svae48_ablation_20261024_state.json")
LOG = Path("/root/zifan/openwam-runs/svae48_train_20261024.log")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def record(stage: str, **details: object) -> None:
    payload = {"stage": stage, "updated_unix": time.time(), **details}
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATE)
    print(json.dumps(payload, sort_keys=True), flush=True)


def wait_for_collector(pid: int) -> None:
    record("waiting_for_features", collector_pid=pid)
    while True:
        proc = Path(f"/proc/{pid}")
        try:
            state = (proc / "stat").read_text().split(") ", 1)[1].split()[0]
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except FileNotFoundError:
            return
        if state == "Z":
            return
        if "scripts/svae_train/collect_svae_features.py" not in command:
            raise RuntimeError(f"PID {pid} is not the registered feature collector")
        time.sleep(30)


def verify_features() -> tuple[Path, dict]:
    candidates = sorted(FEATURE_ROOT.glob("dinov2_*/stats.pt"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one DINOv2 feature archive, found {candidates}")
    stats_path = candidates[0]
    stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    if (stats["encoder_name"], int(stats["raw_dim"]), int(stats["dataset_total_clips"]),
        int(stats["selected_clips"]), int(stats["num_tokens"])) != (
            "dinov2", 768, 124103, 8192, 8192 * 1440):
        raise ValueError(f"Wrong DINO feature selection: {stats_path}")
    if len(_ShardedFeatureDataset(stats_path.parent)) != 8192:
        raise ValueError("Feature shard clip count does not match selection")
    details = {
        "features_dir": str(stats_path.parent),
        "stats_sha256": digest(stats_path),
        "selected_indices_sha256": stats["selected_indices_sha256"],
        "clips": 8192,
        "tokens": int(stats["num_tokens"]),
    }
    return stats_path.parent, details


def train(features_dir: Path) -> Path:
    if LOG.exists():
        raise FileExistsError(LOG)
    if shutil.disk_usage("/data").free < 10_000_000_000:
        raise RuntimeError("Shared NFS has too little free space for S-VAE output")
    before = set(TRAIN_ROOT.glob("svae_*/svae.pt")) if TRAIN_ROOT.exists() else set()
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "2"
    environment["OMP_NUM_THREADS"] = "4"
    environment["PYTHONPATH"] = str(REPO)
    environment["PATH"] = f"{PYTHON.parent}:{environment['PATH']}"
    command = [
        str(PYTHON), "scripts/svae_train/train_svae.py",
        f"train.features_dir={features_dir}",
        f"train.output_dir={TRAIN_ROOT}",
        "svae.latent_dim=48", "train.max_steps=1000", "train.batch_size=8",
        "train.num_workers=2", "train.warmup_epochs=0", "train.seed=42",
        "train.wandb.project=null",
    ]
    record("training_started", features_dir=str(features_dir), command=command)
    with LOG.open("w") as stream:
        result = subprocess.run(command, cwd=REPO, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"S-VAE training exited {result.returncode}; see {LOG}")
    outputs = set(TRAIN_ROOT.glob("svae_*/svae.pt")) - before
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one S-VAE checkpoint, found {sorted(outputs)}")
    return outputs.pop()


def verify_checkpoint(path: Path, features: dict) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    if (int(config["input_dim"]), int(config["latent_dim"]),
        int(checkpoint["num_clips"])) != (768, 48, 8192):
        raise ValueError(f"Wrong S-VAE checkpoint architecture: {path}")
    metrics = checkpoint["metrics"]
    if not metrics or int(metrics[-1]["step"]) != 1000:
        raise ValueError(f"S-VAE did not reach the registered 1000 steps: {path}")
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": digest(path),
        "features": features,
        "last_train_metrics": metrics[-1],
        "scope": "Frozen 48-channel DINO reducer; no WAM or task result yet",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-pid", type=int, required=True)
    args = parser.parse_args()
    try:
        wait_for_collector(args.after_pid)
        features_dir, features = verify_features()
        record("features_verified", **features)
        checkpoint = train(features_dir)
        result = verify_checkpoint(checkpoint, features)
        record("complete", **result)
    except BaseException as exc:
        record("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
