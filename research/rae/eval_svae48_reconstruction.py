#!/usr/bin/env python3
"""Check the frozen 48-channel reducer on episode-disjoint dev DINO features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from openwam.dataloader.robotwin import MultiTaskRoboTwinDataset
from openwam.model.video_backbone.encoder import build_video_encoder
from openwam.model.video_backbone.encoder.svae.model import load_svae, svae_loss
from research.rae.eval_teacher_forced import ACTION_STATS_SHA256, SPLITS, SPLIT_SHA256, selected_windows, sha256

FEATURES = Path("/data/zifan-openwam-rae/svae_features/dinov2_20261024_181345")
REDUCER = Path("/data/zifan-openwam-rae/svae48_runs/svae_20261024_182838/svae.pt")
FEATURE_STATS_SHA256 = "f62dca637273c1c88aed1292615d1cc34285b4995fdc0688009b67479b7d6cb5"
REDUCER_SHA256 = "76519a1dc5f8199c728d892c7551ceb59432eebd60113f55bc692889b26cd1fe"


@torch.inference_mode()
def evaluate(output: Path, per_task: int) -> None:
    if not torch.cuda.is_available() or per_task <= 0:
        raise ValueError("A CUDA GPU and positive --per-task are required")
    if output.exists():
        raise FileExistsError(output)
    manifest = SPLITS / "split_manifest.json"
    action_stats = SPLITS / "train/dataset/meta/robotwin_clean_50_normalization_stats.npy"
    if sha256(manifest) != SPLIT_SHA256 or sha256(action_stats) != ACTION_STATS_SHA256:
        raise ValueError("Frozen data manifest or training normalization changed")
    stats_path = FEATURES / "stats.pt"
    if sha256(stats_path) != FEATURE_STATS_SHA256 or sha256(REDUCER) != REDUCER_SHA256:
        raise ValueError("S-VAE features or reducer checkpoint changed")
    stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    shards = sorted(FEATURES.glob("features_rank*_part*.pt"))
    if not shards:
        raise FileNotFoundError("Missing registered train feature shards")
    train_shape = tuple(torch.load(shards[0], map_location="cpu", mmap=True,
                                   weights_only=True)["features"].shape[1:])
    if train_shape[0] != 768 or np.prod(train_shape[1:]) != 1440:
        raise ValueError(f"Unexpected registered train feature geometry: {train_shape}")

    cfg = OmegaConf.load(FEATURES / "config.yaml")
    with open_dict(cfg.dataloader):
        cfg.dataloader.dataset_dir = str(SPLITS / "dev/dataset")
        cfg.dataloader.normalization_stats_path = str(action_stats)
        cfg.dataloader.color_jitter.enabled = False
    dataset = MultiTaskRoboTwinDataset.from_config(cfg.dataloader, split="val")
    windows = selected_windows(dataset, per_task)
    if len(dataset._sub_datasets) != 10:
        raise ValueError("Expected ten held-out tasks")

    encoder = build_video_encoder(cfg.model.video_backbone.encoder)
    encoder.to(device="cuda:0", dtype=torch.bfloat16).eval()
    svae = load_svae(str(REDUCER)).to(device="cuda:0", dtype=torch.bfloat16).eval()
    if (svae.input_dim, svae.latent_dim) != (768, 48):
        raise ValueError("Reducer is not DINOv2 Base to 48 channels")
    if not torch.equal(svae.input_mean.cpu(), stats["mean"].float()):
        raise ValueError("Reducer mean buffer differs from training feature stats")
    if not torch.equal(svae.input_std.cpu(), stats["std"].float()):
        raise ValueError("Reducer std buffer differs from training feature stats")

    rows = []
    for position, (task, index) in enumerate(windows):
        sample = dataset[index]
        video = encoder.preprocess_video(sample["video"])
        raw = encoder.batch_encode_pooled_for_svae_training(video)
        if tuple(raw.shape[1:]) != train_shape:
            raise ValueError(f"Unexpected DINO feature geometry: {tuple(raw.shape)}")
        outcome = svae(raw)
        _, metrics = svae_loss(outcome, beta=1e-4)
        baseline_mse = outcome["target"].float().square().mean().item()
        row = {
            "task": task,
            "window_index": index,
            "standardized_mse": float(metrics["mse"]),
            "standardized_cosine_loss": float(metrics["cos"]),
            "zero_reconstruction_mse": float(baseline_mse),
        }
        if not np.isfinite(list(row.values())[2:]).all():
            raise FloatingPointError(f"Non-finite S-VAE reconstruction at {task}/{index}")
        rows.append(row)
        if (position + 1) % per_task == 0:
            print(f"evaluated {position + 1}/{len(windows)} dev windows", flush=True)

    per_task_rows = {}
    for task in sorted({row["task"] for row in rows}):
        selected = [row for row in rows if row["task"] == task]
        per_task_rows[task] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in ("standardized_mse", "standardized_cosine_loss", "zero_reconstruction_mse")
        }
    report = {
        "scope": "Episode-disjoint dev DINO feature reconstruction by a frozen reducer; not WAM action loss or task success",
        "split": "dev",
        "windows_per_task": per_task,
        "windows": rows,
        "per_task": per_task_rows,
        "mean_standardized_mse": float(np.mean([r["standardized_mse"] for r in per_task_rows.values()])),
        "mean_standardized_cosine_loss": float(np.mean([r["standardized_cosine_loss"] for r in per_task_rows.values()])),
        "mean_zero_reconstruction_mse": float(np.mean([r["zero_reconstruction_mse"] for r in per_task_rows.values()])),
        "reducer_sha256": REDUCER_SHA256,
        "feature_stats_sha256": FEATURE_STATS_SHA256,
        "split_manifest_sha256": SPLIT_SHA256,
        "dino_model_sha256": "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841",
        "train_feature_shape": train_shape,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in (
        "mean_standardized_mse", "mean_standardized_cosine_loss", "mean_zero_reconstruction_mse"
    )}, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=10)
    args = parser.parse_args()
    evaluate(args.output, args.per_task)
