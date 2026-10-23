#!/usr/bin/env python3
"""Evaluate held-out RoboTwin teacher-forced losses for a trained WAM.

Action loss is an offline proxy, not closed-loop success. Video losses from
native VAE and DINOv2 have different targets and are recorded separately,
never directly ranked against each other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from openwam.dataloader.robotwin import MultiTaskRoboTwinDataset
from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir

SPLIT_SHA256 = "d492addcdd94f41b0c5819561e38bc0f3b0b39cfb650de15161142cd0339a1f2"
ACTION_STATS_SHA256 = "eaa21a50053e846057617aff7b0c1f2d019ed66e6ed34165130ec5148f72f91e"
SPLITS = Path("/root/zifan/openwam-assets/robotwin-splits")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_windows(dataset: MultiTaskRoboTwinDataset, per_task: int) -> list[tuple[str, int]]:
    """Choose evenly spaced dev windows within each task without reading labels."""
    selected = []
    offset = 0
    for task_data in dataset._sub_datasets:
        length = len(task_data)
        if length < per_task:
            raise ValueError(f"Task has only {length} dev windows, need {per_task}")
        task = task_data.task_name
        for local in np.linspace(0, length - 1, per_task, dtype=np.int64):
            selected.append((task, offset + int(local)))
        offset += length
    if len(selected) != len(dataset._sub_datasets) * per_task:
        raise AssertionError("Window selection count changed")
    return selected


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for WAM evaluation")
    if args.per_task <= 0 or args.noise_repeats <= 0:
        raise ValueError("per_task and noise_repeats must be positive")
    manifest = SPLITS / "split_manifest.json"
    stats = SPLITS / "train/dataset/meta/robotwin_clean_50_normalization_stats.npy"
    if sha256(manifest) != SPLIT_SHA256 or sha256(stats) != ACTION_STATS_SHA256:
        raise ValueError("Frozen split or action statistics hash changed")

    cfg = OmegaConf.load(args.checkpoint / "config.yaml")
    with open_dict(cfg.dataloader):
        cfg.dataloader.dataset_dir = str(SPLITS / "dev/dataset")
        cfg.dataloader.variant = "clean_50"
        cfg.dataloader.normalization_stats_path = str(stats)
        cfg.dataloader.color_jitter.enabled = False
    dataset = MultiTaskRoboTwinDataset.from_config(cfg.dataloader, split="val")
    if len(dataset._sub_datasets) != 10:
        raise ValueError("Expected ten task datasets")
    windows = selected_windows(dataset, args.per_task)

    _, model, _ = build_architecture_from_ckpt_dir(str(args.checkpoint), weights_required=True)
    model.set_dtype_device(torch.bfloat16, torch.device("cuda:0"))
    model.init_training_schedulers(1000)
    model.set_training_runtime(
        use_gradient_checkpointing=False,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    )
    model.eval()

    results = []
    for position, (task, index) in enumerate(windows):
        sample = dataset[index]
        losses = []
        for repeat in range(args.noise_repeats):
            seed = args.seed + position * 1009 + repeat
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            inputs = model.prepare_inputs([sample])
            outcome = model.compute_loss(**inputs, lambda_video=1.0, lambda_action=1.0)
            action = float(outcome["loss_action"].float().item())
            video = float(outcome["loss_video"].float().item())
            if not np.isfinite(action) or not np.isfinite(video):
                raise FloatingPointError(f"Non-finite loss at {task} index={index} repeat={repeat}")
            losses.append({"action": action, "video": video})
        results.append({"task": task, "window_index": index, "losses": losses})
        if (position + 1) % args.per_task == 0:
            print(f"evaluated {position + 1}/{len(windows)} dev windows", flush=True)

    per_task = {}
    for task in sorted({item["task"] for item in results}):
        rows = [loss for item in results if item["task"] == task for loss in item["losses"]]
        per_task[task] = {
            "action_mean": float(np.mean([r["action"] for r in rows])),
            "video_mean_within_variant": float(np.mean([r["video"] for r in rows])),
            "evaluations": len(rows),
        }
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(next(args.checkpoint.glob("checkpoint_step_*.safetensors"))),
        "split_manifest_sha256": SPLIT_SHA256,
        "action_stats_sha256": ACTION_STATS_SHA256,
        "seed": args.seed,
        "windows_per_task": args.per_task,
        "noise_repeats": args.noise_repeats,
        "action_mean_across_tasks": float(np.mean([r["action_mean"] for r in per_task.values()])),
        "per_task": per_task,
        "windows": results,
        "scope": "Teacher-forced loss proxy only; no closed-loop success conclusion",
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"action_mean_across_tasks": report["action_mean_across_tasks"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=10)
    parser.add_argument("--noise-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261023)
    evaluate(parser.parse_args())
