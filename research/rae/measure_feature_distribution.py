#!/usr/bin/env python3
"""Measure frozen DINOv2 channel statistics on train episodes only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from openwam.model.video_backbone.encoder.dinov2 import DinoV2VideoEncoder
from research.rae.train_decoder import DINO_SHA256, RoboTwinFrames, sha256


@torch.no_grad()
def collect(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if sha256(args.encoder / "model.safetensors") != DINO_SHA256:
        raise ValueError("DINOv2 hash mismatch")
    manifest = args.splits / "split_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    dataset = RoboTwinFrames(args.splits / "train", stride=args.stride)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    encoder = DinoV2VideoEncoder.from_pretrained(str(args.encoder)).cuda().eval()
    encoder.requires_grad_(False)
    channels = encoder.properties.z_dim
    total = torch.zeros(channels, dtype=torch.float64)
    total_square = torch.zeros_like(total)
    count = 0
    for batch in loader:
        raw = encoder._batch_encode_pooled_raw(batch.cuda(non_blocking=True).to(torch.bfloat16).unsqueeze(2))
        values = raw.float().permute(1, 0, 2, 3, 4).reshape(channels, -1).double().cpu()
        total += values.sum(dim=1)
        total_square += values.square().sum(dim=1)
        count += values.shape[1]
    mean = total / count
    variance = (total_square / count - mean.square()).clamp_min(0)
    std = variance.sqrt()
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise FloatingPointError("Non-finite or zero feature variance")
    stats = {
        "mean": mean.float(),
        "std": std.float(),
        "count_per_channel": count,
        "frames": len(dataset),
        "stride": args.stride,
        "encoder_sha256": DINO_SHA256,
        "split_manifest_sha256": sha256(manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.save(stats, args.output)
    summary = {
        "output": str(args.output),
        "sha256": sha256(args.output),
        "frames": len(dataset),
        "count_per_channel": count,
        "channel_mean_min": mean.min().item(),
        "channel_mean_max": mean.max().item(),
        "channel_std_min": std.min().item(),
        "channel_std_max": std.max().item(),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=Path("/root/zifan/openwam-assets/robotwin-splits"))
    parser.add_argument("--encoder", type=Path, default=Path("/root/zifan/openwam-assets/dinov2-base"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    collect(parser.parse_args())
