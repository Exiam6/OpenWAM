#!/usr/bin/env python3
"""Train the pixel decoder on frozen DINOv2 features from train episodes only.

The decoder is a stage-one representation-autoencoder component. This script
does not train the world-action model or count as a closed-loop evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.model.video_backbone.encoder.dinov2 import DinoV2VideoEncoder
from openwam.model.video_backbone.encoder.representation_decoder import RepresentationPixelDecoder

CAMERAS = ("head_camera", "left_camera", "right_camera")
MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
DINO_SHA256 = "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RoboTwinFrames(Dataset):
    """Fixed frame stride from episode-disjoint symlink roots."""

    def __init__(self, split_root: Path, *, stride: int) -> None:
        if stride <= 0:
            raise ValueError("stride must be positive")
        files = sorted((split_root / "dataset").glob("*/aloha-agilex_clean_50/data/episode*.hdf5"))
        if len(files) != (400 if split_root.name == "train" else 50):
            raise ValueError(f"Wrong episode count under {split_root}: {len(files)}")
        self.index: list[tuple[Path, int]] = []
        for path in files:
            with h5py.File(path, "r") as handle:
                lengths = [len(handle[f"observation/{camera}/rgb"]) for camera in CAMERAS]
            if len(set(lengths)) != 1:
                raise ValueError(f"Camera frame-count mismatch in {path}: {lengths}")
            self.index.extend((path, t) for t in range(0, lengths[0], stride))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> torch.Tensor:
        path, frame_index = self.index[index]
        with h5py.File(path, "r") as handle:
            frames = {}
            for camera in CAMERAS:
                raw = handle[f"observation/{camera}/rgb"][frame_index]
                # RoboTwin encoded RGB with cv2.imencode's BGR convention;
                # cv2.imdecode returns the original RGB ordering here.
                rgb = cv2.imdecode(np.frombuffer(bytes(raw), np.uint8), cv2.IMREAD_COLOR)
                if rgb is None:
                    raise ValueError(f"Cannot decode {path} camera={camera} frame={frame_index}")
                frames[camera] = Image.fromarray(rgb)
        canvas = assemble_multiview_layout(frames, list(CAMERAS), 384, 320)
        pixel = torch.from_numpy(np.asarray(canvas, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        return (pixel - MEAN) / STD


@torch.no_grad()
def dev_loss(encoder, decoder, loader, device) -> float:
    decoder.eval()
    total = 0.0
    count = 0
    for batch in loader:
        pixels = batch.to(device, non_blocking=True)
        latent = encoder.batch_encode(pixels.to(torch.bfloat16).unsqueeze(2)).float()
        predicted = decoder(latent).squeeze(2)
        total += F.l1_loss(predicted, pixels, reduction="sum").item()
        count += pixels.numel()
    decoder.train()
    return total / count


def train(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DINOv2 decoder training")
    repo = Path(__file__).resolve().parents[2]
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    if dirty.strip():
        raise RuntimeError("Commit the research code and plan before a GPU run")
    manifest = args.splits / "split_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Episode split manifest is required: {manifest}")
    if sha256(args.encoder / "model.safetensors") != DINO_SHA256:
        raise ValueError("DINOv2 model hash mismatch")
    if args.steps <= 0 or args.eval_every <= 0 or args.batch_size <= 0:
        raise ValueError("steps, eval_every and batch_size must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    train_data = RoboTwinFrames(args.splits / "train", stride=4)
    dev_data = RoboTwinFrames(args.splits / "dev", stride=16)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    encoder = DinoV2VideoEncoder.from_pretrained(str(args.encoder)).to(device).eval()
    encoder.requires_grad_(False)
    decoder = RepresentationPixelDecoder(in_channels=encoder.properties.z_dim, width=args.width).to(device).train()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=0.01)

    args.output.mkdir(parents=True, exist_ok=True)
    run = {
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "width": args.width,
        "eval_every": args.eval_every,
        "train_frames": len(train_data),
        "dev_frames": len(dev_data),
        "split_manifest_sha256": sha256(manifest),
        "dino_model_sha256": DINO_SHA256,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
    }
    (args.output / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    stream = iter(train_loader)
    best = float("inf")
    for step in range(1, args.steps + 1):
        try:
            batch = next(stream)
        except StopIteration:
            stream = iter(train_loader)
            batch = next(stream)
        pixels = batch.to(device, non_blocking=True)
        with torch.no_grad():
            latent = encoder.batch_encode(pixels.to(torch.bfloat16).unsqueeze(2)).float()
        predicted = decoder(latent).squeeze(2)
        loss = F.l1_loss(predicted, pixels) + 0.1 * F.mse_loss(predicted, pixels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite decoder loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 and not any(p.grad is not None and torch.count_nonzero(p.grad) for p in decoder.parameters()):
            raise RuntimeError("Decoder received no gradients")
        if step % args.eval_every == 0 or step == args.steps:
            validation = dev_loss(encoder, decoder, dev_loader, device)
            record = {"step": step, "train_loss": loss.item(), "dev_l1": validation}
            with (args.output / "metrics.jsonl").open("a") as out:
                out.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            if validation < best:
                best = validation
                temp = args.output / "decoder_best.pt.tmp"
                torch.save({"state_dict": decoder.state_dict(), "config": run, "step": step}, temp)
                temp.replace(args.output / "decoder_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=Path("/root/zifan/openwam-assets/robotwin-splits"))
    parser.add_argument("--encoder", type=Path, default=Path("/root/zifan/openwam-assets/dinov2-base"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())
