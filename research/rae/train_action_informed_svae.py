#!/usr/bin/env python3
"""Fit a 48-channel DINO reducer with feature and action supervision.

This is an action-informed *offline* S-VAE ablation, not ReWAM's online
action-gradient routing. Only the S-VAE is loaded by the later WAM run; the
action readout is a stage-one training head and is saved for auditing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from openwam.model.video_backbone.encoder.svae import SVAE, svae_loss
from openwam.model.video_backbone.encoder.svae.model import _CHECKPOINT_FORMAT_VERSION
from scripts.svae_train.train_svae import _ShardedFeatureDataset


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class PairedFeatureActionDataset(Dataset):
    def __init__(self, features_dir: Path, action_path: Path) -> None:
        self.features = _ShardedFeatureDataset(features_dir)
        self.actions = torch.load(action_path, map_location="cpu", weights_only=True)
        stats = torch.load(features_dir / "stats.pt", map_location="cpu", weights_only=True)
        if len(self.features) != int(stats["selected_clips"]):
            raise ValueError("Feature count differs from collector registration")
        if len(self.features) != int(self.actions["action"].shape[0]):
            raise ValueError("Feature/action clip counts differ")
        if int(stats["world_size"]) != 1:
            raise ValueError("Paired labels require the one-rank feature archive")
        if self.actions["feature_selection_sha256"] != stats["selected_indices_sha256"]:
            raise ValueError("Feature/action selection hashes differ")
        if self.actions["feature_stats_sha256"] != digest(features_dir / "stats.pt"):
            raise ValueError("Feature statistics changed after action extraction")
        if tuple(self.actions["action"].shape[1:]) != (32, 80):
            raise ValueError("Unexpected action shape")
        if self.actions["action_mask"].shape != self.actions["action"].shape:
            raise ValueError("Action mask shape differs")
        self.mean = stats["mean"]
        self.std = stats["std"]
        self.selection_sha256 = stats["selected_indices_sha256"]
        self.action_sha256 = digest(action_path)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.features[index],
            self.actions["action"][index],
            self.actions["action_mask"][index],
        )


class ActionReadout(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int = 80, hidden_dim: int = 256) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, mu: torch.Tensor, steps: int) -> torch.Tensor:
        if mu.ndim != 5:
            raise ValueError("Expected B,C,T,H,W posterior mean")
        temporal = mu.float().mean(dim=(-2, -1))
        temporal = F.interpolate(temporal, size=steps, mode="linear", align_corners=False)
        temporal = temporal.transpose(1, 2)
        position = torch.linspace(-1, 1, steps, device=mu.device).view(1, steps, 1)
        position = position.expand(mu.shape[0], -1, -1)
        return self.layers(torch.cat((temporal, position), dim=-1))


def masked_action_mse(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if predicted.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Prediction, action and mask shapes must agree")
    valid = mask.float()
    if valid.sum().item() == 0:
        raise ValueError("Action batch has no valid targets")
    return ((predicted.float() - target.float()).square() * valid).sum() / valid.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps < 1 or args.batch_size < 1 or args.action_weight < 0:
        raise ValueError("Steps and batch size must be positive; action weight nonnegative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    dataset = PairedFeatureActionDataset(args.features, args.actions)
    model_config = dict(
        input_dim=int(dataset.features.raw_dim),
        latent_dim=48,
        num_heads=16,
        num_layers=3,
        intermediate_size=2048,
        dropout=0.0,
    )
    model = SVAE(**model_config)
    model.set_input_stats(dataset.mean, dataset.std)
    readout = ActionReadout(model.latent_dim)
    model.to(device)
    readout.to(device)
    model.train()
    readout.train()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(readout.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )
    # Match the registered reconstruction-only reducer's warmup/cosine rule.
    warmup_steps = max(1, len(loader))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(
                optimizer,
                T_max=max(args.steps - warmup_steps, 1),
                eta_min=args.learning_rate * 0.9,
            ),
        ],
        milestones=[warmup_steps],
    )
    metrics: list[dict] = []
    step = 0
    while step < args.steps:
        for features, action, mask in loader:
            features = features.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            beta = 1e-4 * min(1.0, step / max(1, int(0.2 * args.steps)))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(features)
                feature_loss, terms = svae_loss(output, beta=beta, cos_weight=1.0)
                prediction = readout(output["mu"], action.shape[1])
                action_loss = masked_action_mse(prediction, action, mask)
                loss = feature_loss + args.action_weight * action_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(readout.parameters()), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step == 1 or step % 20 == 0 or step == args.steps:
                record = {
                    "step": step,
                    "total": float(loss.detach()),
                    "feature": float(feature_loss.detach()),
                    "action_mse": float(action_loss.detach()),
                    "mse": float(terms["mse"].detach()),
                    "cos": float(terms["cos"].detach()),
                    "kl": float(terms["kl"].detach()),
                    "beta": beta,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                metrics.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
            if step >= args.steps:
                break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": _CHECKPOINT_FORMAT_VERSION,
        "model_config": model.config_dict(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "action_readout_state_dict": {k: v.detach().cpu() for k, v in readout.state_dict().items()},
        "metrics": metrics,
        "raw_dim": model.input_dim,
        "num_clips": len(dataset),
        "world_size": 1,
        "stage_one_method": "offline_action_informed_svae",
        "action_weight": args.action_weight,
        "feature_selection_sha256": dataset.selection_sha256,
        "action_targets_sha256": dataset.action_sha256,
        "steps": step,
        "seed": args.seed,
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    print(json.dumps({"checkpoint": str(args.output), "sha256": digest(args.output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
