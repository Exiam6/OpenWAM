#!/usr/bin/env python3
"""Graft frozen DINOv2 onto the released OpenWAM trunk for a bounded pilot.

Only the two DiT latent projections are reinitialized. Every other shared
parameter must match the released checkpoint by name and shape. The native
VAE is removed and the DINOv2 weights come from a separately verified model.
This produces an initialization artifact, not a trained RAE WAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict
from safetensors import safe_open
from transformers import AutoModel

from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
from openwam.train.utils.visual_projection_reset import reset_visual_projections

DONOR_SHA256 = "180a02653118b0f96da28a9cae9ec7b4c1c1e6cd0e4608b56c7b4884f8001d3d"
DINO_SHA256 = "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841"
FEATURE_STATS_SHA256 = "c582c0c8dca54f0944d4d5a99dd1fc34a25948bd61313ba17ba61fd92d54b838"
PROJECTIONS = (
    "video_backbone.dit.patch_embedding.",
    "video_backbone.dit.head.head.",
    "video_backbone.dit2.patch_embedding.",
    "video_backbone.dit2.head.head.",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(args: argparse.Namespace) -> None:
    donor_weights = args.donor / "checkpoint_step_154000.safetensors"
    if sha256(donor_weights) != DONOR_SHA256:
        raise ValueError("Released donor checkpoint hash mismatch")
    if sha256(args.dino / "model.safetensors") != DINO_SHA256:
        raise ValueError("DINOv2 checkpoint hash mismatch")
    if args.feature_stats is not None and sha256(args.feature_stats) != FEATURE_STATS_SHA256:
        raise ValueError("Train-only feature statistics hash mismatch")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    weight_output = output / "checkpoint_step_0.safetensors"
    if weight_output.exists():
        raise FileExistsError(weight_output)
    temporary_weights = output / "checkpoint_step_0.safetensors.tmp"
    if temporary_weights.exists():
        raise FileExistsError(f"Incomplete earlier save requires inspection: {temporary_weights}")

    cfg = OmegaConf.load(args.donor / "config.yaml")
    with open_dict(cfg):
        cfg.model.video_backbone.from_scratch = True
        cfg.model.video_backbone.encoder = {
            "name": "dinov2",
            "model_path": str(args.dino),
            "feature_stats_path": str(args.feature_stats) if args.feature_stats else None,
        }
    OmegaConf.save(cfg, output / "config.yaml")
    tokenizer = output / "tokenizer"
    if not tokenizer.exists():
        tokenizer.symlink_to((args.donor / "tokenizer").resolve(), target_is_directory=True)
    sidecar = output / "dinov2" / "config.json"
    sidecar.parent.mkdir(exist_ok=True)
    shutil.copy2(args.dino / "config.json", sidecar)

    torch.manual_seed(args.seed)
    _, model, _ = build_architecture_from_ckpt_dir(str(output), weights_required=False)
    target = model.state_dict()
    matched = []
    donor_skipped = []
    with safe_open(donor_weights, framework="pt", device="cpu") as donor:
        for name in donor.keys():
            if name.startswith("video_backbone.vae."):
                donor_skipped.append(name)
                continue
            if name.startswith(PROJECTIONS):
                donor_skipped.append(name)
                continue
            if name not in target:
                raise KeyError(f"Unexpected donor key outside native VAE: {name}")
            shape = tuple(donor.get_slice(name).get_shape())
            if shape != tuple(target[name].shape):
                raise ValueError(
                    f"Shared tensor shape mismatch: {name}: donor={shape}, target={tuple(target[name].shape)}"
                )
            target[name].copy_(donor.get_tensor(name))
            matched.append(name)

    missing = set(target) - set(matched)
    expected_missing = {
        k for k in missing if k.startswith("video_backbone.video_encoder.") or k.startswith(PROJECTIONS)
    }
    if missing != expected_missing:
        raise KeyError(f"Uninitialized shared target keys: {sorted(missing - expected_missing)[:20]}")
    vit = AutoModel.from_pretrained(str(args.dino), torch_dtype=torch.bfloat16).eval()
    model.video_backbone.video_encoder._m.load_state_dict(vit.state_dict(), strict=True)
    del vit
    if args.feature_stats is not None:
        stats = torch.load(args.feature_stats, map_location="cpu", weights_only=True)
        model.video_backbone.video_encoder._set_feature_stats(stats["mean"], stats["std"])
    reset_names = reset_visual_projections(model, args.seed + 1009)
    model.save_checkpoint(str(temporary_weights))
    temporary_weights.replace(weight_output)
    record = {
        "donor_sha256": DONOR_SHA256,
        "dino_sha256": DINO_SHA256,
        "feature_stats_sha256": FEATURE_STATS_SHA256 if args.feature_stats else None,
        "seed": args.seed,
        "matched_shared_tensors": len(matched),
        "dropped_native_vae_and_projection_tensors": len(donor_skipped),
        "new_dino_and_projection_tensors": len(missing),
        "reset_visual_projections": reset_names,
        "output_weights_sha256": sha256(weight_output),
        "scope": "Foundation initialization only; no WAM training or task evaluation",
    }
    (output / "graft_manifest.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--donor",
        type=Path,
        default=Path("/root/zifan/openwam-assets/foundation-checkpoint"),
    )
    parser.add_argument("--dino", type=Path, default=Path("/root/zifan/openwam-assets/dinov2-base"))
    parser.add_argument("--feature-stats", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    build(parser.parse_args())
