#!/usr/bin/env python3
"""Pair the registered train-only DINO feature archive with normalized actions.

The feature collector saved clips in a seeded, unpadded permutation but did
not save action labels. Reconstruct that permutation, verify its SHA256, then
read only the matching HDF5 action slices. This avoids decoding videos again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from openwam.dataloader.registry import build_dataset
from openwam.dataloader.utils.unify_action import UNIFY_DIM, map_to_unify


def selected_indices(stats: dict, dataset_len: int) -> list[int]:
    if dataset_len != int(stats["dataset_total_clips"]):
        raise ValueError("Training dataset length changed since feature collection")
    if int(stats["world_size"]) != 1:
        raise ValueError("This extractor requires the registered one-rank feature archive")
    gen = torch.Generator().manual_seed(int(stats["seed"]))
    indices = torch.randperm(dataset_len, generator=gen)[: int(stats["selected_clips"])].tolist()
    digest = hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
    if digest != stats["selected_indices_sha256"]:
        raise ValueError("Feature/archive selection hash mismatch")
    return indices


def locate(dataset, index: int):
    if index < 0 or index >= len(dataset):
        raise IndexError(index)
    import bisect

    sub_index = bisect.bisect_right(dataset._cumulative_lengths, index)
    offset = dataset._cumulative_lengths[sub_index - 1] if sub_index else 0
    sub = dataset._sub_datasets[sub_index]
    ep_index, start = sub._window_index[index - offset]
    return sub, ep_index, start


def action_only(sub, ep_index: int, start: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path = sub._episode_files[ep_index]
    ep_len = sub._episode_lengths[ep_index]
    actual_end = min(start + sub._raw_window_len, ep_len)
    actual_len = actual_end - start
    if actual_len < 2:
        raise ValueError(f"No action target in {path} at {start}")
    with h5py.File(path, "r") as stream:
        raw = sub._read_raw_actions(stream, start, actual_end)
    if actual_len < sub._raw_window_len:
        raw = np.concatenate(
            [raw, np.repeat(raw[-1:], sub._raw_window_len - actual_len, axis=0)], axis=0
        )
    if sub._normalizer is not None:
        raw = sub._normalizer.normalize(raw)
    dim_mask = None
    if sub._unify_dst_index is not None:
        raw, dim_mask = map_to_unify(raw.astype(np.float32), sub._unify_dst_index, UNIFY_DIM)
    action = torch.from_numpy(raw[1 : sub.num_frames].astype(np.float32))
    proprio = torch.from_numpy(raw[0:1].astype(np.float32))
    valid = torch.tensor(
        [(t + 1) < actual_len for t in range(sub.num_action_steps)], dtype=torch.bool
    ).unsqueeze(-1).expand(-1, sub._action_dim_value).contiguous()
    if dim_mask is not None:
        valid &= torch.from_numpy(dim_mask).unsqueeze(0)
    return action, valid, proprio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    stats = torch.load(args.features / "stats.pt", map_location="cpu", weights_only=True)
    cfg = OmegaConf.load(args.features / "config.yaml")
    dataset = build_dataset(cfg.dataloader, split="train")
    indices = selected_indices(stats, len(dataset))

    actions = []
    masks = []
    proprio = []
    for archive_index, dataset_index in enumerate(indices):
        sub, ep_index, start = locate(dataset, dataset_index)
        action, mask, state = action_only(sub, ep_index, start)
        if archive_index in (0, len(indices) // 2, len(indices) - 1):
            reference = dataset[dataset_index]
            if not torch.equal(action, reference["action"]):
                raise ValueError(f"Action mismatch at archive index {archive_index}")
            if not torch.equal(mask, reference["action_mask"]):
                raise ValueError(f"Action mask mismatch at archive index {archive_index}")
            if not torch.equal(state, reference["proprio"]):
                raise ValueError(f"Proprio mismatch at archive index {archive_index}")
        actions.append(action)
        masks.append(mask)
        proprio.append(state)
        if (archive_index + 1) % 1024 == 0:
            print(f"{archive_index + 1}/{len(indices)} labels", flush=True)

    payload = {
        "action": torch.stack(actions),
        "action_mask": torch.stack(masks),
        "proprio": torch.stack(proprio),
        "dataset_index": torch.tensor(indices, dtype=torch.int64),
        "feature_selection_sha256": stats["selected_indices_sha256"],
        "feature_stats_sha256": hashlib.sha256((args.features / "stats.pt").read_bytes()).hexdigest(),
        "scope": "Train-only labels paired with the registered DINO feature archive",
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, temporary)
    temporary.replace(args.output)
    manifest = {
        "output": str(args.output),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "clips": len(indices),
        "action_shape": list(payload["action"].shape),
        "feature_selection_sha256": stats["selected_indices_sha256"],
        "scope": payload["scope"],
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
