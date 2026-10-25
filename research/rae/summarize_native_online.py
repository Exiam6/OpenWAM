#!/usr/bin/env python3
"""Audit seed-42 native success for VAE, raw DINO, and online action adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.rae.summarize_native_ablation import summarize_rows as summarize_single_seed
from research.rae.summarize_native_paired import TASKS, labtasker_rows

VARIANTS = ("vae", "rae", "online_action")
PAIRS = (("online_action", "vae"), ("online_action", "rae"))


def summarize_rows(rows: list[dict], *, tasks: tuple[str, ...] = TASKS,
                   episodes_per_task: int = 100, draws: int = 20_000,
                   rng_seed: int = 20261025) -> dict:
    return summarize_single_seed(
        rows, tasks=tasks, episodes_per_task=episodes_per_task,
        draws=draws, rng_seed=rng_seed, variants=VARIANTS, pairs=PAIRS,
        scope="Native RTX/Vulkan RT manifest-paired seed-42 online action adapter ablation",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    registry = json.loads(args.registry.read_text())
    rows = labtasker_rows(registry, tasks=TASKS, episodes=args.episodes,
                          seeds=(42,), variants=VARIANTS)
    summary = summarize_rows(rows, episodes_per_task=args.episodes,
                             draws=args.bootstrap_draws)
    summary["submissions"] = registry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"success_rates": summary["success_rates"],
                      "comparisons": summary["comparisons"]}, sort_keys=True))


if __name__ == "__main__":
    main()
