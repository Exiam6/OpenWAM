"""Recompute saved RAE sigma curves on CPU, without loading any checkpoint."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from sigma_metrics import recover_unweighted_curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("audited_flow", args.scheduler)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = {
        "scope": "CPU reweighting of saved single-index video losses; no model execution",
        "scheduler_sha256": hashlib.sha256(args.scheduler.read_bytes()).hexdigest(),
        "curves": {},
        "paired": {},
        "limits": [
            "Recovered values retain the old nearest-grid sigma mismatch and model-dtype rounding.",
            "Original per-window values were not saved, so paired uncertainty cannot be reconstructed.",
            "Video MSE does not establish robot closed-loop success.",
        ],
    }
    for seed in [42, 43]:
        records = []
        for arm in ["raw768_native", "raw768_shift16"]:
            path = args.curves / f"curve-{arm}-seed{seed}.json"
            record = json.loads(path.read_text())
            scheduler = module.FlowMatchScheduler("Wan")
            scheduler.set_timesteps(1000, training=True, shift=record["shift_video"])
            record["curve"] = recover_unweighted_curve(record["curve"], scheduler)
            record["original_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            output["curves"][f"{arm}-seed{seed}"] = record
            records.append(record)
        pairs = []
        for baseline, candidate in zip(records[0]["curve"], records[1]["curve"], strict=True):
            if baseline["target_sigma"] != candidate["target_sigma"] or baseline["n"] != candidate["n"]:
                raise ValueError("Curves do not share a target grid/population size")
            pairs.append(
                {
                    "target_sigma": baseline["target_sigma"],
                    "actual_sigma_difference": candidate["actual_sigma"] - baseline["actual_sigma"],
                    "weighted_loss_ratio": candidate["loss_video"] / baseline["loss_video"],
                    "weight_ratio": candidate["training_weight"] / baseline["training_weight"],
                    "unweighted_mse_ratio": candidate["video_mse"] / baseline["video_mse"],
                }
            )
        output["paired"][str(seed)] = pairs
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["paired"], indent=2))


if __name__ == "__main__":
    main()
