"""Exact shared-sigma, unweighted video MSE; diagnostic only, not policy success.

Run only in an authorized experiment allocation. This tool loads a checkpoint
and uses the selected study's trusted common.py; importing this module does not.
The old evaluator is retained unchanged for historical reproducibility.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from sigma_metrics import fixed_video_sigma

TARGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def evaluate_curve(model, cache, variant, inputs_for, paired_loss, *, targets=TARGETS, draws=3, guard=None):
    """Evaluate dev windows with paired noise seeds and preserve individual losses.

    paired_loss must call native compute_loss with lambda_video=lambda_action=1
    and independent video/action noise streams (the frozen common.py contract).
    Action losses retain their native training weights and sampling distribution.
    """
    if guard is not None:
        guard()
    if draws < 1 or draws > 10:
        raise ValueError("draws must be in [1, 10] to keep noise seeds unique")
    if model.training:
        raise ValueError("model must be in evaluation mode")
    dev_ids = [i for i, meta in enumerate(cache["metadata"]) if meta["split"] == "dev"]
    if not dev_ids:
        raise ValueError("cache has no dev windows")
    scheduler = model.video_backbone.scheduler
    if scheduler is model.action_backbone.scheduler:
        raise ValueError("video and action schedulers must be independent")
    rows = []
    with torch.no_grad():
        for target in targets:
            values = []
            with fixed_video_sigma(scheduler, target):
                actual_sigma = float(scheduler.sigmas[0].to(dtype=model.dtype))
                actual_timestep = float(scheduler.timesteps[0].to(dtype=model.dtype))
                for i in dev_ids:
                    for draw in range(draws):
                        if guard is not None:
                            guard()
                        seed = 7000000 + i * 10 + draw
                        inputs = inputs_for(cache, [i], variant, checkpoint=False)
                        inputs["min_timestep_boundary"] = 0.0
                        inputs["max_timestep_boundary"] = 1.0
                        loss = paired_loss(model, inputs, seed)
                        video = float(loss["loss_video"])
                        action = float(loss["loss_action"])
                        if not all(torch.isfinite(torch.tensor([video, action]))):
                            raise ValueError("nonfinite evaluation loss")
                        values.append(
                            {
                                "window_index": i,
                                "metadata": cache["metadata"][i],
                                "draw": draw,
                                "noise_seed": seed,
                                "video_mse": video,
                                "action_training_weighted_loss": action,
                            }
                        )
            rows.append(
                {
                    "target_sigma": target,
                    "model_sigma": actual_sigma,
                    "model_timestep": actual_timestep,
                    "video_weight": 1.0,
                    "video_mse": sum(v["video_mse"] for v in values) / len(values),
                    "dev_windows": len(dev_ids),
                    "draws": draws,
                    "values": values,
                }
            )
    return rows


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("variant")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--draws", type=int, default=3, choices=range(1, 11))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    study = args.study.resolve()
    helper_path = study / "common.py"
    spec = importlib.util.spec_from_file_location("sigma_study_common", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.guard()
    cache_path = study / "cache.pt"
    provenance = {
        str(p.resolve()): sha256(p)
        for p in [
            helper_path,
            cache_path,
            args.checkpoint,
            Path(__file__),
            Path(__file__).with_name("sigma_metrics.py"),
            study / "templates" / args.variant / "config.yaml",
        ]
    }
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    model, _ = helper.build(args.variant, 42)
    helper.activate(model)
    model.load_checkpoint(str(args.checkpoint))
    model.eval()
    curve = evaluate_curve(
        model, cache, args.variant, helper.inputs_for, helper.paired_loss, draws=args.draws, guard=helper.guard
    )
    result = {
        "metric_version": 2,
        "study": str(study),
        "variant": args.variant,
        "checkpoint": str(args.checkpoint.resolve()),
        "sha256": provenance,
        "model_dtype": str(model.dtype),
        "torch_version": torch.__version__,
        "shift_video": float(model.video_backbone.shift_video),
        "note": "Video MSE uses unit weights at exact shared sigma before dtype rounding. "
        "Action loss remains training-weighted. Neither measures closed-loop success. "
        "Repeat draws are not independent demonstrations.",
        "curve": curve,
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
