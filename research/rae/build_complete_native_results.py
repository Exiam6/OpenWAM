#!/usr/bin/env python3
"""Combine verified offline and native RoboTwin reports after every run finishes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from research.rae.summarize_native_official import MODES, OFFICIAL_CHECKPOINT_SHA256
from research.rae.summarize_native_paired import TASKS

RUNS = Path("/root/zifan/openwam-runs")


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def validate_paired(report: dict) -> None:
    if (report.get("mode") != "demo_clean" or report.get("instruction_type") != "unseen"
            or report.get("training_seeds") != [42, 43, 44]
            or report.get("episodes_per_task_per_seed") != 100
            or report.get("episodes_per_variant") != 3000
            or report.get("registered_rows") != 6000
            or report.get("tasks") != list(TASKS)
            or report.get("bootstrap", {}).get("draws") != 20_000
            or len(report.get("per_task", [])) != len(TASKS)):
        raise ValueError("Native paired result does not cover the registered protocol")
    if sum(report["discordant_pairs"].values()) != 3000:
        raise ValueError("Native paired outcome counts do not reconcile")
    vae_successes = report["discordant_pairs"]["both_success"] + report["discordant_pairs"]["vae_only"]
    rae_successes = report["discordant_pairs"]["both_success"] + report["discordant_pairs"]["rae_only"]
    if (not math.isclose(report["vae_success_rate"], vae_successes / 3000, abs_tol=1e-12)
            or not math.isclose(report["rae_success_rate"], rae_successes / 3000, abs_tol=1e-12)):
        raise ValueError("Native paired success counts do not reconcile")
    for row in report["per_task"]:
        if row["task"] not in TASKS or row["episodes_per_variant"] != 300:
            raise ValueError("Native per-task coverage mismatch")
    low, high = report["bootstrap"]["delta_95pct_ci"]
    for value in (low, high, report["vae_success_rate"], report["rae_success_rate"], report["rae_minus_vae"]):
        if not math.isfinite(value):
            raise ValueError("Non-finite native paired estimate")
    if not math.isclose(report["rae_minus_vae"], report["rae_success_rate"] - report["vae_success_rate"], abs_tol=1e-12):
        raise ValueError("Native paired delta does not reconcile")


def validate_official(report: dict) -> None:
    if (report.get("instruction_type") != "unseen" or report.get("episodes_per_task") != 100
            or report.get("total_episodes") != 10_000
            or len(report.get("per_task", [])) != 100
            or len(report.get("by_mode", [])) != 2
            or set(report.get("submissions", {})) != set(MODES)):
        raise ValueError("Official native result does not cover both full modes")
    if {row["mode"] for row in report["by_mode"]} != set(MODES):
        raise ValueError("Official modes mismatch")
    for mode in MODES:
        spec = report["submissions"][mode]
        if spec["checkpoint_sha256"] != OFFICIAL_CHECKPOINT_SHA256:
            raise ValueError("Official checkpoint digest mismatch")
        row = next(item for item in report["by_mode"] if item["mode"] == mode)
        if row["tasks"] != 50 or row["episodes"] != 5000:
            raise ValueError(f"Official {mode} coverage mismatch")
        selected = [item for item in report["per_task"] if item["mode"] == mode]
        if len(selected) != 50 or len({item["task"] for item in selected}) != 50:
            raise ValueError(f"Official {mode} task coverage mismatch")
        if any(item["episodes"] != 100 for item in selected):
            raise ValueError(f"Official {mode} episode coverage mismatch")
        if row["successes"] != sum(item["successes"] for item in selected):
            raise ValueError(f"Official {mode} successes do not reconcile")
        if not math.isclose(row["task_balanced_success_rate"], row["successes"] / 5000, abs_tol=1e-12):
            raise ValueError(f"Official {mode} rate does not reconcile")


def build(paired: dict, official: dict, offline: dict, width: dict, action: dict,
          *, base_report: Path, paired_path: Path, official_path: Path) -> str:
    validate_paired(paired)
    validate_official(official)
    if (offline.get("training_seeds") != [42, 43, 44] or offline.get("split") != "test"
            or width.get("stage") != "complete" or action.get("stage") != "complete"
            or not base_report.is_file()):
        raise ValueError("Offline training, ablations, or consolidated base report incomplete")
    low, high = paired["bootstrap"]["delta_95pct_ci"]
    lines = [
        "# OpenWAM visual representation: complete experiment results", "",
        "## Matched native RoboTwin success", "",
        "Ten preselected `demo_clean` tasks, 100 matched feasible scenes per task,",
        "three training seeds (42, 43, 44), unseen instructions. Each VAE/RAE",
        "pair uses the same scene manifest and evaluation settings.", "",
        "| Method | Success rate | Successful episodes / total |",
        "|---|---:|---:|",
        f"| Native VAE | {paired['vae_success_rate']:.2%} | "
        f"{paired['discordant_pairs']['both_success'] + paired['discordant_pairs']['vae_only']}/3000 |",
        f"| DINOv2 RAE | {paired['rae_success_rate']:.2%} | "
        f"{paired['discordant_pairs']['both_success'] + paired['discordant_pairs']['rae_only']}/3000 |", "",
        f"RAE − VAE success difference: **{paired['rae_minus_vae']:+.2%}**; "
        f"paired hierarchical bootstrap 95% interval **[{low:+.2%}, {high:+.2%}]** "
        "(20,000 draws over training seeds, tasks and episodes).", "",
        f"Discordant scene pairs: RAE only {paired['discordant_pairs']['rae_only']}; "
        f"VAE only {paired['discordant_pairs']['vae_only']}.", "",
        "| Task | VAE | RAE | RAE − VAE |", "|---|---:|---:|---:|",
    ]
    for row in paired["per_task"]:
        lines.append(f"| {row['task']} | {row['vae_success_rate']:.1%} | "
                     f"{row['rae_success_rate']:.1%} | {row['rae_minus_vae']:+.1%} |")
    lines += ["", "## Official released-checkpoint reproduction", "",
              "The released RoboTwin-Full checkpoint is a separate reference; its",
              "training data and model provenance differ from the matched experiment.", "",
              "| Mode | Successes | Episodes | Success rate |",
              "|---|---:|---:|---:|"]
    for mode in MODES:
        row = next(item for item in official["by_mode"] if item["mode"] == mode)
        lines.append(f"| {mode} | {row['successes']} | {row['episodes']} | "
                     f"{row['task_balanced_success_rate']:.2%} |")
    lines += ["", "## Offline action prediction", "",
              "Teacher-forced action loss is a proxy; lower is better. The table uses",
              "the same three matched training seeds as the native comparison.", "",
              "| Split | VAE loss | RAE loss | RAE − VAE |", "|---|---:|---:|---:|",
              f"| Test | {offline['vae_action_loss_mean']:.7f} | "
              f"{offline['rae_action_loss_mean']:.7f} | {offline['rae_minus_vae']:+.7f} |", "",
              "The seed-42 width and offline action-informed bottleneck ablations,",
              f"reconstruction checks, dev results and CPU diagnostics are in `{base_report}`.", "",
              "## Evidence", "",
              f"- Native paired audit and per-task summaries: `{paired_path}`",
              f"- Native official full-benchmark audit: `{official_path}`",
              f"- Offline and ablation report: `{base_report}`",
              f"- Official released checkpoint SHA256: `{OFFICIAL_CHECKPOINT_SHA256}`", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=RUNS)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.run_root
    base = root / "OPENWAM_RAE_EXPERIMENT_RESULTS_WITH_ACTION.md"
    report = build(read(args.paired), read(args.official),
                   read(root / "wam_multiseed_20261024/aggregate_test.json"),
                   read(root / "wam_svae48_seed42/state.json"),
                   read(root / "wam_action_svae48_seed42/state.json"),
                   base_report=base, paired_path=args.paired, official_path=args.official)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
