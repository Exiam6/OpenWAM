#!/usr/bin/env python3
"""Combine verified offline and native RoboTwin reports after every run finishes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from research.rae.summarize_native_official import MODES, OFFICIAL_CHECKPOINT_SHA256
from research.rae.summarize_native_paired import TASKS
from research.rae.summarize_native_ablation import VARIANTS as ABLATION_VARIANTS, PAIRS as ABLATION_PAIRS

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


def validate_ablation(report: dict, paired: dict) -> None:
    if (report.get("mode") != "demo_clean" or report.get("instruction_type") != "unseen"
            or report.get("training_seed") != 42 or report.get("tasks") != list(TASKS)
            or report.get("variants") != list(ABLATION_VARIANTS)
            or report.get("episodes_per_task_per_variant") != 100
            or report.get("episodes_per_variant") != 1000
            or report.get("registered_rows") != 3000
            or report.get("bootstrap", {}).get("draws") != 20_000
            or len(report.get("per_task", [])) != len(TASKS)
            or len(report.get("comparisons", [])) != len(ABLATION_PAIRS)):
        raise ValueError("Native S-VAE ablation does not cover the registered protocol")
    if report["manifest_hashes"] != paired["manifest_hashes"]:
        raise ValueError("S-VAE ablation did not reuse the matched scene manifests")
    if report["submissions"]["42"]["vae"] != paired["submissions"]["42"]["vae"]:
        raise ValueError("S-VAE ablation changed the seed-42 VAE control")
    if set(report["successes"]) != set(ABLATION_VARIANTS) or set(report["success_rates"]) != set(ABLATION_VARIANTS):
        raise ValueError("Native ablation variant coverage mismatch")
    for variant in ABLATION_VARIANTS:
        successes = report["successes"][variant]
        if (type(successes) is not int or not 0 <= successes <= 1000
                or not math.isclose(report["success_rates"][variant], successes / 1000, abs_tol=1e-12)):
            raise ValueError(f"Native ablation success count mismatch: {variant}")
    if not math.isclose(report["success_rates"]["vae"],
                        sum(row["vae_successes"] for row in paired["per_seed_task"]
                            if row["training_seed"] == 42) / 1000, abs_tol=1e-12):
        raise ValueError("Native ablation VAE outcomes differ from matched seed 42")
    for row in report["per_task"]:
        if (row["task"] not in TASKS or row["episodes_per_variant"] != 100
                or set(row["successes"]) != set(ABLATION_VARIANTS)):
            raise ValueError("Native ablation per-task coverage mismatch")
    for variant in ABLATION_VARIANTS:
        if sum(row["successes"][variant] for row in report["per_task"]) != report["successes"][variant]:
            raise ValueError(f"Native ablation per-task counts do not reconcile: {variant}")
    for item, (treatment, control) in zip(report["comparisons"], ABLATION_PAIRS):
        if (item["treatment"], item["control"]) != (treatment, control):
            raise ValueError("Native ablation comparison order mismatch")
        if sum(item[key] for key in ("treatment_only", "control_only", "both_success", "both_failure")) != 1000:
            raise ValueError("Native ablation paired counts do not reconcile")
        if (item["treatment_only"] + item["both_success"] != report["successes"][treatment]
                or item["control_only"] + item["both_success"] != report["successes"][control]
                or not math.isclose(item["difference"],
                                    report["success_rates"][treatment] - report["success_rates"][control],
                                    abs_tol=1e-12)
                or len(item["bootstrap_95pct_ci"]) != 2
                or not all(math.isfinite(value) for value in item["bootstrap_95pct_ci"])):
            raise ValueError("Native ablation paired outcomes do not reconcile")


def build(paired: dict, official: dict, ablation: dict, offline: dict, width: dict, action: dict,
          *, base_report: Path, paired_path: Path, official_path: Path,
          ablation_path: Path) -> str:
    validate_paired(paired)
    validate_official(official)
    validate_ablation(ablation, paired)
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
    lines += ["", "## Native S-VAE48 ablations (seed 42)", "",
              "These three policies used the same 1,000 scenes and manifest as",
              "the matched seed-42 comparison. This is a single-seed exploratory",
              "comparison; its interval resamples tasks and paired episodes but",
              "does not measure variation across training seeds.", "",
              "| Policy | Successes / 1,000 | Success rate |",
              "|---|---:|---:|"]
    for variant in ABLATION_VARIANTS:
        lines.append(f"| {variant} | {ablation['successes'][variant]}/1000 | "
                     f"{ablation['success_rates'][variant]:.2%} |")
    lines += ["", "| Comparison | Success difference | Paired 95% interval |",
              "|---|---:|---:|"]
    for item in ablation["comparisons"]:
        low, high = item["bootstrap_95pct_ci"]
        lines.append(f"| {item['treatment']} − {item['control']} | "
                     f"{item['difference']:+.2%} | [{low:+.2%}, {high:+.2%}] |")
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
              f"- Native S-VAE48 ablation audit: `{ablation_path}`",
              f"- Native official full-benchmark audit: `{official_path}`",
              f"- Offline and ablation report: `{base_report}`",
              f"- Official released checkpoint SHA256: `{OFFICIAL_CHECKPOINT_SHA256}`", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=RUNS)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.run_root
    base = root / "OPENWAM_RAE_EXPERIMENT_RESULTS_WITH_ACTION.md"
    report = build(read(args.paired), read(args.official), read(args.ablation),
                   read(root / "wam_multiseed_20261024/aggregate_test.json"),
                   read(root / "wam_svae48_seed42/state.json"),
                   read(root / "wam_action_svae48_seed42/state.json"),
                   base_report=base, paired_path=args.paired, official_path=args.official,
                   ablation_path=args.ablation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
