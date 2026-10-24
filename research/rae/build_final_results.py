#!/usr/bin/env python3
"""Validate completed RAE experiments and write one evidence-backed report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from research.rae.summarize_robotwin_software_rt import summarize

RUNS = Path("/root/zifan/openwam-runs")
PILOT = RUNS / "wam_pilot_20261023"
MULTISEED = RUNS / "wam_multiseed_20261024"
WIDTH = RUNS / "wam_svae48_seed42"
ROBOTWIN = RUNS / "robotwin_rt_eval_20261024"


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-10):
        raise ValueError(f"{label}: {actual} != {expected}")


def load_offline() -> tuple[dict[str, dict], dict[str, list[tuple[int, dict]]]]:
    aggregates = {}
    pairs = {}
    for split in ("dev", "test"):
        aggregate = read(MULTISEED / f"aggregate_{split}.json")
        if aggregate["split"] != split or aggregate["training_seeds"] != [42, 43, 44]:
            raise ValueError(f"Incomplete or mismatched {split} training-seed aggregate")
        if len(aggregate["tasks"]) != 10 or aggregate["windows_per_task"] != 10:
            raise ValueError(f"Wrong {split} task/window coverage")
        if aggregate["noise_repeats_per_window"] != 4 or aggregate["bootstrap"]["draws"] != 20_000:
            raise ValueError(f"Wrong {split} evaluation or uncertainty settings")
        reports = []
        for seed in (42, 43, 44):
            path = (
                PILOT / f"paired_{split}_comparison.json" if seed == 42
                else MULTISEED / f"seed{seed}" / f"paired_{split}_comparison.json"
            )
            report = read(path)
            if report["split"] != split or sorted(report["per_task"]) != aggregate["tasks"]:
                raise ValueError(f"Seed {seed} {split} task mismatch")
            if report["windows"] != 100 or report["noise_repeats_per_window"] != 4:
                raise ValueError(f"Seed {seed} {split} evaluation coverage mismatch")
            if report["split_manifest_sha256"] != aggregate["split_manifest_sha256"]:
                raise ValueError(f"Seed {seed} {split} split manifest mismatch")
            row = next(r for r in aggregate["per_seed"] if r["training_seed"] == seed)
            for variant in ("vae", "rae"):
                if row[f"{variant}_checkpoint_sha256"] != report[f"{variant}_checkpoint_sha256"]:
                    raise ValueError(f"Seed {seed} {split} {variant} checkpoint mismatch")
                require_close(
                    row[f"{variant}_action_loss"],
                    report[f"{variant}_action_mean_across_tasks"],
                    f"Seed {seed} {split} {variant} action loss",
                )
            if seed > 42:
                state = read(MULTISEED / f"seed{seed}" / "state.json")
                if state["stage"] != "complete":
                    raise ValueError(f"Seed {seed} training incomplete")
                for variant in ("vae", "rae"):
                    if state["checkpoint_hashes"][variant] != report[f"{variant}_checkpoint_sha256"]:
                        raise ValueError(f"Seed {seed} {variant} state/report hash mismatch")
                    checkpoint = Path(state["checkpoints"][variant]) / "checkpoint_step_1000.safetensors"
                    if not checkpoint.is_file() or checkpoint.stat().st_size < 20_000_000_000:
                        raise ValueError(f"Seed {seed} {variant} checkpoint missing or truncated")
            reports.append((seed, report))
        for variant in ("vae", "rae"):
            require_close(
                aggregate[f"{variant}_action_loss_mean"],
                sum(report[f"{variant}_action_mean_across_tasks"] for _, report in reports) / 3,
                f"Three-seed {split} {variant} action loss",
            )
        aggregates[split] = aggregate
        pairs[split] = reports
    return aggregates, pairs


def load_width() -> tuple[dict, dict[str, dict], dict]:
    state = read(WIDTH / "state.json")
    if state["stage"] != "complete":
        raise ValueError("48-channel WAM ablation incomplete")
    checkpoint = Path(state["checkpoint"]) / "checkpoint_step_1000.safetensors"
    if not checkpoint.is_file() or checkpoint.stat().st_size < 20_000_000_000:
        raise ValueError("48-channel WAM checkpoint missing or truncated")
    reconstruction = read(RUNS / "svae48_dev_reconstruction.json")
    if reconstruction["reducer_sha256"] != state["svae_sha256"]:
        raise ValueError("48-channel reducer hash mismatch")
    if reconstruction["split"] != "dev" or reconstruction["windows_per_task"] != 10:
        raise ValueError("48-channel feature reconstruction coverage mismatch")
    reports = {}
    for split in ("dev", "test"):
        report = read(WIDTH / f"paired_{split}_comparison.json")
        pilot = read(PILOT / f"paired_{split}_comparison.json")
        if report["split"] != split or report["windows"] != 100:
            raise ValueError(f"48-channel {split} coverage mismatch")
        if report["rae_checkpoint_sha256"] != state["checkpoint_sha256"]:
            raise ValueError(f"48-channel {split} WAM checkpoint mismatch")
        if report["vae_checkpoint_sha256"] != pilot["vae_checkpoint_sha256"]:
            raise ValueError(f"48-channel {split} VAE control mismatch")
        if report["split_manifest_sha256"] != pilot["split_manifest_sha256"]:
            raise ValueError(f"48-channel {split} split mismatch")
        reports[split] = report
    return state, reports, reconstruction


def load_rollouts() -> dict[str, dict]:
    paths = {
        "paired": ROBOTWIN / "paired_1spp_cpu_oidn_mesa2625",
        "official": ROBOTWIN / "official_1spp_cpu_oidn_mesa2625",
    }
    reports = {label: summarize(path) for label, path in paths.items()}
    for label, report in reports.items():
        if any(row["state"] == "pending" for row in report["episodes"]):
            raise ValueError(f"{label} CPU rollout sweep incomplete")
        if report["episodes_complete"] + report["episodes_with_terminal_technical_error"] != report["episodes_planned"]:
            raise ValueError(f"{label} CPU rollout counts do not reconcile")
    return reports


def build() -> str:
    aggregates, pairs = load_offline()
    width_state, width, reconstruction = load_width()
    rollouts = load_rollouts()
    decoder = read(RUNS / "rae_decoder_20261023_seed42" / "test_metrics.json")
    calibrated = read(RUNS / "rae_decoder_calibrated_20261023_seed42" / "test_metrics.json")
    if decoder["test_frames"] != calibrated["test_frames"] or decoder["test_frames"] != 1002:
        raise ValueError("Pixel decoder test-frame coverage mismatch")
    shifts = {shift: read(PILOT / "video_shift_sensitivity" / f"paired_shift{shift}_dev.json")
              for shift in (1, 10)}
    shift5 = read(PILOT / "paired_dev_comparison.json")
    for shift, report in shifts.items():
        if report["evaluation_video_shift"] != shift or report["windows"] != 100:
            raise ValueError(f"Video shift {shift} evaluation mismatch")
        for variant in ("vae", "rae"):
            if report[f"{variant}_checkpoint_sha256"] != shift5[f"{variant}_checkpoint_sha256"]:
                raise ValueError(f"Video shift {shift} {variant} checkpoint mismatch")
    lines = [
        "# OpenWAM visual-representation experiments",
        "",
        "Ten preselected RoboTwin clean_50 tasks; episode-disjoint 400/50/50 train/dev/test split.",
        "Three matched 1,000-step Foundation-initialized seeds compare the native VAE with",
        "frozen, train-calibrated DINOv2 RAE. Action loss is a teacher-forced offline proxy",
        "(lower is better), not robot-task success.", "",
        "## Matched three-seed action loss", "",
        "| Split | VAE | DINOv2 RAE | RAE − VAE | Relative change | 95% paired bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("dev", "test"):
        row = aggregates[split]
        low, high = row["bootstrap"]["delta_95pct_ci"]
        lines.append(
            f"| {split} | {row['vae_action_loss_mean']:.7f} | {row['rae_action_loss_mean']:.7f} | "
            f"{row['rae_minus_vae']:+.7f} | {row['relative_change']:+.2%} | "
            f"[{low:+.7f}, {high:+.7f}] |"
        )
    lines += ["", "Bootstrap resamples training seeds, tasks, and paired windows (20,000 draws).",
              "It describes the selected offline sample and does not estimate success-rate uncertainty.", "",
              "### Per-seed action loss", "",
              "| Seed | Split | VAE | RAE | RAE − VAE |", "|---:|---|---:|---:|---:|"]
    for split in ("dev", "test"):
        for seed, report in pairs[split]:
            lines.append(
                f"| {seed} | {split} | {report['vae_action_mean_across_tasks']:.7f} | "
                f"{report['rae_action_mean_across_tasks']:.7f} | {report['rae_minus_vae']:+.7f} |"
            )
    lines += ["", "### Test action loss by task (mean over three seeds)", "",
              "| Task | VAE | RAE | RAE − VAE |", "|---|---:|---:|---:|"]
    for task in aggregates["test"]["tasks"]:
        vae = sum(report["per_task"][task]["vae_action_mean"] for _, report in pairs["test"]) / 3
        rae = sum(report["per_task"][task]["rae_action_mean"] for _, report in pairs["test"]) / 3
        lines.append(f"| {task} | {vae:.7f} | {rae:.7f} | {rae - vae:+.7f} |")
    lines += ["", "## Width and video-noise checks", "",
              "Frozen 768-to-48-channel S-VAE reducer: 8,192 train windows, 1,000 reducer steps.",
              f"On 100 dev windows, standardized DINO-feature reconstruction MSE was "
              f"{reconstruction['mean_standardized_mse']:.5f}, versus "
              f"{reconstruction['mean_zero_reconstruction_mse']:.5f} for zero reconstruction; "
              f"cosine loss was {reconstruction['mean_standardized_cosine_loss']:.5f}.", "",
              "| Split | Native VAE 48 | DINO/S-VAE 48 | Difference |",
              "|---|---:|---:|---:|"
              ]
    for split in ("dev", "test"):
        row = width[split]
        lines.append(f"| {split} | {row['vae_action_mean_across_tasks']:.7f} | "
                     f"{row['rae_action_mean_across_tasks']:.7f} | {row['rae_minus_vae']:+.7f} |")
    lines += ["", "The reducer is an added trained module, so this controls visual-head width only",
              "approximately and does not isolate representation quality.", "",
              "Seed-42 fixed-checkpoint dev video-shift sensitivity (both trained at shift 5):", "",
              "| Evaluation shift | VAE | RAE | RAE − VAE |", "|---:|---:|---:|---:|"]
    for shift, row in ((1, shifts[1]), (5, shift5), (10, shifts[10])):
        lines.append(f"| {shift} | {row['vae_action_mean_across_tasks']:.7f} | "
                     f"{row['rae_action_mean_across_tasks']:.7f} | {row['rae_minus_vae']:+.7f} |")
    lines += ["", "## Representation decoder reconstruction", "",
              "Both single-frame decoders were selected at step 1,000 and evaluated on 1,002 test frames.", "",
              "| DINO features | Test RGB MAE | Test PSNR (dB) |",
              "|---|---:|---:|",
              f"| Per-token normalization | {decoder['test_rgb_mae']:.5f} | {decoder['test_rgb_psnr_db']:.2f} |",
              f"| Train-calibrated channels | {calibrated['test_rgb_mae']:.5f} | {calibrated['test_rgb_psnr_db']:.2f} |",
              "", "These metrics are frame reconstruction only; generated-video quality was not measured.", "",
              "## Exploratory CPU-rendered RoboTwin rollouts", "",
              "Mesa Lavapipe RT at 1 sample/pixel and CPU OIDN changes the released render domain.",
              "Exit 139 is a simulator failure and is excluded from success denominators.", ""]
    for label in ("paired", "official"):
        row = rollouts[label]
        lines.append(f"- {label}: {row['episodes_complete']}/{row['episodes_planned']} episodes completed; "
                     f"{row['episodes_with_terminal_technical_error']} terminal technical failures; "
                     f"successes among completed episodes {row['successes_by_variant']}.")
    row = rollouts["paired"]
    lines += [f"- Matched-scene pilot pairs: {row['pairs_complete']}; "
              f"VAE {row['vae_successes_in_pairs']}/{row['pairs_complete']}, "
              f"RAE {row['rae_successes_in_pairs']}/{row['pairs_complete']}.", "",
              "These CPU runs cannot establish native RoboTwin success. Native Vulkan RT",
              "reproduction and a reliable matched-scene success comparison remain required.", "",
              "## Evidence", "",
              f"- Three-seed reports and final checkpoint hashes: `{MULTISEED}`",
              f"- Width ablation report and checkpoint hash: `{WIDTH}`",
              f"- Seed-42 paired reports and shift sensitivity: `{PILOT}`",
              f"- S-VAE reducer SHA256: `{width_state['svae_sha256']}`",
              f"- CPU rollout logs and per-task outcomes: `{ROBOTWIN}`",
              ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUNS / "OPENWAM_RAE_EXPERIMENT_RESULTS.md")
    args = parser.parse_args()
    result = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(result, encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
