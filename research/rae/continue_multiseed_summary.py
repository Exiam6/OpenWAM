#!/usr/bin/env python3
"""Write the three-seed result bundle once the registered trainer exits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from research.rae.aggregate_multiseed import aggregate
from research.rae.summarize_robotwin_software_rt import summarize

PILOT = Path("/root/zifan/openwam-runs/wam_pilot_20261023")
RUNS = Path("/root/zifan/openwam-runs/wam_multiseed_20261024")
ROBOTWIN = Path("/root/zifan/openwam-runs/robotwin_rt_eval_20261024")


def wait_for_trainer(pid: int) -> None:
    while True:
        proc = Path(f"/proc/{pid}")
        try:
            state = (proc / "stat").read_text().split(") ", 1)[1].split()[0]
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except FileNotFoundError:
            return
        if state == "Z":
            return
        if "research/rae/run_additional_seeds.py" not in command:
            raise RuntimeError(f"PID {pid} is not the registered multi-seed trainer")
        time.sleep(30)


def result_path(split: str, training_seed: int) -> Path:
    if training_seed == 42:
        return PILOT / f"paired_{split}_comparison.json"
    return RUNS / f"seed{training_seed}" / f"paired_{split}_comparison.json"


def write_once(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-pid", type=int, required=True)
    args = parser.parse_args()
    wait_for_trainer(args.after_pid)
    for seed in (43, 44):
        path = RUNS / f"seed{seed}" / "state.json"
        state = json.loads(path.read_text())
        if state["stage"] != "complete":
            raise RuntimeError(f"Seed {seed} did not complete: {state}")

    reports = {}
    for split in ("dev", "test"):
        pairs = [(seed, json.loads(result_path(split, seed).read_text())) for seed in (42, 43, 44)]
        report = aggregate(pairs, draws=20_000, bootstrap_seed=20261024)
        write_once(RUNS / f"aggregate_{split}.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
        reports[split] = report

    rollout_reports = {}
    for label, directory in (
        ("paired", ROBOTWIN / "paired_1spp_cpu_oidn_mesa2625"),
        ("official", ROBOTWIN / "official_1spp_cpu_oidn_mesa2625"),
    ):
        if (directory / "status.tsv").is_file():
            report = summarize(directory)
            (directory / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            rollout_reports[label] = report

    lines = [
        "# Matched OpenWAM VAE / DINOv2 RAE results",
        "",
        "Three 1,000-step training seeds (42, 43, 44), ten preselected RoboTwin tasks.",
        "The action loss below is teacher-forced and lower is better. It is not task success.",
        "",
        "| Split | VAE action loss | RAE action loss | RAE − VAE | Relative change | 95% bootstrap interval (difference) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("dev", "test"):
        report = reports[split]
        low, high = report["bootstrap"]["delta_95pct_ci"]
        lines.append(
            f"| {split} | {report['vae_action_loss_mean']:.7f} | "
            f"{report['rae_action_loss_mean']:.7f} | {report['rae_minus_vae']:+.7f} | "
            f"{report['relative_change']:+.2%} | [{low:+.7f}, {high:+.7f}] |"
        )
    lines.extend(["", "## Per training seed", "",
                  "| Seed | Split | VAE | RAE | RAE − VAE |",
                  "|---:|---|---:|---:|---:|"])
    for split in ("dev", "test"):
        for row in reports[split]["per_seed"]:
            lines.append(
                f"| {row['training_seed']} | {split} | {row['vae_action_loss']:.7f} | "
                f"{row['rae_action_loss']:.7f} | {row['rae_minus_vae']:+.7f} |"
            )
    lines.extend(["", "## Exploratory CPU-rendered RoboTwin rollouts", "",
                  "This CPU renderer differs from the released graphics renderer. Segfaults are",
                  "technical failures and are excluded from task-success denominators.", ""])
    for label, report in rollout_reports.items():
        lines.append(
            f"- {label}: {report['episodes_complete']}/{report['episodes_planned']} completed, "
            f"{report['episodes_with_terminal_technical_error']} terminal technical errors, "
            f"successes among completed episodes {report['successes_by_variant']}."
        )
        if label == "paired":
            lines.append(
                f"  Matched-scene VAE/RAE task pairs: {report['pairs_complete']}; "
                f"completed pairs with different accepted scene seeds: "
                f"{len(report['scene_mismatched_completed_pairs'])}."
            )
    lines.extend(["", "Native RoboTwin success remains unmeasured until a stable Vulkan RT graphics host is available.", ""])
    write_once(RUNS / "SUMMARY.md", "\n".join(lines))
    print(RUNS / "SUMMARY.md", flush=True)


if __name__ == "__main__":
    main()
