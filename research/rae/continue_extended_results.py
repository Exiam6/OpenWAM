#!/usr/bin/env python3
"""Write one consolidated report after the action-informed ablation finishes."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from research.rae.build_final_results import build
from research.rae.continue_action_svae48_wam import (
    ACTIONS,
    ACTION_SHA,
    EXPECTED_MAIN_COMMIT,
    REDUCER,
    RUNS as ACTION_RUNS,
    VAE_REPORTS,
    WEIGHTS,
    WIDTH_RUNS,
)
from research.rae.continue_svae48_wam import wait_for
from research.rae.run_additional_seeds import digest

ROOT = Path("/root/zifan/openwam-runs")
BASE_STATE = ROOT / "openwam_rae_experiment_results_state.json"
OUTPUT = ROOT / "OPENWAM_RAE_EXPERIMENT_RESULTS_WITH_ACTION.md"
STATE = ROOT / "openwam_rae_extended_results_state.json"


def record(stage: str, **details: object) -> None:
    data = {"stage": stage, "updated_unix": time.time(), **details}
    temp = STATE.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temp.replace(STATE)
    print(json.dumps(data, sort_keys=True), flush=True)


def require_close(a: float, b: float, label: str) -> None:
    if not math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-10):
        raise ValueError(f"{label}: {a} != {b}")


def append_action_section(base: str) -> str:
    state = json.loads((ACTION_RUNS / "state.json").read_text())
    if state.get("stage") != "complete":
        raise ValueError("Action-informed WAM experiment incomplete")
    if digest(ACTIONS) != ACTION_SHA:
        raise ValueError("Action target archive changed")
    reducer_sha = digest(REDUCER)
    if reducer_sha != state["reducer_sha256"]:
        raise ValueError("Action-informed reducer changed")
    reducer = torch.load(REDUCER, map_location="cpu", weights_only=True)
    if reducer.get("stage_one_method") != "offline_action_informed_svae":
        raise ValueError("Wrong reducer method")
    if reducer.get("action_targets_sha256") != ACTION_SHA or reducer.get("steps") != 1000:
        raise ValueError("Wrong reducer training source or duration")
    checkpoint = Path(state["checkpoint"]) / "checkpoint_step_1000.safetensors"
    if not checkpoint.is_file() or checkpoint.stat().st_size < 20_000_000_000:
        raise ValueError("Action-informed WAM checkpoint missing or truncated")
    marker = WEIGHTS / "rae/source_commit_at_launch.txt"
    if marker.read_text().strip() != EXPECTED_MAIN_COMMIT:
        raise ValueError("Action-informed WAM source commit mismatch")
    if digest(checkpoint) != state["checkpoint_sha256"]:
        raise ValueError("Action-informed WAM checkpoint changed")

    lines = [
        base.rstrip(), "", "## Action-informed visual bottleneck", "",
        "A further seed-42 48-channel DINO/S-VAE reducer used the same 8,192",
        "train-only feature clips, 1,000 reducer steps, and an auxiliary masked",
        "action prediction loss with weight 1.0. Its readout was discarded before",
        "the 1,000-step WAM run. This is an offline auxiliary objective, not",
        "ReWAM's online action-gradient routing or a task-success result.", "",
        "| Split | Control | Control action loss | Action-informed action loss | Difference |",
        "|---|---|---:|---:|---:|",
    ]
    for split in ("dev", "test"):
        for control in ("native_vae", "reconstruction_svae"):
            report_path = ACTION_RUNS / f"paired_{split}_vs_{control}.json"
            report = json.loads(report_path.read_text())
            row = state["comparisons"][split][control]
            if report["split"] != split or report["windows"] != 100 or report["noise_repeats_per_window"] != 4:
                raise ValueError(f"Wrong {split}/{control} evaluation coverage")
            if report["rae_checkpoint_sha256"] != state["checkpoint_sha256"]:
                raise ValueError(f"Wrong {split}/{control} action-informed checkpoint")
            baseline_path = (
                VAE_REPORTS / f"{split}_teacher_forced.json" if control == "native_vae"
                else WIDTH_RUNS / f"rae_{split}_teacher_forced.json"
            )
            baseline = json.loads(baseline_path.read_text())
            if report["vae_checkpoint_sha256"] != baseline["checkpoint_sha256"]:
                raise ValueError(f"Wrong {split}/{control} control checkpoint")
            for field, report_field in (
                ("baseline", "vae_action_mean_across_tasks"),
                ("action_informed", "rae_action_mean_across_tasks"),
                ("difference", "rae_minus_vae"),
            ):
                require_close(row[field], report[report_field], f"{split}/{control}/{field}")
            lines.append(
                f"| {split} | {control} | {row['baseline']:.7f} | "
                f"{row['action_informed']:.7f} | {row['difference']:+.7f} |"
            )
    lines.extend([
        "", f"Reducer SHA256: `{reducer_sha}`.",
        f"WAM checkpoint SHA256: `{state['checkpoint_sha256']}`.",
        f"Detailed reports: `{ACTION_RUNS}`.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-pid", type=int, required=True)
    parser.add_argument("--base-pid", type=int, required=True)
    args = parser.parse_args()
    try:
        record("waiting_for_action", pid=args.action_pid)
        wait_for(args.action_pid, "research/rae/continue_action_svae48_wam.py")
        record("waiting_for_base_report", pid=args.base_pid)
        wait_for(args.base_pid, "research/rae/continue_experiment_results.py")
        base_state = json.loads(BASE_STATE.read_text())
        if base_state.get("stage") != "complete":
            raise ValueError("Base experiment report incomplete")
        result = append_action_section(build())
        if OUTPUT.exists():
            raise FileExistsError(OUTPUT)
        temporary = OUTPUT.with_suffix(".md.tmp")
        temporary.write_text(result, encoding="utf-8")
        temporary.replace(OUTPUT)
        record("complete", output=str(OUTPUT))
    except BaseException as exc:
        record("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
