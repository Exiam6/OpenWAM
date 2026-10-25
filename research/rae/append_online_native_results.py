#!/usr/bin/env python3
"""Append the audited online-action ablation to the complete native report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from research.rae.summarize_native_online import PAIRS, VARIANTS
from research.rae.summarize_native_paired import TASKS


def read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def validate(online: dict, paired: dict, state: dict) -> None:
    if (online.get("mode") != "demo_clean" or online.get("instruction_type") != "unseen"
            or online.get("training_seed") != 42 or online.get("tasks") != list(TASKS)
            or online.get("variants") != list(VARIANTS)
            or online.get("episodes_per_task_per_variant") != 100
            or online.get("episodes_per_variant") != 1000
            or online.get("registered_rows") != 3000
            or online.get("bootstrap", {}).get("draws") != 20_000
            or len(online.get("per_task", [])) != len(TASKS)
            or len(online.get("comparisons", [])) != len(PAIRS)):
        raise ValueError("Online native result does not cover the registered protocol")
    if state.get("stage") != "complete" or online["manifest_hashes"] != paired["manifest_hashes"]:
        raise ValueError("Online training incomplete or native manifests differ")
    if set(online["successes"]) != set(VARIANTS) or set(online["success_rates"]) != set(VARIANTS):
        raise ValueError("Online native variant coverage mismatch")
    submissions = online["submissions"]["42"]
    if any(submissions[name] != paired["submissions"]["42"][name] for name in ("vae", "rae")):
        raise ValueError("Online ablation changed a seed-42 control submission")
    if submissions["online_action"]["checkpoint_sha256"] != state["checkpoint_sha256"]:
        raise ValueError("Online native/offline checkpoint SHA256 mismatch")
    for variant in VARIANTS:
        successes = online["successes"][variant]
        if (type(successes) is not int or not 0 <= successes <= 1000
                or not math.isclose(online["success_rates"][variant], successes / 1000, abs_tol=1e-12)
                or sum(row["successes"][variant] for row in online["per_task"]) != successes):
            raise ValueError(f"Online native success counts do not reconcile: {variant}")
    if ({row["task"] for row in online["per_task"]} != set(TASKS)
            or any(row.get("episodes_per_variant") != 100 or set(row["successes"]) != set(VARIANTS)
                   for row in online["per_task"])):
        raise ValueError("Online native per-task coverage mismatch")
    seed42 = [row for row in paired["per_seed_task"] if row["training_seed"] == 42]
    if len(seed42) != len(TASKS):
        raise ValueError("Paired seed-42 task coverage incomplete")
    for variant in ("vae", "rae"):
        if online["successes"][variant] != sum(row[f"{variant}_successes"] for row in seed42):
            raise ValueError(f"Online {variant} control outcomes differ from matched evaluation")
    for item, (treatment, control) in zip(online["comparisons"], PAIRS):
        if (item["treatment"], item["control"]) != (treatment, control):
            raise ValueError("Online comparison order mismatch")
        if sum(item[key] for key in ("treatment_only", "control_only", "both_success", "both_failure")) != 1000:
            raise ValueError("Online paired outcomes do not reconcile")
        if (item["treatment_only"] + item["both_success"] != online["successes"][treatment]
                or item["control_only"] + item["both_success"] != online["successes"][control]
                or not math.isclose(item["difference"], online["success_rates"][treatment]
                                    - online["success_rates"][control], abs_tol=1e-12)
                or len(item["bootstrap_95pct_ci"]) != 2
                or not all(math.isfinite(x) for x in item["bootstrap_95pct_ci"])):
            raise ValueError("Online paired rates or confidence interval invalid")


def offline_losses(state: dict, paired: dict) -> dict[str, float]:
    results = {}
    for control in ("native_vae", "raw_dino"):
        item = state["comparisons"]["test"][control]
        report = read(Path(item["comparison"]))
        if (report.get("split") != "test" or report.get("windows") != 100
                or report.get("noise_repeats_per_window") != 4
                or report.get("rae_checkpoint_sha256") != state["checkpoint_sha256"]
                or report.get("vae_checkpoint_sha256") != paired["submissions"]["42"]["vae" if control == "native_vae" else "rae"]["checkpoint_sha256"]):
            raise ValueError(f"Wrong online offline test comparison: {control}")
        for field, report_field in (("baseline_loss", "vae_action_mean_across_tasks"),
                                    ("online_loss", "rae_action_mean_across_tasks"),
                                    ("difference", "rae_minus_vae")):
            if not math.isclose(item[field], report[report_field], rel_tol=1e-7, abs_tol=1e-10):
                raise ValueError(f"Online offline loss mismatch: {control}/{field}")
        results[control] = item["baseline_loss"]
        results["online_action"] = item["online_loss"]
    return results


def build(core: str, online: dict, paired: dict, state: dict,
          *, core_path: Path, online_path: Path) -> str:
    validate(online, paired, state)
    losses = offline_losses(state, paired)
    lines = [core.rstrip(), "", "## Online action-grounded DINO adapter (seed 42)", "",
             "The adapter starts from the matched seed-42 DINO initialization and",
             "receives online gradients from the action loss through the current latent.",
             "It retains the 768-channel grid and is not a full ReWAM TRB reproduction.",
             "These three policies use the same 1,000 selected-task scenes and manifests.",
             "This is a single-seed ablation; intervals resample tasks and paired episodes.", "",
             "| Policy | Test action loss ↓ | Successes / 1,000 | Native success ↑ |",
             "|---|---:|---:|---:|"]
    for variant, loss_key in (("vae", "native_vae"), ("rae", "raw_dino"),
                              ("online_action", "online_action")):
        lines.append(f"| {variant} | {losses[loss_key]:.7f} | "
                     f"{online['successes'][variant]}/1000 | {online['success_rates'][variant]:.2%} |")
    lines += ["", "| Comparison | Success difference | Paired 95% interval |",
              "|---|---:|---:|"]
    for row in online["comparisons"]:
        low, high = row["bootstrap_95pct_ci"]
        lines.append(f"| {row['treatment']} − {row['control']} | "
                     f"{row['difference']:+.2%} | [{low:+.2%}, {high:+.2%}] |")
    lines += ["", "| Task | VAE | Raw DINO | Online adapter |", "|---|---:|---:|---:|"]
    for row in online["per_task"]:
        counts = row["successes"]
        lines.append(f"| {row['task']} | {counts['vae'] / 100:.1%} | "
                     f"{counts['rae'] / 100:.1%} | {counts['online_action'] / 100:.1%} |")
    lines += ["", f"Online checkpoint SHA256: `{state['checkpoint_sha256']}`.",
              f"Online native audit: `{online_path}`.",
              f"Core complete report: `{core_path}`.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--online", type=Path, required=True)
    parser.add_argument("--online-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.core.is_file():
        raise FileNotFoundError(args.core)
    report = build(args.core.read_text(), read(args.online), read(args.paired),
                   read(args.online_state), core_path=args.core, online_path=args.online)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(report)
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
