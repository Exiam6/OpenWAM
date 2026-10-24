"""Check that the final report joins offline and native results by checkpoint."""

import copy
import tempfile
import unittest
from pathlib import Path

from research.rae.build_complete_native_results import build
from research.rae.summarize_native_ablation import PAIRS, VARIANTS
from research.rae.summarize_native_official import MODES, OFFICIAL_CHECKPOINT_SHA256
from research.rae.summarize_native_paired import TASKS


def reports():
    losses = {42: (0.020, 0.030), 43: (0.025, 0.024), 44: (0.018, 0.020)}
    successes = {42: (50, 40), 43: (60, 60), 44: (70, 80)}
    submissions = {str(seed): {variant: {"checkpoint_sha256": f"{variant}{seed}"}
                               for variant in ("vae", "rae")}
                   for seed in losses}
    manifest_hashes = {task: "sha256:scene" for task in TASKS}
    paired = {
        "mode": "demo_clean", "instruction_type": "unseen", "training_seeds": [42, 43, 44],
        "episodes_per_task_per_seed": 100, "episodes_per_variant": 3000,
        "registered_rows": 6000, "tasks": list(TASKS), "manifest_hashes": manifest_hashes,
        "bootstrap": {"draws": 20_000, "delta_95pct_ci": [-0.1, 0.1]},
        "discordant_pairs": {"both_success": 1700, "vae_only": 100,
                             "rae_only": 100, "both_failure": 1100},
        "vae_success_rate": 0.6, "rae_success_rate": 0.6, "rae_minus_vae": 0.0,
        "per_task": [{"task": task, "episodes_per_variant": 300,
                      "vae_success_rate": 0.6, "rae_success_rate": 0.6,
                      "rae_minus_vae": 0.0} for task in TASKS],
        "per_seed_task": [{"training_seed": seed, "task": task, "episodes": 100,
                           "vae_successes": successes[seed][0],
                           "rae_successes": successes[seed][1]}
                          for seed in losses for task in TASKS],
        "submissions": submissions,
    }
    offline = {
        "training_seeds": [42, 43, 44], "split": "test",
        "per_seed": [{"training_seed": seed, "vae_action_loss": vae,
                      "rae_action_loss": rae, "vae_checkpoint_sha256": f"vae{seed}",
                      "rae_checkpoint_sha256": f"rae{seed}"}
                     for seed, (vae, rae) in losses.items()],
        "vae_action_loss_mean": sum(row[0] for row in losses.values()) / 3,
        "rae_action_loss_mean": sum(row[1] for row in losses.values()) / 3,
    }
    offline["rae_minus_vae"] = offline["rae_action_loss_mean"] - offline["vae_action_loss_mean"]
    ablation_successes = {"vae": 500, "reconstruction_svae48": 550, "action_svae48": 560}
    ablation = {
        "mode": "demo_clean", "instruction_type": "unseen", "training_seed": 42,
        "tasks": list(TASKS), "variants": list(VARIANTS),
        "episodes_per_task_per_variant": 100, "episodes_per_variant": 1000,
        "registered_rows": 3000, "bootstrap": {"draws": 20_000},
        "manifest_hashes": manifest_hashes, "successes": ablation_successes,
        "success_rates": {key: count / 1000 for key, count in ablation_successes.items()},
        "per_task": [{"task": task, "episodes_per_variant": 100,
                      "successes": {key: count // 10 for key, count in ablation_successes.items()}}
                     for task in TASKS],
        "comparisons": [
            {"treatment": treatment, "control": control,
             "treatment_only": treatment_only, "control_only": control_only,
             "both_success": both_success, "both_failure": both_failure,
             "difference": (ablation_successes[treatment] - ablation_successes[control]) / 1000,
             "bootstrap_95pct_ci": [0.0, 0.1]}
            for (treatment, control), treatment_only, control_only, both_success, both_failure in zip(
                PAIRS, (100, 110, 60), (50, 50, 50), (450, 450, 500), (400, 390, 390), strict=True)
        ],
        "submissions": {"42": {"vae": submissions["42"]["vae"],
                               "reconstruction_svae48": {"checkpoint_sha256": "svae42"},
                               "action_svae48": {"checkpoint_sha256": "action42"}}},
    }
    official = {
        "instruction_type": "unseen", "episodes_per_task": 100, "total_episodes": 10_000,
        "submissions": {mode: {"checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256} for mode in MODES},
        "by_mode": [{"mode": mode, "tasks": 50, "episodes": 5000,
                     "successes": count * 50, "task_balanced_success_rate": count / 100}
                    for mode, count in zip(MODES, (80, 70), strict=True)],
        "per_task": [{"mode": mode, "task": f"task_{index}", "episodes": 100,
                      "successes": count}
                     for mode, count in zip(MODES, (80, 70), strict=True) for index in range(50)],
    }
    width = {"stage": "complete", "checkpoint_sha256": "svae42",
             "comparisons": {"test": {"vae_action_loss": 0.020, "svae48_action_loss": 0.021}}}
    action = {"stage": "complete", "checkpoint_sha256": "action42",
              "comparisons": {"test": {
                  "native_vae": {"baseline": 0.020, "action_informed": 0.019},
                  "reconstruction_svae": {"baseline": 0.021, "action_informed": 0.019}}}}
    return paired, official, ablation, offline, width, action


class NativeLossSuccessReportTests(unittest.TestCase):
    def build_report(self, values):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base.md"
            base.write_text("base report\n")
            return build(*values, base_report=base, paired_path=Path("paired.json"),
                         official_path=Path("official.json"), ablation_path=Path("ablation.json"))

    def test_joins_loss_and_success_and_reports_disagreement(self):
        report = self.build_report(reports())
        self.assertIn("| 42 | vae | 0.0200000 | 50.00% | 500/1000 |", report)
        self.assertIn("| 44 | rae | 0.0200000 | 80.00% | 800/1000 |", report)
        self.assertIn("| action_svae48 | 0.0190000 | 560/1000 | 56.00% |", report)
        self.assertIn("aligned in 1, opposed in 1, and tied in 1", report)

    def test_rejects_checkpoint_mismatch(self):
        values = list(reports())
        values[0]["submissions"]["43"]["rae"]["checkpoint_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "checkpoint or loss mismatch"):
            self.build_report(values)

    def test_rejects_missing_native_seed_task(self):
        values = list(reports())
        values[0]["per_seed_task"].pop()
        with self.assertRaisesRegex(ValueError, "per-seed task coverage"):
            self.build_report(values)

    def test_rejects_ablation_checkpoint_mismatch(self):
        values = list(reports())
        values[4]["checkpoint_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "S-VAE48 checkpoint or loss mismatch"):
            self.build_report(values)


if __name__ == "__main__":
    unittest.main()
