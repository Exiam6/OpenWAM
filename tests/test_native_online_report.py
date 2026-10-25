"""The augmented report rejects incomplete or mismatched native evidence."""

import json
import tempfile
import unittest
from pathlib import Path

from research.rae.append_online_native_results import build, validate
from research.rae.summarize_native_paired import TASKS


class NativeOnlineReportTests(unittest.TestCase):
    def fixture(self, root: Path):
        vae = {"checkpoint_sha256": "a" * 64, "submission_id": "vae-42"}
        rae = {"checkpoint_sha256": "b" * 64, "submission_id": "rae-42"}
        online_spec = {"checkpoint_sha256": "c" * 64, "submission_id": "online-42"}
        paired = {
            "manifest_hashes": {task: f"sha256:{task}" for task in TASKS},
            "submissions": {"42": {"vae": vae, "rae": rae}},
            "per_seed_task": [{"training_seed": 42, "task": task,
                               "vae_successes": 0, "rae_successes": 0} for task in TASKS],
        }
        zero_pair = lambda control: {
            "treatment": "online_action", "control": control, "difference": 0.0,
            "treatment_only": 0, "control_only": 0,
            "both_success": 0, "both_failure": 1000,
            "bootstrap_95pct_ci": [0.0, 0.0],
        }
        online = {
            "mode": "demo_clean", "instruction_type": "unseen", "training_seed": 42,
            "tasks": list(TASKS), "variants": ["vae", "rae", "online_action"],
            "episodes_per_task_per_variant": 100, "episodes_per_variant": 1000,
            "registered_rows": 3000, "bootstrap": {"draws": 20_000},
            "manifest_hashes": paired["manifest_hashes"],
            "submissions": {"42": {"vae": vae, "rae": rae, "online_action": online_spec}},
            "successes": {"vae": 0, "rae": 0, "online_action": 0},
            "success_rates": {"vae": 0.0, "rae": 0.0, "online_action": 0.0},
            "per_task": [{"task": task, "episodes_per_variant": 100,
                          "successes": {"vae": 0, "rae": 0, "online_action": 0}} for task in TASKS],
            "comparisons": [zero_pair("vae"), zero_pair("rae")],
        }
        state = {"stage": "complete", "checkpoint_sha256": "c" * 64,
                 "comparisons": {"test": {}}}
        for name, baseline_sha in (("native_vae", "a" * 64), ("raw_dino", "b" * 64)):
            path = root / f"{name}.json"
            path.write_text(json.dumps({
                "split": "test", "windows": 100, "noise_repeats_per_window": 4,
                "vae_checkpoint_sha256": baseline_sha, "rae_checkpoint_sha256": "c" * 64,
                "vae_action_mean_across_tasks": 0.02,
                "rae_action_mean_across_tasks": 0.019,
                "rae_minus_vae": -0.001,
            }))
            state["comparisons"]["test"][name] = {
                "comparison": str(path), "baseline_loss": 0.02,
                "online_loss": 0.019, "difference": -0.001,
            }
        return online, paired, state

    def test_complete_augmented_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            online, paired, state = self.fixture(root)
            result = build("CORE\n", online, paired, state,
                           core_path=root / "core.md", online_path=root / "online.json")
            self.assertIn("Online action-grounded DINO adapter", result)
            self.assertIn("| online_action | 0.0190000 | 0/1000 | 0.00% |", result)

    def test_mismatched_controls_and_tasks_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            online, paired, state = self.fixture(Path(temporary))
            online["submissions"]["42"]["rae"] = {**online["submissions"]["42"]["rae"],
                                                      "submission_id": "other"}
            with self.assertRaisesRegex(ValueError, "control submission"):
                validate(online, paired, state)
            online, paired, state = self.fixture(Path(temporary))
            online["per_task"][0]["task"] = "not_a_task"
            with self.assertRaisesRegex(ValueError, "per-task coverage"):
                validate(online, paired, state)


if __name__ == "__main__":
    unittest.main()
