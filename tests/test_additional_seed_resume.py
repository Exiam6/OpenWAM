"""Partial-seed recovery must preserve existing training and evaluation work."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.rae import run_additional_seeds as runner


def evaluation(checkpoint, checkpoint_hash, split, value=0.1):
    tasks = [f"task-{i}" for i in range(10)]
    return {
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "split": split, "windows_per_task": 10, "noise_repeats": 4,
        "seed": 20261023, "split_manifest_sha256": "manifest",
        "action_stats_sha256": "stats", "action_mean_across_tasks": value,
        "per_task": {task: {"action_mean": value} for task in tasks},
        "windows": [{"task": task, "window_index": i,
                     "losses": [{"action": value} for _ in range(4)]}
                    for task in tasks for i in range(10)],
    }


class AdditionalSeedResumeTests(unittest.TestCase):
    def test_seed44_reuses_rae_and_does_not_require_deleted_graft(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "weights" / "seed44"
            existing = weights / "rae" / "finished"
            existing.mkdir(parents=True)
            (existing / "checkpoint_step_1000.safetensors").touch()

            def verified(_, variant, log, expected_seed):
                self.assertEqual(expected_seed, 44)
                return weights / variant / "finished", variant + "-hash"

            with (patch.object(runner, "RUNS", root / "reports"),
                  patch.object(runner, "WEIGHTS", root / "weights"),
                  patch.object(runner, "verify_final", side_effect=verified),
                  patch.object(runner, "verify_graft") as graft,
                  patch.object(runner, "require_space"),
                  patch.object(runner, "record"),
                  patch.object(runner, "run_logged") as launch,
                  patch.object(runner, "evaluate_pair") as evaluate):
                runner.run_seed(44)
                graft.assert_not_called()
                launch.assert_called_once()
                self.assertEqual(launch.call_args.args[0],
                                 ["bash", "research/rae/launch_wam_pilot.sh", "vae"])
                self.assertEqual(evaluate.call_args.args[3],
                                 {"rae": "rae-hash", "vae": "vae-hash"})

    def test_complete_seed_launches_no_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for variant in ("rae", "vae"):
                p = root / "weights/seed43" / variant / "finished"
                p.mkdir(parents=True)
                (p / "checkpoint_step_1000.safetensors").touch()
            with (patch.object(runner, "RUNS", root / "reports"),
                  patch.object(runner, "WEIGHTS", root / "weights"),
                  patch.object(runner, "verify_final", return_value=(root, "hash")),
                  patch.object(runner, "verify_graft") as graft,
                  patch.object(runner, "require_space") as space,
                  patch.object(runner, "record"),
                  patch.object(runner, "run_logged") as launch,
                  patch.object(runner, "evaluate_pair")):
                runner.run_seed(43)
                launch.assert_not_called()
                graft.assert_not_called()
                space.assert_not_called()

    def test_unfinished_training_log_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "reports/seed44/rae_train.log"
            log.parent.mkdir(parents=True)
            log.write_text("interrupted attempt\n")
            with (patch.object(runner, "RUNS", root / "reports"),
                  patch.object(runner, "WEIGHTS", root / "weights"),
                  patch.object(runner, "run_logged") as launch):
                with self.assertRaisesRegex(RuntimeError, "Incomplete training attempt"):
                    runner.run_seed(44)
                launch.assert_not_called()
            self.assertEqual(log.read_text(), "interrupted attempt\n")

    def test_reuse_checks_checkpoint_training_seed_before_loading_tensors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ckpt = root / "rae/finished"
            ckpt.mkdir(parents=True)
            weight = ckpt / "checkpoint_step_1000.safetensors"
            weight.touch()
            (ckpt / "config.yaml").write_text("project:\n  seed: 43\n")
            log = root / "train.log"
            log.write_text(f"[checkpoint] Saved final: {weight}\n")
            with self.assertRaisesRegex(ValueError, "training seed mismatch"):
                runner.verify_final(root, "rae", log, expected_seed=44)

    def test_partial_evaluation_reuses_existing_rae_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = root / "seed44"
            reports.mkdir()
            checkpoints = {v: root / v for v in ("rae", "vae")}
            hashes = {v: v + "-hash" for v in checkpoints}
            for split in ("dev", "test"):
                (reports / f"rae_{split}_teacher_forced.json").write_text(
                    json.dumps(evaluation(checkpoints["rae"], hashes["rae"], split))
                )

            def run(command, log, *, env):
                self.assertIn("--checkpoint", command)
                self.assertEqual(command[command.index("--checkpoint") + 1], str(checkpoints["vae"]))
                split = command[command.index("--split") + 1]
                output = Path(command[command.index("--output") + 1])
                output.write_text(json.dumps(evaluation(checkpoints["vae"], hashes["vae"], split)))

            with (patch.object(runner, "RUNS", root),
                  patch.object(runner, "record"),
                  patch.object(runner, "run_logged", side_effect=run) as launch):
                runner.evaluate_pair(44, checkpoints, {}, hashes)
                self.assertEqual(launch.call_count, 2)
                # A second resume audits the same artifacts and launches nothing.
                launch.reset_mock()
                runner.evaluate_pair(44, checkpoints, {}, hashes)
                launch.assert_not_called()

    def test_reused_report_rejects_changed_hash_duplicate_window_or_repeat_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "report.json"
            reference = evaluation(root / "checkpoint", "registered", "test")
            for fault in ("checkpoint", "window", "repeats", "aggregate"):
                with self.subTest(fault=fault):
                    report = copy.deepcopy(reference)
                    if fault == "checkpoint":
                        report["checkpoint_sha256"] = "changed"
                    elif fault == "window":
                        report["windows"][1] = copy.deepcopy(report["windows"][0])
                    elif fault == "repeats":
                        report["windows"][0]["losses"].pop()
                    else:
                        report["action_mean_across_tasks"] = 0.5
                    path.write_text(json.dumps(report))
                    with self.assertRaises(ValueError):
                        runner.validate_existing_evaluation(path, root / "checkpoint", "registered", "test")


if __name__ == "__main__":
    unittest.main()
