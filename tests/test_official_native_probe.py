"""Native diagnostics must be isolated, auditable, and explicitly executed."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from benchmarks.utils.eval_manifest import seal_manifest
from research.rae import run_official_native_probe as probe


class OfficialNativeProbeTests(unittest.TestCase):
    def fixture(self, root, episodes=2):
        registry = {"official": {"directory": str(root / "official"),
                                  "checkpoint_sha256": probe.OFFICIAL_SHA256}}
        registry_file = root / "checkpoints.json"
        registry_file.write_text(json.dumps(registry))
        argv = ["--checkpoints", str(registry_file), "--manifest-dir", str(root / "production/manifests"),
                "--run-dir", str(root / "probe"), "--robotwin-root", str(root / "RoboTwin"),
                "--robotwin-python", str(root / "envs/simulator/bin/python"),
                "--policy-python", str(root / "envs/policy/bin/python"), "--gpu", "0",
                "--tasks", "stack_blocks_three", "--episodes", str(episodes)]
        args = probe.parse(argv)
        return argv, args, registry, probe.build_plan(args, registry)

    def batches(self, root, plan):
        entries = [{"episode": i, "seed": 100000 + i,
                    "instructions": {"seen": "seen instruction", "unseen": f"instruction-{i}"}}
                   for i in range(plan["episodes_per_task"])]
        task = plan["tasks"][0]
        manifest = seal_manifest({"benchmark": "robotwin", "task": task,
                                  "mode": "demo_clean", "entries": entries})
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest))
        rows = [{"episode": e["episode"], "seed": e["seed"], "success": i == 0,
                 "instruction": e["instructions"]["unseen"]} for i, e in enumerate(entries)]
        n = len(rows)
        args = {"operation": "run_eval", "mode": "demo_clean", "task": task,
                "instruction_type": "unseen", "total_episodes": n,
                "episode_start": 0, "num_episodes": n, "manifest_path": str(path),
                "manifest_hash": manifest["manifest_hash"], "model": {"checkpoint": plan["checkpoint"]}}
        result = {k: args[k] for k in ("operation", "mode", "task", "instruction_type",
                                      "total_episodes", "episode_start", "num_episodes", "manifest_hash")}
        result.update(episodes=rows, successes=1)
        metadata = {"submission_task_count": 1, "submission_unit_count": n,
                    "submission_task_index": 0, "range_kind": "episode",
                    "range_start": 0, "range_count": n, "range_total": n}
        return [SimpleNamespace(id="probe-task", args=args, result=result,
                                metadata=metadata, status="succeeded")]

    def test_default_mode_plans_without_submitting_starting_workers_or_creating_run_dir(self):
        with tempfile.TemporaryDirectory() as temporary:
            argv, args, _, _ = self.fixture(Path(temporary))
            with (patch.object(probe, "execute") as execute,
                  patch.object(probe.subprocess, "run") as run,
                  patch.object(probe.subprocess, "Popen") as popen,
                  redirect_stdout(io.StringIO()) as output):
                self.assertEqual(probe.main(argv), 0)
                self.assertIn("diagnostic", json.loads(output.getvalue())["scope"])
                execute.assert_not_called()
                run.assert_not_called()
                popen.assert_not_called()
            self.assertFalse(args.run_dir.exists())

    def test_submit_and_worker_use_same_private_route_with_stable_submission_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, args, registry, plan = self.fixture(Path(temporary))
            for key in ("submit_command", "worker_command"):
                command = plan[key]
                self.assertEqual(command[command.index("--route") + 1], plan["route"])
                self.assertNotIn("openwam-robotwin-run_eval", command)
            self.assertEqual(probe.build_plan(args, registry)["submission_id"], plan["submission_id"])
            args.episodes += 1
            self.assertNotEqual(probe.build_plan(args, registry)["submission_id"], plan["submission_id"])

    def test_wrong_release_and_production_output_directory_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, args, registry, _ = self.fixture(Path(temporary))
            changed = copy.deepcopy(registry)
            changed["official"]["checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "digest"):
                probe.build_plan(args, changed)
            args.run_dir = args.manifest_dir.parent
            with self.assertRaisesRegex(ValueError, "separate"):
                probe.build_plan(args, registry)

    def test_complete_probe_counts_robot_success_and_never_claims_whole_benchmark(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, plan = self.fixture(root)
            summary = probe.summarize_batches(self.batches(root, plan), plan)
            self.assertEqual((summary["successes"], summary["episode_count"]), (1, 2))
            self.assertFalse(summary["whole_benchmark_complete"])
            self.assertFalse(summary["all_zero_success"])

    def test_missing_batch_or_nonterminal_task_is_not_a_completed_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, plan = self.fixture(root)
            with self.assertRaises(ValueError):
                probe.summarize_batches([], plan)
            batches = self.batches(root, plan)
            batches[0].status = "running"
            with self.assertRaisesRegex(ValueError, "incomplete"):
                probe.summarize_batches(batches, plan)

    def test_wrong_model_instruction_or_scene_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, plan = self.fixture(root)
            original = self.batches(root, plan)
            for fault in ("checkpoint", "instruction", "seed"):
                with self.subTest(fault=fault):
                    batches = copy.deepcopy(original)
                    if fault == "checkpoint":
                        batches[0].args["model"]["checkpoint"] = str(root / "wrong-model")
                    else:
                        batches[0].result["episodes"][0][fault] = "changed" if fault == "instruction" else 123456
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        probe.summarize_batches(batches, plan)

    def test_all_robot_failures_remain_successfully_executed_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, plan = self.fixture(root)
            batches = self.batches(root, plan)
            for row in batches[0].result["episodes"]:
                row["success"] = False
            batches[0].result["successes"] = 0
            summary = probe.summarize_batches(batches, plan)
            self.assertTrue(summary["all_zero_success"])
            self.assertEqual(summary["success_rate"], 0)

    def test_timeout_cleans_up_owned_worker_after_verifying_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, args, registry, plan = self.fixture(Path(temporary))
            args.timeout_seconds = 1
            process = MagicMock()
            process.poll.return_value = None
            with (patch.object(probe, "load_checkpoints") as load,
                  patch.object(probe.subprocess, "check_output", return_value="NVIDIA RTX A6000\n"),
                  patch.object(probe.subprocess, "run"),
                  patch.object(probe.subprocess, "Popen", return_value=process),
                  patch.object(probe.rt, "create_client", return_value=MagicMock()),
                  patch.object(probe.rt, "list_robotwin_submission_tasks", return_value=[]),
                  patch.object(probe.time, "monotonic", side_effect=[0, 2]),
                  patch.object(probe, "stop_workers") as stop,
                  redirect_stdout(io.StringIO())):
                with self.assertRaises(TimeoutError):
                    probe.execute(args, plan, registry)
                load.assert_called_once_with(args.run_dir / "official-registry.json", verify_bytes=True)
                stop.assert_called_once_with([process])


if __name__ == "__main__":
    unittest.main()
