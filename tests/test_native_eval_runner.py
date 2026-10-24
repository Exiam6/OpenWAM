"""Protocol invariants for the resumable native evaluation runner."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from research.rae.run_native_eval import MODELS, load_checkpoints, plan, reconcile_state, write_registries


class NativeEvalRunnerTests(unittest.TestCase):
    def test_complete_matrix_has_two_manifests_and_ten_model_submissions(self):
        stages = plan(dict.fromkeys(MODELS, {}))
        self.assertEqual(len(stages), 12)
        self.assertEqual([stage[0] for stage in stages[:2]], ["manifest_clean", "manifest_randomized"])
        self.assertEqual(sum(stage[1] == "run_eval" for stage in stages), 10)
        self.assertEqual(stages[-1], ("official_randomized", "run_eval", "demo_randomized"))

    def test_available_checkpoints_can_start_before_training_finishes(self):
        stages = plan({"seed42_vae": {}, "seed42_rae": {}, "official": {}})
        self.assertEqual([stage[0] for stage in stages],
                         ["manifest_clean", "manifest_randomized", "seed42_vae", "seed42_rae",
                          "official_clean", "official_randomized"])

    def test_resume_adds_stages_without_changing_submitted_ids(self):
        first = {"seed42_vae": {"checkpoint_sha256": "a" * 64},
                 "seed42_rae": {"checkpoint_sha256": "b" * 64}}
        state = reconcile_state(None, first)
        submitted = dict(state["submission_ids"])
        state["completed"] = ["manifest_clean", "manifest_randomized", "seed42_vae"]
        later = {**first, "seed43_vae": {"checkpoint_sha256": "c" * 64}}
        resumed = reconcile_state(state, later)
        self.assertEqual({name: resumed["submission_ids"][name] for name in submitted}, submitted)
        self.assertIn("seed43_vae", resumed["submission_ids"])
        self.assertEqual(resumed["completed"], ["manifest_clean", "manifest_randomized", "seed42_vae"])

    def test_registries_reuse_seed42_vae_and_official_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = {name: {"directory": f"/checkpoints/{name}",
                             "checkpoint": f"/checkpoints/{name}/weights.safetensors",
                             "checkpoint_sha256": f"{index:064x}"}
                      for index, name in enumerate(MODELS)}
            submission_ids = {name: f"submission-{name}" for name, _, _ in plan()}
            write_registries(SimpleNamespace(run_dir=root, submission_ids=submission_ids), models)
            paired = json.loads((root / "paired-submissions.json").read_text())
            ablation = json.loads((root / "ablation-submissions.json").read_text())
            official = json.loads((root / "official-submissions.json").read_text())
            self.assertEqual(ablation["42"]["vae"], paired["42"]["vae"])
            self.assertEqual(official["demo_clean"]["checkpoint_sha256"],
                             official["demo_randomized"]["checkpoint_sha256"])
            self.assertNotEqual(official["demo_clean"]["submission_id"],
                                official["demo_randomized"]["submission_id"])

    def test_checkpoint_hash_is_checked_before_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "seed42_vae"
            checkpoint.mkdir()
            (checkpoint / "tokenizer").mkdir()
            (checkpoint / "config.yaml").write_text("model: fixture\n")
            (checkpoint / "normalization_stats.npy").write_bytes(b"fixture")
            (checkpoint / "checkpoint_step_1000.safetensors").write_bytes(b"incorrect")
            registry = root / "checkpoints.json"
            registry.write_text(json.dumps({"seed42_vae": {"directory": str(checkpoint),
                                                           "checkpoint_sha256": "0" * 64}}))
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_checkpoints(registry, verify_bytes=True)

    def test_wrong_checkpoint_step_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = root / "checkpoints.json"
            registry.write_text(json.dumps({"seed42_vae": {"directory": str(root),
                                                           "weights_file": "checkpoint_step_800.safetensors",
                                                           "checkpoint_sha256": "0" * 64}}))
            with self.assertRaisesRegex(ValueError, "wrong registered checkpoint step"):
                load_checkpoints(registry, verify_bytes=False)


if __name__ == "__main__":
    unittest.main()
