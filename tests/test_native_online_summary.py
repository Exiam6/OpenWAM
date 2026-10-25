"""The optional online ablation reuses exactly the paired seed-42 scenes."""

import unittest

from research.rae.summarize_native_online import VARIANTS, summarize_rows


class NativeOnlineSummaryTests(unittest.TestCase):
    @staticmethod
    def rows():
        rows = []
        for task in ("task_a", "task_b"):
            for episode in range(3):
                for variant in VARIANTS:
                    rows.append({
                        "training_seed": 42, "variant": variant, "task": task,
                        "episode": episode, "scene_seed": 100000 + episode,
                        "manifest_hash": f"sha256:{task}",
                        "success": variant == "online_action" or (variant == "rae" and episode == 0),
                    })
        return rows

    def test_full_scene_paired_report(self):
        report = summarize_rows(self.rows(), tasks=("task_a", "task_b"),
                                episodes_per_task=3, draws=100)
        self.assertEqual(report["registered_rows"], 18)
        self.assertEqual(report["successes"], {"vae": 0, "rae": 2, "online_action": 6})
        self.assertEqual(report["comparisons"][0]["difference"], 1.0)
        self.assertEqual(report["comparisons"][1]["difference"], 2 / 3)

    def test_scene_mismatch_is_rejected(self):
        rows = self.rows()
        rows[0]["scene_seed"] += 1
        with self.assertRaisesRegex(ValueError, "Scene or manifest mismatch"):
            summarize_rows(rows, tasks=("task_a", "task_b"), episodes_per_task=3, draws=100)


if __name__ == "__main__":
    unittest.main()
