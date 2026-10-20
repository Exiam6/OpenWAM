import unittest
import numpy as np
from audit import sensitivity, validate_pair


class AuditTests(unittest.TestCase):
    def test_single_task_effect_is_not_generalized(self):
        left = np.zeros((3, 3, 20))
        left[:, 0] = 1
        result = sensitivity(left, np.zeros_like(left), 921018)
        self.assertAlmostEqual(result["difference"], 1/3)
        self.assertEqual(result["leave_one_task_out"]["adjust_bottle"], 0)
        self.assertEqual(result["exploratory_task_resampled_ci95"][0], 0)
        self.assertEqual(result["seed_cluster_signflip_two_sided_p"], .25)
        self.assertEqual(result["seed_cluster_signflip_positive_one_sided_p"], .125)

    def test_identical_models(self):
        a = np.ones((3, 3, 20))
        result = sensitivity(a, a, 921018)
        self.assertEqual(result["registered_fixed_task_bootstrap_ci95"], [0, 0])
        self.assertEqual(result["gains"], 0)
        self.assertEqual(result["seed_cluster_signflip_two_sided_p"], 1)

    def test_mismatched_state_is_rejected(self):
        row = dict(scene_seed=1, training_seed=42, condition="clean", initial_state="a",
                   evaluation_instruction="test", action_seed=100, step_limit=400,
                   status="complete", success=False,
                   initial_render_mae={"head": 0, "left": 0, "right": 0})
        validate_pair([row, dict(row)])
        with self.assertRaisesRegex(AssertionError, "initial_state"):
            validate_pair([row, dict(row, initial_state="different")])

    def test_missing_camera_is_rejected(self):
        row = dict(scene_seed=1, training_seed=42, condition="clean", initial_state="a",
                   evaluation_instruction="test", action_seed=100, step_limit=400,
                   status="complete", success=False, initial_render_mae={"head": 0})
        with self.assertRaises(AssertionError):
            validate_pair([row])


if __name__ == "__main__":
    unittest.main()
