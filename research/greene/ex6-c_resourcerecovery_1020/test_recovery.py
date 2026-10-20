import unittest
from register import missing_scenes


class RecoveryTests(unittest.TestCase):
    def test_completed_successes_and_failures_are_both_retained(self):
        self.assertEqual(missing_scenes([1,2,3,4,5], [1,2,4]), [3,5])

    def test_no_repeat_if_all_complete(self):
        self.assertEqual(missing_scenes([1,2,3,4,5], [1,2,3,4,5]), [])

    def test_unknown_scene_rejected(self):
        with self.assertRaises(AssertionError):
            missing_scenes([1,2,3,4,5], [6])


if __name__ == '__main__':
    unittest.main()
