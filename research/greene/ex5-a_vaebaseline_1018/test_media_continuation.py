from pathlib import Path
import tempfile
import unittest

from resume_full_media import pending_specs


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'pair.json'
        self.spec = dict(key='pair', output=str(self.path))

    def test_only_missing_pairs_are_scheduled(self):
        self.assertEqual(pending_specs([self.spec], set()), [self.spec])
        self.path.write_text('existing verified pair')
        self.assertEqual(pending_specs([self.spec], {'pair'}), [])

    def test_partial_or_unknown_existing_output_never_retried(self):
        for ext in ['.json', '.mp4', '.jpg']:
            path = self.path.with_suffix(ext)
            path.write_text('partial')
            with self.assertRaises(AssertionError): pending_specs([self.spec], set())
            self.assertEqual(path.read_text(), 'partial')
            path.unlink()

    def test_duplicate_candidates_rejected(self):
        with self.assertRaises(AssertionError): pending_specs([self.spec, self.spec], set())


if __name__ == '__main__':
    unittest.main()
