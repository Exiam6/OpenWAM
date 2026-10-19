"""Real CPU encoding fixture and adversarial evidence reuse checks."""
import copy
import json
from pathlib import Path
import unittest

from build_cache import sha
from paired_video import encode
from prepare_media_resume import verify_reuse
import test_full_evidence as fixtures


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EvidenceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.spec = self.fixture.spec
        self.row = encode(self.spec)
        self.entry = dict(key=self.spec['key'], metadata=self.spec['output'],
                          video=self.row['video'], preview=self.row['preview'])
        for field in ['metadata', 'video', 'preview']:
            self.entry[field + '_sha256'] = sha(self.entry[field])

    def verify(self):
        return verify_reuse([self.entry], [self.spec])

    def rewrite_metadata(self):
        Path(self.entry['metadata']).write_text(json.dumps(self.row))
        self.entry['metadata_sha256'] = sha(self.entry['metadata'])

    def test_verified_pair_reused_without_modification(self):
        before = copy.deepcopy(self.entry)
        self.assertEqual(self.verify(), {self.entry['key']: before})
        for field in ['metadata', 'video', 'preview']:
            self.assertEqual(sha(self.entry[field]), before[field + '_sha256'])

    def test_corrupt_video_rejected(self):
        Path(self.entry['video']).write_bytes(b'corrupt')
        with self.assertRaises(ValueError): self.verify()

    def test_changed_trace_rejected(self):
        (Path(self.spec['left']['result_path']).parent / 'trace.jsonl').write_text('changed')
        with self.assertRaises(AssertionError): self.verify()

    def test_valid_hash_cannot_hide_wrong_identity(self):
        self.row['scene_seed'] += 1
        self.rewrite_metadata()
        with self.assertRaises(ValueError): self.verify()

    def test_valid_hash_cannot_hide_wrong_frames(self):
        self.row['frames'] += 1
        self.rewrite_metadata()
        with self.assertRaises(ValueError): self.verify()

    def test_valid_hash_cannot_hide_missing_decode(self):
        self.row['full_decode_passed'] = False
        self.rewrite_metadata()
        with self.assertRaises(ValueError): self.verify()

    def test_missing_or_duplicate_records_rejected(self):
        with self.assertRaises(ValueError): verify_reuse([], [self.spec])
        with self.assertRaises(ValueError): verify_reuse([self.entry, self.entry], [self.spec])
        with self.assertRaises(ValueError): verify_reuse([self.entry], [self.spec, self.spec])

    def test_cli_does_not_overwrite_existing_inventory(self):
        from unittest.mock import patch
        from prepare_media_resume import main
        destination = self.fixture.root / 'inventory.json'
        destination.write_text('preserve this inventory')
        args = ['prepare_media_resume', '--root', str(self.fixture.root), '--audit', 'unused',
                '--audit-sha256', 'unused', '--output', str(destination)]
        with patch('sys.argv', args), patch('prepare_media_resume.prepare') as prepare:
            with self.assertRaises(ValueError): main()
            prepare.assert_not_called()
        self.assertEqual(destination.read_text(), 'preserve this inventory')


if __name__ == '__main__':
    unittest.main()
