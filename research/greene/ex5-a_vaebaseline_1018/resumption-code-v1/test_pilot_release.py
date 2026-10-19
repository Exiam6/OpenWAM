"""CPU fixtures only: challenge budget estimates and actual paired-video output."""
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import make_pilot_pairs as media
from release_evaluation import estimate


class PilotReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.rows=[]
        for task,cap in [('adjust',400),('hand',800),('place',700)]:
            for scene in range(5):
                path=self.root/task/str(scene)/'result.json';path.parent.mkdir(parents=True)
                frames=path.parent/'frames';frames.mkdir()
                first=frames/'0000-head_camera.jpg';last=frames/'0010-head_camera.jpg'
                first.touch();last.touch();os.utime(first,(1000,1000));os.utime(last,(1002,1002))
                self.rows.append(dict(task=task,steps=10,step_limit=cap,seconds=12.,result_path=str(path)))

    def test_native_caps_not_early_success_steps(self):
        result=estimate(self.rows,200.)
        # 45 full-length episodes, 10s setup each, 0.2s/action with 10% reserve.
        self.assertAlmostEqual(result['projected_remaining_seconds'],7020.)
        self.assertEqual(result['overhead_seconds'],300.)

    def test_invalid_timing_is_rejected(self):
        first=Path(self.rows[0]['result_path']).parent/'frames/0000-head_camera.jpg'
        os.utime(first,(1003,1003))
        with self.assertRaises(AssertionError):estimate(self.rows,200.)

    def test_nonfinite_occupation_is_rejected(self):
        with self.assertRaises(AssertionError):estimate(self.rows,math.nan)

    def test_video_encodes_decodes_and_preserves_corruption(self):
        from PIL import Image
        paths=[]
        for route,count in [('raw768_native',6),('vae48_native',8)]:
            path=self.root/route/'result.json';path.parent.mkdir()
            path.write_text(json.dumps(dict(training_seed=42,task='fixture',scene_seed=1,steps=count-1,success=False)))
            frames=path.parent/'frames';frames.mkdir()
            for index in range(count):
                for camera in media.CAMERAS:Image.new('RGB',(32,32),(index*20,30,70)).save(frames/f'{index:04d}-{camera}.jpg')
            paths.append(path)
        with patch.object(media,'ROOT',self.root/'output'):
            result=media.make(*paths)
            self.assertTrue(result['full_decode_passed']);self.assertEqual(result['frames'],8)
            self.assertEqual(result['raw_frames'],6);self.assertEqual(result['vae_frames'],8)
            self.assertEqual(len(result['preview_indices']),6)
            self.assertEqual(media.make(*paths),result)
            Path(result['preview']).write_bytes(b'corrupted')
            with self.assertRaises(AssertionError):media.make(*paths)


if __name__=='__main__':unittest.main()
