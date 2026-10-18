"""CPU fixtures: actual encoding and strict identity/hash checks, no GPU work."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from paired_video import encode,checked_pair,CAMERAS
from build_cache import sha


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.spec=dict(key='fixture',kind='model',comparison='primary_raw_minus_vae',seed=42,task='fixture',scene_seed=1,labels=['Raw','VAE'],routes=['raw768_native','vae48_native'],output=str(self.root/'output.json'))
        for side,route,count in [('left','raw768_native',6),('right','vae48_native',8)]:
            path=self.root/side/'result.json';path.parent.mkdir()
            row=dict(route=route,status='complete',training_seed=42,task='fixture',scene_seed=1,steps=count-1,success=False,initial_state='same',action_seed=11,step_limit=20,evaluation_instruction='fixture')
            path.write_text(json.dumps(row));(path.parent/'trace.jsonl').write_text('fixture trace\n')
            frames=path.parent/'frames';frames.mkdir()
            for index in range(count):
                for camera in CAMERAS:Image.new('RGB',(32,32),(index*20,40,50)).save(frames/f'{index:04d}-{camera}.jpg')
            self.spec[side]=dict(result_path=str(path),result_sha256=sha(path),trace_sha256=sha(path.parent/'trace.jsonl'),result=row)

    def test_real_encode_and_no_overwrite(self):
        result=encode(self.spec);self.assertTrue(result['full_decode_passed']);self.assertEqual(result['side_frames'],[6,8]);self.assertEqual(result['frames'],8)
        self.assertEqual(result['preview_indices'],[0,1,3,4,6,7])
        with self.assertRaises(AssertionError):encode(self.spec)

    def test_wrong_trace_rejected(self):
        path=Path(self.spec['left']['result_path']).parent/'trace.jsonl';path.write_text('changed')
        with self.assertRaises(AssertionError):checked_pair(self.spec)

    def test_wrong_pairing_rejected_even_if_hash_matches(self):
        item=self.spec['right'];item['result']['scene_seed']=2;path=Path(item['result_path']);path.write_text(json.dumps(item['result']));item['result_sha256']=sha(path)
        with self.assertRaises(AssertionError):checked_pair(self.spec)

    def test_missing_camera_rejected(self):
        path=Path(self.spec['left']['result_path']).parent/'frames/0003-left_camera.jpg';path.unlink()
        with self.assertRaises(AssertionError):encode(self.spec)


    def pilot_metadata(self,record):
        old=dict(seed=42,task='fixture',scene_seed=1,full_decode_passed=True,raw_result=self.spec['left']['result_path'],vae_result=self.spec['right']['result_path'],
            raw_result_sha256=self.spec['left']['result_sha256'],vae_result_sha256=self.spec['right']['result_sha256'],video=record['video'],video_sha256=record['video_sha256'],
            preview=record['preview'],preview_sha256=record['preview_sha256'],frames=record['frames'],raw_frames=record['side_frames'][0],vae_frames=record['side_frames'][1],preview_indices=record['preview_indices'])
        path=self.root/'pilot-metadata.json';path.write_text(json.dumps(old));return path

    def test_verified_pilot_video_is_reused_without_duplicate_encoding(self):
        record=encode(self.spec);path=self.pilot_metadata(record)
        self.spec.update(reuse_metadata=str(path),output=str(self.root/'reused.json'))
        reused=encode(self.spec);self.assertEqual(reused['video'],record['video']);self.assertIsNotNone(reused['reused'])
        self.assertFalse((self.root/'reused.mp4').exists())

    def test_changed_pilot_preview_is_rejected(self):
        record=encode(self.spec);path=self.pilot_metadata(record);Path(record['preview']).write_bytes(b'corrupt')
        self.spec.update(reuse_metadata=str(path),output=str(self.root/'reused.json'))
        with self.assertRaises(AssertionError):encode(self.spec)

    def test_queue_refuses_before_any_real_vae_audit(self):
        from unittest.mock import patch
        import full_media
        (self.root/'gpu-registration.json').write_text('{}')
        with patch.object(full_media,'ROOT',self.root):
            with self.assertRaises(AssertionError):full_media.main()
        self.assertFalse((self.root/'full-media').exists())

    def test_training_figure_from_cpu_metric_fixtures(self):
        from unittest.mock import patch
        import report_complete
        for seed in [42,43,44]:
            path=self.root/'runs'/f'vae48_native-seed{seed}'/'metrics.jsonl';path.parent.mkdir(parents=True)
            values=[dict(step=step,loss=loss,loss_action=loss/2,loss_video=loss/2) for step,loss in [(1,1.),(20000,.1)]]
            path.write_text(''.join(json.dumps(x)+'\n' for x in values))
        record=dict(comparison='primary_raw_minus_vae',seed=42,task='adjust_bottle',scene_seed=6200001,preview=str(self.root/'left/frames/0000-head_camera.jpg'))
        destination=self.root/'figure';destination.mkdir()
        with patch.object(report_complete,'ROOT',self.root):path=report_complete.training_figure(destination,dict(fixture=record))
        self.assertGreater(path.stat().st_size,10000)

if __name__=='__main__':unittest.main()
