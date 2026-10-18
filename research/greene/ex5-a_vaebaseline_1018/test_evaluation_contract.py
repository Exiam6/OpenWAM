"""Challenge the new validator with independently audited existing rollout evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from evaluation_contract import validate

ROOT=Path('/home/zifanz4/openwam-runtime/ex4-g_calibratedfactorial_1017')


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)/'result.json'
        source=ROOT/'runs/full-svae48_native-seed42/evaluation/fixed/adjust_bottle/seed-6200008'
        self.row=json.loads((source/'result.json').read_text());assert self.row['status']=='complete'
        # Rename only this temporary fixture. This is not a VAE evaluation result.
        self.row['route']='vae48_native'
        trace=(source/'trace.jsonl').read_text();(self.path.parent/'trace.jsonl').write_text(trace)
        steps=[json.loads(x) for x in trace.splitlines()]
        historical=json.loads((ROOT/'historical-outcomes.json').read_text())
        self.historical={(x['training_seed'],x['task'],x['scene_seed']):x for x in historical if x['route']=='raw768_native'}
        self.reference={'adjust_bottle|6200008':dict(training_seed=42,task='adjust_bottle',scene_seed=6200008,
            action_seed=self.row['action_seed'],initial_state=self.row['initial_state'],instruction=self.row['evaluation_instruction'],actions=[x['action'] for x in steps[:8]])}

    def check(self):
        self.path.write_text(json.dumps(self.row));return validate(self.path,self.historical,self.reference)

    def test_real_fixture_passes(self):
        self.assertTrue(self.check()['verified'])

    def test_one_wrong_camera_is_rejected(self):
        self.row['initial_render_mae']['left_camera']=2.
        with self.assertRaises(AssertionError):self.check()

    def test_wrong_initial_state_is_rejected(self):
        self.row['initial_state']='different'
        with self.assertRaises(AssertionError):self.check()

    def test_reference_action_corruption_is_rejected(self):
        self.reference['adjust_bottle|6200008']['actions'][0][0]+=.01
        with self.assertRaises(AssertionError):self.check()

    def test_wrong_inference_seed_is_rejected(self):
        self.reference['adjust_bottle|6200008']['action_seed']+=1
        with self.assertRaises(AssertionError):self.check()


if __name__=='__main__':unittest.main()
