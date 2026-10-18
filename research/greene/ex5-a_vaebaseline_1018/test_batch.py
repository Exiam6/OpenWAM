"""Test real concurrent CPU children with mocked resource admission, never GPUs."""
import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_batch


class BatchTests(unittest.TestCase):
    def check_batch(self, fail_last):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'code';source.mkdir()
            script=f"""import sys,json,time
from pathlib import Path
seed=int(sys.argv[sys.argv.index('--seed')+1]);root=Path({str(root)!r})
d=root/'runs'/f'vae48_native-seed{{seed}}';d.mkdir(parents=True,exist_ok=False)
(d/'attempt').write_text('one attempt')
time.sleep(.05)
if seed==44 and {fail_last!r}:raise SystemExit(7)
(d/'result.json').write_text(json.dumps(dict(complete=True)))
"""
            (source/'train_job.py').write_text(script)
            p=dict(pause_paths=[],gpus={str(x):f'CPU-TEST-{x}' for x in [5,6,7]})
            samples={x:dict(uuid=f'CPU-TEST-{x}',pci='CPU-TEST',idle=True) for x in [5,6,7]}
            with patch.object(run_batch,'ROOT',root),patch.object(run_batch,'SOURCE',source),patch.object(run_batch,'ADMISSION_POLL_SECONDS',.001),\
                 patch.object(run_batch,'admission',return_value=(p,{42:2.,43:2.,44:2.})),patch.object(run_batch,'capacity',return_value=samples),\
                 patch.object(run_batch,'allocation_end',return_value=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=30)):
                # Actual production helper reserves10s cleanup: use12s test limits without waiting for them.
                with patch.object(run_batch,'admission',return_value=(p,{42:12.,43:12.,44:12.})):
                    run_batch.main('training',[42,43,44],[5,6,7])
            result=json.loads((root/'batches/training-42-43-44/result.json').read_text())
            self.assertEqual(len(result['children']),3)
            self.assertEqual(sum(x['complete'] for x in result['children']),2 if fail_last else 3)
            self.assertEqual(result['complete'],not fail_last)
            for seed in [42,43,44]:self.assertEqual((root/'runs'/f'vae48_native-seed{seed}'/'attempt').read_text(),'one attempt')
            self.assertGreater(result['occupied_gpu_seconds'],0)
            # All this elapsed time came from CPU test children; no CUDA library was imported.
            self.assertTrue(all('CPU-TEST' in x['uuid'] for x in result['capacity_samples'].values()))

    def test_three_children_complete(self):self.check_batch(False)
    def test_one_failure_preserves_other_completed_children(self):self.check_batch(True)


if __name__=='__main__':unittest.main()
