import ast
import datetime as dt
from pathlib import Path
import unittest
from unittest.mock import patch
import corrected_protocol as c
import run_batch


class CorrectedProtocolTests(unittest.TestCase):
    def capacity(self,apps='',used=38,uuid='GPU-test'):
        row=f'1, {uuid}, 00000000:07:00.0, 97213, {used}, 637, 0\n'
        with patch.object(c.subprocess,'check_output',side_effect=[apps,row]):
            return c.idle_capacity({'gpus':{'1':'GPU-test'}},[1])[1]

    def test_driver_reserved_memory_does_not_block_idle_gpu(self):
        self.assertTrue(self.capacity()['idle'])

    def test_process_or_allocated_memory_blocks_gpu(self):
        self.assertFalse(self.capacity('GPU-test, 9001\n')['idle'])
        self.assertFalse(self.capacity(used=512)['idle'])

    def test_gpu_identity_change_rejected(self):
        with self.assertRaises(AssertionError):self.capacity(uuid='GPU-other')

    def test_full_endpoint_window_not_only_training(self):
        now=dt.datetime.now(dt.timezone.utc)
        p={'deadline':(now+dt.timedelta(seconds=22700)).isoformat()}
        self.assertGreater(run_batch.allocation_end(p,1,10800+11700),now)
        p['deadline']=(now+dt.timedelta(seconds=22000)).isoformat()
        with self.assertRaises(AssertionError):run_batch.allocation_end(p,1,10800+11700)

    def test_wrong_training_lineage_rejected(self):
        with patch.object(c,'verify_training_release'),patch.object(c,'sha',return_value='expected'):
            good=dict(complete=True,steps=20000,corrected_protocol_sha256='expected',runtime_manifest_sha256='expected')
            c.verify_training_lineage(good)
            with self.assertRaises(AssertionError):c.verify_training_lineage(dict(good,runtime_manifest_sha256='wrong'))
            with self.assertRaises(AssertionError):c.verify_training_lineage(dict(good,steps=19999))

    def test_scheduler_requires_time_for_full_endpoint_on_extras(self):
        from wait_for_capacity import eligible_gpus
        now=dt.datetime.now(dt.timezone.utc)
        p={'deadline':(now+dt.timedelta(hours=8)).isoformat(),'extra_window_start':(now-dt.timedelta(hours=1)).isoformat(),'extra_window_end':(now+dt.timedelta(hours=5)).isoformat()}
        samples={1:{'idle':True},2:{'idle':False},4:{'idle':True},5:{'idle':True}}
        self.assertEqual(eligible_gpus(p,samples,now),[1,4])
        p['extra_window_end']=(now+dt.timedelta(hours=7)).isoformat()
        self.assertEqual(eligible_gpus(p,samples,now),[1,4,5])

    def test_training_computation_is_unchanged(self):
        source=Path(__file__).parent
        def main(path):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
        old=main(source.parent/'code/train_job.py');new=main(source/'train_job.py')
        for node in ast.walk(new):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict':
                node.keywords=[k for k in node.keywords if k.arg not in ['corrected_protocol_sha256','runtime_manifest_sha256']]
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))

    def test_evaluation_computation_only_changes_runtime_path(self):
        source=Path(__file__).parent
        def main(path):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
        old=main(source.parent/'code/evaluate_job.py');new=main(source/'evaluate_job.py')
        for node in ast.walk(old):
            if isinstance(node,ast.Constant) and node.value==str(c.ORIGINAL_ENGINE):node.value=str(c.ENGINE)
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))


if __name__=='__main__':unittest.main()
