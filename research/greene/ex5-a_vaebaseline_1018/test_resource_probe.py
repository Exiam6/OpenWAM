"""CPU-only resource measurement and original-parent delegation checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import resource_probe as probe
import resource_preflight
import resource_batch
import resource_schedule


class ResourceTests(unittest.TestCase):
    def sample(self,used=86,reserved=637,util=0,occupied='',uuid='GPU-5'):
        with patch.object(probe.subprocess,'check_output',side_effect=[occupied,f'5, {uuid}, PCI-5, 97165, {used}, {reserved}, {util}\n']):
            return probe.capacity(dict(gpus={'5':'GPU-5'}),[5])[5]

    def test_driver_reserved_memory_does_not_count_as_allocated(self):
        row=self.sample();self.assertTrue(row['idle']);self.assertEqual(row['used_mib'],86);self.assertEqual(row['driver_reserved_mib'],637)

    def test_actual_allocated_memory_boundary(self):
        self.assertTrue(self.sample(used=255)['idle']);self.assertFalse(self.sample(used=256)['idle'])

    def test_compute_process_blocks_even_with_zero_used_and_utilization(self):
        self.assertFalse(self.sample(used=0,occupied='GPU-5, 123\n')['idle'])

    def test_nonzero_utilization_blocks(self):
        self.assertFalse(self.sample(util=1)['idle'])

    def test_physical_identity_mismatch_rejected(self):
        with self.assertRaises(AssertionError):self.sample(uuid='GPU-other')

    def test_preflight_delegates_to_original_parent(self):
        original=resource_preflight.frozen
        with patch.object(resource_preflight,'verify',return_value=dict(files={str(Path(resource_preflight.__file__).resolve()):'fixture'})),patch.object(original,'main') as run,patch.object(original,'idle_gpus'):
            resource_preflight.main(5);run.assert_called_once_with(5);self.assertIs(original.idle_gpus,probe.idle_gpus)
        self.assertEqual(Path(original.__file__).parent,probe.ROOT/'code')

    def test_batch_delegates_to_original_parent(self):
        original=resource_batch.frozen
        with patch.object(resource_batch,'verify',return_value=dict(files={str(Path(resource_batch.__file__).resolve()):'fixture'})),patch.object(original,'main') as run,patch.object(original,'capacity'):
            resource_batch.main('training',[42,43,44],[0,5,6]);run.assert_called_once_with('training',[42,43,44],[0,5,6]);self.assertIs(original.capacity,probe.capacity)
        self.assertEqual(Path(original.__file__).parent,probe.ROOT/'code')

    def test_scheduler_cannot_launch_before_window(self):
        class EndObservation(Exception):pass
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);registration=root/'gpu-registration.json'
            registration.write_text(json.dumps(dict(gpu_launch_permitted=True,preflight_launch_permitted=True,extra_window_start='2099-01-01T00:00:00+00:00',deadline='2099-01-02T00:00:00+00:00',pause_paths=[])))
            (root/'resource-registration-v2.json').write_text('{}')
            with patch.object(resource_schedule,'verify',return_value=dict(files={str(Path(resource_schedule.__file__).resolve()):'fixture'})),patch.object(resource_schedule.time,'sleep',side_effect=EndObservation),patch.object(resource_schedule.subprocess,'Popen') as spawn:
                with self.assertRaises(EndObservation):resource_schedule.main(registration,5)
                spawn.assert_not_called()
            self.assertIsNone(json.loads((root/'preflight-schedule-v2.json').read_text())['child_pid'])


if __name__=='__main__':unittest.main()
