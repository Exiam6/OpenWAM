import datetime as dt
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import controller


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        for name in ['jobs','claims']:(self.root/name).mkdir()
        (self.root/'policy.json').write_text('{}')
        self.job=dict(id='fixture',scientific_identity='study/phase/seed42',max_seconds=600)
        (self.root/'jobs/fixture.json').write_text(json.dumps(self.job))
        self.row=dict(index=1,uuid='CPU-ONLY',idle=True)
    def dispatch(self,idle=True):
        now=dt.datetime.now(dt.timezone.utc)
        with patch.object(controller,'ROOT',self.root),patch.object(controller,'eligible',return_value=True),\
             patch.object(controller,'window',return_value=(now,now+dt.timedelta(hours=2),True)),\
             patch.object(controller,'sample',return_value={'gpus':[dict(self.row,idle=idle)]}),\
             patch.object(controller,'command',return_value=subprocess.CompletedProcess([],0,'','')) as command:
            result=controller.dispatch({},self.job,'gpu01',1,self.row)
            return result,command.call_count
    def test_one_atomic_claim_prevents_duplicate_submission(self):
        result,calls=self.dispatch();self.assertEqual(result['status'],'submitted');self.assertEqual(calls,2)
        with self.assertRaises(FileExistsError):self.dispatch()
    def test_last_moment_occupancy_prevents_submission(self):
        result,calls=self.dispatch(False);self.assertEqual(result['status'],'submission_uncertain');self.assertEqual(calls,0)
    def test_remote_worker_completion_is_authoritative(self):
        path=self.root/'claims/fixture.json';path.write_text(json.dumps(dict(self.job,status='submitted',host='gpu01',gpu_uuid='CPU-ONLY')))
        worker=dict(id='fixture',gpu_uuid='CPU-ONLY',status='complete')
        with patch.object(controller,'ROOT',self.root),patch.object(controller,'command',return_value=subprocess.CompletedProcess([],0,json.dumps(worker),'')):
            self.assertEqual(controller.refresh_claim({},path)['status'],'complete')
    def test_dead_confirmed_unit_is_not_reported_running(self):
        path=self.root/'claims/fixture.json'
        path.write_text(json.dumps(dict(self.job,status='submitted',host='gpu01',gpu_uuid='CPU-ONLY',unit='test-unit',created=(dt.datetime.now(dt.timezone.utc)-dt.timedelta(minutes=1)).isoformat())))
        replies=[subprocess.CompletedProcess([],0,'null',''),subprocess.CompletedProcess([],0,'LoadState=not-found\nActiveState=inactive\n','')]
        with patch.object(controller,'ROOT',self.root),patch.object(controller,'command',side_effect=replies):
            self.assertEqual(controller.refresh_claim({},path)['status'],'failed')


if __name__=='__main__':unittest.main()
