import datetime as dt
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from policy import validate_job,window
from worker import run,sha,owned_process


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        auth=self.root/'auth.json';auth.write_text('{"cpu_fixture":true}')
        script=self.root/'job.py';script.write_text('from pathlib import Path\nPath("child-output").write_text("one execution")\n')
        self.job=dict(id='fixture',status='ready',scientific_identity='cpu-fixture-only',max_seconds=120,
                      authorization_evidence=str(auth),authorization_sha256=sha(auth),
                      host_commands={'gpu01':dict(argv=[sys.executable,str(script)],admission_argv=[sys.executable,'-c','pass'],
                                                 cwd=str(self.root),input_sha256={str(script):sha(script)},outputs_absent=[str(self.root/'child-output')])})
        self.spec=self.root/'spec.json';self.spec.write_text(json.dumps(self.job))
        import worker
        source=Path(worker.__file__).resolve()
        (self.root/'source-manifest.json').write_text(json.dumps({'files':{str(source):sha(source)}}))
        self.now=dt.datetime.now(dt.timezone.utc)
        self.row=dict(uuid='CPU-FIXTURE',index=0,idle=True,processes=[])
    def execute(self,row=None):
        end=self.now+dt.timedelta(hours=1)
        with patch('worker.window',return_value=(self.now-dt.timedelta(minutes=1),end+dt.timedelta(seconds=60),True)),\
             patch('worker.probe',return_value=dict(gpus=[row or self.row])),patch('worker.time.sleep',return_value=None):
            run(self.spec,'gpu01','CPU-FIXTURE',end.isoformat(),self.root)
        return json.loads((self.root/'runs/fixture/result.json').read_text())
    def test_real_cpu_child_only_once(self):
        self.assertEqual(self.execute()['status'],'complete')
        self.assertEqual((self.root/'child-output').read_text(),'one execution')
        with self.assertRaises(FileExistsError):self.execute()
    def test_busy_gpu_refuses_before_child(self):
        self.assertEqual(self.execute(dict(self.row,idle=False))['status'],'failed')
        self.assertFalse((self.root/'child-output').exists())
    def test_changed_inputs_refuse_before_child(self):
        (self.root/'job.py').write_text('changed')
        self.assertEqual(self.execute()['status'],'failed')
        self.assertFalse((self.root/'child-output').exists())
    def test_expired_allocation_refuses(self):
        with patch('worker.probe') as probe:
            run(self.spec,'gpu01','CPU-FIXTURE',(self.now-dt.timedelta(seconds=1)).isoformat(),self.root)
            probe.assert_not_called()
        self.assertFalse((self.root/'child-output').exists())
    def test_invalid_job_scope_rejected(self):
        for changes in [dict(id='../bad'),dict(status='proposal'),dict(max_seconds=0),dict(host_commands={})]:
            with self.assertRaises(AssertionError):validate_job(dict(self.job,**changes))
    def test_descendant_with_new_session_still_owned(self):
        import os,subprocess
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)'],start_new_session=True)
        try:
            self.assertTrue(owned_process(child.pid,child.pid))
            self.assertFalse(owned_process(1,child.pid))
        finally:child.terminate();child.wait(timeout=5)

    def test_priority_worker_outside_nightly_window(self):
        policy={'always_available':[['gpu01',1],['gpu01',2]],
                'hosts':{'gpu01':{'gpus':{'1':'CPU-FIXTURE'}}}}
        (self.root/'policy.json').write_text(json.dumps(policy))
        end=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=300)
        with patch('worker.window',return_value=(self.now,self.now,False)),\
             patch('worker.probe',return_value={'gpus':[dict(self.row,index=1)]}),\
             patch('worker.time.sleep',return_value=None):
            run(self.spec,'gpu01','CPU-FIXTURE',end.isoformat(),self.root)
        self.assertEqual(json.loads((self.root/'runs/fixture/result.json').read_text())['status'],'complete')

    def test_other_gpu_worker_refuses_daytime(self):
        policy={'always_available':[['gpu01',1],['gpu01',2]],
                'hosts':{'gpu01':{'gpus':{'3':'CPU-FIXTURE'}}}}
        (self.root/'policy.json').write_text(json.dumps(policy))
        end=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=300)
        with patch('worker.window',return_value=(self.now,self.now,False)),patch('worker.probe') as probe:
            run(self.spec,'gpu01','CPU-FIXTURE',end.isoformat(),self.root)
            probe.assert_not_called()
        self.assertFalse((self.root/'child-output').exists())


if __name__=='__main__':unittest.main()
