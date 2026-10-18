"""Exercise real CPU children; no model import or GPU work."""
import datetime as dt
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

from stage_runner import run_child


class RunnerTests(unittest.TestCase):
    def run_case(self, code, root, **kwargs):
        return run_child([sys.executable, '-c', code], log=Path(root)/'log', seconds=2.,
            cleanup_seconds=.4, deadline=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=10), poll_seconds=.02, **kwargs)

    def test_success_charged_and_failure_not_retried(self):
        with tempfile.TemporaryDirectory() as d:
            result=self.run_case("print('done')",d)
            self.assertTrue(result['complete']);self.assertGreater(result['occupied_seconds'],0)
        with tempfile.TemporaryDirectory() as d:
            marker=Path(d)/'marker'
            result=self.run_case(f"from pathlib import Path; Path({str(marker)!r}).write_text('one attempt'); raise SystemExit(7)",d)
            self.assertFalse(result['complete']);self.assertEqual(result['exit_code'],7)
            self.assertEqual(marker.read_text(),'one attempt')

    def test_timeout_stops_owned_grandchild(self):
        with tempfile.TemporaryDirectory() as d:
            marker=Path(d)/'pid'
            code=f"import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);Path({str(marker)!r}).write_text(str(p.pid));time.sleep(60)"
            result=self.run_case(code,d)
            self.assertFalse(result['complete']);self.assertIn('budget',result['error'])
            self.assertLess(result['occupied_seconds'],2.5)
            pid=int(marker.read_text());status=Path('/proc',str(pid),'stat')
            def running():
                try:return status.read_text().split()[2]!='Z'
                except (FileNotFoundError, ProcessLookupError):return False
            for _ in range(20):
                if not running():break
                time.sleep(.02)
            self.assertFalse(running())

    def test_pause_refuses_before_spawn(self):
        with tempfile.TemporaryDirectory() as d:
            pause=Path(d)/'pause';pause.touch()
            result=self.run_case("raise RuntimeError('must not launch')",d,pause_paths=[pause])
            self.assertFalse(result['complete']);self.assertNotIn('pid',result)
            self.assertFalse((Path(d)/'log').exists())


    def test_expired_deadline_refuses_before_spawn(self):
        with tempfile.TemporaryDirectory() as d:
            result=run_child([sys.executable,'-c',"raise RuntimeError('must not launch')"],log=Path(d)/'log',seconds=2.,cleanup_seconds=.4,
                deadline=dt.datetime.now(dt.timezone.utc)-dt.timedelta(seconds=1))
            self.assertFalse(result['complete']);self.assertIn('deadline',result['error']);self.assertNotIn('pid',result)


if __name__=='__main__':unittest.main()
