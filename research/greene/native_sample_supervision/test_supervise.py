import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from supervise import sample, SampleError


class SampleTests(unittest.TestCase):
    def run_code(self, code, **kw):
        return sample([sys.executable, '-c', code], **kw)

    def failure(self, code, **kw):
        began = time.monotonic()
        with self.assertRaises(SampleError) as cm:
            self.run_code(code, **kw)
        self.assertLess(time.monotonic() - began, 2)
        json.dumps(cm.exception.evidence)
        return cm.exception.evidence

    def test_success_raw_evidence_and_reap(self):
        r = self.run_code('import sys; print(\'{"complete":true}\'); print("raw",file=sys.stderr)')
        self.assertEqual(r['value'], {'complete': True})
        self.assertEqual(r['evidence']['stderr'], 'raw\n')
        self.assertTrue(r['evidence']['cleanup_reaped'])
        self.assertEqual(r['evidence']['returncode'], 0)

    def test_hung_worker_retains_partial_and_reaps(self):
        e = self.failure('import time; print("partial",flush=True); time.sleep(30)', seconds=.15)
        self.assertIn('deadline', e['error'])
        self.assertEqual(e['stdout'], 'partial\n')
        self.assertTrue(e['cleanup_reaped'])
        self.assertEqual(e['returncode'], -signal.SIGKILL)

    def test_blocking_fifo_identity_read(self):
        with tempfile.TemporaryDirectory() as d:
            fifo = Path(d)/'proc-status'; os.mkfifo(fifo)
            e = self.failure('from pathlib import Path; print("before identity",flush=True); Path('+repr(str(fifo))+').read_text()',seconds=.15)
        self.assertIn('before identity',e['stdout'])
        self.assertTrue(e['cleanup_reaped'])

    def test_descendant_holds_pipe_and_is_stopped(self):
        code = 'import subprocess,sys; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); print(p.pid,flush=True)'
        e = self.failure(code, seconds=.2)
        pid = int(e['stdout'].strip())
        # A killed orphan can remain a zombie until init reaps it. It must
        # not remain live or continue holding our output pipes.
        for _ in range(20):
            stat = Path('/proc')/str(pid)/'stat'
            if not stat.exists() or stat.read_text().split(') ',1)[1][0] == 'Z':
                break
            time.sleep(.01)
        else:
            self.fail('Own descendant remained live after cleanup')
        self.assertTrue(e['cleanup_reaped'])

    def test_closed_pipes_do_not_bypass_deadline(self):
        e = self.failure('import os,time; os.close(1); os.close(2); time.sleep(30)', seconds=.15)
        self.assertIn('deadline',e['error'])

    def test_nonzero_even_with_json(self):
        e = self.failure('print("{}"); raise SystemExit(7)')
        self.assertEqual(e['returncode'],7)

    def test_signal_exit(self):
        e = self.failure('import os,signal; os.kill(os.getpid(),signal.SIGTERM)')
        self.assertEqual(e['returncode'],-signal.SIGTERM)

    def test_invalid_json_and_empty_and_nonobject(self):
        for code in ('print("bad")', 'pass', 'print("[]")'):
            with self.subTest(code=code): self.failure(code)

    def test_byte_cap_covers_stderr(self):
        e = self.failure('import sys; print("x"*10000,file=sys.stderr)',max_bytes=80)
        self.assertLessEqual(len(e['stdout'])+len(e['stderr']),80)
        self.assertIn('output limit',e['error'])

    def test_flood_cannot_block_parent(self):
        e = self.failure('import os\nwhile True: os.write(1,b"x"*65536)', max_bytes=2000)
        self.assertEqual(len(e['stdout']),2000)
        self.assertTrue(e['cleanup_reaped'])

    def test_invalid_budgets(self):
        for v in (True,0,11,float('nan'),float('inf')):
            with self.subTest(v=v): self.failure('print("{}")',seconds=v)
        for v in (True,0,1024*1024+1):
            with self.subTest(v=v): self.failure('print("{}")',max_bytes=v)

    def test_unrelated_sibling_untouched(self):
        import subprocess
        sibling = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        try:
            self.failure('import time; time.sleep(30)', seconds=.1)
            self.assertIsNone(sibling.poll())
        finally:
            sibling.kill(); sibling.wait(timeout=1)

    def test_invalid_command(self):
        for argv in (None, '', 'echo', [], [None]):
            with self.subTest(argv=argv), self.assertRaises(SampleError): sample(argv)

    def test_actual_observer_peer_context_roundtrip(self):
        observer_dir = Path(__file__).resolve().parents[1]/'native_observer_prep'
        code = ('import sys,json; sys.path.insert(0,'+repr(str(observer_dir))+'); '
                'from test_observer import ObserverTests; '
                'from test_inventory import fixture; '
                'print(json.dumps(ObserverTests().run_observer(fixture([(1,2)]))))')
        r = self.run_code(code)
        self.assertTrue(r['value']['complete'])
        self.assertEqual(len(r['value']['contexts']),1)
        sys.path.insert(0,str(observer_dir.parent/'native_inventory_prep'))
        from test_inventory import MAP
        self.assertEqual(r['value']['contexts'][0]['uuid'],MAP[1])

    def test_actual_observer_blocked_identity(self):
        observer_dir = Path(__file__).resolve().parents[1]/'native_observer_prep'
        with tempfile.TemporaryDirectory() as d:
            fifo = Path(d)/'identity'; os.mkfifo(fifo)
            code = ('import sys; from pathlib import Path; sys.path.insert(0,'+repr(str(observer_dir))+'); '
                    'from test_observer import ObserverTests; from test_inventory import fixture; '
                    'ObserverTests().run_observer(fixture([(0,2)]),reader=lambda pid: Path('+repr(str(fifo))+').read_text())')
            e = self.failure(code,seconds=.2)
        self.assertIn('deadline',e['error'])
        self.assertTrue(e['cleanup_reaped'])

    def test_spawn_failure(self):
        with self.assertRaises(SampleError) as cm:
            sample(['/does/not/exist'])
        self.assertNotIn('pid',cm.exception.evidence)
        json.dumps(cm.exception.evidence)

if __name__ == '__main__': unittest.main()
