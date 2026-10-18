"""Integration-test handoff signals using disposable CPU-only coordinators."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

SOURCE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('handoff',SOURCE/'handoff_supervisor.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
ORIGINAL=module.ROOT,module.HERE,module.P

OLD='''import argparse,json,os,signal,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int);p.add_argument('--variant');a=p.parse_args()
r=Path(__file__).resolve().parents[1];path=r/f'lane-gpu{a.gpu}.json'
record={'key':f'{a.variant}-seed42','complete':True,'server_pid':999999999}
d={'pid':os.getpid(),'phase':'waiting_capacity','seed':43,'admitted_samples':1,'records':[record],'active_seconds':100}
def save():
 t=path.with_suffix('.tmp');t.write_text(json.dumps(d));t.replace(path)
def stop(*args):
 d['phase']='stopped';d['records'].append({'error':"RuntimeError('Lane terminated')"});save();raise SystemExit(0)
signal.signal(signal.SIGTERM,stop);save()
while True:time.sleep(1)
'''
NEW='''import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int);p.add_argument('--variant');p.add_argument('--seed',type=int);p.add_argument('--allocated-seconds',type=int);a=p.parse_args()
r=Path(__file__).resolve().parents[1];path=r/f'lane-gpu{a.gpu}-{a.variant}-seed{a.seed}.json'
t=path.with_suffix('.tmp');t.write_text(json.dumps({'pid':os.getpid(),'phase':'complete','budget':a.allocated_seconds}));t.replace(path)
'''

class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(prefix='openwam-handoff-test-');self.root=Path(self.directory.name)
        module.ROOT=self.root;module.HERE=self.root/'resource-handoff';module.P={'deadline':'2099-01-01T00:00:00+00:00','pause_paths':[]}
        module.HERE.mkdir();(self.root/'code').mkdir()
        (self.root/'code/run_lane.py').write_text(OLD);(module.HERE/'run_cell.py').write_text(NEW)
        (module.HERE/'freeze.json').write_text(json.dumps({'files':{}}))
        self.children=[]
        for gpu,variant in module.ARMS.items():
            gate=self.root/'evaluation'/f'{variant}-seed42';gate.mkdir(parents=True);(gate/'L2-gate.json').write_text('{"passed":true}')
            self.children.append(subprocess.Popen(['/usr/bin/python3',str(self.root/'code/run_lane.py'),'--gpu',str(gpu),'--variant',variant]))
        begin=time.monotonic()
        while not all((self.root/f'lane-gpu{gpu}.json').exists() for gpu in module.ARMS):
            assert time.monotonic()-begin<5;time.sleep(.05)

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:child.terminate()
            child.wait(timeout=5)
        module.ROOT,module.HERE,module.P=ORIGINAL
        self.directory.cleanup()

    def test_only_between_cell_handoff_and_budget_conserved(self):
        module.main();state=json.loads((module.HERE/'status.json').read_text())
        self.assertEqual(state['phase'],'complete');self.assertEqual(len(state['stopped']),2);self.assertEqual(len(state['jobs']),4)
        occupation=sum(x['final_record']['active_seconds'] for x in state['stopped'].values())
        allocated=sum(x['allocated_seconds'] for x in state['jobs'])
        self.assertEqual(occupation+allocated,21600)
        self.assertTrue(all(x['phase']=='complete' for x in state['jobs']))

    def test_attempted_next_cell_is_never_interrupted(self):
        (self.root/'evaluation/raw768_native-seed43').mkdir()
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(AssertionError):module.main()
        self.assertTrue(all(child.poll() is None for child in self.children))
        state=json.loads((module.HERE/'status.json').read_text());self.assertEqual(state['stopped'],{})

if __name__=='__main__':unittest.main()
