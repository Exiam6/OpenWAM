import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from integration import run, observer, outcome, timing, HELPER, OPERATIONS, DEPENDENCIES
sys.path.insert(0,str(DEPENDENCIES/'native_evidence_adapter'))
import test_adapter as fixtures

class Backend:
    def __init__(self, fault=None):self.calls=[];self.fault=fault
    def record(self,n):
        self.calls.append(n)
        if n==self.fault:raise RuntimeError('injected native failure')
        return True
    def prewarm(self):return self.record('prewarm')
    def sapien_import(self):return self.record('sapien_import')
    def explicit_device(self):return self.record('explicit_device')
    def helper_load(self):return self.record('helper_load')
    def helper_call(self,n):
        self.record(n)
        return 0 if n in ['headless_init','headless_support'] else 1

class IntegrationTests(unittest.TestCase):
    def compose(self,root,index=0,backend=None,guard=lambda:None):
        c=dict(host='host',device_map=fixtures.MAP,assigned=fixtures.MAP[0],own_group=fixtures.OWN,gdm_uid=120)
        o=observer(c,fixtures.worker(index),Path(root)/'journal',seconds=2)
        b=backend or Backend()
        r=run(b,o,guard,fixtures.MAP[0],seconds=15,sample_timeout=2,interval=.001)
        worker=dict(id='test',scientific_identity='unique/attempt1',gpu_uuid=fixtures.MAP[0],status='complete',exit_code=0)
        claim=dict(worker)
        return r,b,o,worker,claim
    def test_real_subprocess_contract_journal_outcome(self):
        with tempfile.TemporaryDirectory() as d:
            r,b,o,w,c=self.compose(d)
            self.assertEqual(r['status'],'supervisor_sequence_passed',r)
            self.assertEqual(b.calls,list(OPERATIONS[:4])+[h[1] for h in HELPER])
            self.assertGreaterEqual(o.journal.records,39)
            v=outcome(r,w,c);self.assertTrue(v['GPU_diagnostic_validated'],v)
            self.assertFalse(v['full_release_allowed']);self.assertFalse(v['native_L2_L3_passed'])
            # Real packet evidence cannot be erased after a positive result.
            first=Path(r['samples'][0]['evidence_receipt']['path']);first.write_text('{}')
            self.assertFalse(outcome(r,w,c)['GPU_diagnostic_validated'])
    def test_real_peer_contract_prevents_every_native_operation(self):
        with tempfile.TemporaryDirectory() as d:
            r,b,o,w,c=self.compose(d,index=1)
            self.assertEqual(r['status'],'failed');self.assertEqual(b.calls,[])
            record=json.loads((o.journal.directory/'0000.json').read_text())
            self.assertEqual(record['kind'],'contract_failure')
            self.assertIn('nvidia_smi_log',record['packet']['value']['evidence']['queries'][0]['stdout'])
            self.assertFalse(outcome(r,w,c)['GPU_diagnostic_validated'])
    def test_native_failure_stops_at_each_boundary(self):
        names=list(OPERATIONS[:4])+[h[1] for h in HELPER]
        for name in names:
            with self.subTest(name=name):
                b=Backend(name)
                r=run(b,lambda:dict(complete=True,contexts=[dict(uuid='assigned')]),lambda:None,'assigned',seconds=2,sample_timeout=.2,interval=.001)
                self.assertEqual(r['status'],'failed');self.assertEqual(b.calls,names[:names.index(name)+1])
    def test_pause_stops_before_native(self):
        def guard():raise RuntimeError('paused')
        with tempfile.TemporaryDirectory() as d:
            r,b,o,w,c=self.compose(d,guard=guard)
            self.assertEqual(r['status'],'failed');self.assertEqual(b.calls,[])
    def test_wrong_native_return_codes(self):
        for bad in [None,-1,2]:
            b=Backend();b.helper_call=lambda _:bad
            r=run(b,lambda:dict(complete=True,contexts=[dict(uuid='assigned')]),lambda:None,'assigned',seconds=2,sample_timeout=.2,interval=.001)
            self.assertEqual(r['status'],'failed')
    def test_outcome_pending_and_wrong_identity(self):
        self.assertFalse(outcome(None,None,None)['GPU_diagnostic_validated'])
        a=dict(id='x',scientific_identity='a',gpu_uuid='GPU-a',status='complete',exit_code=0)
        b=dict(a,scientific_identity='b')
        self.assertEqual(outcome({'status':'supervisor_sequence_passed'},a,b)['assessment'],'identity mismatch')
    def test_dispatcher_72_seconds_underfits_observed_cadence(self):
        p=timing(72,sample_seconds=1.1,native_seconds=9,setup_seconds=5)
        self.assertEqual(p['stable_samples_required'],39)
        self.assertEqual(p['child_latest_elapsed_seconds'],27)
        self.assertFalse(p['projected_to_fit'])
        self.assertTrue(timing(150,sample_seconds=1.5,native_seconds=10,setup_seconds=10)['projected_to_fit'])

if __name__=='__main__':unittest.main()
