import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from adapter import DurableObserver, Journal, EvidenceError, ContractError, SampleError
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'native_observer_prep'))
import test_observer as fixtures
MAP,OWN=fixtures.MAP,fixtures.OWN


def worker(index=0):
    code=('import sys,json;sys.path.insert(0,'+repr(str(ROOT/'native_observer_prep'))+');'
          'import test_observer as f;print(json.dumps(f.ObserverTests().run_observer(root=f.fixture([('+str(index)+',2)]))))')
    return [sys.executable,'-c',code]


def observer(directory,argv=None,**kw):
    return DurableObserver(worker() if argv is None else argv,Journal(directory),
                 host='host',device_map=MAP,assigned=MAP[0],own_group=OWN,
                 gdm_uid=120,seconds=2,**kw)


class AdapterTests(unittest.TestCase):
    def test_real_worker_success_is_durable_before_return(self):
        with tempfile.TemporaryDirectory() as d:
            o=observer(Path(d)/'journal')
            with patch('adapter.os.fsync',wraps=__import__('os').fsync) as sync:
                value=o()
            self.assertEqual(sync.call_count,2)
            receipt=value['evidence_receipt'];raw=Path(receipt['path']).read_bytes()
            self.assertEqual(receipt['sha256'],hashlib.sha256(raw).hexdigest())
            record=json.loads(raw);self.assertEqual(record['kind'],'validated_sample')
            self.assertIn('nvidia_smi_log',record['packet']['value']['evidence']['queries'][0]['stdout'])
            self.assertTrue(record['packet']['evidence']['cleanup_reaped'])

    def test_actual_peer_failure_survives_supervisor_repr(self):
        with tempfile.TemporaryDirectory() as d:
            o=observer(Path(d)/'journal',worker(1));calls=[]
            operations={n:lambda n=n:calls.append(n) or True for n in fixtures.OPERATIONS}
            result=fixtures.Supervisor().run(operations,o,lambda:None,MAP[0],seconds=3,sample_timeout=1,interval=.001)
            self.assertEqual(result['status'],'failed');self.assertEqual(calls,[])
            record=json.loads((o.journal.directory/'0000.json').read_text())
            self.assertEqual(record['kind'],'contract_failure')
            self.assertEqual(record['packet']['value']['contexts'][0]['uuid'],MAP[1])
            self.assertIn('nvidia_smi_log',record['packet']['value']['evidence']['queries'][0]['stdout'])
            with self.assertRaises(EvidenceError):o()
            self.assertEqual(o.journal.records,1)

    def test_actual_worker_failure_keeps_stderr(self):
        with tempfile.TemporaryDirectory() as d:
            o=observer(Path(d)/'journal',[sys.executable,'-c',"import sys;print('native failure',file=sys.stderr);sys.exit(7)"])
            with self.assertRaises(SampleError):o()
            record=json.loads((o.journal.directory/'0000.json').read_text())
            self.assertEqual(record['kind'],'worker_failure')
            self.assertEqual(record['packet']['returncode'],7)
            self.assertIn('native failure',record['packet']['stderr'])

    def test_actual_timeout_and_cleanup_retained(self):
        with tempfile.TemporaryDirectory() as d:
            o=observer(Path(d)/'journal',[sys.executable,'-c',"import time;print('partial',flush=True);time.sleep(10)"])
            o.seconds=.15
            with self.assertRaises(SampleError):o()
            record=json.loads((o.journal.directory/'0000.json').read_text())
            self.assertIn('partial',record['packet']['stdout'])
            self.assertTrue(record['packet']['cleanup_reaped'])

    def test_actual_output_limit_retained(self):
        with tempfile.TemporaryDirectory() as d:
            o=observer(Path(d)/'journal',[sys.executable,'-c',"print('x'*1024)"],max_bytes=100)
            with self.assertRaises(SampleError):o()
            record=json.loads((o.journal.directory/'0000.json').read_text())
            self.assertLessEqual(len(record['packet']['stdout'].encode()),100)
            self.assertIn('output limit',record['packet']['error'])

    def test_replayed_packet_rejected_and_retained(self):
        with tempfile.TemporaryDirectory() as d:
            first=observer(Path(d)/'first');first()
            packet=json.loads((first.journal.directory/'0000.json').read_text())['packet']
            second=observer(Path(d)/'second')
            with patch('adapter.sample',return_value=packet):
                with self.assertRaises(ContractError):second()
            record=json.loads((second.journal.directory/'0000.json').read_text())
            self.assertEqual(record['kind'],'contract_failure')
            self.assertIn('pre-phase',record['error'])

    def test_existing_file_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            j=Journal(Path(d)/'journal');p=j.directory/'0000.json';p.write_text('preserve')
            with self.assertRaises(EvidenceError):j.append({'record':1})
            self.assertEqual(p.read_text(),'preserve')
            with self.assertRaises(EvidenceError):j.append({'record':2})
            with self.assertRaises(FileExistsError):Journal(j.directory)

    def test_limits_poison_journal(self):
        for kw,first in [({'max_records':1},True),({'max_bytes':20},False),({'record_bytes':20},False)]:
            with self.subTest(kw=kw),tempfile.TemporaryDirectory() as d:
                j=Journal(Path(d)/'journal',**kw)
                if first:j.append({'x':1})
                with self.assertRaises(EvidenceError):j.append({'large':'z'*100})
                with self.assertRaises(EvidenceError):j.append({'x':2})
                self.assertEqual(j.records,int(first))

    def test_serialize_failure_stops_without_inventing_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            j=Journal(Path(d)/'journal')
            with self.assertRaises(EvidenceError):j.append({'nan':float('nan')})
            self.assertTrue(j.failed);self.assertEqual(j.records,0)

    def test_fsync_failure_retains_file_and_blocks_callbacks(self):
        with tempfile.TemporaryDirectory() as d:
            o=observer(Path(d)/'journal');calls=[]
            operations={n:lambda n=n:calls.append(n) or True for n in fixtures.OPERATIONS}
            with patch('adapter.os.fsync',side_effect=OSError('disk failure')):
                result=fixtures.Supervisor().run(operations,o,lambda:None,MAP[0],seconds=3,sample_timeout=1,interval=.001)
            self.assertEqual(result['status'],'failed');self.assertEqual(calls,[])
            self.assertTrue((o.journal.directory/'0000.json').exists())
            with self.assertRaises(EvidenceError):o()

    def test_positive_composition_cannot_release_full_stage(self):
        with tempfile.TemporaryDirectory() as d:
            o=observer(Path(d)/'journal');calls=[]
            def operation(name):
                self.assertGreater(o.journal.records,0)
                self.assertTrue((o.journal.directory/('%04d.json'%(o.journal.records-1))).exists())
                calls.append(name);return True
            operations={n:lambda n=n:operation(n) for n in fixtures.OPERATIONS}
            result=fixtures.Supervisor().run(operations,o,lambda:None,MAP[0],seconds=15,sample_timeout=2,interval=.001)
            self.assertEqual(result['status'],'supervisor_sequence_passed',result)
            self.assertEqual(calls,list(fixtures.OPERATIONS))
            for flag in ('full_release_allowed','native_L2_L3_passed','full_paper_reproduced'):
                self.assertFalse(result[flag])
            self.assertEqual(len(result['samples']),o.journal.records)

    def test_invalid_journal_bounds(self):
        for kw in ({'max_records':True},{'max_bytes':0},{'record_bytes':4*1024*1024+1}):
            with tempfile.TemporaryDirectory() as d,self.assertRaises(ValueError):Journal(Path(d)/'journal',**kw)

if __name__=='__main__':unittest.main()
