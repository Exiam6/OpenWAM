import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from verify_raw import verify_samples, OPERATIONS, DEPENDENCIES, audit
sys.path.insert(0,str(DEPENDENCIES/'native_evidence_adapter'))
import test_adapter as fixtures
from integration import run, observer, outcome

class Backend:
    def prewarm(self):return True
    sapien_import=explicit_device=helper_load=instance_helper_load=native_Instance_Get=native_scene=prewarm
    instance_helper_load=native_Instance_Get=native_scene=prewarm
    def helper_call(self,symbol):return 0 if symbol in ['headless_init','headless_support'] else 1

class RawAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.cfg=dict(host='host',device_map={str(k):v for k,v in fixtures.MAP.items()},assigned=fixtures.MAP[0],own_group=fixtures.OWN,gdm_uid=120,seconds=3)
        obs=observer(cls.cfg,fixtures.worker(),cls.root/'journal',seconds=2)
        cls.science=run(Backend(),obs,lambda:None,fixtures.MAP[0],seconds=15,sample_timeout=2,interval=.001)
        assert cls.science['status']=='supervisor_sequence_passed',cls.science
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_real_subprocess_journal_reconstructs(self):
        proof=verify_samples(json.loads(json.dumps(self.science)),self.cfg,self.root/'journal')
        self.assertGreaterEqual(proof['distinct_raw_packets'],39)
        self.assertFalse(proof['continuous_isolation_proven'])
    def reject(self,science=None,config=None,journal=None):
        with self.assertRaises(Exception):verify_samples(json.loads(json.dumps(science or self.science)),config or self.cfg,journal or self.root/'journal')
    def baseline_accepts(self,s):
        terminal=dict(id='fixture',scientific_identity='fixture/attempt1',gpu_uuid=fixtures.MAP[0],status='complete',exit_code=0)
        self.assertTrue(outcome(s,terminal,terminal)['GPU_diagnostic_validated'])
    def test_duplicate_packet_rejected(self):
        s=copy.deepcopy(self.science);s['samples'][1]=copy.deepcopy(s['samples'][0]);self.baseline_accepts(s);self.reject(s)
    def test_forged_summary_rejected(self):
        s=copy.deepcopy(self.science);s['samples'][0]['contexts']=[];self.baseline_accepts(s);self.reject(s)
    def test_wrong_assignment_rejected(self):
        c=copy.deepcopy(self.cfg);c['assigned']=fixtures.MAP[1];self.reject(config=c)
    def test_wrong_host_rejected(self):
        c=copy.deepcopy(self.cfg);c['host']='wrong';self.reject(config=c)
    def test_foreign_journal_rejected(self):
        self.reject(journal=self.root/'other')
    def test_phase_replay_rejected(self):
        s=copy.deepcopy(self.science);s['samples'][-1].update(phase='before_prewarm',finished_phase='before_prewarm',stable_phase=True);self.reject(s)
    def test_unreferenced_failure_record_rejected(self):
        f=self.root/'journal'/'0127.json';f.write_text('{"kind":"worker_failure"}')
        try:self.reject()
        finally:f.unlink()
    def test_raw_tampering_even_with_matching_digest_rejected(self):
        s=copy.deepcopy(self.science);receipt=s['samples'][0]['evidence_receipt'];f=Path(receipt['path']);original=f.read_bytes()
        packet=json.loads(original);packet['packet']['value']['evidence']['queries'][0]['stdout']='invalid XML'
        raw=json.dumps(packet).encode();f.write_bytes(raw);receipt.update(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
        try:self.reject(s)
        finally:f.write_bytes(original)
    def test_persisted_wrong_map_rejected(self):
        s=json.loads(json.dumps(self.science));s['samples'][0]['device_map']['0']='GPU-wrong'
        self.reject(s)
    def test_accepted_prefix_cannot_satisfy_new_instance_gate(self):
        old=Path('/home/zifanz4/openwam-runtime/ex8-m_hostidentity_1022')
        s=json.loads((old/'result.json').read_text());cfg=json.loads((old/'observer-config.json').read_text())
        with self.assertRaises(Exception):verify_samples(s,cfg,old/'journal')
    def test_unknown_phase_rejected(self):
        s=copy.deepcopy(self.science);s['samples'][0].update(phase='unknown',finished_phase='unknown');self.reject(s)
    def test_pending_never_releases_stage(self):
        out=audit();self.assertFalse(out['full_release_allowed']);self.assertFalse(out['native_L2_L3_passed'])

if __name__=='__main__':unittest.main()
