import copy
import json
import unittest
from verify_chain import verify_chain,sha,N,audit

class ChainTests(unittest.TestCase):
    def setUp(self):
        self.name='ex8d-full-001-beat-block-hammer'
        paths=[N/'jobs'/(self.name+'.json'),N/'claims'/(self.name+'.json'),N/'runs'/self.name/'result.json',N/'incoming'/(self.name+'.json')]
        self.job,self.claim,self.worker,self.spec=[json.loads(p.read_text()) for p in paths]
        from pathlib import Path
        self.kw=dict(job_sha=sha(paths[0]),spec_sha=sha(paths[3]),protocol_sha=sha(Path(self.job['authorization_evidence'])),expected_id=self.name)
    def run_check(self):return verify_chain(self.job,self.claim,self.worker,self.spec,**self.kw)
    def reject(self):
        with self.assertRaises(Exception):self.run_check()
    def test_real_completed_worker(self):self.assertTrue(self.run_check()['dispatcher_chain_validated'])
    def test_job_digest(self):self.claim['job_sha256']='wrong';self.reject()
    def test_spec_digest(self):self.worker['spec_sha256']='wrong';self.claim['worker_result']=copy.deepcopy(self.worker);self.reject()
    def test_spec_command(self):self.spec['host_commands']['gpu01']['argv']=['different'];self.reject()
    def test_executed_command(self):self.worker['command']=['different'];self.claim['worker_result']=copy.deepcopy(self.worker);self.reject()
    def test_final_claim_disagreement(self):self.claim['worker_result']['exit_code']=5;self.reject()
    def test_wrong_authorization(self):self.kw['protocol_sha']='wrong';self.reject()
    def test_wrong_allocation(self):self.claim['gpu_index']=2;self.reject()
    def test_short_idle(self):self.claim['idle_evidence']['idle_seconds']=1799;self.reject()
    def test_contradictory_idle(self):self.claim['idle_evidence']['since']-=100;self.reject()
    def test_occupied_idle(self):self.claim['idle_evidence']['processes']=[{'pid':123}];self.reject()
    def test_nonfinite_idle(self):self.claim['idle_evidence']['idle_seconds']=float('nan');self.reject()
    def test_policy_disagreement(self):self.claim['idle_evidence']['policy_hash']='wrong';self.reject()
    def test_over_budget(self):self.worker['occupied_parent_seconds']=self.job['max_seconds']+1;self.claim['worker_result']=copy.deepcopy(self.worker);self.reject()
    def test_nonfinite_duration(self):self.worker['occupied_parent_seconds']=float('inf');self.claim['worker_result']=copy.deepcopy(self.worker);self.reject()
    def test_bool_exit(self):self.worker['exit_code']=False;self.claim['worker_result']=copy.deepcopy(self.worker);self.reject()
    def test_timestamp_order(self):self.worker['finished']=self.claim['created'];self.claim['worker_result']=copy.deepcopy(self.worker);self.reject()
    def test_pending_cannot_release(self):
        r=audit();self.assertFalse(r['native_L2_L3_passed']);self.assertFalse(r['full_release_allowed']);self.assertFalse(r['full_paper_reproduced'])

if __name__=='__main__':unittest.main()
