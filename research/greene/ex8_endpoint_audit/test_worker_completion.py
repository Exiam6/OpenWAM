import copy
import unittest
from worker_completion import terminal_state, select_completed


class TerminalEvidence(unittest.TestCase):
    def setUp(self):
        self.job = dict(id='job', scientific_identity='attempt/row/seed0', max_seconds=100,
                        host_commands={'gpu01': dict(argv=['run', '--key', 'row'])})
        self.worker = dict(id='job', scientific_identity='attempt/row/seed0', host='gpu01',
                           gpu_uuid='GPU-A', gpu_index=1, spec_sha256='spec', command=['run','--key','row'],
                           status='complete', exit_code=0, started='2026-10-21T19:00:00+00:00',
                           finished='2026-10-21T19:01:00+00:00', occupied_parent_seconds=60)
        self.claim = dict(id='job', scientific_identity='attempt/row/seed0', host='gpu01',
                          gpu_uuid='GPU-A', gpu_index=1, job_sha256='jobhash', status='complete',
                          policy_sha256='policy', worker_result=copy.deepcopy(self.worker),
                          idle_evidence=dict(idle=True, qualified=True, idle_seconds=1800,
                                             uuid='GPU-A', index=1, policy_hash='policy'))
        self.row = dict(host='gpu01', gpu_uuid='GPU-A', started='2026-10-21T19:00:01+00:00',
                        finished='2026-10-21T19:00:59+00:00')

    def check(self):
        return terminal_state(self.job,self.claim,self.worker,self.row,'jobhash','spec')

    def test_successful_final_worker(self):
        self.assertEqual(self.check(),'complete')

    def test_missing_claim_or_worker_is_pending(self):
        self.assertEqual(terminal_state(self.job,None,None,self.row,'jobhash','spec'),'pending')

    def test_row_complete_during_worker_cleanup_is_pending(self):
        self.worker['status']='running';self.claim['status']='running'
        self.assertEqual(self.check(),'pending')

    def test_failed_worker_cannot_be_counted(self):
        self.worker['status']='failed'
        self.assertEqual(self.check(),'failed')

    def test_claim_refresh_lag_is_pending(self):
        self.claim['status']='running'
        self.assertEqual(self.check(),'pending')

    def test_scientific_identity_mismatch_rejected(self):
        self.worker['scientific_identity']='another/attempt'
        with self.assertRaises(AssertionError):self.check()

    def test_nonzero_exit_rejected(self):
        self.worker['exit_code']=1
        with self.assertRaises(AssertionError):self.check()

    def test_spec_or_job_hash_mismatch_rejected(self):
        for target,field in [(self.worker,'spec_sha256'),(self.claim,'job_sha256')]:
            old=target[field];target[field]='changed'
            with self.assertRaises(AssertionError):self.check()
            target[field]=old

    def test_wrong_gpu_rejected(self):
        self.row['gpu_uuid']='GPU-B'
        with self.assertRaises(AssertionError):self.check()

    def test_finish_before_row_rejected(self):
        self.worker['finished']='2026-10-21T19:00:50+00:00'
        self.claim['worker_result']=copy.deepcopy(self.worker)
        with self.assertRaises(AssertionError):self.check()

    def test_insufficient_idle_rejected(self):
        self.claim['idle_evidence']['idle_seconds']=1799
        with self.assertRaises(AssertionError):self.check()

    def test_cap_overrun_rejected(self):
        self.worker['occupied_parent_seconds']=101
        self.claim['worker_result']=copy.deepcopy(self.worker)
        with self.assertRaises(AssertionError):self.check()



class HeadlineGate(unittest.TestCase):
    def test_full_artifacts_with_pending_worker_cannot_yield_headline(self):
        rows=[dict(key=str(i)) for i in range(100)]
        result=dict(completed_rows=100,pending=[],per_task=rows,rates_pct={'mean':100},
                    gaps_pp={'mean':6.4},practical_agreement_passed=True,
                    independent_native_outcome_checks=[dict(key='99',included_in_endpoint=True)])
        checks=[dict(key=str(i),status='complete' if i<99 else 'pending') for i in range(100)]
        select_completed(result,rows[:99],checks)
        self.assertEqual(result['artifact_completed_rows'],100)
        self.assertEqual(result['completed_rows'],99)
        self.assertEqual(result['scored_episodes'],9900)
        self.assertFalse(result['full_endpoint_complete'])
        self.assertFalse(result['practical_agreement_passed'])
        self.assertNotIn('rates_pct',result)
        self.assertNotIn('gaps_pp',result)
        self.assertEqual(result['pending'],['99'])
        self.assertFalse(result['independent_native_outcome_checks'][0]['included_in_endpoint'])

    def test_failed_worker_separate_from_capacity_wait(self):
        result=dict(completed_rows=1,pending=[],per_task=[dict(key='row')])
        select_completed(result,[],[dict(key='row',status='failed')])
        self.assertEqual(result['dispatcher_failed_rows'],['row'])
        self.assertEqual(result['pending'],[])
        self.assertEqual(result['scored_episodes'],0)


if __name__=='__main__':unittest.main(verbosity=2)
