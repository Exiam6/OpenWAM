import copy
import datetime as dt
import unittest

from resource_amendment import validate_authorization, check_remaining, limit


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.a=dict(status='authorized_resource_resumption',seeds=[42,43],gpus=[1,2],
            evaluation_limits_seconds={'42':10920,'43':10920,'44':10800},study_limit_seconds=73800,
            fixed_remaining_episodes_per_seed=45,checkpoint_steps=20000,no_retraining=True,no_automatic_retry=True,
            start='2026-10-19T06:15:13+00:00',deadline='2026-10-19T12:15:13+00:00',
            allocation_end='2026-10-19T12:14:13+00:00',window_seconds=21600)

    def test_exact_window(self):
        validate_authorization(self.a,dt.datetime.fromisoformat(self.a['start']))
        for time in ['2026-10-19T06:15:12+00:00',self.a['deadline']]:
            with self.assertRaises(AssertionError):validate_authorization(self.a,dt.datetime.fromisoformat(time))

    def test_scope_or_budget_drift_rejected(self):
        for key,value in [('study_limit_seconds',73801),('gpus',[1,3]),('seeds',[42,43,44]),
                          ('fixed_remaining_episodes_per_seed',44),('checkpoint_steps',19999),
                          ('no_automatic_retry',False),('window_seconds',21601)]:
            with self.subTest(key=key),self.assertRaises(AssertionError):
                validate_authorization(dict(self.a,**{key:value}))

    def test_original_refusal_and_amended_fit(self):
        for prior,projected in [(2427.972110762028,8388.554448720446),(2486.9386035960633,8333.71372520317)]:
            with self.assertRaises(AssertionError):check_remaining(10800,prior,10800-prior,projected)
            check_remaining(10920,prior,10920-prior,projected)

    def test_reserve_and_charged_pilot_preserved(self):
        for cap,prior,remaining,projected in [(10920,2400,8521,8000),(10920,2400,8520,8460),
                                             (10920,2400,8520,float('nan')),(10920,0,10920,8000)]:
            with self.assertRaises(AssertionError):check_remaining(cap,prior,remaining,projected)

    def test_seed44_and_original_default_unchanged(self):
        p=dict(evaluation_seconds_per_seed=10800)
        self.assertEqual(limit(p,42),10800)
        p['evaluation_limits_seconds']=self.a['evaluation_limits_seconds']
        self.assertEqual(limit(p,44),10800)
        self.assertEqual(limit(p,42),10920)

    def test_training_or_pilot_relaunch_rejected(self):
        from run_batch import admission
        from evaluate_job import admission as evaluator
        for phase,seeds in [('training',[42,43]),('pilot_evaluation',[42,43]),('full_evaluation',[44])]:
            with self.assertRaises(AssertionError):admission(phase,seeds)
        for seed,stage in [(42,'pilot'),(44,'full')]:
            with self.assertRaises(AssertionError):evaluator(seed,stage)


if __name__=='__main__':unittest.main()
