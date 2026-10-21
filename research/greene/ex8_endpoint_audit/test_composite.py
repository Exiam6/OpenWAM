import copy
import unittest
from verify_composite import native_rows, statistics


def log(successes, total, seed, outcome):
    return f'{outcome}!\nSuccess rate: {successes}/{total} => 50.0%, current seed: {seed}\n'


class NativeCounts(unittest.TestCase):
    def test_failure_counts_as_accepted_episode(self):
        rows = native_rows(log(1,1,100000,'Success')+log(1,2,100004,'Fail'), final=True)
        self.assertEqual([x['success'] for x in rows], [True,False])
        self.assertEqual([x['scene_seed'] for x in rows], [100000,100004])

    def test_outcome_disagreement_is_rejected(self):
        with self.assertRaises(AssertionError):
            native_rows(log(0,1,100000,'Success'))

    def test_counter_jump_is_rejected(self):
        with self.assertRaises(AssertionError):
            native_rows(log(1,2,100000,'Success'))

    def test_duplicate_seed_is_rejected(self):
        with self.assertRaises(AssertionError):
            native_rows(log(1,1,100000,'Success')+log(2,2,100000,'Success'))

    def test_missing_outcome_is_rejected(self):
        with self.assertRaises(AssertionError):
            native_rows('Success rate: 1/1 => 100%, current seed: 100000\n')

    def test_partial_final_outcome_not_complete(self):
        self.assertEqual(native_rows('Success!\n'), [])
        with self.assertRaises(AssertionError):
            native_rows('Success!\n', final=True)

    def test_ansi_and_carriage_return(self):
        self.assertEqual(len(native_rows('step: 1/400\r\x1b[92m'+log(1,1,100000,'Success')+'\x1b[0m')),1)

    def test_live_partial_counter_waits_but_final_rejects(self):
        self.assertEqual(native_rows('Success!\nSuccess rate: 1/'), [])
        with self.assertRaises(AssertionError):
            native_rows('Success!\nSuccess rate: 1/', final=True)

    def test_success_regression_is_rejected(self):
        with self.assertRaises(AssertionError):
            native_rows(log(1,1,100000,'Success')+log(0,2,100001,'Fail'))


class PairedEndpoint(unittest.TestCase):
    def setUp(self):
        self.tasks = [f'task{i}' for i in range(50)]
        self.rows = [dict(task=t,mode=m,episodes=100,successes=(90+i%11 if m=='demo_clean' else 100-i%11))
                     for m in ('demo_clean','demo_randomized') for i,t in enumerate(self.tasks)]

    def test_paired_draw_preserves_constant_pair_mean(self):
        result = statistics(self.rows,self.tasks)
        self.assertEqual(result['rates_pct']['mean'],95)
        self.assertEqual(result['task_bootstrap_95pct']['mean'],[95,95])
        self.assertLess(*result['task_bootstrap_95pct']['clean'])
        self.assertAlmostEqual(result['gaps_pp']['mean'],1.4)
        self.assertFalse(result['full_paper_reproduced'])
        self.assertFalse(result['exact_published_numbers_matched'])

    def test_missing_row_cannot_yield_headline(self):
        with self.assertRaises(AssertionError):
            statistics(self.rows[:-1],self.tasks)

    def test_duplicate_row_cannot_substitute_missing_task(self):
        self.rows[-1] = copy.deepcopy(self.rows[0])
        with self.assertRaises(AssertionError):
            statistics(self.rows,self.tasks)

    def test_unaccepted_episode_count_is_rejected(self):
        self.rows[0]['episodes'] = 99
        with self.assertRaises(AssertionError):
            statistics(self.rows,self.tasks)

    def test_impossible_successes_rejected(self):
        for bad in [-1,101,True,2.5]:
            self.rows[0]['successes'] = bad
            with self.assertRaises(AssertionError):
                statistics(self.rows,self.tasks)


if __name__ == '__main__':
    unittest.main(verbosity=2)
