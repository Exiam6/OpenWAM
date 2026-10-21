import importlib.util,json,pathlib,tempfile,unittest,hashlib
import run_full,prepare,aggregate_full
from unittest.mock import patch
LOG=prepare.A/'pilot/demo_randomized-place_object_basket/client.log'
class FailureClassification(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.text=LOG.read_text();cls.block=cls.text[cls.text.index('[eval_policy_wrapper] exception'):]
 def test_actual_native_catch(self):
  c=run_full.classify_full(self.text,'place_object_basket',True)
  self.assertEqual(c['rejected_candidates'],1);self.assertEqual(c['fatal'],[])
 def test_other_task_rejects_same_block(self):
  self.assertTrue(run_full.classify_full(self.text,'adjust_bottle',True)['fatal'])
 def test_unrelated_traceback_still_fatal(self):
  self.assertTrue(run_full.classify_full(self.text+'Traceback (most recent call last):\nRuntimeError: policy error\n','place_object_basket',True)['fatal'])
 def test_planner_errors_still_fatal(self):
  for error in ['CUDA/Curobo prewarm failed:', 'planner fallback engaged:']:
   self.assertTrue(run_full.classify_full(self.text+error,'place_object_basket',True)['fatal'])
 def test_planner_error_inside_candidate_traceback_is_fatal(self):
  text=self.text.replace('AssertionError: target_pose', 'planner fallback engaged: forbidden\nAssertionError: target_pose')
  self.assertTrue(run_full.classify_full(text,'place_object_basket',True)['fatal'])
 def test_changed_exception_fatal(self):
  self.assertTrue(run_full.classify_full(self.text.replace('AssertionError: target_pose cannot be None for move action.','RuntimeError: other error'),'place_object_basket',True)['fatal'])
 def test_missing_native_catch_final_fatal(self):
  self.assertTrue(run_full.classify_full(self.text.replace('error occurs !',''),'place_object_basket',True)['fatal'])
 def test_partial_traceback_is_not_final_pass(self):
  t=self.text[:self.text.index('AssertionError: target_pose')]
  self.assertFalse(run_full.classify_full(t,'place_object_basket')['fatal'])
  self.assertTrue(run_full.classify_full(t,'place_object_basket',True)['fatal'])
 def test_wrong_method_fatal(self):
  self.assertTrue(run_full.classify_full(self.text.replace('exception in place_object_basket.play_once','exception in place_object_basket.take_action'),'place_object_basket',True)['fatal'])
class CountsAndCosts(unittest.TestCase):
 def test_native_environment_rejects_alternate_evaluator(self):
  with patch.dict(run_full.os.environ,{'ROBOTWIN_LABTASKER_OPERATION':'run_eval'}):
   with self.assertRaises(AssertionError):run_full.native_environment_guard()
 def test_native_version_check_cannot_be_bypassed(self):
  with patch.dict(run_full.os.environ,{'ROBOTWIN_SKIP_VERSION_CHECK':'1'}):
   with self.assertRaises(AssertionError):run_full.native_environment_guard()
 def test_preserved_partial_not_complete(self):
  self.assertEqual(len(run_full.score(LOG.read_text())),2)
 def test_duplicate_or_decreasing_seed_rejected(self):
  for seed in [100000,99999]:
   with self.assertRaises(AssertionError):run_full.score(f'Success rate: 1/1 current seed: 100000\nSuccess rate: 2/2 current seed: {seed}')
 def test_missing_episode_rejected(self):
  with self.assertRaises(AssertionError):run_full.score('Success rate: 1/1 current seed: 100000\nSuccess rate: 2/3 current seed: 100001')
 def test_success_regression_rejected(self):
  with self.assertRaises(AssertionError):run_full.score('Success rate: 1/1 current seed: 100000\nSuccess rate: 0/2 current seed: 100001')
 def test_budget_allocation_and_complete_suite(self):
  base=prepare.read(prepare.A/'protocol.json')
  for ceiling in [200,280]:
   rows=prepare.allocation(base,ceiling)
   self.assertEqual(sum(r['episodes'] for r in rows),10000)
   self.assertEqual(len({r['key'] for r in rows}),100)
   self.assertLessEqual(sum(r['max_seconds'] for r in rows),ceiling*3600)
 def test_zero_recovery_lower_bound_exceeds_existing_budget(self):
  groups=prepare.read(prepare.A/'pilot/result.json')['groups']
  self.assertGreater(sum(g['seconds'] for g in groups)/30*10000/3600+5,200)
 def test_manifest_duplicate_row_rejected(self):
  rows=prepare.allocation(prepare.read(prepare.A/'protocol.json'),200);rows[-1]=rows[0]
  with self.assertRaises(AssertionError):run_full.validate_full_suite({'rows':rows,'planned_episodes':10000})
 def test_manifest_changed_seed_rejected(self):
  rows=prepare.allocation(prepare.read(prepare.A/'protocol.json'),200);rows[0]['seed']=1
  with self.assertRaises(AssertionError):run_full.validate_full_suite({'rows':rows,'planned_episodes':10000})
 def test_manifest_wrong_task_rejected(self):
  rows=prepare.allocation(prepare.read(prepare.A/'protocol.json'),200);rows[0]['task']='other'
  with self.assertRaises(AssertionError):run_full.validate_full_suite({'rows':rows,'planned_episodes':10000})
 def test_incomplete_aggregate_no_headline(self):
  previous=aggregate_full.ROOT
  with tempfile.TemporaryDirectory() as d:
   aggregate_full.ROOT=pathlib.Path(d);(aggregate_full.ROOT/'protocol.json').write_text(json.dumps({'rows':prepare.allocation(prepare.read(prepare.A/'protocol.json'),200),'planned_episodes':10000}))
   try:
    r=aggregate_full.main();self.assertFalse(r['full_endpoint_complete']);self.assertNotIn('rates_pct',r)
   finally:aggregate_full.ROOT=previous
if __name__=='__main__':unittest.main(verbosity=2)
