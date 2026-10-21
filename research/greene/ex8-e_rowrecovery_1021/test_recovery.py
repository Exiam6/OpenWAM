import copy,json,pathlib,unittest
from run_recovery import account,KEY,R
class RecoverySafety(unittest.TestCase):
 def setUp(self):
  self.p=json.loads((R/'ex8-d_nativefull_1021/protocol.json').read_text())
  self.w=json.loads((R/'nightly-gpu/runs/ex8d-full-000-adjust-bottle/result.json').read_text())
 def test_total_includes_actual_failed_usage_and_all99_caps(self):
  total=account(self.p,self.w);self.assertLessEqual(total,280*3600)
  self.assertGreater(total, sum(x['max_seconds'] for x in self.p['rows'] if x['key']!=KEY)+9000)
 def test_running_parent_cannot_release_reservation(self):
  self.w['status']='running'
  with self.assertRaises(AssertionError):account(self.p,self.w)
 def test_unexplained_failure_is_not_blindly_retried(self):
  self.w['error']='Native technical error'
  with self.assertRaises(AssertionError):account(self.p,self.w)
 def test_invalid_parent_cost_is_rejected(self):
  self.w['occupied_parent_seconds']=0
  with self.assertRaises(AssertionError):account(self.p,self.w)
 def test_parent_overrun_is_rejected(self):
  self.w['occupied_parent_seconds']=10459
  with self.assertRaises(AssertionError):account(self.p,self.w)
if __name__=='__main__':unittest.main(verbosity=2)
