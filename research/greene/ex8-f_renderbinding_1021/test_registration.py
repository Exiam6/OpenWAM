import unittest
from run_probe import peers
from register import reserve
class Safety(unittest.TestCase):
 def test_uncertain_peer_blocks(self):
  self.assertEqual(peers([dict(id='ex8d-full-2',status='submission_uncertain')]),['ex8d-full-2'])
 def test_terminal_peers_clear(self):
  self.assertEqual(peers([dict(id='ex8d-full-2',status='failed'),dict(id='ex8d-full-1',status='complete')]),[])
 def test_unrelated_job_not_native_peer(self):
  self.assertEqual(peers([dict(id='ex6c-other',status='running')]),[])
 def test_round_occupied_time_up(self):
  f=dict(status='failed',finished='now',error='Another process started using this GPU',occupied_parent_seconds=267.1)
  self.assertEqual(reserve(dict(combined_reserved_seconds=1007528),f),1007976)
 def test_unfinished_or_nonresource_rejected(self):
  for f in [dict(status='running'),dict(status='failed',finished='now',error='OOM')]:
   with self.assertRaises(AssertionError):reserve(dict(combined_reserved_seconds=1),f)
if __name__=='__main__':unittest.main()
