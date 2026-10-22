import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import observer_worker as worker
from integration import DEPENDENCIES
sys.path.insert(0,str(DEPENDENCIES/'native_observer_prep'))
import test_observer as f
from observer import observe,ObservationError

class LatencyRecoveryTests(unittest.TestCase):
 def load(self,seconds):
  cfg=dict(host='host',device_map={str(k):v for k,v in f.MAP.items()},assigned=f.MAP[0],own_group=f.OWN,gdm_uid=120,seconds=seconds,pause_paths=[worker.GLOBAL_PAUSE])
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'config.json';raw=json.dumps(cfg).encode();p.write_bytes(raw)
   return worker.load_config(p,hashlib.sha256(raw).hexdigest())
 def test_bounded_eight_seconds_configuration(self):
  self.assertEqual(self.load(8)['seconds'],8)
  for value in [8.01,9,0,-1,True,float('nan'),float('inf')]:
   with self.subTest(value=value),self.assertRaises(worker.WorkerError):self.load(value)
 def telemetry(self,seconds,delay):
  now=[0.0];outputs=iter([f.xml(f.fixture()),f.csv()])
  def runner(argv,**kw):
   if delay>kw['timeout']:raise subprocess.TimeoutExpired(argv,kw['timeout'],output=b'partial')
   now[0]+=delay
   return SimpleNamespace(stdout=next(outputs),stderr='',returncode=0)
  return observe('host',f.MAP,f.MAP[0],f.OWN,120,seconds=seconds,clock=lambda:now[0],hostname=lambda:'host',runner=runner,read_identity=f.proc)
 def test_delayed_complete_inventory_old_fails_new_accepts(self):
  with self.assertRaises(ObservationError) as caught:self.telemetry(3,3.2)
  self.assertTrue(caught.exception.evidence['queries'][0]['timed_out'])
  result=self.telemetry(8,3.2)
  self.assertTrue(result['complete']);self.assertEqual(len(result['evidence']['queries']),2)
  self.assertAlmostEqual(result['evidence']['monotonic_end'],6.4)
 def test_combined_deadline_still_rejects_and_retains_partial(self):
  with self.assertRaises(ObservationError) as caught:self.telemetry(8,4.1)
  self.assertEqual(len(caught.exception.evidence['queries']),2)
  self.assertTrue(caught.exception.evidence['queries'][1]['timed_out'])
 def test_entry_propagates_registered_timing_and_local_worker(self):
  import probe_candidate as probe
  from test_host_identity import HostIdentityTests,HOST
  seen,result,exists=HostIdentityTests().entry(HOST,'gpu01')
  self.assertEqual(seen[0]['seconds'],8);self.assertTrue(exists)

if __name__=='__main__':unittest.main()
