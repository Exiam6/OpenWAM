import importlib.util,json,os,socket,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from host_identity import checked_host
import probe_candidate as probe
HOST='shenlong-gpu-01.cs.illinois.edu'
class HostIdentityTests(unittest.TestCase):
 def test_live_host_matches_frozen_controller(self):
  p=json.loads(Path('/home/zifanz4/openwam-runtime/nightly-gpu/policy.json').read_text())
  self.assertEqual(checked_host(socket.gethostname(),'gpu01',{'gpu01':p['hosts']['gpu01']['hostname']}),HOST)
  self.assertNotEqual(socket.gethostname(),'gpu01')
 def test_negative_identities(self):
  for actual,alias,mapping in [(HOST,'gpu02',{'gpu01':HOST}),('wrong','gpu01',{'gpu01':HOST}),(HOST,'gpu01',{}),(HOST,'gpu01',{'gpu01':''}),(HOST,'gpu01',{'gpu01':HOST,'gpu02':'other'})]:
   with self.subTest(actual=actual,alias=alias,mapping=mapping),self.assertRaises(ValueError):checked_host(actual,alias,mapping)
 def entry(self,actual,alias):
  import datetime as dt
  class StopObserver(Exception):pass
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);p=dict(registered_for_dispatch=True,kind='FROZEN_NATIVE_HEADLESS_PREFIX',max_seconds=150,probe_alarm_seconds=70,pause_paths=[],input_sha256={},deadline='2026-10-28T18:20:27.005560+00:00',hostnames={'gpu01':HOST},device_map={'1':'assigned'})
   (root/'protocol.json').write_text(json.dumps(p));seen=[]
   def stop(config,*args,**kw):seen.append(config);raise StopObserver('CPU observer boundary; no native calls')
   with patch.object(probe,'F',root),patch.object(probe,'native_guard',return_value='assigned'),patch.object(probe.socket,'gethostname',return_value=actual),patch.object(probe,'observer',side_effect=stop),patch.dict(os.environ,{'OPENWAM_GPU_HOST':alias,'OPENWAM_ALLOCATION_END':(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=330)).isoformat()}),patch('inventory.unit_path',return_value='/fake/dispatcher.service'):
    self.assertEqual(probe.main(),1)
   return seen,json.loads((root/'result.json').read_text()),(root/'observer-config.json').exists()
 def test_actual_entry_reaches_observer_with_physical_host(self):
  seen,result,config=self.entry(HOST,'gpu01');self.assertEqual(seen[0]['host'],HOST);self.assertTrue(config);self.assertIn('StopObserver',result['traceback'])
 def test_actual_entry_refuses_wrong_host_before_observer(self):
  for host,alias in [('wrong','gpu01'),(HOST,'gpu02')]:
   seen,result,config=self.entry(host,alias);self.assertEqual(seen,[]);self.assertFalse(config);self.assertIn('hostname mismatch',result['error'])
if __name__=='__main__':unittest.main()
