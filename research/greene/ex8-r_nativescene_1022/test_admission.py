import datetime as dt,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run_probe
class AdmissionChecks(unittest.TestCase):
 def setup_files(self,root):
  f=Path(root)/'stage';n=Path(root)/'dispatcher';f.mkdir();n.mkdir();(n/'claims').mkdir()
  pause=f/'pause';source=f/'source';source.write_text('fixed')
  p=dict(kind='FROZEN_NATIVE_INSTANCE_AUDIT',registered_for_dispatch=True,max_seconds=600,probe_alarm_seconds=480,deadline='2026-10-28T18:20:27.005560+00:00',pause_paths=[str(pause)],input_sha256={str(source):run_probe.sha(source)})
  (f/'protocol.json').write_text(json.dumps(p));(n/'latest.json').write_text(json.dumps(dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),errors={})))
  return f,n,pause,source
 def test_ready_and_stale_inventory(self):
  with tempfile.TemporaryDirectory() as d:
   f,n,_,_=self.setup_files(d)
   with patch.object(run_probe,'F',f),patch.object(run_probe,'N',n),patch.dict(os.environ,{},clear=True):
    self.assertEqual(run_probe.admit()['max_seconds'],600)
    (n/'latest.json').write_text(json.dumps(dict(time='2026-01-01T00:00:00+00:00',errors={})))
    with self.assertRaises(AssertionError):run_probe.admit()
 def test_pause_changed_input_existing_result_and_active_peer(self):
  for fault in ['pause','input','result','peer']:
   with self.subTest(fault=fault),tempfile.TemporaryDirectory() as d:
    f,n,pause,source=self.setup_files(d)
    if fault=='pause':pause.write_text('{}')
    if fault=='input':source.write_text('changed')
    if fault=='result':(f/'result.json').write_text('{}')
    if fault=='peer':(n/'claims/peer.json').write_text(json.dumps(dict(id='ex8-peer',status='running')))
    with patch.object(run_probe,'F',f),patch.object(run_probe,'N',n),patch.dict(os.environ,{},clear=True):
     with self.assertRaises(AssertionError):run_probe.admit()
if __name__=='__main__':unittest.main()
