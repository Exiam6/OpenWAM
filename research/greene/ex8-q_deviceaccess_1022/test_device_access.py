import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from device_access import abi,assigned_nodes,native_rules
class AccessChecks(unittest.TestCase):
 def test_actual_kernel_support_only(self):self.assertGreaterEqual(abi(),3)
 def test_real_restriction_threads_children_and_aliases(self):
  script=r"""
import json,os,threading,subprocess,sys
from pathlib import Path
from device_access import restrict_paths
root=Path(sys.argv[1]);yes=root/'allowed';no=root/'denied'
ready=threading.Event();go=threading.Event();seen=[]
def observer():
 ready.set();go.wait();seen.append(no.read_text())
t=threading.Thread(target=observer);t.start();ready.wait()
restrict_paths([yes,Path(sys.executable).resolve(),Path('/usr'),Path('/lib'),Path('/lib64'),Path('/etc'),Path('/proc'),Path(sys.base_prefix),Path(sys.prefix)],list_dirs=[root])
assert yes.read_text()=='allowed'
for p in [no,root/'alias',Path('/proc/self/root')/str(no).lstrip('/')]:
 try:p.read_text()
 except PermissionError:pass
 else:raise AssertionError('Unselected read succeeded')
go.set();t.join();assert seen==['denied']
q=subprocess.run([sys.executable,'-c','import pathlib,sys;pathlib.Path(sys.argv[1]).read_text()',str(no)],capture_output=True,text=True)
assert q.returncode!=0 and 'PermissionError' in q.stderr,q.stderr
print(json.dumps({'restricted':True,'observer_unrestricted':True,'child_denied':True}))
"""
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'allowed').write_text('allowed');(p/'denied').write_text('denied');(p/'alias').symlink_to(p/'denied')
   q=subprocess.run([sys.executable,'-c',script,d],cwd=Path(__file__).parent,capture_output=True,text=True,timeout=15)
   self.assertEqual(q.returncode,0,q.stderr);self.assertTrue(json.loads(q.stdout)['observer_unrestricted'])
 def test_pci_uuid_minor_and_drm_mapping(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);proc=root/'proc';drm=root/'drm';dev=root/'dev';(proc/'0000:07:00.0').mkdir(parents=True);drm.mkdir();(dev/'dri').mkdir(parents=True)
   info=proc/'0000:07:00.0/information';info.write_text('GPU UUID: GPU-assigned\nDevice Minor: 3\n');(dev/'nvidia3').touch()
   for n in ['card4','renderD131']:
    (drm/n).mkdir();(drm/n/'device').symlink_to(proc/'0000:07:00.0');(dev/'dri'/n).touch()
   out=assigned_nodes('00000000:07:00.0','GPU-assigned',proc_root=proc,drm_root=drm,dev_root=dev)
   self.assertEqual(len(out),3)
   with self.assertRaises(AssertionError):assigned_nodes('0000:07:00.0','GPU-wrong',proc_root=proc,drm_root=drm,dev_root=dev)
 def test_rule_construction_excludes_peer_nodes_and_ancestor_grants(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);dev=root/'dev';dev.mkdir();(root/'usr').mkdir();(dev/'dri').mkdir();(dev/'nvidia0').touch();(dev/'nvidia1').touch();(dev/'nvidiactl').touch();(dev/'null').touch();(dev/'alias').symlink_to(dev/'nvidia1')
   selected=[str(dev/'nvidia0')];out=list(map(str,native_rules(selected,root=root,dev=dev)))
   self.assertIn(str(dev/'nvidia0'),out)
   for p in [dev,dev/'dri',dev/'nvidia1',dev/'alias']:self.assertNotIn(str(p),out)
if __name__=='__main__':unittest.main()
