import sys,types,unittest
from pathlib import Path
from unittest.mock import patch
from integration import NativeBackend
class BackendChecks(unittest.TestCase):
 def test_device_import_order_and_lifetime(self):
  calls=[]
  torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:True,device_count=lambda:1),zeros=lambda *a,**k:calls.append('cuda') or object())
  dev=types.SimpleNamespace(is_cuda=lambda:True,can_render=lambda:True,is_cpu=lambda:False,cuda_id=0,pci_string='0000:75:00.0',name='mock')
  sapien=types.SimpleNamespace(Device=lambda s:calls.append(s) or dev)
  with patch.dict(sys.modules,torch=torch,sapien=sapien):
   b=NativeBackend('unused','00000000:75:00.0');self.assertTrue(b.sapien_import());self.assertTrue(b.explicit_device())
  self.assertEqual(calls,['cuda','cuda:0']);self.assertIs(b.device,dev);self.assertTrue(b.tensor)
 def test_no_device_on_wrong_cuda_cardinality(self):
  t=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:True,device_count=lambda:2))
  with patch.dict(sys.modules,torch=t):
   with self.assertRaises(AssertionError):NativeBackend('unused','0000:75:00.0').sapien_import()
 def test_wrong_pci_stops(self):
  b=NativeBackend('unused','0000:75:00.0');d=types.SimpleNamespace(is_cuda=lambda:True,can_render=lambda:True,is_cpu=lambda:False,cuda_id=0,pci_string='0000:76:00.0',name='mock')
  b.sapien=types.SimpleNamespace(Device=lambda _:d)
  with self.assertRaises(AssertionError):b.explicit_device()
 def test_helper_lazy_load_and_raw_failure(self):
  class Fn:
   def __init__(self,value):self.value=value
   def __call__(self):return self.value
  from integration import HELPER
  lib=types.SimpleNamespace(**{s:Fn(1) for _,s,_ in HELPER},headless_error=Fn(b'raw error'))
  with patch('ctypes.CDLL',return_value=lib) as load:
   b=NativeBackend('unused','0000:75:00.0');load.assert_not_called();self.assertTrue(b.helper_load());load.assert_called_once()
   lib.headless_loader.value=-1;self.assertEqual(b.helper_call('headless_loader'),-1)
   self.assertEqual(b.details['native_error'],'raw error')
if __name__=='__main__':unittest.main()
