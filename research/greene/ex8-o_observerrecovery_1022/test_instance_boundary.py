import types,unittest
from unittest.mock import patch
from integration import NativeBackend,operations
from supervisor import OPERATIONS
class Fn:
 def __init__(self,value):self.value=value;self.calls=0
 def __call__(self):self.calls+=1;return self.value
class InstanceTests(unittest.TestCase):
 def test_laziness_and_exact_single_constructor(self):
  create=Fn(0);lib=types.SimpleNamespace(instance_boundary_create=create,instance_boundary_error=Fn(b'error'))
  with patch('ctypes.CDLL',return_value=lib) as load:
   b=NativeBackend('/tmp/libheadless.so','0000:07:00.0');load.assert_not_called()
   self.assertEqual(tuple(operations(b)),OPERATIONS)
   b.instance_helper_load();load.assert_called_once_with('/tmp/libinstance_boundary.so')
   self.assertTrue(b.native_Instance_Get());self.assertEqual(create.calls,1)
   self.assertEqual(b.details['instance_boundary_create'],0)
 def test_failure_keeps_native_message(self):
  lib=types.SimpleNamespace(instance_boundary_create=Fn(-1),instance_boundary_error=Fn(b'actual native error'))
  with patch('ctypes.CDLL',return_value=lib):
   b=NativeBackend('/tmp/libheadless.so','0000:07:00.0');b.instance_helper_load()
   with self.assertRaisesRegex(RuntimeError,'actual native error'):b.native_Instance_Get()
   self.assertEqual(b.details['instance_error'],'actual native error')
if __name__=='__main__':unittest.main()
