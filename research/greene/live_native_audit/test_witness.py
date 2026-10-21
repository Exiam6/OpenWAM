import pathlib,unittest
from witness import validate_environment
class EnvironmentChecks(unittest.TestCase):
 def test_valid(self):
  self.assertEqual(validate_environment({'CUDA_VISIBLE_DEVICES':'GPU-x','TMPDIR':'/out/tmp'},'GPU-x',pathlib.Path('/out'),'same','same')['CUDA_VISIBLE_DEVICES'],'GPU-x')
 def test_forbidden_overrides(self):
  for flag in ['ROBOTWIN_TEST_NUM','ROBOTWIN_ENABLE_PLANNER_FALLBACK','ROBOTWIN_LABTASKER_OPERATION','ROBOTWIN_SKIP_VERSION_CHECK']:
   with self.subTest(flag=flag),self.assertRaises(AssertionError):
    validate_environment({'CUDA_VISIBLE_DEVICES':'GPU-x','TMPDIR':'/out/tmp',flag:'1'},'GPU-x',pathlib.Path('/out'),'same','same')
 def test_wrong_gpu(self):
  with self.assertRaises(AssertionError):validate_environment({'CUDA_VISIBLE_DEVICES':'GPU-y'},'GPU-x',pathlib.Path('/out'),'same','same')
 def test_wrong_cgroup(self):
  with self.assertRaises(AssertionError):validate_environment({'CUDA_VISIBLE_DEVICES':'GPU-x'},'GPU-x',pathlib.Path('/out'),'other','same')
 def test_shared_tmp(self):
  with self.assertRaises(AssertionError):validate_environment({'CUDA_VISIBLE_DEVICES':'GPU-x','TMPDIR':'/tmp'},'GPU-x',pathlib.Path('/out'),'same','same')
if __name__=='__main__':unittest.main(verbosity=2)
