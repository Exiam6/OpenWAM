import unittest
from prewarm import checked_prewarm,native_environment
class Gates(unittest.TestCase):
 def test_swallowed_failure_stops(self):
  with self.assertRaises(RuntimeError):checked_prewarm(lambda:print("CUDA/Curobo prewarm failed: Ninja is required"))
 def test_silent_failure_stops(self):
  with self.assertRaises(RuntimeError):checked_prewarm(lambda:None)
 def test_mixed_success_failure_stops(self):
  with self.assertRaises(RuntimeError):checked_prewarm(lambda:print("prewarmed CUDA/Curobo before SAPIEN import\nCUDA/Curobo prewarm failed:"))
 def test_original_exception_retained(self):
  def fail():raise ValueError("native import failed")
  with self.assertRaises(ValueError):checked_prewarm(fail)
 def test_explicit_native_success(self):
  self.assertIn("prewarmed",checked_prewarm(lambda:print("prewarmed CUDA/Curobo before SAPIEN import")))
 def test_native_toolchain(self):
  e=native_environment("/runtime","/output")
  self.assertIn("/runtime/benchmark-env/bin",e["PATH"])
  self.assertEqual(e["CUDA_HOME"],"/runtime/cuda-toolkit")
  self.assertEqual(e["TORCH_EXTENSIONS_DIR"],"/output/cache/extensions")
if __name__=="__main__":unittest.main()
