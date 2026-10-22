"""Compile actual helper against CPU-only fake symbols; never load GPU libs."""
import ctypes, pathlib, subprocess, tempfile, unittest
S=pathlib.Path(__file__).resolve().parent
class Prefix(unittest.TestCase):
 def check(self,init,support,sequence,expected):
  with tempfile.TemporaryDirectory(prefix="openwam-fake-glfw-") as d:
   d=pathlib.Path(d); fake=d/'fake.cpp'; out=d/'fake.so'
   fake.write_text('extern "C" int glfwInit(){return %d;} extern "C" int glfwVulkanSupported(){return %d;}'%(init,support))
   subprocess.run(['g++','-shared','-fPIC',str(S/'glfw_prefix.cpp'),str(fake),'-o',str(out)],check=True,capture_output=True)
   deps=subprocess.check_output(['readelf','-d',str(out)],text=True)
   self.assertFalse(any(x in deps.lower() for x in ['libcuda','libvulkan','libsvulkan','libsapien','libglfw']))
   lib=ctypes.CDLL(str(out))
   self.assertEqual([getattr(lib,'glfw_prefix_'+name)() for name in sequence],expected)
 def test_good_and_repeated_init(self):self.check(1,1,['init','supported','init'],[1,1,-1])
 def test_bad_order(self):self.check(1,1,['supported','init'],[-1,-1])
 def test_init_failure(self):self.check(0,1,['init','supported'],[0,-1])
 def test_support_failure(self):self.check(1,0,['init','supported','supported'],[1,0,-1])
if __name__=='__main__':unittest.main()
