import unittest
from pci_visibility import selector,apply,BLOCKED
class SelectionTests(unittest.TestCase):
 def test_uuid_pci_normalization_and_exclusive_suffix(self):
  self.assertEqual(selector('00000000:07:00.0'),'pci-0000_07_00_0!')
  self.assertEqual(selector('0000:AB:1F.7'),'pci-0000_ab_1f_7!')
 def test_each_bus_maps_to_its_own_selector(self):
  for bus in [6,7,8,9,10,11,12,13]:
   env={'CUDA_VISIBLE_DEVICES':'assigned','KEEP':'fixed'}
   self.assertEqual(apply(env,f'0000:{bus:02x}:00.0'),f'pci-0000_{bus:02x}_00_0!')
   self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'assigned');self.assertEqual(env['KEEP'],'fixed')
 def test_rejects_malformed_or_out_of_range_identity(self):
  for value in ['',None,'7','0000:07:00','0000:07:00.8','10000:07:00.0','0000:07:20.0','0000:07:00.0!']:
   with self.subTest(value=value),self.assertRaises((AssertionError,TypeError)):selector(value)
 def test_any_ambient_selection_override_stops_before_mutation(self):
  for key in BLOCKED:
   env={key:'bad'};before=dict(env)
   with self.subTest(key=key),self.assertRaises(AssertionError):apply(env,'0000:07:00.0')
   self.assertEqual(env,before)
if __name__=='__main__':unittest.main()
