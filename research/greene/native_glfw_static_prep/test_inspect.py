import struct
import unittest
from pathlib import Path
from inspect_elf import symbol,literal,destination

class EvidenceChecks(unittest.TestCase):
    def test_actual_dynamic_symbol_format(self):
        self.assertEqual(symbol('0000000000c8a680 g DF .text 0000000000000106 Base glfwInit','glfwInit'),(0xc8a680,0x106))
    def test_missing_or_ambiguous_rejected(self):
        line='0000000000c8a680 g DF .text 0000000000000106 Base glfwInit'
        for table in ['',line+'\n'+line]:
            with self.assertRaises(AssertionError):symbol(table,'glfwInit')
    def fixture(self,kind=1):
        raw=bytearray(512);raw[:6]=b'\x7fELF\x02\x01';struct.pack_into('<Q',raw,40,64);struct.pack_into('<HH',raw,58,64,1)
        struct.pack_into('<IIQQQQIIQQ',raw,64,0,kind,0,0x1000,256,32,0,0,1,0)
        raw[256:270]=b'libvulkan.so.1';return bytes(raw)
    def test_resolves_exact_virtual_address(self):
        self.assertEqual(literal(self.fixture(),0x1000),'libvulkan.so.1')
        self.assertIsNone(literal(self.fixture(),0x2000))
    def test_nobits_not_read_as_file_data(self):
        self.assertIsNone(literal(self.fixture(8),0x1000))
    def test_invalid_elf_rejected(self):
        with self.assertRaises(AssertionError):literal(bytes(512),0)
    def test_output_cannot_escape_evidence(self):
        for p in ['/tmp/a','/home/zifanz4/openwam-runtime/a','/home/zifanz4/.local/state/openwam-selfcheck/evidence/../handoff.json']:
            with self.assertRaises(AssertionError):destination(Path(p))

if __name__=='__main__':unittest.main(verbosity=2)
