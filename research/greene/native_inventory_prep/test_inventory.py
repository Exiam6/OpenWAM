import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from inventory import InventoryError, belongs, process_group, snapshot, unit_path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native_loader_supervisor_prep"))
from supervisor import OPERATIONS, Supervisor

MAP = {i: "GPU-%08x-1111-2222-3333-444444444444" % i for i in range(8)}
OWN = "0::/user.slice/openwam-nightly-job-test.service/child\n"
OTHER = "0::/user.slice/another.service\n"

def fixture(processes=()):
    root = ET.Element("nvidia_smi_log")
    for i in range(8):
        gpu = ET.SubElement(root, "gpu")
        ET.SubElement(gpu, "uuid").text = MAP[i]
        procs = ET.SubElement(gpu, "processes")
        for index, pid in processes:
            if i != index:
                continue
            p = ET.SubElement(procs, "process_info")
            for key, value in dict(pid=str(pid), type="C", used_memory="12 MiB", process_name="python").items():
                ET.SubElement(p, key).text = value
    return root

def xml(root):
    return ET.tostring(root, encoding="unicode")

def csv():
    return "\n".join("%d, %s" % (i, u) for i, u in MAP.items())

def run(root=None, text=None, groups=None):
    return snapshot(xml(fixture() if root is None else root), csv() if text is None else text,
                    MAP, MAP[0], OWN, lambda pid: (groups or {2: OWN})[pid])

class InventoryTests(unittest.TestCase):
    def test_complete_empty(self):
        self.assertEqual(run()["contexts"], [])
    def test_exact_and_nested_unit(self):
        unit = unit_path(OWN)
        self.assertTrue(belongs("0::"+unit, unit))
        self.assertTrue(belongs(OWN, unit))
        self.assertFalse(belongs("0::"+unit+"-other/child", unit))
    def test_invalid_own_cgroups(self):
        for text in (OTHER, "1:name=systemd:/x", OWN+"1::/x", "0::/../openwam-nightly-job-x.service"):
            with self.subTest(text=text), self.assertRaises(InventoryError):
                unit_path(text)
    def test_own_assigned_and_peer_preserved(self):
        s = run(fixture([(0, 2), (1, 2)]))
        self.assertEqual({c["uuid"] for c in s["contexts"]}, {MAP[0], MAP[1]})
    def test_foreign_peer_retained(self):
        s = run(fixture([(1, 3)]), groups={3: OTHER})
        self.assertEqual(len(s["foreign_contexts"]), 1)
        self.assertEqual(s["contexts"], [])
    def test_foreign_assigned_rejected(self):
        with self.assertRaises(InventoryError):
            run(fixture([(0, 3)]), groups={3: OTHER})
    def test_unverified_display_rejected(self):
        r = fixture([(0, 3)]); p = r.find("./gpu/processes/process_info")
        p.find("type").text = "G"; p.find("process_name").text = "/usr/lib/xorg/Xorg"
        with self.assertRaises(InventoryError):
            run(r, groups={3: OTHER})
    def test_missing_xml_gpu(self):
        r = fixture(); r.remove(r[-1])
        with self.assertRaises(InventoryError): run(r)
    def test_duplicate_xml_gpu(self):
        r = fixture(); r[-1].find("uuid").text = MAP[0]
        with self.assertRaises(InventoryError): run(r)
    def test_unknown_xml_uuid(self):
        r = fixture(); r[-1].find("uuid").text = "GPU-ffffffff-1111-2222-3333-444444444444"
        with self.assertRaises(InventoryError): run(r)
    def test_missing_csv_gpu(self):
        with self.assertRaises(InventoryError): run(text="\n".join(csv().splitlines()[:-1]))
    def test_duplicate_csv_index(self):
        with self.assertRaises(InventoryError): run(text=csv().replace("7, ", "0, "))
    def test_csv_mapping_swap(self):
        with self.assertRaises(InventoryError): run(text=csv().replace(MAP[0], "TEMP").replace(MAP[1], MAP[0]).replace("TEMP", MAP[1]))
    def test_csv_extra_column(self):
        with self.assertRaises(InventoryError): run(text=csv()+", garbage")
    def test_missing_process_list(self):
        r = fixture(); r[0].remove(r[0].find("processes"))
        with self.assertRaises(InventoryError): run(r)
    def test_unsupported_process_inventory(self):
        for value in ("N/A", "Not Supported"):
            r = fixture(); r[0].find("processes").text = value
            with self.subTest(value=value), self.assertRaises(InventoryError): run(r)
    def test_unrecognized_process_node(self):
        r = fixture(); ET.SubElement(r[0].find("processes"), "error")
        with self.assertRaises(InventoryError): run(r)
    def test_duplicate_process(self):
        with self.assertRaises(InventoryError): run(fixture([(0, 2), (0, 2)]))
    def test_malformed_process_fields(self):
        for field, value in (("pid", "-1"), ("pid", "N/A"), ("type", "?"), ("used_memory", "N/A"), ("process_name", "")):
            r = fixture([(0, 2)]); r.find("./gpu/processes/process_info/"+field).text = value
            with self.subTest(field=field, value=value), self.assertRaises(InventoryError): run(r)
    def test_process_read_denied(self):
        with self.assertRaises(InventoryError):
            snapshot(xml(fixture([(0, 2)])), csv(), MAP, MAP[0], OWN,
                     lambda pid: (_ for _ in ()).throw(PermissionError()))
    def test_disappeared_process(self):
        with tempfile.TemporaryDirectory() as root, self.assertRaises(InventoryError):
            process_group(2, Path(root))
    def test_pid_reuse(self):
        stat = lambda ticks: "2 (name with ) parens) S " + "0 "*18 + str(ticks)
        with patch.object(Path, "read_text", side_effect=[stat(1), OWN, stat(2)]), self.assertRaises(InventoryError):
            process_group(2)
    def test_stable_proc_identity(self):
        stat = "2 (name) S " + "0 "*18 + "123"
        with patch.object(Path, "read_text", side_effect=[stat, OWN, stat]):
            self.assertEqual(process_group(2), OWN)
    def test_integration_peer_blocks_operations(self):
        calls = []
        operations = {name: lambda name=name: calls.append(name) or True for name in OPERATIONS}
        result = Supervisor().run(operations, lambda: run(fixture([(1, 2)])), lambda: None,
                                  MAP[0], seconds=1, sample_timeout=.1, interval=.001)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(calls, [])
        self.assertIn("Peer GPU context", str(result))

if __name__ == "__main__": unittest.main()
