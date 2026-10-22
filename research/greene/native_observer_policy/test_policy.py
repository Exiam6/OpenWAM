import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"native_observer_prep"))
from observer import identity, system_display, XORG, ObservationError
import test_observer as f
from test_observer import proc, display_fixture, OTHER, OWN, MAP, fixture

class PolicyTests(unittest.TestCase):
    def test_display_policy_boundaries(self):
        for kind,name,uid,memory in itertools.product(("G","C","C+G"),(XORG,"/tmp/Xorg"),(0,120,1000),(-1,0,4,16,17,100)):
            row=dict(type=kind,name=name,memory_mib=memory)
            value=proc(3,OTHER,uid,None);value["executable_permission_denied"]=True
            expected=kind=="G" and name==XORG and uid in (0,120) and 0<=memory<=16
            self.assertEqual(system_display(row,value,120),expected)
    def test_no_unexplained_exe_absence(self):
        row=dict(type="G",name=XORG,memory_mib=4)
        self.assertFalse(system_display(row,proc(3,OTHER,120,None),120))
    def test_readable_mismatch_never_exempt(self):
        row=dict(type="G",name=XORG,memory_mib=4)
        value=proc(3,OTHER,120,"/tmp/Xorg");value["executable_permission_denied"]=True
        self.assertFalse(system_display(row,value,120))
    def test_permission_denied_identity_retained(self):
        stat="2 (test) S "+"0 "*18+"1"
        with patch.object(Path,"read_text",side_effect=[stat,OTHER,"Uid: 120 120 120 120",stat]),patch.object(Path,"resolve",side_effect=PermissionError(13,"denied")):
            value=identity(2)
            self.assertIsNone(value["executable"])
            self.assertTrue(value["executable_permission_denied"])
            self.assertEqual(value["cgroup"],OTHER)
    def test_missing_exe_still_rejected(self):
        stat="2 (test) S "+"0 "*18+"1"
        with patch.object(Path,"read_text",side_effect=[stat,OTHER,"Uid: 120 120 120 120"]),patch.object(Path,"resolve",side_effect=FileNotFoundError()):
            with self.assertRaises(RuntimeError):identity(2)
    def test_composed_denied_display_and_overcap(self):
        harness=f.ObserverTests()
        for memory in (4,16,17):
            root=display_fixture();root.find("./gpu/processes/process_info/used_memory").text=str(memory)+" MiB"
            value=proc(3,OTHER,120,None);value["executable_permission_denied"]=True
            if memory<=16:
                result=harness.run_observer(root,reader=lambda pid:value)
                self.assertEqual(len(result["system_displays"]),1)
            else:
                with self.assertRaises(ObservationError):harness.run_observer(root,reader=lambda pid:value)
    def test_own_unreadable_graphics_not_hidden(self):
        value=proc(3,OWN,120,None);value["executable_permission_denied"]=True
        result=f.ObserverTests().run_observer(display_fixture(),reader=lambda pid:value)
        self.assertEqual(len(result["contexts"]),1)
        self.assertEqual(result["system_displays"],[])
    def test_foreign_compute_denied_exe_remains_foreign(self):
        value=proc(2,OTHER,1000,None);value["executable_permission_denied"]=True
        with self.assertRaises(ObservationError):f.ObserverTests().run_observer(fixture([(0,2)]),reader=lambda pid:value)

if __name__=="__main__":unittest.main()
