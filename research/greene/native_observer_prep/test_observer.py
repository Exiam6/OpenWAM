import copy
import json
import itertools
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from observer import ObservationError, identity, observe, system_display, XORG
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native_inventory_prep"))
from test_inventory import MAP, OWN, OTHER, fixture, xml, csv, OPERATIONS, Supervisor


def proc(pid=2, group=OWN, uid=1000, exe="/usr/bin/python3"):
    return dict(pid=pid, start_ticks=1, cgroup=group, uids=(uid,)*4, executable=exe)

def display_fixture():
    root = fixture([(0, 3)])
    p = root.find("./gpu/processes/process_info")
    p.find("type").text = "G"
    p.find("process_name").text = XORG
    return root

class ObserverTests(unittest.TestCase):
    def run_observer(self, root=None, reader=None, runner=None, **kwargs):
        replies = iter([xml(fixture() if root is None else root), csv()])
        def fake(argv, **kw):
            self.assertGreater(kw["timeout"], 0)
            self.assertLessEqual(kw["timeout"], 4)
            return SimpleNamespace(stdout=next(replies), stderr="", returncode=0)
        return observe("host", MAP, MAP[0], OWN, 120, runner=runner or fake,
                       hostname=kwargs.pop("hostname", lambda: "host"),
                       read_identity=reader or proc, **kwargs)
    def test_complete_empty_raw_evidence(self):
        result = self.run_observer()
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["evidence"]["queries"]), 2)
        json.dumps(result)
    def test_wrong_host_no_query(self):
        calls=[]
        with self.assertRaises(ObservationError):
            self.run_observer(hostname=lambda:"wrong", runner=lambda *a,**k:calls.append(1))
        self.assertEqual(calls, [])
    def test_query_failure_retains_raw(self):
        with self.assertRaises(ObservationError) as cm:
            self.run_observer(runner=lambda *a,**k:SimpleNamespace(stdout="raw",stderr="denied",returncode=9))
        self.assertEqual(cm.exception.evidence["queries"][0]["stderr"], "denied")
    def test_timeout_retains_partial_serializable(self):
        def timeout(*a,**k):
            raise subprocess.TimeoutExpired(a[0],k["timeout"],output=b"partial",stderr=b"late")
        with self.assertRaises(ObservationError) as cm:
            self.run_observer(runner=timeout)
        self.assertTrue(cm.exception.evidence["queries"][0]["timed_out"])
        json.dumps(cm.exception.evidence)
    def test_total_deadline_rejects_late_return(self):
        with self.assertRaises(ObservationError):
            self.run_observer(clock=iter([0,0,0,5,5,5]).__next__)
    def test_remaining_timeout_decreases(self):
        calls=[]
        replies=iter([xml(fixture()),csv()])
        def runner(*a,**k):
            calls.append(k["timeout"])
            return SimpleNamespace(stdout=next(replies),stderr="",returncode=0)
        t=itertools.count(0,.1)
        self.run_observer(runner=runner, clock=t.__next__)
        self.assertGreater(calls[0],calls[1])
    def test_invalid_budgets(self):
        for seconds in (0,-1,11,float("nan"),float("inf"),True):
            with self.subTest(seconds=seconds),self.assertRaises(ObservationError):
                self.run_observer(seconds=seconds)
    def test_trusted_root_and_gdm_display(self):
        for uid in (0,120):
            result=self.run_observer(display_fixture(), reader=lambda pid:proc(pid,OTHER,uid,XORG))
            self.assertEqual(len(result["system_displays"]),1)
            self.assertEqual(result["contexts"],[])
    def test_display_spoof_rejected(self):
        for uid,exe in ((1000,XORG),(0,"/tmp/Xorg")):
            with self.subTest(uid=uid,exe=exe),self.assertRaises(ObservationError):
                self.run_observer(display_fixture(),reader=lambda pid:proc(pid,OTHER,uid,exe))
    def test_compute_root_not_exempt(self):
        with self.assertRaises(ObservationError):
            self.run_observer(fixture([(0,3)]),reader=lambda pid:proc(pid,OTHER,0,XORG))
    def test_mixed_uid_not_exempt(self):
        value=proc(3,OTHER,0,XORG);value["uids"]=(0,120,120,120)
        with self.assertRaises(ObservationError):
            self.run_observer(display_fixture(),reader=lambda pid:value)
    def test_own_display_never_hidden(self):
        result=self.run_observer(display_fixture(),reader=lambda pid:proc(pid,OWN,0,XORG))
        self.assertEqual(len(result["contexts"]),1)
        self.assertEqual(result["system_displays"],[])
    def test_peer_context_preserved(self):
        result=self.run_observer(fixture([(1,2)]))
        self.assertEqual(result["contexts"][0]["uuid"],MAP[1])
    def test_integration_peer_stops_native_operations(self):
        calls=[]
        operations={name:lambda name=name:calls.append(name) or True for name in OPERATIONS}
        result=Supervisor().run(operations,lambda:self.run_observer(fixture([(1,2)])),
                                lambda:None,MAP[0],seconds=1,sample_timeout=.1,interval=.001)
        self.assertEqual(result["status"],"failed")
        self.assertEqual(calls,[])
        self.assertIn("Peer GPU context",str(result))
    def test_real_subprocess_timeout(self):
        def runner(argv,**kwargs):
            return subprocess.run([sys.executable,"-c","import time; time.sleep(1)"],**kwargs)
        with self.assertRaises(ObservationError) as cm:
            self.run_observer(seconds=.03,runner=runner)
        self.assertTrue(cm.exception.evidence["queries"][0]["timed_out"])
    def test_identity_changes_rejected(self):
        calls=iter([proc(),dict(proc(),start_ticks=2)])
        with self.assertRaises(ObservationError):
            self.run_observer(fixture([(0,2)]),reader=lambda pid:next(calls))
    def test_identity_changes_across_cards_rejected(self):
        calls=iter([proc(),dict(proc(),start_ticks=2)])
        with self.assertRaises(ObservationError):
            self.run_observer(fixture([(0,2),(1,2)]),reader=lambda pid:next(calls))
    def test_unavailable_proc_rejected(self):
        with tempfile.TemporaryDirectory() as d,self.assertRaises(RuntimeError):
            identity(2,Path(d))
    def test_trusted_proc_fields_and_reuse(self):
        stat=lambda ticks:"2 (test) S "+"0 "*18+str(ticks)
        for after in (1,2):
            with patch.object(Path,"read_text",side_effect=[stat(1),OWN,"Uid: 120 120 120 120",stat(after)]),patch.object(Path,"resolve",return_value=Path(XORG)):
                if after==1:
                    self.assertEqual(identity(2)["uids"],(120,)*4)
                else:
                    with self.assertRaises(RuntimeError):identity(2)

if __name__ == "__main__":unittest.main()
