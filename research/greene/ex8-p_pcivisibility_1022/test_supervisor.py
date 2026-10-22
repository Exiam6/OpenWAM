"""Adverse-event CPU tests; never import or call native GPU libraries."""
import collections
import signal
import time
import threading
import unittest
from supervisor import OPERATIONS, Supervisor

class Checks(unittest.TestCase):
    def fixture(self, observe=None, guard=None, fault=None, seconds=1.2, sample_timeout=.2):
        calls=[]
        def op(name):
            calls.append(name)
            if fault and name == fault[0]:
                return fault[1]()
            return True
        operations=collections.OrderedDict((n,lambda n=n:op(n)) for n in OPERATIONS)
        supervisor=Supervisor()
        result=supervisor.run(operations,
            observe or (lambda:dict(complete=True,contexts=[dict(uuid="assigned")])),
            guard or (lambda:None),"assigned",seconds=seconds,
            sample_timeout=sample_timeout,interval=.001)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.0,0.0))
        return result,calls,supervisor,operations

    def test_success_and_single_use(self):
        r,c,s,o=self.fixture();self.assertEqual(r["status"],"supervisor_sequence_passed")
        self.assertEqual(c,list(OPERATIONS));self.assertFalse(r["full_release_allowed"])
        self.assertEqual(r["scored_episodes"],0)
        phases=["before_prewarm"]+["after_"+n for n in OPERATIONS]
        for p in phases:
            self.assertGreaterEqual(sum(x["phase"]==p and x["stable_phase"] for x in r["samples"]),3)
        with self.assertRaisesRegex(RuntimeError,"Single-use"):
            s.run(o,lambda:None,lambda:None,"assigned",seconds=1,sample_timeout=.1)

    def test_peer_before_native(self):
        r,c,_,_=self.fixture(observe=lambda:dict(complete=True,contexts=[dict(uuid="peer")]))
        self.assertEqual(r["status"],"failed");self.assertFalse(c)
        self.assertTrue(r["monitor_errors"])

    def test_peer_during_native(self):
        active=[False]
        def op():
            active[0]=True;time.sleep(.3);return True
        def observe():return dict(complete=True,contexts=[dict(uuid="peer" if active[0] else "assigned")])
        r,c,_,_=self.fixture(observe=observe,fault=("native_loader",op))
        self.assertEqual(r["status"],"failed");self.assertEqual(c[-1],"native_loader")
        self.assertEqual(r["failed_or_final_phase"],"native_loader")

    def test_incomplete_inventory(self):
        r,c,_,_=self.fixture(observe=lambda:dict(complete=False,contexts=[]))
        self.assertEqual(r["status"],"failed");self.assertFalse(c)

    def test_malformed_inventory(self):
        for row in [dict(complete=True),dict(complete=True,contexts=[{}])]:
            r,c,_,_=self.fixture(observe=lambda row=row:row)
            self.assertEqual(r["status"],"failed");self.assertFalse(c)

    def test_monitor_exception(self):
        def bad():raise OSError("probe failed")
        r,c,_,_=self.fixture(observe=bad)
        self.assertEqual(r["status"],"failed");self.assertFalse(c)

    def test_pause_before_native(self):
        def guard():raise RuntimeError("paused")
        r,c,_,_=self.fixture(guard=guard)
        self.assertEqual(r["status"],"failed");self.assertFalse(c)

    def test_pause_during_native(self):
        paused=[False]
        def guard():
            if paused[0]:raise RuntimeError("paused")
        def op():paused[0]=True;time.sleep(.3);return True
        r,c,_,_=self.fixture(guard=guard,fault=("native_getproc",op))
        self.assertEqual(r["status"],"failed");self.assertEqual(c[-1],"native_getproc")

    def test_each_native_boundary_failure(self):
        for i,name in enumerate(OPERATIONS):
            with self.subTest(name=name):
                r,c,_,_=self.fixture(fault=(name,lambda:False))
                self.assertEqual(r["status"],"failed");self.assertEqual(c,list(OPERATIONS[:i+1]))
                self.assertEqual(r["failed_or_final_phase"],name)

    def test_native_exception(self):
        def bad():raise ValueError("native failure")
        r,c,_,_=self.fixture(fault=("native_dispatcher",bad))
        self.assertEqual(r["status"],"failed");self.assertEqual(c[-1],"native_dispatcher")

    def test_missing_assigned_witness(self):
        r,_,_,_=self.fixture(observe=lambda:dict(complete=True,contexts=[]))
        self.assertEqual(r["status"],"failed");self.assertIn("No assigned",r["error"])

    def test_wall_alarm_during_native(self):
        r,c,_,_=self.fixture(fault=("prewarm",lambda:time.sleep(.5)),seconds=.08,sample_timeout=.04)
        self.assertEqual(r["status"],"failed");self.assertIn("expired",r["error"])
        self.assertEqual(c,["prewarm"])

    def test_missing_phase_samples(self):
        def slow():time.sleep(.015);return dict(complete=True,contexts=[dict(uuid="assigned")])
        r,c,_,_=self.fixture(observe=slow,sample_timeout=.02)
        self.assertEqual(r["status"],"failed");self.assertFalse(c)
        self.assertIn("Missing three",r["error"])

    def test_phase_crossing_not_counted(self):
        armed=threading.Event();entered=threading.Event();release=threading.Event()
        def observe():
            if armed.is_set():
                armed.clear();entered.set();release.wait(.1)
            return dict(complete=True,contexts=[dict(uuid="assigned")])
        def operation():
            armed.set()
            if not entered.wait(.1):raise RuntimeError("test barrier failed")
            threading.Timer(.01,release.set).start()
            return True
        r,_,_,_=self.fixture(observe=observe,fault=("native_loader",operation))
        self.assertEqual(r["status"],"supervisor_sequence_passed")
        crossing=[s for s in r["samples"] if not s["stable_phase"]]
        self.assertTrue(any(s["phase"]=="native_loader" and s["finished_phase"]=="after_native_loader" for s in crossing))

    def test_order_rejected_before_native(self):
        o=collections.OrderedDict((n,lambda:True) for n in reversed(OPERATIONS))
        with self.assertRaisesRegex(AssertionError,"ordered"):
            Supervisor().run(o,lambda:None,lambda:None,"assigned",seconds=1,sample_timeout=.1)

    def test_existing_timer_preserved(self):
        signal.setitimer(signal.ITIMER_REAL,1)
        try:
            with self.assertRaisesRegex(AssertionError,"Existing timer"):
                self.fixture()
            self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0],0)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)

    def test_delayed_observer_cannot_signal_after_cleanup(self):
        release=threading.Event();exited=threading.Event();signals=[]
        restored_handler=lambda *_:signals.append(True)
        old=signal.signal(signal.SIGUSR1,restored_handler)
        def observe():
            release.wait(.3);exited.set();raise OSError("late observer error")
        try:
            r,c,_,_=self.fixture(observe=observe,sample_timeout=.015)
            self.assertEqual(r["status"],"failed");self.assertFalse(c)
            snapshot=list(r["monitor_errors"])
            release.set();self.assertTrue(exited.wait(.3));time.sleep(.01)
            self.assertFalse(signals);self.assertEqual(r["monitor_errors"],snapshot)
            self.assertIs(signal.getsignal(signal.SIGUSR1),restored_handler)
        finally:
            release.set();signal.signal(signal.SIGUSR1,old)

if __name__=="__main__":unittest.main()
