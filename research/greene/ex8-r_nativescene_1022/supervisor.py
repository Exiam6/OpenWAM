"""Ordered native boundary supervision; caller enforces frozen registration.
Observer contract: bounded call, complete host inventory, own-cgroup contexts only.
A separately reviewed frozen adapter must establish that contract before GPU use.
"""
import copy
import math
import os
import signal
import threading
import time

OPERATIONS = ("prewarm", "sapien_import", "explicit_device", "helper_load",
              "native_glfwInit", "native_headless_support", "native_loader",
              "native_getproc", "native_dispatcher", "native_layer_symbol",
              "native_layer_count", "native_layer_data", "instance_helper_load", "native_Instance_Get", "native_scene")

class Supervisor:
    def __init__(self):
        self.used = False

    def run(self, operations, observe, guard, assigned, *, seconds,
            sample_timeout, interval=0.02):
        if self.used:
            raise RuntimeError("Single-use supervisor; never retry")
        self.used = True
        assert tuple(operations) == OPERATIONS, "Exact ordered native operations required"
        assert threading.current_thread() is threading.main_thread()
        assert assigned and math.isfinite(seconds) and 0 < seconds <= 480
        assert math.isfinite(sample_timeout) and 0 < sample_timeout < seconds
        assert math.isfinite(interval) and 0 < interval < sample_timeout
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "Existing timer"
        start = time.monotonic()
        lock = threading.Lock()
        done = threading.Event()
        samples, errors, results = [], [], {}
        wakeup_enabled = True
        phase = "before_prewarm"
        result = dict(status="running", scored_episodes=0, full_release_allowed=False,
                      native_L2_L3_passed=False, full_paper_reproduced=False)

        def stop(*_):
            raise RuntimeError("Monitor requested stop: " + repr(errors))

        def expired(*_):
            raise TimeoutError("Bounded supervisor expired")

        old_usr = signal.signal(signal.SIGUSR1, stop)
        old_alarm = signal.signal(signal.SIGALRM, expired)

        def monitor():
            while not done.is_set():
                try:
                    guard()
                    with lock:
                        observed_phase = phase
                    row = copy.deepcopy(observe())
                    assert row.get("complete") is True, "Incomplete inventory"
                    assert isinstance(row.get("contexts"), list), "Missing contexts"
                    assert all(isinstance(c.get("uuid"), str) and c["uuid"]
                               for c in row["contexts"]), "Malformed context"
                    with lock:
                        row.update(phase=observed_phase, finished_phase=phase,
                                   stable_phase=(observed_phase == phase),
                                   monotonic=time.monotonic())
                        samples.append(row)
                    assert not any(c["uuid"] != assigned for c in row["contexts"]), "Peer GPU context"
                except BaseException as exc:
                    with lock:
                        errors.append(repr(exc))
                        if wakeup_enabled and not done.is_set():
                            done.set()
                            os.kill(os.getpid(), signal.SIGUSR1)
                    return
                done.wait(interval)

        def checkpoint(name):
            nonlocal phase
            guard()
            with lock:
                phase = name
            until = min(start + seconds, time.monotonic() + sample_timeout)
            while True:
                if errors or done.is_set():
                    raise RuntimeError("Monitor stopped: " + repr(errors))
                with lock:
                    count = sum(s["phase"] == name and s["stable_phase"] for s in samples)
                if count >= 3:
                    return
                if time.monotonic() >= until:
                    raise TimeoutError("Missing three complete phase samples: " + name)
                done.wait(interval)

        thread = threading.Thread(target=monitor, daemon=True)
        try:
            signal.setitimer(signal.ITIMER_REAL, seconds)
            guard()
            thread.start()
            checkpoint("before_prewarm")
            for name, operation in operations.items():
                guard()
                with lock:
                    phase = name
                value = operation()
                results[name] = value
                assert value is True, "Native boundary failed: " + name
                checkpoint("after_" + name)
            guard()
            assert not errors, errors
            assert any(c["uuid"] == assigned for s in samples for c in s["contexts"]), "No assigned witness"
            result["status"] = "supervisor_sequence_passed"
        except BaseException as exc:
            result.update(status="failed", error=repr(exc))
        finally:
            # Disable wakeups before restoring handlers, even if an observer stalls.
            signal.signal(signal.SIGUSR1, lambda *_: None)
            with lock:
                wakeup_enabled = False
                done.set()
            if thread.ident is not None:
                try:
                    thread.join(timeout=sample_timeout)
                except BaseException as exc:
                    errors.append("Monitor cleanup: " + repr(exc))
                if thread.is_alive():
                    errors.append("Observer did not stop; bounded adapter contract violated")
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGUSR1, old_usr)
            signal.signal(signal.SIGALRM, old_alarm)
            if errors:
                result.update(status="failed", error="Context monitor failed")
            result.update(samples=copy.deepcopy(samples), monitor_errors=list(errors), results=dict(results),
                          failed_or_final_phase=phase,
                          occupied_seconds=time.monotonic() - start)
        return result
