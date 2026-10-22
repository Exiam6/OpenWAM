"""CPU preparation; supervise trusted observation commands, never native work."""
import json
import math
import os
import selectors
import signal
import subprocess
import time


class SampleError(RuntimeError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def sample(argv, *, seconds=4.0, max_bytes=1024*1024, env=None):
    """Return JSON and bounded raw evidence, or fail closed.

    Linux-only. Command must be trusted and remain in its new process group.
    Deadline includes normal spawn time; uninterruptible spawn/kernel calls
    still require the dispatcher's independent cgroup/wall-clock supervision.
    Cleanup has a separate 0.2s reap allowance and never waits indefinitely.
    """
    evidence = dict(argv=repr(argv), stdout='', stderr='', cleanup_reaped=False)
    p = None
    sel = selectors.DefaultSelector()
    streams = {'stdout': bytearray(), 'stderr': bytearray()}
    began = time.monotonic()
    evidence['monotonic_start'] = began
    error = None
    value = None
    try:
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or not 0 < seconds <= 10):
            raise ValueError('Invalid sample deadline')
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 1 <= max_bytes <= 1024*1024:
            raise ValueError('Invalid output limit')
        if not isinstance(argv, (list, tuple)) or not argv or any(not isinstance(a, str) for a in argv):
            raise ValueError('Invalid command')
        evidence['argv'] = list(argv)
        deadline = began + seconds
        def remaining():
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError('Independent sample deadline exceeded')
            return left
        p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, start_new_session=True, env=env)
        evidence['pid'] = p.pid
        for name, pipe in [('stdout', p.stdout), ('stderr', p.stderr)]:
            os.set_blocking(pipe.fileno(), False)
            sel.register(pipe, selectors.EVENT_READ, name)
        terminal = None
        while sel.get_map() or terminal is None:
            remaining()
            # WNOWAIT keeps our child unreaped, preventing PID/PGID reuse
            # until dedicated-group cleanup has completed.
            terminal = os.waitid(os.P_PID, p.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            for key, _ in sel.select(min(.02, remaining())):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    sel.unregister(key.fileobj)
                    continue
                available = max_bytes - sum(len(v) for v in streams.values())
                streams[key.data].extend(chunk[:available])
                if len(chunk) > available:
                    raise ValueError('Combined output limit exceeded')
        remaining()
        if terminal.si_code != os.CLD_EXITED or terminal.si_status != 0:
            raise ValueError('Observation worker exited unsuccessfully')
        value = json.loads(bytes(streams['stdout']))
        if not isinstance(value, dict):
            raise ValueError('Observation must be a JSON object')
        remaining()
    except Exception as exc:
        error = exc
    finally:
        if p is not None:
            # Only the session created by this call; never observed GPU PIDs.
            try:
                os.killpg(p.pid, signal.SIGKILL)
                evidence['group_kill_requested'] = True
            except ProcessLookupError:
                evidence['group_kill_requested'] = False
            except OSError as exc:
                evidence['cleanup_error'] = repr(exc)
                error = error or exc
            try:
                evidence['returncode'] = p.wait(timeout=.2)
                evidence['cleanup_reaped'] = True
            except subprocess.TimeoutExpired as exc:
                evidence['cleanup_error'] = repr(exc)
                error = error or exc
            for pipe in (p.stdout, p.stderr):
                if pipe:
                    pipe.close()
        sel.close()
        evidence.update({k: bytes(v).decode('utf-8', errors='replace') for k, v in streams.items()})
        evidence['monotonic_end'] = time.monotonic()
    if error is not None:
        evidence['error'] = repr(error)
        raise SampleError(str(error), evidence) from error
    return dict(value=value, evidence=evidence)
