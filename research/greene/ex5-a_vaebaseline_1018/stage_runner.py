"""Bound a child process group by elapsed time, deadline and pause files."""
import datetime as dt
import os
from pathlib import Path
import signal
import subprocess
import time


def run_child(command, *, log, seconds, deadline, pause_paths=(), env=None, cwd=None, cleanup_seconds=10., poll_seconds=.2, health_check=None, on_started=None):
    assert seconds > cleanup_seconds > 0 and poll_seconds > 0
    started = time.monotonic()
    child = None
    record = dict(command=list(command), started=dt.datetime.now(dt.timezone.utc).isoformat(), limit_seconds=seconds, complete=False)
    def reason():
        if health_check is not None:
            health = health_check()
            if health:
                return str(health)
        if time.monotonic()-started >= seconds-cleanup_seconds:
            return 'elapsed budget reached; cleanup reserve'
        if dt.datetime.now(dt.timezone.utc) >= deadline-dt.timedelta(seconds=cleanup_seconds):
            return 'allocation deadline reached; cleanup reserve'
        for path in pause_paths:
            if Path(path).exists():
                return 'pause requested: '+str(path)
        return None
    try:
        stop = reason()
        if stop:
            record['error'] = stop
            return record
        with Path(log).open('x') as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, env=env, cwd=cwd)
            record['pid'] = child.pid
            if on_started is not None:
                on_started(child.pid)
            while child.poll() is None:
                stop = reason()
                if stop:
                    record['error'] = stop
                    break
                time.sleep(poll_seconds)
            if child.poll() is not None:
                record['exit_code'] = child.returncode
                record['complete'] = child.returncode == 0 and not record.get('error')
                if child.returncode:
                    record['error'] = 'child exited unsuccessfully'
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        if child is not None:
            # Every child starts a new session; only that owned process group is targeted.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=max(.01, cleanup_seconds/2))
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=max(.01, cleanup_seconds/2))
            record['exit_code'] = child.returncode
        record['occupied_seconds'] = time.monotonic()-started
        record['finished'] = dt.datetime.now(dt.timezone.utc).isoformat()
    return record
