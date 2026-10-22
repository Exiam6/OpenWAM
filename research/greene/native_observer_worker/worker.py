"""Read-only observer child; CPU preparation, not a native experiment entrypoint.

Caller must freeze this worker/dependencies and supervise through sample().
No native libraries, allocations, process signals, service or queue mutations.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import re
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native_observer_prep"))
from observer import observe, ObservationError
from inventory import UUID, unit_path

GLOBAL_PAUSE = "/home/zifanz4/.local/state/openwam-selfcheck/paused.json"
KEYS = {"host", "device_map", "assigned", "own_group", "gdm_uid", "seconds", "pause_paths"}


class WorkerError(RuntimeError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence


def check(ok, message):
    if not ok:
        raise WorkerError(message)


def unambiguous(pairs):
    result = {}
    for key, value in pairs:
        check(key not in result, "Duplicate configuration key")
        result[key] = value
    return result


def guard(paths):
    for name in paths:
        check(not os.path.lexists(name), "Paused: " + name)


def load_config(path, digest):
    guard([GLOBAL_PAUSE])
    check(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, "Invalid configuration hash")
    # Bound reads even if the file grows concurrently. The verified bytes are
    # the only configuration used for this invocation.
    with Path(path).open("rb") as handle:
        raw = handle.read(65537)
    check(len(raw) <= 65536, "Configuration exceeds byte limit")
    check(hashlib.sha256(raw).hexdigest() == digest, "Configuration hash mismatch")
    config = json.loads(raw, object_pairs_hook=unambiguous)
    check(type(config) is dict and set(config) == KEYS, "Unexpected configuration schema")
    check(type(config["host"]) is str and bool(config["host"]), "Missing expected host")
    mapping = config["device_map"]
    check(type(mapping) is dict and set(mapping) == {str(i) for i in range(8)}, "Expected eight-card map")
    check(all(type(v) is str and UUID.fullmatch(v) for v in mapping.values()), "Invalid device UUID")
    check(len(set(mapping.values())) == 8 and config["assigned"] in mapping.values(), "Invalid assignment")
    check(type(config["own_group"]) is str, "Missing frozen dispatcher cgroup")
    unit_path(config["own_group"])
    check(type(config["gdm_uid"]) is int and config["gdm_uid"] > 0, "Invalid frozen display UID")
    seconds = config["seconds"]
    check(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 4,
          "Invalid inner observation deadline")
    paths = config["pause_paths"]
    check(type(paths) is list and 1 <= len(paths) <= 64, "Invalid pause paths")
    check(all(type(v) is str and Path(v).is_absolute() and str(Path(v)) == v
              and ".." not in Path(v).parts for v in paths), "Invalid pause path")
    check(len(set(paths)) == len(paths) and GLOBAL_PAUSE in paths, "Mandatory global pause missing")
    config["device_map"] = {int(k): v for k, v in mapping.items()}
    return config


def run(config):
    guard(config["pause_paths"])
    check(socket.gethostname() == config["host"], "Worker host mismatch")
    own = Path("/proc/self/cgroup").read_text()
    check(unit_path(own) == unit_path(config["own_group"]), "Worker dispatcher unit mismatch")
    check(pwd.getpwnam("gdm").pw_uid == config["gdm_uid"], "Worker display UID mismatch")
    guard(config["pause_paths"])
    result = observe(config["host"], config["device_map"], config["assigned"],
                     config["own_group"], config["gdm_uid"], seconds=config["seconds"])
    # A pause arriving during a query invalidates even an otherwise safe sample.
    try:
        guard(config["pause_paths"])
        check(socket.gethostname() == config["host"], "Worker host changed")
        check(unit_path(Path("/proc/self/cgroup").read_text()) == unit_path(config["own_group"]),
              "Worker dispatcher unit changed")
    except Exception as exc:
        raise WorkerError(str(exc), evidence=result) from exc
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    try:
        result = run(load_config(args.config, args.sha256))
        sys.stdout.write(json.dumps(result, allow_nan=False) + "\n")
        return 0
    except Exception as exc:
        failure = {"worker_error": repr(exc), "complete": False}
        if isinstance(exc, ObservationError):
            failure["observer_evidence"] = exc.evidence
        if isinstance(exc, WorkerError) and exc.evidence is not None:
            failure["rejected_observation"] = exc.evidence
        sys.stdout.write(json.dumps(failure, allow_nan=False) + "\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
