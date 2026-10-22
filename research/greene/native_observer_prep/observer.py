"""CPU preparation; no native imports or command-line entrypoint."""
from pathlib import Path
import math
import socket
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native_inventory_prep"))
from inventory import InventoryError, require, snapshot, start_ticks

SMI = "/usr/bin/nvidia-smi"
XORG = "/usr/lib/xorg/Xorg"

class ObservationError(InventoryError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence

def identity(pid, proc=Path("/proc")):
    """Read trusted OS fields, bracketing all reads with start ticks."""
    try:
        root = proc / str(pid)
        first = start_ticks((root / "stat").read_text(), pid)
        group = (root / "cgroup").read_text()
        status = (root / "status").read_text()
        rows = [r.split()[1:] for r in status.splitlines() if r.startswith("Uid:")]
        require(len(rows) == 1 and len(rows[0]) == 4, "Missing UID identity")
        uids = tuple(int(v) for v in rows[0])
        executable = str((root / "exe").resolve(strict=True))
        second = start_ticks((root / "stat").read_text(), pid)
        require(first == second, "PID reused during identity read")
        return dict(pid=pid, start_ticks=first, cgroup=group, uids=uids, executable=executable)
    except (OSError, ValueError, IndexError) as exc:
        raise InventoryError("Unavailable trusted process identity") from exc

def system_display(row, process, gdm_uid):
    # Ownership is checked by snapshot before this exemption is considered.
    return (isinstance(gdm_uid, int) and not isinstance(gdm_uid, bool) and gdm_uid > 0
            and row["type"] == "G" and row["name"] == XORG
            and process["executable"] == XORG
            and len(process["uids"]) == 4
            and (all(uid == 0 for uid in process["uids"])
                 or all(uid == gdm_uid for uid in process["uids"])))

def observe(expected_host, expected_map, assigned, own_group, gdm_uid,
            seconds=4.0, *, clock=time.monotonic, hostname=socket.gethostname,
            runner=subprocess.run, read_identity=identity):
    """One deadline for both queries, parsing and ownership reads.

    A slow /proc or kernel syscall cannot be interrupted here; a future
    production caller MUST enforce an independent outer sample deadline.
    Raw failure evidence is attached to ObservationError for caller persistence.
    """
    evidence = dict(queries=[], identities={})
    began = clock()
    def remaining():
        left = seconds - (clock() - began)
        require(math.isfinite(left) and left > 0, "Observation deadline exceeded")
        return left
    try:
        require(isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
                and math.isfinite(seconds) and 0 < seconds <= 10, "Invalid sample budget")
        require(hostname() == expected_host, "Observer host mismatch")
        evidence.update(host=expected_host, monotonic_start=began)
        outputs = []
        for argv in ((SMI, "-q", "-x"),
                     (SMI, "--query-gpu=index,uuid", "--format=csv,noheader")):
            record = dict(argv=list(argv), started=clock())
            evidence["queries"].append(record)
            result = runner(list(argv), capture_output=True, text=True, check=False,
                            timeout=remaining())
            record.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode,
                          finished=clock())
            remaining()
            require(result.returncode == 0, "GPU inventory query failed")
            outputs.append(result.stdout)
        cache = {}
        def group(pid):
            remaining()
            value = read_identity(pid)
            remaining()
            require(pid not in cache or cache[pid] == value, "Process identity changed between contexts")
            cache[pid] = value
            evidence["identities"][str(pid)] = value
            return value["cgroup"]
        result = snapshot(*outputs, expected_map, assigned, own_group, group,
                          lambda row: system_display(row, cache[row["pid"]], gdm_uid))
        remaining()
        # Recheck each unique identity after parsing and both queries.
        for pid, previous in cache.items():
            require(read_identity(pid) == previous, "Process identity changed across sample")
            remaining()
        require(hostname() == expected_host, "Observer host changed")
        remaining()
        evidence["monotonic_end"] = clock()
        result["evidence"] = evidence
        return result
    except Exception as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            decode = lambda v: v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
            record.update(stdout=decode(exc.stdout), stderr=decode(exc.stderr), timed_out=True)
        evidence["monotonic_end"] = clock()
        evidence["error"] = repr(exc)
        raise ObservationError(str(exc), evidence) from exc
