"""Read-only CPU audit of trusted /proc fields; no driver/native calls."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import pwd
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native_observer_prep"))
from observer import identity
from inventory import start_ticks, unit_path

ROOT = Path("/home/zifanz4/openwam-runtime/nightly-gpu")
PAUSES = ["/home/zifanz4/.local/state/openwam-selfcheck/paused.json",
          "/home/zifanz4/openwam-runtime/ex8-d_nativefull_1021/paused.json",
          "/home/zifanz4/openwam-runtime/ex8-k_glfwprefix_1021/paused.json"]

def guard():
    if any(os.path.lexists(p) for p in PAUSES):
        raise RuntimeError("Paused")

def read(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()

def audit():
    guard()
    began = time.monotonic()
    policy, policy_hash = read(ROOT / "policy.json")
    inventory, inventory_hash = read(ROOT / "latest.json")
    now = dt.datetime.now(dt.timezone.utc)
    age = (now - dt.datetime.fromisoformat(inventory["time"])).total_seconds()
    if not 0 <= age <= 25:
        raise RuntimeError("Stale controller inventory")
    host = socket.gethostname()
    host_keys = [k for k,v in policy["hosts"].items() if v["hostname"] == host]
    if len(host_keys) != 1:
        raise RuntimeError("Unexpected local host")
    key = host_keys[0]
    pids = {os.getpid(): ["self"], inventory["pid"]: ["controller"]}
    for index, card in inventory["hosts"][key].items():
        if card["uuid"] != policy["hosts"][key]["gpus"][index]:
            raise RuntimeError("Inventory device mismatch")
        for source in ("processes", "system_display_processes"):
            for row in card.get(source, []):
                pids.setdefault(row["pid"], []).append(dict(index=index, source=source, row=row))
    result = dict(scope="CPU_ONLY_REAL_LOCAL_PROC_VISIBILITY", time=now.isoformat(),
                  hostname=host, uid=os.getuid(), gdm_uid=pwd.getpwnam("gdm").pw_uid,
                  policy_sha256=policy_hash, inventory_sha256=inventory_hash,
                  inventory_time=inventory["time"], inventory_age_seconds=age,
                  GPU_seconds=0, identities=[], dispatcher_service_validated=False,
                  native_L2_L3_passed=False, full_release_allowed=False)
    for pid, sources in sorted(pids.items()):
        guard()
        if time.monotonic() - began > 60:
            raise RuntimeError("CPU audit deadline")
        entry = dict(pid=pid, sources=sources, fields={})
        root = Path("/proc") / str(pid)
        for name, callback in {"stat_before": lambda: (root/"stat").read_text(),
                               "cgroup": lambda: (root/"cgroup").read_text(),
                               "uids": lambda: [r for r in (root/"status").read_text().splitlines() if r.startswith("Uid:")],
                               "executable": lambda: str((root/"exe").resolve(strict=True)),
                               "stat_after": lambda: (root/"stat").read_text()}.items():
            try:
                entry["fields"][name] = dict(ok=True,value=callback())
            except Exception as exc:
                entry["fields"][name] = dict(ok=False,error=repr(exc),errno=getattr(exc,"errno",None))
        before, after = [entry["fields"][n] for n in ("stat_before","stat_after")]
        entry["stable_start_ticks"] = bool(before["ok"] and after["ok"] and
            start_ticks(before["value"],pid) == start_ticks(after["value"],pid))
        try:
            entry["unchanged_identity"] = dict(ok=True,value=identity(pid))
        except Exception as exc:
            entry["unchanged_identity"] = dict(ok=False,error=repr(exc),cause=repr(exc.__cause__))
        result["identities"].append(entry)
    result["passed"] = all(e["stable_start_ticks"] and e["unchanged_identity"]["ok"] for e in result["identities"])
    result["failed_pids"] = [e["pid"] for e in result["identities"] if not e["unchanged_identity"]["ok"]]
    result["elapsed_seconds"] = time.monotonic()-began
    guard()
    return result

def main():
    result=audit()
    destination=Path(sys.argv[1])
    guard()
    destination.parent.mkdir(parents=True,exist_ok=True)
    guard()
    with destination.open("x") as handle:
        json.dump(result,handle,indent=2);handle.write("\n")
    print(json.dumps({k:result[k] for k in ["scope","time","passed","failed_pids","elapsed_seconds"]}))

if __name__=="__main__":
    main()
