"""Real read-only NVML observation, not a dispatcher worker or native run."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import pwd
import socket
import sys

base=Path(sys.argv[1])
sys.path.insert(0,str(base/"native_observer_prep"))
from observer import observe, ObservationError
pauses=["/home/zifanz4/.local/state/openwam-selfcheck/paused.json",
        "/home/zifanz4/openwam-runtime/ex8-d_nativefull_1021/paused.json",
        "/home/zifanz4/openwam-runtime/ex8-k_glfwprefix_1021/paused.json"]
def guard():
    if any(os.path.lexists(p) for p in pauses):raise RuntimeError("Paused")
guard()
root=Path("/home/zifanz4/openwam-runtime/nightly-gpu")
raw=(root/"policy.json").read_bytes();policy=json.loads(raw)
current=json.loads((root/"latest.json").read_text())
assert (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(current["time"])).total_seconds()<25
host=socket.gethostname();assert host==policy["hosts"]["gpu01"]["hostname"]
mapping={int(k):v for k,v in policy["hosts"]["gpu01"]["gpus"].items()}
# Explicit inactive unit: probes ownership classification only, never claims
# this read-only process runs inside a dispatcher service.
controller_group=Path("/proc",str(current["pid"]),"cgroup").read_text().strip()
intent=controller_group.rsplit("/",1)[0]+"/openwam-nightly-job-ex8k-native-glfw-prefix.service\n"
result=dict(scope="REAL_READ_ONLY_NVML_WITH_INACTIVE_DISPATCHER_UNIT",
            time=dt.datetime.now(dt.timezone.utc).isoformat(),
            observer_base=str(base),policy_sha256=hashlib.sha256(raw).hexdigest(),
            own_group_intent=intent,assigned=mapping[2],native_executed=False,
            dispatcher_service_validated=False,native_L2_L3_passed=False,
            full_release_allowed=False,GPU_seconds=0)
try:
    result["observation"]=observe(host,mapping,mapping[2],intent,pwd.getpwnam("gdm").pw_uid,seconds=2)
    result["passed"]=True
except ObservationError as exc:
    result.update(passed=False,error=repr(exc),evidence=exc.evidence)
guard()
with Path(sys.argv[2]).open("x") as handle:json.dump(result,handle,indent=2);handle.write("\n")
print(json.dumps({k:v for k,v in result.items() if k not in ["observation","evidence"]}))
