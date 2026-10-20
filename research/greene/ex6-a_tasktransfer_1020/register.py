"""Submit a verified pilot to the existing idle-history controller exactly once."""
import argparse
import json
from pathlib import Path
import sys

from run_task import ROOT, admission, sha


def register(task):
    p = admission(task)
    verified = json.loads((ROOT / "input-verification.json").read_text())
    assert verified["passed"] and verified["tests_passed"] == 5
    assert {(a["name"], a["sha256"]) for a in p["arms"]} == {
        (a["arm"], a["sha256"]) for a in verified["checkpoints"] if a["verified"]}
    sys.path.insert(0, str(ROOT.parent / "nightly-gpu/code-v2"))
    from policy import validate_job
    protocol = ROOT / "protocol.json"
    inputs = dict(p["code_sha256"])
    inputs[str(protocol)] = sha(protocol)
    job = {
        "id": f"ex6a-{task.replace('_', '-')}",
        "status": "ready",
        "scientific_identity": f"{ROOT.name}/transfer-pilot/{task}/seed42",
        "max_seconds": p["job_max_seconds"],
        "authorization_evidence": str(protocol),
        "authorization_sha256": sha(protocol),
        "pause_paths": [str(ROOT / "paused.json"),
                        "/home/zifanz4/.local/state/openwam-selfcheck/paused.json"],
        "host_commands": {"gpu01": {
            "argv": ["/usr/bin/python3", str(ROOT / "run_task.py"), "--task", task],
            "admission_argv": ["/usr/bin/python3", str(ROOT / "run_task.py"), "--task", task,
                               "--check-admission"],
            "cwd": str(ROOT),
            "input_sha256": inputs,
            "outputs_absent": [str(ROOT / "tasks" / task)],
        }},
    }
    validate_job(job)
    target = ROOT.parent / "nightly-gpu/jobs" / f"{job['id']}.json"
    # Atomic, no-overwrite publication: the continuously running controller
    # must never observe a partially written descriptor.
    temporary = target.with_suffix(".prepared")
    with temporary.open("x") as handle:
        handle.write(json.dumps(job, indent=2) + "\n")
    import os
    os.link(temporary, target)
    temporary.unlink()
    print(json.dumps({"registered": task, "path": str(target)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    register(args.task)
