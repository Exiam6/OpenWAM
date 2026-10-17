"""Amendment 5 (2026-10-17 03:08 CDT, before any learned policy was scored).

place_object_basket screening stopped at seed 6202023 because RoboTwin's expert raised
AssertionError('target_pose cannot be None for move action.') inside play_once (no grasp pose for
the basket). RoboTwin's own collect_data.py skips a seed on ANY exception from setup_demo/play_once.
Screening now records an exception raised by the expert's play_once as outcome "expert_exception"
(not accepted) and moves to the next fixed candidate. Timeouts, the render-device check and every
replay/render consistency check still stop the task. The place_object_basket attempt is archived
and the task is re-screened from its first candidate. adjust_bottle and handover_block are kept:
none of their attempts raised inside play_once, so the new rule gives them identical outcomes.
"""
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

R = Path("/home/zifanz4/openwam-runtime/rae-factorial-20261010")
TASK = "place_object_basket"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    archive = R / "resource-amendment-20261017-expert-exception"
    assert not archive.exists()
    assert not list((R / "post-claims").glob("evaluate-*.json")), "evaluation already started"
    for task in ["adjust_bottle", "handover_block"]:
        m = json.loads((R / "scenes" / task / "manifest.json").read_text())
        assert m["complete"] and all(a["outcome"] in ("accepted", "expert_infeasible", "scene_unstable") for a in m["attempts"]), task
    archive.mkdir()
    for name in ["collect_fresh.py", "code-freeze.json", "eval-freeze.json"]:
        shutil.copy2(R / name, archive / name)
    failed = archive / "failed-attempt-place_object_basket"
    failed.mkdir()
    shutil.move(str(R / "post-claims" / f"collect-{TASK}.json"), failed / f"collect-{TASK}.json")
    shutil.move(str(R / f"collect-{TASK}.log"), failed / f"collect-{TASK}.log")
    shutil.move(str(R / "scenes" / TASK), failed / "scenes")
    s = (R / "collect_fresh.py").read_text()
    old = "            env.play_once()\n            feasible = bool(env.plan_success and env.check_success())\n"
    new = ('            try:\n'
           '                env.play_once()\n'
           '                expert_error = None\n'
           '            except TimeoutError:\n'
           '                raise\n'
           '            except Exception as exc:  # RoboTwin collect_data.py skips a seed on any expert exception\n'
           '                expert_error = repr(exc)\n'
           '            feasible = expert_error is None and bool(env.plan_success and env.check_success())\n'
           '            if expert_error is not None:\n'
           '                row["error"] = expert_error\n')
    assert s.count(old) == 1
    s = s.replace(old, new)
    old2 = '            row.update(accepted=feasible, outcome="accepted" if feasible else "expert_infeasible")\n'
    new2 = '            row.update(accepted=feasible, outcome="accepted" if feasible else ("expert_exception" if expert_error else "expert_infeasible"))\n'
    assert s.count(old2) == 1
    s = s.replace(old2, new2)
    compile(s, "collect_fresh.py", "exec")
    (R / "collect_fresh.py").write_text(s)
    for name in ["code-freeze.json", "eval-freeze.json"]:
        m = json.loads((R / name).read_text())
        m["files"][str(R / "collect_fresh.py")] = sha(R / "collect_fresh.py")
        m["resource_amendment_expert_exception"] = str(archive)
        (R / name).write_text(json.dumps(m, indent=2) + "\n")
    rec = dict(applied_at=dt.datetime.now(dt.timezone.utc).isoformat(), trigger="expert AssertionError in play_once at place_object_basket seed 6202023",
               change="exception inside the expert's play_once -> outcome expert_exception, candidate skipped (RoboTwin collect_data.py rule)",
               rescreened=[TASK], kept=["adjust_bottle", "handover_block"], collect_fresh_sha256=sha(R / "collect_fresh.py"),
               archived=str(archive), learned_policy_scored=False)
    (R / "resource-amendment-gpu01-expert-exception.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
