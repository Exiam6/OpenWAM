"""Amendment 3 (2026-10-17 02:50 CDT, before any learned policy was loaded for scoring).

collect_fresh.py stopped a whole task when RoboTwin raised UnStableError for a candidate seed
("objects unstable in seed(6200000)"). RoboTwin's own collect_data.py and eval_policy.py treat that
exception as "skip this seed". Screening now records it as outcome "scene_unstable" (not accepted)
and moves to the next fixed candidate, exactly like expert infeasibility. Any other exception still
stops the task. The failed first attempt is archived, not deleted. Candidate range, 20-per-task rule
and every other field are unchanged.
"""
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

R = Path("/home/zifanz4/openwam-runtime/rae-factorial-20261010")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    archive = R / "resource-amendment-20261017-unstable-seed"
    assert not archive.exists()
    assert not list((R / "post-claims").glob("evaluate-*.json")), "evaluation already started"
    archive.mkdir()
    for name in ["collect_fresh.py", "code-freeze.json", "eval-freeze.json"]:
        shutil.copy2(R / name, archive / name)
    failed = archive / "failed-attempt-1"
    failed.mkdir()
    for claim in (R / "post-claims").glob("collect-*.json"):
        shutil.move(str(claim), failed / claim.name)
    for log in R.glob("collect-*.log"):
        shutil.move(str(log), failed / log.name)
    if (R / "scenes").exists():
        shutil.move(str(R / "scenes"), failed / "scenes")
    s = (R / "collect_fresh.py").read_text()
    old = '''        except BaseException as exc:
            row.update(accepted=False, outcome="infrastructure_failure", error=repr(exc))
            raise
'''
    new = '''        except BaseException as exc:
            if type(exc).__name__ == "UnStableError":
                # RoboTwin's native rule (collect_data.py, eval_policy.py): an unstable initial
                # scene is skipped. Decided before any policy runs; treated like expert infeasibility.
                row.update(accepted=False, outcome="scene_unstable", error=repr(exc))
            else:
                row.update(accepted=False, outcome="infrastructure_failure", error=repr(exc))
                raise
'''
    assert s.count(old) == 1
    s = s.replace(old, new)
    compile(s, "collect_fresh.py", "exec")
    (R / "collect_fresh.py").write_text(s)
    for name in ["code-freeze.json", "eval-freeze.json"]:
        m = json.loads((R / name).read_text())
        m["files"][str(R / "collect_fresh.py")] = sha(R / "collect_fresh.py")
        m["resource_amendment_unstable_seed"] = str(archive)
        (R / name).write_text(json.dumps(m, indent=2) + "\n")
    rec = dict(applied_at=dt.datetime.now(dt.timezone.utc).isoformat(), host="shenlong-gpu-01",
               trigger="UnStableError for adjust_bottle seed 6200000 stopped screening as an infrastructure failure",
               change="UnStableError -> outcome scene_unstable, candidate skipped (RoboTwin native rule)",
               collect_fresh_sha256=sha(R / "collect_fresh.py"), archived=str(archive), learned_policy_loaded=False)
    (R / "resource-amendment-gpu01-unstable-seed.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
