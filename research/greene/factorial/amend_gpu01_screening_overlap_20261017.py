"""Scheduler-only amendment 2 (2026-10-17 02:45 CDT): let fresh-scene screening start before training ends.

collect_fresh.py loads no learned policy (expert planner only), so the three screening jobs can use GPUs
that training does not occupy. Evaluation cells still require all six trainings complete and all three
screenings complete. No scientific field changes. Archive, patch, re-hash, record; relaunch separately.
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
    archive = R / "resource-amendment-20261017-screening-overlap"
    assert not archive.exists()
    assert not list((R / "post-claims").glob("*.json")), "post-claims already exist; amend before any post work"
    archive.mkdir()
    for name in ["post_worker.py", "code-freeze.json", "eval-freeze.json"]:
        shutil.copy2(R / name, archive / name)
    s = (R / "post_worker.py").read_text()
    old = '''            if not all(r.get("status") == "complete" for r in rows):
                state("waiting_for_training")
                time.sleep(30)
                continue
            collected = [record(j) for j in jobs[:3]]
            if any(r and r.get("status") == "failed" for r in collected):
                raise RuntimeError("fresh scene preparation failed; no policy scoring")
            eligible = jobs if all(r and r.get("status") == "complete" for r in collected) else jobs[:3]
            job = next((j for j in eligible if record(j) is None), None)
            if job is None:
                allrows = [record(j) for j in jobs]
                if all(r and r.get("status") in ("complete", "failed") for r in allrows):
                    state("finished", records=allrows)
                    return
                state("waiting_for_peer")
                time.sleep(30)
                continue
'''
    new = '''            training_done = all(r.get("status") == "complete" for r in rows)
            collected = [record(j) for j in jobs[:3]]
            if any(r and r.get("status") == "failed" for r in collected):
                raise RuntimeError("fresh scene preparation failed; no policy scoring")
            collected_done = all(r and r.get("status") == "complete" for r in collected)
            # Scene screening uses only the expert planner (no learned policy), so it may start before
            # training finishes; evaluation cells still wait for all trainings and all screenings.
            eligible = jobs if (training_done and collected_done) else jobs[:3]
            job = next((j for j in eligible if record(j) is None), None)
            if job is None:
                allrows = [record(j) for j in jobs]
                if all(r and r.get("status") in ("complete", "failed") for r in allrows):
                    state("finished", records=allrows)
                    return
                state("waiting_for_training" if not training_done else "waiting_for_peer")
                time.sleep(30)
                continue
'''
    assert s.count(old) == 1
    s = s.replace(old, new)
    compile(s, "post_worker.py", "exec")
    (R / "post_worker.py").write_text(s)
    for name in ["code-freeze.json", "eval-freeze.json"]:
        m = json.loads((R / name).read_text())
        m["files"][str(R / "post_worker.py")] = sha(R / "post_worker.py")
        m["resource_amendment_screening_overlap"] = str(archive)
        (R / name).write_text(json.dumps(m, indent=2) + "\n")
    rec = dict(applied_at=dt.datetime.now(dt.timezone.utc).isoformat(), change="screening may precede training completion; evaluation unchanged",
               post_worker_sha256=sha(R / "post_worker.py"), archived=str(archive), scientific_fields_unchanged=True)
    (R / "resource-amendment-gpu01-screening-overlap.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
