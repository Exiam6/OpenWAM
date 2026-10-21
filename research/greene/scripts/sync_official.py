"""Pull official OpenWAM main without mutating frozen scientific worktrees."""
import datetime as dt
import json
from pathlib import Path
import subprocess

CHECKOUT = Path("/home/zifanz4/openwam-official-main")
RECORDS = Path("/home/zifanz4/openwam-runtime/official-sync")
URL = "https://github.com/OpenWAM-Official/OpenWAM.git"


def git(*args):
    return subprocess.check_output(["git", "-C", str(CHECKOUT), *args],
                                   text=True, stderr=subprocess.STDOUT, timeout=180).strip()


def main():
    RECORDS.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc)
    record = {"time": stamp.isoformat(), "checkout": str(CHECKOUT), "official_url": URL,
              "command": ["git", "-C", str(CHECKOUT), "pull", "--ff-only", "origin", "main"],
              "passed": False}
    try:
        assert git("remote", "get-url", "origin") == URL
        assert not git("status", "--porcelain"), "Official tracking checkout has local edits"
        before = git("rev-parse", "HEAD")
        record.update(before=before, output=git("pull", "--ff-only", "origin", "main"))
        after = git("rev-parse", "HEAD")
        assert after == git("rev-parse", "refs/remotes/origin/main")
        assert not git("status", "--porcelain")
        record.update(passed=True, after=after,
                      changed_files=git("diff", "--name-only", before, after).splitlines(),
                      source_date=git("show", "-s", "--format=%cI", after),
                      note="Official source synchronized. Frozen experiment worktrees are not merged or rebased. New scientific runtimes require a recorded compatibility check.")
    except BaseException as error:
        record["error"] = repr(error)
        raise
    finally:
        dest = RECORDS / (stamp.strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        with dest.open("x") as handle:
            handle.write(json.dumps(record, indent=2) + "\n")
        temporary = RECORDS / "latest.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(RECORDS / "latest.json")
        print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
