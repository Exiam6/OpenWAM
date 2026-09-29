#!/usr/bin/env python3
"""Restore portable endtoend aliases after the standard Greene unpack job."""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--robotwin", type=Path, required=True)
    ap.add_argument("--manifest", type=Path,
                    default=Path(__file__).with_name("portable-links.json"))
    a = ap.parse_args()
    dest, robotwin = a.dest.resolve(), a.robotwin.resolve()
    if not dest.is_dir() or not robotwin.is_dir():
        raise SystemExit("Both the unpacked destination and Greene RoboTwin directory must exist")
    links = json.loads(a.manifest.read_text())
    plan = []
    for entry in links:
        rel, target_rel = Path(entry["path"]), Path(entry["target"])
        if any(p.is_absolute() or ".." in p.parts for p in (rel, target_rel)):
            raise SystemExit("Unsafe relative path in link manifest")
        p = dest / rel
        if not p.parent.resolve().is_relative_to(dest):
            raise SystemExit("Link parent escapes destination: %s" % p)
        root = {"destination": dest, "robotwin": robotwin}[entry["root"]]
        target = root / target_rel
        if p.is_symlink():
            if p.resolve() != target.resolve():
                raise SystemExit("Refusing to replace existing different link: %s" % p)
        elif p.exists():
            raise SystemExit("Refusing to replace existing file or directory: %s" % p)
        plan.append((p, target))
    missing = set()
    for p, target in plan:
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.is_symlink():
            p.symlink_to(os.path.relpath(target, p.parent))
        if not target.exists():
            missing.add(str(target))
    print("Restored/verified %d portable links" % len(plan))
    if missing:
        print("Targets absent on Greene (check before evaluation):")
        print("\n".join(sorted(missing)))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
