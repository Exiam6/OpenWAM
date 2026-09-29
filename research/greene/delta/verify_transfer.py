#!/usr/bin/env python3
"""Verify a non-main delta transfer (e.g. wan48) and write its record.

1. Every entry of --frozen (hashes copied from frozen study records in git,
   independent of the source side) must appear in the pack's SHA256SUMS with the
   SAME hash — so the sender can't substitute a different checkpoint.
2. Every file in the pack's SHA256SUMS is re-hashed on disk under --dest.
Files hashed only at pack time (no prior record) are listed as such.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys


def sums(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                h, rel = line.rstrip("\n").split(None, 1)
                rel = rel.lstrip("*")
                out[rel[2:] if rel.startswith("./") else rel] = h.lower()
    return out


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(64 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--handoff", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--pack-head", default="")
    a = ap.parse_args()

    frozen, got = sums(a.frozen), sums(os.path.join(a.handoff, "SHA256SUMS"))
    bad = [r for r, h in frozen.items() if got.get(r) != h]
    if bad:
        sys.exit("FROZEN HASH MISMATCH / MISSING: %s" % bad)
    print("frozen hashes: %d/%d match the pack manifest" % (len(frozen), len(frozen)))

    failures, total = [], 0
    for rel, h in sorted(got.items()):
        p = os.path.join(a.dest, rel)
        if not os.path.isfile(p) or sha(p) != h:
            failures.append(rel)
        else:
            total += os.path.getsize(p)
    if failures:
        sys.exit("ON-DISK MISMATCH: %d files, e.g. %s" % (len(failures), failures[:10]))
    print("ALL FILES VERIFIED: %d files, %.2f GB" % (len(got), total / 1e9))

    rec = {
        "verified_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "target_root": a.dest,
        "pack_head": a.pack_head or None,
        "file_count": len(got),
        "total_bytes": total,
        "frozen_hash_files": sorted(frozen),
        "frozen_hash_source": os.path.relpath(a.frozen, os.path.dirname(os.path.dirname(a.record))),
        "hashed_at_pack_only": len(got) - len(frozen),
        "all_sha256_verified": True,
        "note": "Frozen-hash files match hashes recorded by the original study; the rest "
                "(configs, training data) are verified for transport only, against hashes "
                "computed by the sender at pack time.",
    }
    os.makedirs(os.path.dirname(a.record), exist_ok=True)
    with open(a.record, "w") as fh:
        json.dump(rec, fh, indent=2)
        fh.write("\n")
    print("record:", a.record)


if __name__ == "__main__":
    main()
