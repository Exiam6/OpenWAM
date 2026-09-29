#!/usr/bin/env python3
"""Step 3 (Greene): rebuild the 363 original files byte-for-byte from a delta pack.

    python3 unpack_delta.py --pack /scratch/zz4330/openwam-runtime/delta-pack \
        --reference-index .../reference-index.json \
        --dest /scratch/zz4330/openwam-runtime/assets-source \
        --handoff /scratch/zz4330/openwam-runtime/handoff

Order of checks:
  1. PACK_SHA256SUMS — catches corruption on the relay before anything is built;
  2. every reused/packed tensor range is re-hashed as it is written;
  3. every rebuilt file must equal its ORIGINAL sha256 from GPU02's SHA256SUMS,
     else it is deleted and the run fails. (finalize_private_assets.sh then
     re-runs `sha256sum -c` independently over the whole set.)
Re-runnable: files already present with the right sha256 are skipped.
Standard library only.
"""
import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
import time

CHUNK = 64 << 20


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(CHUNK), b""):
            h.update(buf)
    return h.hexdigest()


def copy_range(src_fh, offset, size, out_fh, hashers, what):
    """Copy [offset, offset+size) into out_fh, feeding every hasher in `hashers`."""
    src_fh.seek(offset)
    left = size
    while left:
        buf = src_fh.read(min(CHUNK, left))
        if not buf:
            raise SystemExit("short read from %s" % what)
        for h in hashers:
            h.update(buf)
        out_fh.write(buf)
        left -= len(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--reference-index", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--handoff", required=True)
    a = ap.parse_args()

    log("checking pack integrity (PACK_SHA256SUMS)")
    bad = []
    with open(os.path.join(a.pack, "PACK_SHA256SUMS")) as fh:
        for line in fh:
            digest, rel = line.rstrip("\n").split("  ", 1)
            p = os.path.join(a.pack, rel)
            if not os.path.isfile(p) or file_sha(p) != digest:
                bad.append(rel)
    if bad:
        raise SystemExit("pack corrupted/incomplete, re-copy: %s" % bad[:20])
    log("pack OK")

    with open(os.path.join(a.pack, "manifest.json")) as fh:
        m = json.load(fh)
    with open(a.reference_index) as fh:
        refidx = json.load(fh)
    if refidx["reference_sha256"] != m["reference_sha256"]:
        raise SystemExit("pack was built against a different reference index")
    ref_path = refidx["reference_path_on_greene"]
    if os.path.getsize(ref_path) != refidx["reference_bytes"]:
        raise SystemExit("reference checkpoint size changed: %s" % ref_path)
    ref = refidx["tensors"]

    os.makedirs(a.handoff, exist_ok=True)
    for f in os.listdir(os.path.join(a.pack, "share-eval")):
        shutil.copy2(os.path.join(a.pack, "share-eval", f), os.path.join(a.handoff, f))

    parts = {}
    ref_fh = open(ref_path, "rb")
    done = skipped = 0
    for i, e in enumerate(m["entries"], 1):
        dst = os.path.join(a.dest, e["path"])
        if os.path.isfile(dst) and os.path.getsize(dst) == e["bytes"] \
                and file_sha(dst) == e["sha256"]:
            skipped += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".partial"
        whole = hashlib.sha256()
        if e["kind"] == "plain":
            shutil.copyfile(os.path.join(a.pack, "files", e["path"]), tmp)
            got = file_sha(tmp)
        else:
            log("[%d/%d] rebuilding %s" % (i, len(m["entries"]), e["path"]))
            with open(tmp, "wb") as out:
                head = base64.b64decode(e["header_b64"])
                whole.update(head)
                out.write(head)
                for digest, size, src in e["segments"]:
                    if src == "ref":
                        r = ref[digest]
                        pieces, fhs = [(r["offset"], size)], [ref_fh]
                    else:
                        pieces, fhs = [], []
                        for part, off, psize in m["blobs"][digest]:
                            if part not in parts:
                                parts[part] = open(os.path.join(a.pack, "pack", part), "rb")
                            pieces.append((off, psize))
                            fhs.append(parts[part])
                    th = hashlib.sha256()
                    for fh_, (off, psize) in zip(fhs, pieces):
                        copy_range(fh_, off, psize, out, (th, whole), src)
                    if th.hexdigest() != digest:
                        raise SystemExit("tensor hash mismatch (%s) in %s" % (src, e["path"]))
                tail = base64.b64decode(e["tail_b64"])
                whole.update(tail)
                out.write(tail)
            got = whole.hexdigest()
        if got != e["sha256"]:
            os.remove(tmp)
            raise SystemExit("REBUILD MISMATCH %s: %s != %s" % (e["path"], got, e["sha256"]))
        os.replace(tmp, dst)
        done += 1
    log("rebuilt %d, already-correct %d, total %d — all match original sha256"
        % (done, skipped, len(m["entries"])))


if __name__ == "__main__":
    main()
