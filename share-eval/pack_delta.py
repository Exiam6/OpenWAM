#!/usr/bin/env python3
"""Step 2 (GPU02): pack the private handoff as a byte-exact delta.

Reads the exact 363-file handoff (share-eval/files.txt, SHA256SUMS) and writes
a pack that contains:
  * every non-safetensors file, copied as-is;
  * for every .safetensors file: its original header bytes, plus only the
    tensor byte-ranges whose SHA-256 is NOT in the Greene reference index
    (and not already packed from an earlier checkpoint — the six seeds share
    their frozen modules).
Greene's unpack_delta.py rebuilds each original file byte-for-byte and checks
it against the original SHA256SUMS.

READ-ONLY on the source: nothing under --root is written, moved or deleted.
The source files are re-hashed during packing and the pack aborts if any file
disagrees with SHA256SUMS. Run at low priority next to the live worker:

    nice -n 19 ionice -c3 python3 pack_delta.py \
        --root /home/zifanz4/openwam-runtime \
        --reference reference-index.json \
        --out /home/zifanz4/owam-delta

Standard library only (python >= 3.6).
"""
import argparse
import base64
import hashlib
import json
import os
import shutil
import struct
import sys
import time

CHUNK = 64 << 20


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def norm(p):
    p = p.strip()
    return p[2:] if p.startswith("./") else p


def read_sums(path):
    sums = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            digest, rel = line.split(None, 1)
            sums[norm(rel.lstrip("*"))] = digest.lower()
    return sums


class Pack:
    """Append-only blob store split into parts of about part_size bytes."""

    def __init__(self, outdir, part_size):
        self.dir = os.path.join(outdir, "pack")
        os.makedirs(self.dir)
        self.part_size = part_size
        self.idx = -1
        self.fh = None
        self.hasher = None
        self.part_hashes = {}
        self.blobs = {}          # sha -> [[part_name, offset, size], ...]
        self.bytes = 0
        self._roll()

    def _name(self):
        return "part-%03d.bin" % self.idx

    def _roll(self):
        self._close()
        self.idx += 1
        self.fh = open(os.path.join(self.dir, self._name()), "wb")
        self.hasher = hashlib.sha256()
        self.pos = 0

    def _close(self):
        if self.fh:
            self.fh.close()
            self.part_hashes["pack/" + self._name()] = self.hasher.hexdigest()

    def add(self, digest, src_fh, offset, size):
        """Copy one tensor range; it may span several parts (no part exceeds part_size)."""
        src_fh.seek(offset)
        check = hashlib.sha256()
        pieces, left = [], size
        while left:
            room = self.part_size - self.pos
            if room <= 0:
                self._roll()
                room = self.part_size
            take = min(left, room)
            start, got = self.pos, 0
            while got < take:
                buf = src_fh.read(min(CHUNK, take - got))
                if not buf:
                    raise SystemExit("source shrank while packing")
                check.update(buf)
                self.hasher.update(buf)
                self.fh.write(buf)
                got += len(buf)
            pieces.append([self._name(), start, take])
            self.pos += take
            left -= take
        if check.hexdigest() != digest:
            raise SystemExit("source changed between hash and copy")
        self.blobs[digest] = pieces
        self.bytes += size

    def finish(self):
        self._close()
        self.fh = None


def pack_safetensors(src, ref, pack):
    """Return (recipe, whole_sha256). Single hashing pass, then copy only new ranges."""
    whole = hashlib.sha256()
    with open(src, "rb") as fh:
        head8 = fh.read(8)
        (n,) = struct.unpack("<Q", head8)
        raw = fh.read(n)
        whole.update(head8)
        whole.update(raw)
        header = json.loads(raw)
        base = 8 + n
        items = sorted(
            ((k, v) for k, v in header.items() if k != "__metadata__"),
            key=lambda kv: kv[1]["data_offsets"][0],
        )
        segs, pos = [], 0
        for name, meta in items:
            start, end = meta["data_offsets"]
            if start != pos:
                raise SystemExit("%s: non-contiguous tensor data at %s" % (src, name))
            h = hashlib.sha256()
            left = end - start
            while left:
                buf = fh.read(min(CHUNK, left))
                if not buf:
                    raise SystemExit("%s: truncated at %s" % (src, name))
                h.update(buf)
                whole.update(buf)
                left -= len(buf)
            segs.append((h.hexdigest(), base + start, end - start, name))
            pos = end
        tail = fh.read()
        whole.update(tail)

        recipe_segs = []
        stats = {"ref": 0, "dup": 0, "new": 0}
        for digest, off, size, name in segs:
            if digest in ref:
                kind = "ref"
            elif digest in pack.blobs:
                kind = "dup"
            else:
                pack.add(digest, fh, off, size)
                kind = "new"
            stats[kind] += size
            recipe_segs.append([digest, size, "ref" if kind == "ref" else "blob"])
    recipe = {
        "kind": "safetensors",
        "header_b64": base64.b64encode(head8 + raw).decode("ascii"),
        "segments": recipe_segs,
        "tail_b64": base64.b64encode(tail).decode("ascii"),
        "stats": stats,
    }
    return recipe, whole.hexdigest()


def copy_plain(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    h = hashlib.sha256()
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while True:
            buf = fi.read(CHUNK)
            if not buf:
                break
            h.update(buf)
            fo.write(buf)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/zifanz4/openwam-runtime")
    ap.add_argument("--reference", required=True, help="reference-index.json from Greene")
    ap.add_argument("--out", required=True, help="new, empty output directory")
    ap.add_argument("--part-mib", type=float, default=1024,
                    help="max part size; use 95 for a plain-git transport (GitHub 100 MB/file)")
    ap.add_argument("--manifest-dir", help="dir with files.txt + SHA256SUMS (default ROOT/share-eval)")
    ap.add_argument("--src", help="dir the manifest paths are relative to (default ROOT/assets-source)")
    ap.add_argument("--add-tree", action="append", default=[],
                    help="also pack every file under SRC/REL that is not already listed; "
                         "hashed at pack time (no prior checksum). Unlisted *.safetensors are skipped.")
    ap.add_argument("--add-tree-all", action="append", default=[],
                    help="like --add-tree but also packs *.safetensors (delta-deduplicated, hashed at pack time)")
    ap.add_argument("--dry-run", action="store_true", help="list files and sizes, write nothing")
    ap.add_argument("--no-follow-links", action="store_true",
                    help="omit symlinks; restore them separately using the portable link manifest")
    a = ap.parse_args()

    share = a.manifest_dir or os.path.join(a.root, "share-eval")
    src_root = a.src or os.path.join(a.root, "assets-source")
    with open(os.path.join(share, "files.txt")) as fh:
        files = [norm(l) for l in fh if l.strip()]
    sums = read_sums(os.path.join(share, "SHA256SUMS"))
    listed, added, skipped = set(files), [], []
    trees = [(t, False) for t in a.add_tree] + [(t, True) for t in a.add_tree_all]
    for tree, keep_st in trees:
        top = os.path.join(src_root, norm(tree))
        if not os.path.exists(top):
            raise SystemExit("--add-tree path does not exist: %s" % top)
        if os.path.isfile(top):
            rel = os.path.relpath(top, src_root)
            if rel not in listed:
                listed.add(rel); added.append(rel)
            continue
        for dp, dns, fs in os.walk(top, followlinks=not a.no_follow_links):
            dns.sort()
            for f in sorted(fs):
                if a.no_follow_links and os.path.islink(os.path.join(dp, f)):
                    continue
                rel = os.path.relpath(os.path.join(dp, f), src_root)
                if rel in listed:
                    continue
                if f.endswith(".safetensors") and not keep_st:
                    skipped.append(rel)
                    continue
                listed.add(rel)
                added.append(rel)
    files += added
    size_of = {rel: os.path.getsize(os.path.join(src_root, rel)) for rel in files}
    log("%d listed files (+%d from --add-tree, %.2f GB total); %d unlisted safetensors skipped"
        % (len(files) - len(added), len(added), sum(size_of.values()) / 1e9, len(skipped)))
    for rel in skipped:
        log("  skipped %s (%.2f GB)" % (rel, os.path.getsize(os.path.join(src_root, rel)) / 1e9))
    if a.dry_run:
        log("largest files:")
        for rel in sorted(files, key=lambda r: -size_of[r])[:15]:
            log("  %8.3f GB  %s" % (size_of[rel] / 1e9, rel))
        per_top = {}
        for rel in files:
            k = "/".join(rel.split("/")[:2])
            per_top[k] = per_top.get(k, 0) + size_of[rel]
        log("by top-level dir:")
        for k, v in sorted(per_top.items(), key=lambda kv: -kv[1]):
            log("  %8.3f GB  %s" % (v / 1e9, k))
        for rel in files:
            print("%14d  %s%s" % (size_of[rel], rel, "" if rel in sums else "   [hash at pack time]"))
        return

    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit("--out %s exists and is not empty; use a fresh directory" % a.out)
    os.makedirs(a.out, exist_ok=True)
    with open(a.reference) as fh:
        refidx = json.load(fh)
    ref = refidx["tensors"]
    log("%d checksums, %d reference tensors" % (len(sums), len(ref)))

    out_hashes = {}
    pack = Pack(a.out, int(a.part_mib * 2**20))
    entries, total_src = [], 0
    totals = {"ref": 0, "dup": 0, "new": 0, "plain": 0}
    for i, rel in enumerate(files, 1):
        src = os.path.join(src_root, rel)
        size = os.path.getsize(src)
        total_src += size
        want = sums.get(rel)
        if want is None and rel not in added:
            raise SystemExit("no SHA256SUMS entry for %s" % rel)
        if rel.endswith(".safetensors"):
            log("[%d/%d] %s (%.2f GB)" % (i, len(files), rel, size / 1e9))
            recipe, got = pack_safetensors(src, ref, pack)
            for k, v in recipe["stats"].items():
                totals[k] += v
            log("    reused-from-reference %.2f GB, dup %.2f GB, packed %.3f GB"
                % (recipe["stats"]["ref"] / 1e9, recipe["stats"]["dup"] / 1e9,
                   recipe["stats"]["new"] / 1e9))
        elif size > pack.part_size:
            # The existing unpacker reconstructs any segmented entry using
            # header + segments + tail. Empty framing also supports raw files.
            got = copy_hash(src)
            if got not in pack.blobs:
                with open(src, "rb") as fh:
                    pack.add(got, fh, 0, size)
            recipe = {"kind": "blob", "header_b64": "", "tail_b64": "",
                      "segments": [[got, size, "blob"]]}
            totals["plain"] += size
        else:
            dst = os.path.join(a.out, "files", rel)
            got = copy_plain(src, dst)
            out_hashes["files/" + rel] = got
            recipe = {"kind": "plain"}
            totals["plain"] += size
        if want is None:
            want = got
            recipe["sha256_origin"] = "computed-at-pack"
            sums[rel] = got
        elif got != want:
            raise SystemExit("SOURCE CHECKSUM MISMATCH %s: %s != %s" % (rel, got, want))
        recipe.update({"path": rel, "bytes": size, "sha256": want})
        entries.append(recipe)
    pack.finish()
    out_hashes.update(pack.part_hashes)

    # share-eval in the pack: the given manifest, extended with --add-tree files
    se = os.path.join(a.out, "share-eval")
    shutil.copytree(share, se)
    with open(os.path.join(se, "files.txt"), "w") as fh:
        fh.write("".join(rel + "\n" for rel in files))
    with open(os.path.join(se, "SHA256SUMS"), "w") as fh:
        fh.write("".join("%s  ./%s\n" % (sums[rel], rel) for rel in files))
    for dp, _, fs in os.walk(se):
        for f in fs:
            p = os.path.join(dp, f)
            out_hashes[os.path.relpath(p, a.out)] = copy_hash(p)

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_root": src_root,
        "hashed_at_pack": len(added),
        "reference_file": refidx["reference_file"],
        "reference_sha256": refidx["reference_sha256"],
        "reference_bytes": refidx["reference_bytes"],
        "file_count": len(entries),
        "source_bytes": total_src,
        "blobs": pack.blobs,
        "entries": entries,
        "totals": totals,
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh)
    out_hashes["manifest.json"] = copy_hash(os.path.join(a.out, "manifest.json"))
    with open(os.path.join(a.out, "PACK_SHA256SUMS"), "w") as fh:
        for rel in sorted(out_hashes):
            fh.write("%s  %s\n" % (out_hashes[rel], rel))

    pack_bytes = sum(os.path.getsize(os.path.join(dp, f))
                     for dp, _, fs in os.walk(a.out) for f in fs)
    log("DONE  source %.2f GB -> pack %.2f GB  (reused from reference %.2f GB, "
        "deduplicated %.2f GB)" % (total_src / 1e9, pack_bytes / 1e9,
                                   totals["ref"] / 1e9, totals["dup"] / 1e9))
    log("carry the whole directory %s to Greene" % a.out)


def copy_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(CHUNK), b""):
            h.update(buf)
    return h.hexdigest()


if __name__ == "__main__":
    main()
