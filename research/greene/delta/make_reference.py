#!/usr/bin/env python3
"""Step 1 (Greene): per-tensor SHA-256 index of a public reference checkpoint.

The output (reference-index.json, tens of KB) is carried to GPU02, where
pack_delta.py drops every private-checkpoint tensor whose bytes already exist
here. Only hashes, names, offsets and sizes are written — no weights.

    python3 make_reference.py REFERENCE.safetensors OUT/reference-index.json

Standard library only.
"""
import hashlib
import json
import os
import struct
import sys

CHUNK = 64 << 20


def read_header(fh):
    (n,) = struct.unpack("<Q", fh.read(8))
    raw = fh.read(n)
    return n, raw, json.loads(raw)


def main():
    src, out = sys.argv[1], sys.argv[2]
    size = os.path.getsize(src)
    whole = hashlib.sha256()
    tensors = {}
    with open(src, "rb") as fh:
        n, raw, header = read_header(fh)
        whole.update(struct.pack("<Q", n))
        whole.update(raw)
        base = 8 + n
        items = sorted(
            ((k, v) for k, v in header.items() if k != "__metadata__"),
            key=lambda kv: kv[1]["data_offsets"][0],
        )
        pos = 0
        for i, (name, meta) in enumerate(items):
            start, end = meta["data_offsets"]
            if start != pos:
                raise SystemExit("non-contiguous tensor data at %s" % name)
            h = hashlib.sha256()
            left = end - start
            while left:
                buf = fh.read(min(CHUNK, left))
                if not buf:
                    raise SystemExit("truncated file at %s" % name)
                h.update(buf)
                whole.update(buf)
                left -= len(buf)
            pos = end
            digest = h.hexdigest()
            # first occurrence wins; the offset is all unpack needs
            tensors.setdefault(digest, {
                "name": name, "offset": base + start, "size": end - start,
                "dtype": meta["dtype"], "shape": meta["shape"],
            })
            if (i + 1) % 200 == 0:
                print("  %d/%d tensors" % (i + 1, len(items)), flush=True)
        rest = fh.read()
        whole.update(rest)
    index = {
        "reference_file": os.path.basename(src),
        "reference_path_on_greene": os.path.abspath(src),
        "reference_bytes": size,
        "reference_sha256": whole.hexdigest(),
        "tensor_count": len(items),
        "unique_tensor_hashes": len(tensors),
        "tensors": tensors,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(index, fh)
    print("reference sha256 %s  (%d tensors, %d unique)"
          % (index["reference_sha256"], len(items), len(tensors)))
    print("wrote", out)


if __name__ == "__main__":
    main()
