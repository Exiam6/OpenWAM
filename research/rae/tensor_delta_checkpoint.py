"""Transfer exact safetensors checkpoints by reusing identical tensor bytes.

The packer writes the target header and only tensors whose bytes differ from a
verified base checkpoint. The reconstruction verifies both complete files by
SHA256; no tensor value is approximated or converted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

BLOCK = 16 * 1024 * 1024
PART_BYTES = 128 * 1024 * 1024


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            value.update(block)
    return value.hexdigest()


def header(path: Path) -> tuple[bytes, dict]:
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Incomplete safetensors header: {path}")
        size = struct.unpack("<Q", length_bytes)[0]
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError(f"Incomplete safetensors header: {path}")
    return length_bytes + raw, json.loads(raw)


def tensors(info: dict, file_size: int, prefix_bytes: int) -> list[tuple[str, dict]]:
    ordered = sorted(((key, value) for key, value in info.items() if key != "__metadata__"),
                     key=lambda item: item[1]["data_offsets"][0])
    end = 0
    for key, value in ordered:
        start, following = value["data_offsets"]
        if start != end or following < start:
            raise ValueError(f"Noncontiguous tensor data at {key}: {start}, {end}")
        end = following
    if prefix_bytes + end != file_size:
        raise ValueError(f"Safetensors file size mismatch: {prefix_bytes + end} != {file_size}")
    return ordered


class PartWriter:
    def __init__(self, directory: Path):
        self.directory = directory
        self.stream = None
        self.size = 0
        self.total = 0
        self.index = 0
        self.hash = hashlib.sha256()

    def write(self, block: bytes) -> None:
        view = memoryview(block)
        while view:
            if self.stream is None:
                self.stream = (self.directory / f"part{self.index:03d}").open("wb")
                self.size = 0
            take = min(len(view), PART_BYTES - self.size)
            piece = view[:take]
            self.stream.write(piece)
            self.hash.update(piece)
            self.size += take
            self.total += take
            view = view[take:]
            if self.size == PART_BYTES:
                self.stream.close()
                self.stream = None
                self.index += 1

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()


class PartReader:
    def __init__(self, directory: Path):
        self.directory = directory
        self.stream = None
        self.index = -1
        self.offset = 0
        self.hash = hashlib.sha256()

    def read(self, size: int) -> bytes:
        if size <= 0:
            return b""
        index = self.offset // PART_BYTES
        position = self.offset % PART_BYTES
        if index != self.index:
            if self.stream is not None:
                self.stream.close()
            self.stream = (self.directory / f"part{index:03d}").open("rb")
            self.index = index
        self.stream.seek(position)
        block = self.stream.read(min(size, PART_BYTES - position))
        if not block:
            raise EOFError(f"Missing delta data at offset {self.offset}")
        self.hash.update(block)
        self.offset += len(block)
        return block

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()


def pack(base: Path, target: Path, base_hashes: Path, target_hashes: Path,
         output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    base_prefix, base_info = header(base)
    target_prefix, target_info = header(target)
    target_items = tensors(target_info, target.stat().st_size, len(target_prefix))
    tensors(base_info, base.stat().st_size, len(base_prefix))
    base_audit = json.loads(base_hashes.read_text())
    target_audit = json.loads(target_hashes.read_text())
    if len(base_prefix) != base_audit["header_size"] + 8 or len(target_prefix) != target_audit["header_size"] + 8:
        raise ValueError("Tensor audit header differs from the checkpoint")
    (output / "header.bin").write_bytes(target_prefix)
    parts = output / "parts"
    parts.mkdir(exist_ok=True)
    writer = PartWriter(parts)
    plan = {
        "format": 1, "base_sha256": base_audit["weights_sha256"],
        "target_sha256": target_audit["weights_sha256"],
        "target_bytes": target.stat().st_size, "header_sha256": hashlib.sha256(target_prefix).hexdigest(),
        "segments": [],
    }
    reused = 0
    with target.open("rb") as source:
        for key, value in target_items:
            start, end = value["data_offsets"]
            size = end - start
            base_value = base_info.get(key)
            base_hash = base_audit["tensors"].get(key)
            target_hash = target_audit["tensors"].get(key)
            if target_hash is None or target_hash["bytes"] != size:
                raise ValueError(f"Missing tensor audit: {key}")
            common = (base_value is not None and base_hash is not None and
                      base_hash["bytes"] == size and
                      base_hash["sha256"] == target_hash["sha256"])
            segment = {"key": key, "bytes": size, "source": "base" if common else "delta"}
            if common:
                segment["offset"] = len(base_prefix) + base_value["data_offsets"][0]
                reused += size
            else:
                segment["offset"] = writer.total
                source.seek(len(target_prefix) + start)
                remaining = size
                while remaining:
                    block = source.read(min(BLOCK, remaining))
                    if not block:
                        raise EOFError(f"Incomplete target tensor: {key}")
                    writer.write(block)
                    remaining -= len(block)
            plan["segments"].append(segment)
    writer.close()
    plan["delta_bytes"] = writer.total
    plan["delta_sha256"] = writer.hash.hexdigest()
    plan["reused_bytes"] = reused
    (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def reconstruct(base: Path, package: Path, output: Path) -> dict:
    plan = json.loads((package / "plan.json").read_text())
    if plan.get("format") != 1:
        raise ValueError("Unsupported delta format")
    if digest(base) != plan["base_sha256"]:
        raise ValueError("Base checkpoint SHA256 mismatch")
    prefix = (package / "header.bin").read_bytes()
    if hashlib.sha256(prefix).hexdigest() != plan["header_sha256"]:
        raise ValueError("Target header SHA256 mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".reconstructing")
    reader = PartReader(package / "parts")
    try:
        with base.open("rb") as base_stream, temporary.open("wb") as target_stream:
            target_stream.write(prefix)
            for segment in plan["segments"]:
                size = segment["bytes"]
                if segment["source"] == "base":
                    base_stream.seek(segment["offset"])
                    while size:
                        block = base_stream.read(min(BLOCK, size))
                        if not block:
                            raise EOFError(f"Incomplete base tensor: {segment['key']}")
                        target_stream.write(block)
                        size -= len(block)
                elif segment["source"] == "delta":
                    if reader.offset != segment["offset"]:
                        raise ValueError(f"Out-of-order delta tensor: {segment['key']}")
                    while size:
                        block = reader.read(min(BLOCK, size))
                        target_stream.write(block)
                        size -= len(block)
                else:
                    raise ValueError(f"Unknown tensor source: {segment['source']}")
        if reader.offset != plan["delta_bytes"] or reader.hash.hexdigest() != plan["delta_sha256"]:
            raise ValueError("Delta stream SHA256 mismatch")
        if temporary.stat().st_size != plan["target_bytes"] or digest(temporary) != plan["target_sha256"]:
            raise ValueError("Reconstructed checkpoint SHA256 mismatch")
        os.replace(temporary, output)
    finally:
        reader.close()
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_subparsers(dest="operation", required=True)
    packer = operation.add_parser("pack")
    packer.add_argument("--base", type=Path, required=True)
    packer.add_argument("--target", type=Path, required=True)
    packer.add_argument("--base-hashes", type=Path, required=True)
    packer.add_argument("--target-hashes", type=Path, required=True)
    packer.add_argument("--output", type=Path, required=True)
    restore = operation.add_parser("reconstruct")
    restore.add_argument("--base", type=Path, required=True)
    restore.add_argument("--package", type=Path, required=True)
    restore.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "pack":
        result = pack(args.base, args.target, args.base_hashes, args.target_hashes, args.output)
    else:
        result = reconstruct(args.base, args.package, args.output)
    print(json.dumps({key: result[key] for key in ("target_sha256", "delta_bytes", "reused_bytes")}, sort_keys=True))


if __name__ == "__main__":
    main()
