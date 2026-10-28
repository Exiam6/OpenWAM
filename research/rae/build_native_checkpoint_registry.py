"""Verify checkpoint copies on the RTX host and write the native eval registry.

Copy each complete checkpoint directory to CHECKPOINT_ROOT/<model name> first.
The transfer manifest is produced on the training host; this command verifies
the host-local bytes before exposing a checkpoint to the evaluation runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from research.rae.run_native_eval import MODELS, OFFICIAL_SHA256, write_json


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def build_registry(manifest: dict, checkpoint_root: Path,
                   only: set[str] | None = None) -> tuple[dict, list[str]]:
    entries = dict(manifest["checkpoints"])
    official = manifest["official_checkpoint"]
    if official["registered_sha256"] != OFFICIAL_SHA256:
        raise ValueError("Official checkpoint SHA256 differs from the released checkpoint")
    entries["official"] = {**official, "weights_file": Path(official["file"]).name,
                           "weights_bytes": official["bytes"]}
    if not set(entries) <= set(MODELS):
        raise ValueError(f"Unknown checkpoint names: {sorted(set(entries) - set(MODELS))}")
    if only is not None and not only <= set(entries):
        raise ValueError(f"Models absent from transfer manifest: {sorted(only - set(entries))}")
    registered = {}
    missing = []
    for name in MODELS:
        if only is not None and name not in only:
            continue
        if name not in entries:
            missing.append(name)
            continue
        directory = checkpoint_root / name
        if not directory.is_dir():
            missing.append(name)
            continue
        entry = entries[name]
        expected_file = "checkpoint_step_118655.safetensors" if name == "official" else "checkpoint_step_1000.safetensors"
        if entry["weights_file"] != expected_file:
            raise ValueError(f"Wrong checkpoint step in transfer manifest: {name}")
        weight = directory / expected_file
        if not weight.is_file() or weight.stat().st_size != entry["weights_bytes"]:
            raise ValueError(f"Missing or incomplete checkpoint copy: {name}: {weight}")
        for relative in entry["required_sidecars"]:
            sidecar = directory / relative
            if not sidecar.exists() or (relative == "tokenizer" and not sidecar.is_dir()):
                raise ValueError(f"Missing checkpoint sidecar: {name}: {sidecar}")
        expected_sha = entry["registered_sha256"]
        if sha256(weight) != expected_sha:
            raise ValueError(f"Checkpoint copy SHA256 mismatch: {name}: {weight}")
        registered[name] = {"directory": str(directory.resolve()),
                            "weights_file": expected_file,
                            "checkpoint_sha256": expected_sha}
    return registered, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", nargs="+", choices=MODELS,
                        help="Register only these verified models while other copies are in progress")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = args.checkpoint_root.expanduser().resolve(strict=True)
    registry, missing = build_registry(manifest, root, set(args.only) if args.only else None)
    if not registry:
        raise ValueError("No complete checkpoint copies found")
    write_json(args.output, registry)
    print(json.dumps({"registered": list(registry), "pending": missing, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
