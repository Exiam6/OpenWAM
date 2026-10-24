#!/usr/bin/env python3
"""Register finished matched checkpoints in the native-host transfer manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from safetensors import safe_open

MANIFEST = Path("/data/zifan-openwam-rae/native_eval_20261024/transfer_manifest.json")
WEIGHTS = Path("/data/zifan-openwam-rae/wam_multiseed_20261024")
LOGS = Path("/root/zifan/openwam-runs/wam_multiseed_20261024")
EXPECTED = {f"seed{seed}_{variant}" for seed in (42, 43, 44) for variant in ("rae", "vae")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate(seed: int, variant: str) -> dict[str, object] | None:
    matches = list((WEIGHTS / f"seed{seed}" / variant).glob("*/checkpoint_step_1000.safetensors"))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"Expected one final checkpoint for seed{seed}_{variant}: {matches}")
    weights = matches[0]
    marker = f"[checkpoint] Saved final: {weights}"
    log = LOGS / f"seed{seed}" / f"{variant}_train.log"
    if not log.is_file() or marker not in log.read_text(errors="replace"):
        return None
    required = ["config.yaml", "normalization_stats.npy", "tokenizer"]
    if variant == "rae":
        required.insert(2, "dinov2/config.json")
    for sidecar in required:
        path = weights.parent / sidecar
        if not path.exists() or (sidecar == "tokenizer" and not path.is_dir()):
            raise FileNotFoundError(path)
    if weights.stat().st_size < 20_000_000_000:
        raise ValueError(f"Incomplete checkpoint: {weights}")
    with safe_open(weights, framework="pt", device="cpu") as stream:
        if len(stream.keys()) < 1000:
            raise ValueError(f"Incomplete safetensors header: {weights}")
    return {
        "directory": str(weights.parent),
        "weights_file": weights.name,
        "weights_bytes": weights.stat().st_size,
        "registered_sha256": sha256(weights),
        "required_sidecars": required,
    }


def register_once(manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text())
    checkpoints = manifest["checkpoints"]
    if not set(checkpoints) <= EXPECTED:
        raise ValueError("Unexpected checkpoint key in transfer manifest")
    added: list[str] = []
    for seed in (43, 44):
        for variant in ("rae", "vae"):
            key = f"seed{seed}_{variant}"
            if key in checkpoints:
                continue
            entry = candidate(seed, variant)
            if entry is not None:
                checkpoints[key] = entry
                added.append(key)
    if added:
        manifest["pending_checkpoints"] = sorted(EXPECTED - set(checkpoints))
        manifest["updated_unix"] = time.time()
        temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, manifest_path)
        finally:
            temporary.unlink(missing_ok=True)
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--watch", action="store_true", help="Poll until all six checkpoints are registered")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    while True:
        added = register_once(args.manifest)
        manifest = json.loads(args.manifest.read_text())
        print(json.dumps({"registered": added, "pending": manifest["pending_checkpoints"]}), flush=True)
        if not args.watch or not manifest["pending_checkpoints"]:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
