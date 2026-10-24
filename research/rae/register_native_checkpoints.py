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
EXPECTED = ({f"seed{seed}_{variant}" for seed in (42, 43, 44) for variant in ("rae", "vae")}
            | {"seed42_reconstruction_svae48", "seed42_action_svae48"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate(seed: int, variant: str) -> dict[str, object] | None:
    if variant in ("vae", "rae"):
        root = WEIGHTS / f"seed{seed}" / variant
        log = LOGS / f"seed{seed}" / f"{variant}_train.log"
    elif seed == 42 and variant in ("reconstruction_svae48", "action_svae48"):
        run_name = "wam_svae48_seed42" if variant == "reconstruction_svae48" else "wam_action_svae48_seed42"
        root = Path("/data/zifan-openwam-rae") / run_name / "rae"
        log = Path("/root/zifan/openwam-runs") / run_name / "rae_train.log"
    else:
        raise ValueError(f"Unknown native checkpoint: seed{seed}_{variant}")
    matches = list(root.glob("*/checkpoint_step_1000.safetensors"))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"Expected one final checkpoint for seed{seed}_{variant}: {matches}")
    weights = matches[0]
    marker = f"[checkpoint] Saved final: {weights}"
    if not log.is_file() or marker not in log.read_text(errors="replace"):
        return None
    required = ["config.yaml", "normalization_stats.npy", "tokenizer"]
    if variant != "vae":
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
    targets = ((43, "rae"), (43, "vae"), (44, "rae"), (44, "vae"),
               (42, "reconstruction_svae48"), (42, "action_svae48"))
    for seed, variant in targets:
        key = f"seed{seed}_{variant}"
        if key in checkpoints:
            continue
        entry = candidate(seed, variant)
        if entry is not None:
            checkpoints[key] = entry
            added.append(key)
    pending = sorted(EXPECTED - set(checkpoints))
    if added or manifest.get("pending_checkpoints") != pending:
        manifest["pending_checkpoints"] = pending
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
    parser.add_argument("--watch", action="store_true", help="Poll until all eight checkpoints are registered")
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
