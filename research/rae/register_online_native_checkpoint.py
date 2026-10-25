#!/usr/bin/env python3
"""Create a separate transfer manifest containing the optional online model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open

ONLINE = "seed42_online_action"
CORE_CHECKPOINTS = ({f"seed{seed}_{variant}" for seed in (42, 43, 44) for variant in ("vae", "rae")}
                    | {"seed42_reconstruction_svae48", "seed42_action_svae48"})
REQUIRED_SIDECARS = ("config.yaml", "normalization_stats.npy", "dinov2/config.json", "tokenizer")
ADAPTER_UP = "video_backbone.video_encoder._online_action_adapter.up.weight"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def build_manifest(base: dict, state: dict, *, train_log: Path, source_marker: Path) -> dict:
    if set(base["checkpoints"]) != CORE_CHECKPOINTS or base.get("pending_checkpoints"):
        raise ValueError("Core transfer manifest is not complete")
    if ONLINE in base["checkpoints"]:
        raise ValueError("Do not mutate the core transfer manifest")
    if state.get("stage") != "complete":
        raise ValueError("Online WAM training and offline evaluation are not complete")
    weight = Path(state["checkpoint"])
    if weight.name != "checkpoint_step_1000.safetensors":
        raise ValueError("Wrong online checkpoint step")
    if not train_log.is_file() or f"[checkpoint] Saved final: {weight}" not in train_log.read_text(errors="replace"):
        raise ValueError("Online final-save marker missing")
    if source_marker.read_text().strip() != state["source_commit"]:
        raise ValueError("Online model source commit differs from training state")
    for relative in REQUIRED_SIDECARS:
        sidecar = weight.parent / relative
        if not sidecar.exists() or (relative == "tokenizer" and not sidecar.is_dir()):
            raise FileNotFoundError(sidecar)
    if weight.stat().st_size < 20_000_000_000:
        raise ValueError("Incomplete online checkpoint")
    with safe_open(weight, framework="pt", device="cpu") as stream:
        if len(stream.keys()) < 2000 or ADAPTER_UP not in stream.keys():
            raise ValueError("Online checkpoint lacks adapter tensors")
    actual = sha256(weight)
    if actual != state["checkpoint_sha256"]:
        raise ValueError("Online checkpoint SHA256 changed")
    output = json.loads(json.dumps(base))
    output["checkpoints"][ONLINE] = {
        "directory": str(weight.parent), "weights_file": weight.name,
        "weights_bytes": weight.stat().st_size, "registered_sha256": actual,
        "required_sidecars": list(REQUIRED_SIDECARS),
    }
    output["optional_model_source_commit"] = state["source_commit"]
    output["optional_model_key"] = ONLINE
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--online-state", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--source-marker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = build_manifest(json.loads(args.base_manifest.read_text()),
                              json.loads(args.online_state.read_text()),
                              train_log=args.train_log, source_marker=args.source_marker)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "model": ONLINE,
                      "sha256": manifest["checkpoints"][ONLINE]["registered_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
