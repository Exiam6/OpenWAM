#!/usr/bin/env python3
"""Register the online native checkpoint after its guarded training completes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from research.rae.continue_svae48_wam import wait_for
from research.rae.register_online_native_checkpoint import build_manifest

RUNS = Path("/root/zifan/openwam-runs/wam_online_action_seed42")
SOURCE = Path("/data/zifan-openwam-rae/online_action_adapter_20261025/wam_seed42")
BASE = Path("/data/zifan-openwam-rae/native_eval_20261024/transfer_manifest.json")
OUTPUT = Path("/data/zifan-openwam-rae/online_action_adapter_20261025/transfer_manifest_with_online.json")
STATE = RUNS / "native_transfer_state.json"


def record(stage: str, **details: object) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    data = {"stage": stage, "updated_unix": time.time(), **details}
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATE)
    print(json.dumps(data, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-pid", type=int, required=True)
    args = parser.parse_args()
    try:
        record("waiting_for_online_training", pid=args.after_pid)
        wait_for(args.after_pid, "research/rae/continue_online_action.py")
        online = json.loads((RUNS / "state.json").read_text())
        if online.get("stage") != "complete":
            raise RuntimeError(f"Online training did not complete: {online}")
        if OUTPUT.exists():
            raise FileExistsError(OUTPUT)
        record("verifying_checkpoint", checkpoint=online["checkpoint"])
        manifest = build_manifest(json.loads(BASE.read_text()), online,
                                  train_log=RUNS / "train.log",
                                  source_marker=SOURCE / "source_commit_at_launch.txt")
        temporary = OUTPUT.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(OUTPUT)
        record("complete", manifest=str(OUTPUT),
               checkpoint_sha256=online["checkpoint_sha256"])
    except BaseException as exc:
        record("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
