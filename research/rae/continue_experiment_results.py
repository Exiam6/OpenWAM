#!/usr/bin/env python3
"""Build the consolidated report after both registered training chains finish."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from research.rae.build_final_results import RUNS, WIDTH, build

MULTISEED = RUNS / "wam_multiseed_20261024"
STATE = RUNS / "openwam_rae_experiment_results_state.json"
OUTPUT = RUNS / "OPENWAM_RAE_EXPERIMENT_RESULTS.md"


def record(stage: str, **details: object) -> None:
    payload = {"stage": stage, "updated_unix": time.time(), **details}
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATE)
    print(json.dumps(payload, sort_keys=True), flush=True)


def wait_for(pid: int, marker: str) -> None:
    while True:
        proc = Path(f"/proc/{pid}")
        try:
            state = (proc / "stat").read_text().split(") ", 1)[1].split()[0]
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except FileNotFoundError:
            return
        if state == "Z":
            return
        if marker not in command:
            raise RuntimeError(f"PID {pid} is no longer {marker}")
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiseed-summary-pid", type=int, required=True)
    parser.add_argument("--width-pid", type=int, required=True)
    args = parser.parse_args()
    try:
        record("waiting_for_multiseed_summary", pid=args.multiseed_summary_pid)
        wait_for(args.multiseed_summary_pid, "research/rae/continue_multiseed_summary.py")
        if not (MULTISEED / "SUMMARY.md").is_file():
            raise RuntimeError("Three-seed summary controller exited without its report")
        record("waiting_for_width_ablation", pid=args.width_pid)
        wait_for(args.width_pid, "research/rae/continue_svae48_wam.py")
        width_state = json.loads((WIDTH / "state.json").read_text())
        if width_state["stage"] != "complete":
            raise RuntimeError(f"48-channel WAM ablation did not complete: {width_state}")
        report = build()
        temporary = OUTPUT.with_suffix(".md.tmp")
        temporary.write_text(report, encoding="utf-8")
        temporary.replace(OUTPUT)
        record("complete", output=str(OUTPUT))
    except BaseException as exc:
        record("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
