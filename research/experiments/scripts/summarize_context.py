"""Read-only complete/partial context pilot audit; all planned trials retained."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from repeat_common import array_hash, payload_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", type=Path, required=True)
    a = p.parse_args()
    study = a.study
    plan = json.loads((study / "run-plan.json").read_text())
    fixed = json.loads((study / "commands.json").read_text())
    trials = {}
    source_files = {}
    for spec in plan["trials"]:
        d = study / spec["id"]
        status = (
            json.loads((d / "trial-status.json").read_text())
            if (d / "trial-status.json").exists()
            else {"status": "not_run_or_preflight_failed"}
        )
        traces = d / "seed-400000-trace.jsonl"
        rows = (
            [json.loads(s) for s in traces.read_text().splitlines()]
            if traces.exists()
            else []
        )
        trial = {
            "mode": spec["mode"],
            "status": status,
            "records": rows,
            "commands_recorded": len(rows),
        }
        for i, row in enumerate(rows):
            assert (
                row["step"] == i
                and row["fixed_command_index"] == i
                and row["seed"] == 400000
            )
            payload = json.loads(
                gzip.decompress((d / row["request_file"]).read_bytes())
            )
            assert payload_hash(payload) == row["request_sha256"]
            assert (
                array_hash(np.asarray(row["selected_action"], np.float32))
                == fixed["commands"][i]["server_action_hash"]
            )
            assert row["env_action_hash"] == fixed["commands"][i]["env_action_hash"]
            assert "server_step" not in row and "server_action" not in row
            online = row["online_response"]
            assert (online is None) == (spec["mode"] == "resident_idle")
            if online is not None:
                assert online["step"] == i + 1 and row["online_counter_valid"]
        trial["complete"] = status["status"] == "completed" and len(rows) == 2
        trials[spec["id"]] = trial
        if d.exists():
            for path in d.glob("*.json*"):
                source_files[str(path.relative_to(study))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    comparisons = []
    for left, right in [("A0", "A1"), ("B0", "B1"), ("A0", "B0"), ("A1", "B1")]:
        x, y = trials[left], trials[right]
        row = {"left": left, "right": right, "valid": x["complete"] and y["complete"]}
        if row["valid"]:
            a0, a1 = x["records"]
            b0, b1 = y["records"]
            row.update(
                initial_request_equal=a0["request_sha256"] == b0["request_sha256"],
                first_post_action_request_equal=a1["request_sha256"]
                == b1["request_sha256"],
                first_post_action_cameras_equal=a1["raw_cameras"] == b1["raw_cameras"],
                first_post_action_eef_equal=a1["eef_hash"] == b1["eef_hash"],
                before_first_action_state_equal=a0["before_execution"]["sha256"]
                == b0["before_execution"]["sha256"],
                after_first_action_state_equal=a0["after_execution"]["sha256"]
                == b0["after_execution"]["sha256"],
                first_planners_equal=payload_hash(a0["planner_calls"])
                == payload_hash(b0["planner_calls"]),
            )
        comparisons.append(row)
    result = {
        "role": "bounded context diagnostic, not policy success evaluation",
        "completed_trials": sum(t["complete"] for t in trials.values()),
        "commands_recorded": sum(t["commands_recorded"] for t in trials.values()),
        "trials": trials,
        "comparisons": comparisons,
        "limits": [
            "Only two trials per condition and one fixed scene.",
            "Online inference/RPC/waiting differ jointly; GPU load is not isolated.",
            "Finite equality does not establish determinism or exclude hidden-state effects.",
            "No RAE-vs-VAE comparison or representation benefit estimate.",
        ],
    }
    (study / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (study / "result-manifest.json").write_text(
        json.dumps(source_files, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "trials"}, indent=2))


if __name__ == "__main__":
    main()
