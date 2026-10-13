"""Read-only 30-minute experiment checks; stop this monitor when work is terminal."""

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

from worker import resource_policy

ROOT = Path(__file__).resolve().parent
TIMER = "openwam-rae-factorial-monitor-20261010.timer"


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"read_error": repr(exc)}


def command(args):
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=30)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"error": "observation timeout; not treated as stopped"}


def main():
    now = dt.datetime.now(dt.timezone.utc)
    p = read(ROOT / "protocol.json")
    policy = resource_policy(ROOT, p)
    deadline = dt.datetime.fromisoformat(p["deadline"])
    output = ROOT / "monitor"
    output.mkdir(exist_ok=True)
    services = {}
    changes = {}
    alerts = []
    for prefix in ["", "post-"]:
        for gpu in policy["gpus"]:
            unit = f"openwam-rae-factorial-{prefix}gpu{gpu}-20261010.service"
            observed = command(
                [
                    "systemctl",
                    "--user",
                    "show",
                    unit,
                    "-p",
                    "ActiveState",
                    "-p",
                    "SubState",
                    "-p",
                    "MainPID",
                    "-p",
                    "Result",
                ]
            )
            status = {}
            if observed.get("returncode") == 0:
                status = dict(line.split("=", 1) for line in observed["stdout"].splitlines())
            state = read(ROOT / f"{prefix}worker-gpu{gpu}.json")
            services[unit] = {"systemd": status, "observation": observed, "worker": state}
            job = state.get("job", state.get("next_job", {}))
            changes[unit] = [
                status.get("ActiveState"),
                state.get("phase"),
                job.get("name"),
                job.get("stage"),
                job.get("variant"),
                job.get("seed"),
                state.get("error"),
            ]
            if "error" in observed or observed.get("returncode") != 0:
                alerts.append({"unit": unit, "reason": "service observation failed; no restart"})
            if status.get("ActiveState") == "failed" or state.get("phase") == "stopped":
                alerts.append({"unit": unit, "reason": "service stopped or failed", "error": state.get("error")})
            stamp = state.get("time")
            if stamp and status.get("ActiveState") == "active":
                age = (now - dt.datetime.fromisoformat(stamp)).total_seconds()
                if age > 300:
                    alerts.append(
                        {"unit": unit, "reason": "worker heartbeat older than five minutes", "age_seconds": age}
                    )
    claims = {}
    for directory in ["claims", "post-claims"]:
        claims[directory] = {file.stem: read(file) for file in sorted((ROOT / directory).glob("*.json"))}
    failures = [name for records in claims.values() for name, row in records.items() if row.get("status") == "failed"]
    if failures:
        alerts.append({"reason": "failed attempts retained; no automatic retry", "attempts": failures})
    progress = {file.parent.name: read(file) for file in sorted((ROOT / "runs").glob("*_shift16-seed*/status.json"))}
    summary = read(ROOT / "success-summary.json") if (ROOT / "success-summary.json").exists() else None
    integrity_errors = []
    for manifest_name in ["code-freeze.json", "eval-freeze.json"]:
        manifest = read(ROOT / manifest_name)
        if "files" not in manifest:
            integrity_errors.append(manifest_name)
            continue
        for file, expected in manifest["files"].items():
            try:
                actual = hashlib.sha256(Path(file).read_bytes()).hexdigest()
                if actual != expected:
                    integrity_errors.append(file)
            except OSError:
                integrity_errors.append(file)
    if integrity_errors:
        alerts.append({"reason": "frozen artifact changed or missing", "paths": sorted(set(integrity_errors))})
    remaining = max(0, (deadline - now).total_seconds() / 3600)
    if not claims["claims"] and remaining < 28:
        alerts.append({"reason": "less than estimated 28-hour two-GPU budget remains; experiment has not started"})
    terminal_states = {"inactive", "failed"}
    all_terminal = all(row["systemd"].get("ActiveState") in terminal_states for row in services.values())
    expired = now >= deadline
    changes["claim_status"] = {
        directory: {name: row.get("status") for name, row in records.items()} for directory, records in claims.items()
    }
    changes["summary"] = {"complete": summary.get("complete"), "counts": summary.get("counts")} if summary else None
    changes["alerts"] = [{k: v for k, v in alert.items() if k != "age_seconds"} for alert in alerts]
    previous = read(output / "latest.json")
    meaningful_change = previous.get("change_key") != changes
    record = {
        "checked_at": now.isoformat(),
        "deadline": p["deadline"],
        "remaining_hours": remaining,
        "services": services,
        "claims": claims,
        "training_progress": progress,
        "success_summary": summary,
        "gpu": command(
            [
                "nvidia-smi",
                "-i",
                ",".join(policy["gpus"]),
                "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ]
        ),
        "alerts": alerts,
        "change_key": changes,
        "meaningful_change": meaningful_change,
        "experiment_actions_taken": [],
        "deadline_reached": expired,
        "all_services_terminal": all_terminal,
        "chat_notifications_available": False,
    }
    text = json.dumps(record, indent=2) + "\n"
    (output / (now.strftime("%Y%m%dT%H%M%SZ") + ".json")).write_text(text)
    tmp = output / "latest.tmp"
    tmp.write_text(text)
    tmp.replace(output / "latest.json")
    if meaningful_change:
        with (output / "events.jsonl").open("a") as handle:
            handle.write(json.dumps({"time": now.isoformat(), "changes": changes}) + "\n")
        print(json.dumps({"checked_at": now.isoformat(), "alerts": alerts, "changes": changes}), flush=True)
    if expired or all_terminal:
        result = command(["systemctl", "--user", "stop", TIMER])
        (output / "monitor-stopped.json").write_text(
            json.dumps(
                {
                    "time": now.isoformat(),
                    "reason": "deadline" if expired else "all_services_terminal",
                    "stop_timer": result,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
