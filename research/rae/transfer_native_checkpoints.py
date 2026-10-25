#!/usr/bin/env python3
"""Copy registered checkpoint directories over the Hestia Tailscale SSH proxy.

The remote copy is accepted only after its weight SHA256 and sidecars match the
training-host transfer manifest. Re-running this command resumes partial rsync
copies and leaves already verified directories intact.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time
from pathlib import Path


def checked(command: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(command, text=True, capture_output=capture, check=False)
    if result.returncode:
        detail = result.stderr.strip() if capture else "see command output"
        raise RuntimeError(f"Command failed ({result.returncode}): {shlex.join(command)}: {detail}")
    return result.stdout.strip() if capture else ""


def registered_models(manifest: dict) -> dict[str, dict]:
    models = dict(manifest.get("checkpoints", {}))
    official = manifest.get("official_checkpoint")
    if official:
        models["official"] = {
            "directory": official["directory"],
            "weights_file": Path(official["file"]).name,
            "weights_bytes": official["bytes"],
            "registered_sha256": official["registered_sha256"],
            "required_sidecars": official["required_sidecars"],
        }
    return models


def validate_entry(name: str, entry: dict) -> tuple[Path, Path]:
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError(f"Unsafe checkpoint name: {name!r}")
    directory = Path(entry["directory"])
    weight = directory / entry["weights_file"]
    if not weight.is_file() or weight.stat().st_size != entry["weights_bytes"]:
        raise ValueError(f"Missing or wrong-size weight: {weight}")
    if not re.fullmatch(r"[0-9a-f]{64}", entry["registered_sha256"]):
        raise ValueError(f"Invalid registered SHA256 for {name}")
    for relative in entry["required_sidecars"]:
        path = directory / relative
        if not path.exists():
            raise ValueError(f"Missing checkpoint sidecar: {path}")
    return directory, weight


def write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--remote-root", type=Path, required=True,
                        help="Absolute path on Hestia, containing a checkpoints/ directory")
    parser.add_argument("--host", required=True, help="Current Hestia Tailscale IPv4 address")
    parser.add_argument("--user", default="grail_zifan")
    parser.add_argument("--key", type=Path, default=Path("/root/.ssh/id_ed25519_xarm"))
    parser.add_argument("--known-hosts", type=Path,
                        default=Path("/root/zifan/openwam-runs/tailscale-userspace/hestia_known_hosts"))
    parser.add_argument("--proxy-python", type=Path,
                        default=Path("/root/zifan/openwam-envs/policy/bin/python"))
    parser.add_argument("--proxy-script", type=Path,
                        default=Path(__file__).with_name("socks5_ssh_proxy.py"))
    parser.add_argument("--models", help="Comma-separated model names; default: all registered")
    parser.add_argument("--state", type=Path, help="Local JSON progress record")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.remote_root.is_absolute() or not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(args.remote_root)):
        parser.error("--remote-root must be an absolute path without spaces or shell characters")
    if ".." in args.remote_root.parts:
        parser.error("--remote-root must not contain '..'")
    if not re.fullmatch(r"100\.\d{1,3}\.\d{1,3}\.\d{1,3}", args.host):
        parser.error("--host must be a Tailscale IPv4 address")
    if not re.fullmatch(r"[a-z_][a-z0-9_-]*", args.user):
        parser.error("Invalid SSH username")

    manifest = json.loads(args.manifest.read_text())
    models = registered_models(manifest)
    selected = args.models.split(",") if args.models else sorted(models)
    if not selected or any(name not in models for name in selected):
        parser.error(f"Unknown model in --models; available: {', '.join(sorted(models))}")
    selected = list(dict.fromkeys(selected))
    for name in selected:
        validate_entry(name, models[name])

    remote_checkpoints = args.remote_root / "checkpoints"
    plan = {
        "host": args.host,
        "user": args.user,
        "remote_root": str(args.remote_root),
        "models": {name: {"source": models[name]["directory"],
                          "destination": str(remote_checkpoints / name),
                          "weights_bytes": models[name]["weights_bytes"],
                          "registered_sha256": models[name]["registered_sha256"]}
                   for name in selected},
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    if not all(path.is_file() for path in (args.key, args.known_hosts, args.proxy_python, args.proxy_script)):
        parser.error("SSH key, known-hosts file, proxy Python, or proxy script is missing")
    proxy = shlex.join([str(args.proxy_python), str(args.proxy_script), "%h", "%p"])
    ssh = ["ssh", "-i", str(args.key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
           "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=yes",
           "-o", f"UserKnownHostsFile={args.known_hosts}", "-o", f"ProxyCommand={proxy}"]
    target = f"{args.user}@{args.host}"

    def remote(command: str) -> str:
        return checked([*ssh, target, command], capture=True)

    state_path = args.state or args.manifest.with_name("hestia_checkpoint_transfer_state.json")
    state = {"plan": plan, "verified": {}, "started_unix": time.time()}
    write_state(state_path, state)
    remote(f"mkdir -p -- {shlex.quote(str(remote_checkpoints))} && df -h -- {shlex.quote(str(remote_checkpoints))}")
    for name in selected:
        entry = models[name]
        source, _ = validate_entry(name, entry)
        destination = remote_checkpoints / name
        print(f"Transferring {name}: {source} -> {target}:{destination}", flush=True)
        remote(f"mkdir -p -- {shlex.quote(str(destination))}")
        checked(["rsync", "-a", "--partial", "--append-verify", "--info=progress2",
                 "-e", shlex.join(ssh), f"{source}/", f"{target}:{destination}/"])
        weight = destination / entry["weights_file"]
        digest = remote(f"sha256sum -- {shlex.quote(str(weight))}").split()[0]
        if digest != entry["registered_sha256"]:
            raise RuntimeError(f"Remote SHA256 mismatch for {name}: {digest}")
        for relative in entry["required_sidecars"]:
            remote(f"test -e {shlex.quote(str(destination / relative))}")
        sidecar_diff = checked(
            ["rsync", "-a", "--checksum", "--dry-run", "--itemize-changes",
             f"--exclude=/{entry['weights_file']}", "-e", shlex.join(ssh),
             f"{source}/", f"{target}:{destination}/"], capture=True)
        if sidecar_diff:
            raise RuntimeError(f"Remote sidecars differ for {name}: {sidecar_diff[:1000]}")
        state["verified"][name] = {"remote_weight": str(weight), "sha256": digest,
                                    "verified_unix": time.time()}
        write_state(state_path, state)
        print(f"Verified {name}: {digest}", flush=True)

    remote_manifest = args.remote_root / "transfer_manifest.json"
    checked(["rsync", "-a", "-e", shlex.join(ssh), str(args.manifest), f"{target}:{remote_manifest}"])
    state["remote_manifest"] = str(remote_manifest)
    state["completed_unix"] = time.time()
    write_state(state_path, state)
    print(f"Transferred {len(selected)} checkpoint directories and manifest.")


if __name__ == "__main__":
    main()
