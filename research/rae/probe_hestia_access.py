#!/usr/bin/env python3
"""Wait for Tailscale login and audit the known Hestia RTX host over SSH."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = Path("/root/zifan/openwam-runs/tailscale-userspace/hestia_access_state.json")
SOCKET = Path("/root/zifan/openwam-runs/tailscale-userspace/tailscaled.sock")
TAILSCALE = Path("/root/zifan/openwam-runs/tailscale-userspace/tailscale_1.104.2_amd64/tailscale")
PROXY = ROOT / "research/rae/socks5_ssh_proxy.py"
POLICY_PYTHON = Path("/root/zifan/openwam-envs/policy/bin/python")
KNOWN_HOSTS = STATE.parent / "hestia_known_hosts"
HOST_NAME = "hestia"
HOST_USER = "grail_zifan"
KEYS = (Path("/root/.ssh/id_ed25519_xarm"), Path("/root/.ssh/id_rsa"))
AUDIT = (
    "set -eu; "
    "printf 'HOST\\n'; hostname; "
    "printf 'GPU\\n'; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader; "
    "printf 'DISK\\n'; df -h \"$HOME\" /tmp; "
    "for volume in /data /scratch /mnt /workspace; do "
    "if [ -d \"$volume\" ]; then df -h \"$volume\"; fi; done; "
    "printf 'TOOLS\\n'; "
    "for name in python3.10 python3.12 git rsync conda micromamba uv vulkaninfo; do "
    "location=$(command -v \"$name\" || true); "
    "printf '%s: %s\\n' \"$name\" \"${location:-missing}\"; done; "
    "printf 'OS\\n'; uname -srm"
)


def record(stage: str, **details: object) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "updated_unix": time.time(), **details}
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATE)
    print(json.dumps(payload, sort_keys=True), flush=True)


def tailscale_status() -> dict:
    result = subprocess.run([str(TAILSCALE), f"--socket={SOCKET}", "status", "--json"],
                            text=True, capture_output=True, timeout=15)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout)


def find_hestia(status: dict) -> dict | None:
    for peer in (status.get("Peer") or {}).values():
        identity = str(peer.get("DNSName") or peer.get("HostName") or "")
        if identity.split(".", 1)[0].lower() == HOST_NAME:
            return peer
    return None


def ssh_audit(key: Path, host_ip: str) -> subprocess.CompletedProcess:
    proxy = f"{POLICY_PYTHON} {PROXY} %h %p"
    command = [
        "ssh", "-i", str(key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={KNOWN_HOSTS}", "-o", f"ProxyCommand={proxy}",
        f"{HOST_USER}@{host_ip}", AUDIT,
    ]
    return subprocess.run(command, text=True, capture_output=True, timeout=40)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    record("waiting_for_tailscale_login", host_name=HOST_NAME)
    last_state = None
    while True:
        try:
            status = tailscale_status()
            backend = status.get("BackendState")
            if backend != "Running":
                stage = "waiting_for_tailscale_login"
                if stage != last_state:
                    record(stage, backend_state=backend, auth_url=status.get("AuthURL"))
                    last_state = stage
                time.sleep(args.poll_seconds)
                continue
            peer = find_hestia(status)
            if peer is None:
                stage = "waiting_for_hestia_peer"
                if stage != last_state:
                    record(stage, tailnet_peers=len(status.get("Peer") or {}))
                    last_state = stage
                time.sleep(args.poll_seconds)
                continue
            identity = str(peer.get("DNSName") or peer.get("HostName") or peer.get("Name") or "")
            host_ips = peer.get("TailscaleIPs") or []
            host_ip = next((ip for ip in host_ips if "." in ip), None)
            if host_ip is None:
                raise RuntimeError(f"Hestia peer has no IPv4 address: {host_ips!r}")
            record("hestia_peer_verified", peer_name=identity,
                   tailscale_ips=host_ips, online=peer.get("Online"))
            if not peer.get("Online", False):
                time.sleep(args.poll_seconds)
                continue
            failures = []
            for key in KEYS:
                if not key.is_file():
                    continue
                outcome = ssh_audit(key, host_ip)
                if outcome.returncode == 0:
                    fingerprint = subprocess.run(["ssh-keygen", "-lf", str(KNOWN_HOSTS)],
                                                 text=True, capture_output=True, check=True).stdout.strip()
                    record("ssh_audit_complete", peer_name=identity, host_ip=host_ip, key=str(key),
                           host_key_fingerprint=fingerprint, output=outcome.stdout.strip())
                    return
                failures.append({"key": str(key), "returncode": outcome.returncode,
                                 "stderr": outcome.stderr.strip()[-1000:]})
            record("waiting_for_hestia_ssh", peer_name=identity, host_ip=host_ip,
                   failures=failures)
            time.sleep(args.poll_seconds)
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            # A transient daemon/status timeout is observation failure, not a
            # terminal login or SSH result. Poll the same daemon again.
            record("tailscale_observation_retry", error=f"{type(exc).__name__}: {exc}")
            time.sleep(args.poll_seconds)
        except BaseException as exc:
            record("failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
