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
HOST_IP = "100.114.186.17"
HOST_USER = "grail_zifan"
KEYS = (Path("/root/.ssh/id_ed25519_xarm"), Path("/root/.ssh/id_rsa"))
AUDIT = "set -e; hostname; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader; df -h /; uname -srm"


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
        if HOST_IP in peer.get("TailscaleIPs", []):
            return peer
    return None


def ssh_audit(key: Path) -> subprocess.CompletedProcess:
    proxy = f"{POLICY_PYTHON} {PROXY} %h %p"
    command = [
        "ssh", "-i", str(key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={KNOWN_HOSTS}", "-o", f"ProxyCommand={proxy}",
        f"{HOST_USER}@{HOST_IP}", AUDIT,
    ]
    return subprocess.run(command, text=True, capture_output=True, timeout=40)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    record("waiting_for_tailscale_login", host_ip=HOST_IP)
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
            if "hestia" not in identity.lower():
                raise RuntimeError(f"Expected Hestia identity at {HOST_IP}, got {identity!r}")
            record("hestia_peer_verified", peer_name=identity,
                   tailscale_ips=peer.get("TailscaleIPs"), online=peer.get("Online"))
            if not peer.get("Online", False):
                time.sleep(args.poll_seconds)
                continue
            failures = []
            for key in KEYS:
                if not key.is_file():
                    continue
                outcome = ssh_audit(key)
                if outcome.returncode == 0:
                    fingerprint = subprocess.run(["ssh-keygen", "-lf", str(KNOWN_HOSTS)],
                                                 text=True, capture_output=True, check=True).stdout.strip()
                    record("ssh_audit_complete", peer_name=identity, key=str(key),
                           host_key_fingerprint=fingerprint, output=outcome.stdout.strip())
                    return
                failures.append({"key": str(key), "returncode": outcome.returncode,
                                 "stderr": outcome.stderr.strip()[-1000:]})
            raise RuntimeError(f"Hestia SSH audit failed with available keys: {failures}")
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
