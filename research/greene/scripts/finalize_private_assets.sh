#!/usr/bin/env bash
# Steps 4-6 of the private-asset transfer, shared by both transports:
#   fetch_private_assets.sh  (direct rsync pull)   and
#   delta/unpack_delta.py    (byte-exact delta rebuild, relayed pack).
# Verifies all files with the ORIGINAL GPU02 SHA256SUMS, restores scene
# aliases and writes records/asset-transfer.json.
#   finalize_private_assets.sh <transport-description>
set -euo pipefail
TRANSPORT="${1:?transport description}"
HOST="${OPENWAM_SRC_HOST:-zifanz4@shenlong-gpu-02.cs.illinois.edu}"
ROOT="${OPENWAM_SRC_ROOT:-/home/zifanz4/openwam-runtime}"
RUNTIME=/scratch/zz4330/openwam-runtime
DEST="${RUNTIME}/assets-source"
HANDOFF="${RUNTIME}/handoff"
LOG="${RUNTIME}/logs/finalize-private-assets.$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
say() { printf '\n[%s] %s\n' "$(date -Is)" "$*"; }

# ── 4. verify ────────────────────────────────────────────────────────────────
say "verifying SHA-256 over all 363 files (takes a few minutes at this size)"
cp "${HANDOFF}/SHA256SUMS" "${DEST}/"
if ( cd "$DEST" && sha256sum -c SHA256SUMS ) > "${RUNTIME}/logs/sha256-check.txt" 2>&1; then
  echo "  ALL FILES VERIFIED"
  tail -3 "${RUNTIME}/logs/sha256-check.txt"
else
  echo "  CHECKSUM FAILURES — see ${RUNTIME}/logs/sha256-check.txt" >&2
  grep -v ': OK$' "${RUNTIME}/logs/sha256-check.txt" | head -20 >&2
  echo "  Re-run the transport (fetch script or unpack_delta.py); only bad files are redone." >&2
  exit 1
fi

# ── 5. scene aliases ─────────────────────────────────────────────────────────
say "restoring portable scene aliases"
python3 "${HANDOFF}/restore-aliases.py" "$DEST"

# ── 6. record ────────────────────────────────────────────────────────────────
say "writing transfer record"
python3 - "$DEST" "$HOST" "$ROOT" "$LOG" "$TRANSPORT" <<'PYEOF'
import datetime, json, os, subprocess, sys
dest, host, root, log, transport = sys.argv[1:6]
files = [os.path.join(dp, f) for dp, _, fs in os.walk(dest) for f in fs
         if f != "SHA256SUMS" and not f.endswith(".partial")]
total = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
rec = {
    "transferred_at": datetime.datetime.now().astimezone().isoformat(),
    "target_machine": "NYU Greene",
    "target_root": dest,
    "source": f"{host}:{root}/assets-source",
    "transport": transport,
    "manifest": "share-eval/files.txt",
    "file_count": len(files),
    "total_bytes": total,
    "total_gib": round(total / 2**30, 2),
    "all_sha256_verified": True,
    "verification_log": "logs/sha256-check.txt",
    "transfer_log": log,
    "scope": "evaluation reproduction only; full training trajectories NOT included",
    "note": "Old-machine outputs are not merged with any Greene run. "
            "Hand over evaluation cells explicitly before launching.",
}
out = "/scratch/zz4330/OpenWAM/research/greene/records/asset-transfer.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    json.dump(rec, fh, indent=2)
    fh.write("\n")
print(json.dumps(rec, indent=2))
PYEOF


say "FINALIZE COMPLETE"
