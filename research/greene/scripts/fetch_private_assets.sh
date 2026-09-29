#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OpenWAM private-asset transfer  —  shenlong-gpu-02  ->  NYU Greene
#
# Pulls ONLY the 363 files named in the handoff manifest (73.27 GB / 68.24 GiB):
# six final temporal checkpoints + their configs/tokenizers + the sealed scene
# cohort. Never recurses into runtime/, logs/, caches or GPU locks.
#
# Run it from a Greene login node inside tmux/screen so a dropped SSH session
# does not kill the transfer:
#
#     tmux new -s owam
#     bash /scratch/zz4330/OpenWAM/research/greene/scripts/fetch_private_assets.sh
#     # detach with Ctrl-b d ; reattach later with: tmux attach -t owam
#
# Safe to re-run: rsync resumes partial files, and already-correct files are
# skipped. Nothing on the source machine is written to or deleted.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

HOST="${OPENWAM_SRC_HOST:-zifanz4@shenlong-gpu-02.cs.illinois.edu}"
ROOT="${OPENWAM_SRC_ROOT:-/home/zifanz4/openwam-runtime}"

RUNTIME=/scratch/zz4330/openwam-runtime
DEST="${RUNTIME}/assets-source"
HANDOFF="${RUNTIME}/handoff"
LOG="${RUNTIME}/logs/fetch-private-assets.$(date +%Y%m%dT%H%M%S).log"

REQUIRED_GIB=80          # 68.24 GiB payload + headroom
SSH_OPTS=(-o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

mkdir -p "$DEST" "$HANDOFF" "${RUNTIME}/logs"
exec > >(tee -a "$LOG") 2>&1

say() { printf '\n[%s] %s\n' "$(date -Is)" "$*"; }

say "OpenWAM private-asset transfer"
echo "  source : ${HOST}:${ROOT}"
echo "  dest   : ${DEST}"
echo "  log    : ${LOG}"

# ── 1. preflight ─────────────────────────────────────────────────────────────
say "preflight: ssh reachability"
if ! ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$HOST" true 2>/dev/null; then
  echo "  key-based login not available; rsync will prompt for a password."
  echo "  (To avoid repeated prompts: ssh-copy-id -i ~/.ssh/id_ed25519.pub ${HOST})"
  ssh "${SSH_OPTS[@]}" "$HOST" 'echo "  reached $(hostname)"'
else
  echo "  reached $(ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$HOST" hostname)"
fi

say "preflight: free space on /scratch"
FREE_GIB=$(df -BG --output=avail /scratch/zz4330 | tail -1 | tr -dc '0-9')
QUOTA_LINE=$(myquota 2>/dev/null | grep -E '^/scratch' || true)
echo "  filesystem avail : ${FREE_GIB} GiB"
[ -n "$QUOTA_LINE" ] && echo "  quota            : ${QUOTA_LINE}"
if [ "${FREE_GIB:-0}" -lt "$REQUIRED_GIB" ]; then
  echo "  ABORT: need >= ${REQUIRED_GIB} GiB free" >&2; exit 1
fi
echo "  NOTE: /scratch quota is 5 TB and was 82.7% used at setup time."
echo "        This transfer adds ~68 GiB (~1.4%). Check the line above."

# ── 2. handoff manifest ──────────────────────────────────────────────────────
say "fetching handoff manifest (files.txt, SHA256SUMS, restore-aliases.py)"
rsync -a --info=progress2 -e "ssh ${SSH_OPTS[*]}" \
  "${HOST}:${ROOT}/share-eval/" "${HANDOFF}/"

for f in files.txt SHA256SUMS restore-aliases.py; do
  [ -f "${HANDOFF}/${f}" ] || { echo "ABORT: handoff is missing ${f}" >&2; exit 1; }
done
echo "  manifest entries : $(grep -cve '^\s*$' "${HANDOFF}/files.txt")"
echo "  checksum entries : $(grep -cve '^\s*$' "${HANDOFF}/SHA256SUMS")"

# ── 3. payload ───────────────────────────────────────────────────────────────
# -L resolves the scene symlinks into real files; --no-recursive guarantees we
# copy exactly what files.txt names and never a whole directory.
say "transferring payload (resumable — re-run this script if it drops)"
rsync -aLh --no-recursive --no-owner --no-group \
  --partial --partial-dir=.rsync-partial --info=progress2 \
  -e "ssh ${SSH_OPTS[*]}" \
  --files-from="${HANDOFF}/files.txt" \
  "${HOST}:${ROOT}/assets-source/" "${DEST}/"

# ── 4-6. verify, aliases, record ─────────────────────────────────────────────
bash "$(dirname "$0")/finalize_private_assets.sh" "rsync pull over ssh"

say "TRANSFER COMPLETE"
echo "  payload : ${DEST}"
echo "  record  : /scratch/zz4330/OpenWAM/research/greene/records/asset-transfer.json"
echo "  next    : tell Claude the transfer finished; it will cross-check the"
echo "            checkpoint hashes against the frozen training audits."
