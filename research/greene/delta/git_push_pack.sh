#!/usr/bin/env bash
# GPU02: push a delta pack (from pack_delta.py --part-mib 95) to a PRIVATE
# GitHub repo with plain git — no LFS, every file < 100 MB, pushed in batches
# of ~1.5 GB so no single push is near GitHub's 2 GB limit. Re-runnable: already
# committed files are skipped, and an interrupted batch is simply pushed again.
#
#   bash git_push_pack.sh /home/zifanz4/owam-delta git@github.com:Exiam6/openwam-greene-transfer.git
set -euo pipefail
PACK="${1:?pack dir}"
REPO="${2:?git remote, e.g. git@github.com:OWNER/NAME.git}"
BATCH_BYTES=$((1500 * 1024 * 1024))

# ── refuse a public repo: anonymous ls-remote only succeeds on public repos ──
slug=$(printf '%s' "$REPO" | sed -E 's#^(git@github.com:|https://github.com/)##; s#\.git$##')
if GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/false git -c credential.helper= \
     ls-remote "https://github.com/${slug}.git" >/dev/null 2>&1; then
  echo "ABORT: github.com/${slug} is readable anonymously (PUBLIC). The pack holds" >&2
  echo "       private weights, scenes and configs; use a private repo." >&2
  exit 1
fi
echo "repo ${slug}: not publicly readable — OK"

[ -f "${PACK}/manifest.json" ] && [ -f "${PACK}/PACK_SHA256SUMS" ] \
  || { echo "ABORT: ${PACK} is not a finished pack" >&2; exit 1; }
big=$(find "$PACK" -path "${PACK}/.git" -prune -o -type f -size +99M -print)
if [ -n "$big" ]; then
  echo "ABORT: files >= 99 MB (re-pack with --part-mib 95):" >&2; echo "$big" >&2; exit 1
fi

cd "$PACK"
if [ ! -d .git ]; then
  git init -q -b main
  git remote add origin "$REPO"
fi
git config user.name  "openwam-transfer"
git config user.email "openwam-transfer@localhost"
git config core.compression 0        # bf16 weights do not compress; save CPU

push() { git push -q origin main 2>&1 | tail -3; echo "  pushed: $(git log --oneline | wc -l) commits"; }

# metadata first, so Greene can verify / see progress early
git add manifest.json PACK_SHA256SUMS share-eval files 2>/dev/null || git add manifest.json PACK_SHA256SUMS share-eval
if ! git diff --cached --quiet; then git commit -q -m "delta pack: metadata + plain files"; fi
push

batch=() ; bytes=0 ; n=0
flush() {
  [ ${#batch[@]} -eq 0 ] && return
  n=$((n + 1))
  git add -- "${batch[@]}"
  git commit -q -m "delta pack: ${batch[0]##*/} .. ${batch[-1]##*/}"
  echo "[$(date +%H:%M:%S)] batch ${n}: ${#batch[@]} parts, $((bytes >> 20)) MiB"
  push
  batch=() ; bytes=0
}
for f in pack/part-*.bin; do
  if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then continue; fi
  sz=$(stat -c %s "$f")
  if [ $((bytes + sz)) -gt "$BATCH_BYTES" ]; then flush; fi
  batch+=("$f"); bytes=$((bytes + sz))
done
flush
push   # retries a final push if the last one was interrupted

echo "DONE: $(git ls-files pack | wc -l) parts committed; HEAD $(git rev-parse --short HEAD)"
echo "Tell Greene: repo ${REPO}, HEAD $(git rev-parse HEAD)"
