#!/usr/bin/env bash
#SBATCH --job-name=owam-delta-unpack
#SBATCH --account=torch_pr_222_courant
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/zz4330/openwam-runtime/logs/%x.%j.out
#SBATCH --error=/scratch/zz4330/openwam-runtime/logs/%x.%j.out

# Delta transfer step 3: rebuild the 363 files from the relayed pack (~73 GB
# written), then the shared finalize (independent sha256sum -c, scene aliases,
# records/asset-transfer.json). Pack must be at $PACK, or give PACK_REPO
# (private git repo pushed by delta/git_push_pack.sh) and it is cloned/pulled:
#   sbatch --export=ALL,PACK_REPO=git@github.com:Exiam6/openwam-greene-transfer.git \
#          research/greene/sbatch/delta_unpack.sbatch
# Optional PACK_BRANCH (default main) and PACK_SHA (abort unless HEAD equals it).
set -euo pipefail
export TZ="America/New_York"
RUNTIME=/scratch/zz4330/openwam-runtime
# TRANSFER=main (default): the 363-file handoff -> assets-source + finalize.
# TRANSFER=wan48: endtoend Wan48 checkpoints + training data -> assets-wan48,
#   verified against the FROZEN hashes in delta/wan48/SHA256SUMS.
TRANSFER="${TRANSFER:-main}"
if [ "$TRANSFER" = main ]; then
  PACK="${PACK:-${RUNTIME}/delta/pack}"; DEST="${RUNTIME}/assets-source"; HANDOFF="${RUNTIME}/handoff"
else
  PACK="${PACK:-${RUNTIME}/delta/pack-${TRANSFER}}"; DEST="${RUNTIME}/assets-${TRANSFER}"; HANDOFF="${RUNTIME}/handoff-${TRANSFER}"
fi
G=/scratch/zz4330/OpenWAM/research/greene
echo "node $(hostname) job ${SLURM_JOB_ID} start $(date -Is)"
if [ -n "${PACK_REPO:-}" ]; then
  PACK_BRANCH="${PACK_BRANCH:-main}"
  if [ -d "${PACK}/.git" ]; then git -C "$PACK" pull -q --ff-only origin "$PACK_BRANCH"
  else git clone -q --single-branch -b "$PACK_BRANCH" "$PACK_REPO" "$PACK"; fi
  head_sha="$(git -C "$PACK" rev-parse HEAD)"
  echo "pack repo ${PACK_REPO} ${PACK_BRANCH} @ ${head_sha}"
  if [ -n "${PACK_SHA:-}" ] && [ "$head_sha" != "$PACK_SHA" ]; then
    echo "PACK HEAD MISMATCH: expected ${PACK_SHA}"; exit 1; fi
fi
python3 "$G/delta/unpack_delta.py" --pack "$PACK" \
  --reference-index "${RUNTIME}/delta/reference-index.json" \
  --dest "$DEST" --handoff "$HANDOFF"
if [ "$TRANSFER" = main ]; then
  bash "$G/scripts/finalize_private_assets.sh" \
    "byte-exact delta pack relayed off-line; frozen tensors taken from the public OpenWAM-Alpha-Sim-RoboTwin-Full checkpoint (see delta/README.md)"
else
  python3 "$G/delta/verify_transfer.py" --dest "$DEST" --handoff "$HANDOFF" \
    --frozen "$G/delta/${TRANSFER}/SHA256SUMS" \
    --record "$G/records/${TRANSFER}-transfer.json" --pack-head "${head_sha:-}"
fi
echo "finished $(date -Is)"
