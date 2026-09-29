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
PACK="${PACK:-${RUNTIME}/delta/pack}"
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
  --dest "${RUNTIME}/assets-source" --handoff "${RUNTIME}/handoff"
bash "$G/scripts/finalize_private_assets.sh" \
  "byte-exact delta pack relayed off-line; frozen tensors taken from the public OpenWAM-Alpha-Sim-RoboTwin-Full checkpoint (see delta/README.md)"
echo "finished $(date -Is)"
