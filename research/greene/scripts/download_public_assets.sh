#!/usr/bin/env bash
# Greene deployment: fetch the PUBLIC assets the official-checkpoint reference
# baseline needs. Nothing here is a private experiment artifact.
#   1. RoboTwin 2.0 task assets (TianxingChen/RoboTwin2.0)
#   2. Released OpenWAM-Alpha-Sim-RoboTwin-Full checkpoint (~24.8 GB)
set -euo pipefail

RUNTIME=/scratch/zz4330/openwam-runtime
REPO=/scratch/zz4330/OpenWAM
ROBOTWIN_DIR="${RUNTIME}/benchmarks/RoboTwin"
BENCH_PY=/scratch/zz4330/conda/envs/openwam-bench/bin/python
POLICY_PY=/scratch/zz4330/conda/envs/openwam-policy/bin/python

export HF_HOME=/scratch/zz4330/hf_cache
export HF_HUB_ENABLE_HXET=1
export TMPDIR="${RUNTIME}/tmp"

echo "[$(date -Is)] === RoboTwin 2.0 assets ==="
cd "${ROBOTWIN_DIR}/assets"
"$POLICY_PY" _download.py
for z in background_texture embodiments objects; do
  if [ -f "${z}.zip" ]; then
    echo "[$(date -Is)] unzip ${z}.zip"
    unzip -q -o "${z}.zip" && rm -f "${z}.zip"
  fi
done
cd "${ROBOTWIN_DIR}"
echo "[$(date -Is)] configuring embodiment paths"
"$BENCH_PY" ./script/update_embodiment_config_path.py

echo "[$(date -Is)] === released OpenWAM checkpoint ==="
cd "${REPO}"
"$POLICY_PY" scripts/download_assets/download_openwam_checkpoints.py \
  --family alpha --name OpenWAM-Alpha-Sim-RoboTwin-Full --yes

echo "[$(date -Is)] done"
du -sh "${ROBOTWIN_DIR}/assets" "${REPO}/assets/openwam_ckpt" 2>/dev/null || true
