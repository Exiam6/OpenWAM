#!/usr/bin/env bash
# Greene deployment: build the RoboTwin benchmark-client env (base pip layer).
# Mirrors research/migration/benchmark-requirements-lock.txt (torch 2.4.1+cu124,
# sapien 3.0.0b1, mplib 0.2.1) and the pinned commits in benchmark-setup.json.
# cuRobo CUDA extensions are built separately on a GPU node (build_curobo.sbatch).
set -euo pipefail

CONDA_ROOT=/scratch/zz4330/conda
ENV_NAME=openwam-bench
ENV_PREFIX="${CONDA_ROOT}/envs/${ENV_NAME}"
REPO=/scratch/zz4330/OpenWAM
LOCK="${REPO}/research/migration/benchmark-requirements-lock.txt"
RUNTIME=/scratch/zz4330/openwam-runtime
ROBOTWIN_DIR="${RUNTIME}/benchmarks/RoboTwin"
ROBOTWIN_COMMIT=0aeea2d669c0f8516f4d5785f0aa33ba812c14b4

export CONDA_PKGS_DIRS=/scratch/zz4330/conda_pkgs
export PIP_CACHE_DIR=/scratch/zz4330/.pip_cache
export TMPDIR="${RUNTIME}/tmp"
mkdir -p "$TMPDIR" "${RUNTIME}/benchmarks"

echo "[$(date -Is)] creating ${ENV_PREFIX} (python 3.10 + cuda-toolkit 12.4 for the cuRobo build)"
"${CONDA_ROOT}/bin/conda" create -y -p "${ENV_PREFIX}" -c nvidia -c conda-forge \
  python=3.10 cuda-toolkit=12.4.1

PY="${ENV_PREFIX}/bin/python"
# setuptools 78.1.0 is pinned upstream: SAPIEN 3.0.0b1 imports pkg_resources.
"$PY" -m pip install --upgrade "pip==25.2" "setuptools==78.1.0" wheel

echo "[$(date -Is)] installing locked benchmark requirements (cuRobo line deferred)"
grep -v '^-e git' "$LOCK" > "${TMPDIR}/bench-lock.txt"
"$PY" -m pip install -r "${TMPDIR}/bench-lock.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu124

echo "[$(date -Is)] OpenWAM client-side deps"
"$PY" -m pip install --no-deps websockets pyyaml msgpack

echo "[$(date -Is)] cloning RoboTwin @ ${ROBOTWIN_COMMIT}"
if [ ! -d "${ROBOTWIN_DIR}/.git" ]; then
  git clone https://github.com/RoboTwin-Platform/RoboTwin.git "${ROBOTWIN_DIR}"
fi
git -C "${ROBOTWIN_DIR}" fetch --all --tags
git -C "${ROBOTWIN_DIR}" checkout --detach "${ROBOTWIN_COMMIT}"

echo "[$(date -Is)] verifying base layer"
"$PY" - <<'PYEOF'
import torch, numpy, scipy
print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("numpy", numpy.__version__, "scipy", scipy.__version__)
import sapien; print("sapien", sapien.__version__)
import mplib; print("mplib ok")
import warp; print("warp", warp.config.version)
PYEOF
echo "[$(date -Is)] bench env base done: ${ENV_PREFIX}"
echo "NEXT: apply SAPIEN/mplib patches, then sbatch research/greene/sbatch/build_curobo.sbatch"
