#!/usr/bin/env bash
# Greene deployment: build the OpenWAM policy-server env.
# Mirrors research/migration/policy-requirements-lock.txt (torch 2.7.0+cu126).
# Frozen originals are never modified; this is a target-machine adaptation.
set -euo pipefail

CONDA_ROOT=/scratch/zz4330/conda
ENV_NAME=openwam-policy
ENV_PREFIX="${CONDA_ROOT}/envs/${ENV_NAME}"
REPO=/scratch/zz4330/OpenWAM
LOCK="${REPO}/research/migration/policy-requirements-lock.txt"
RUNTIME=/scratch/zz4330/openwam-runtime

export CONDA_PKGS_DIRS=/scratch/zz4330/conda_pkgs
export PIP_CACHE_DIR=/scratch/zz4330/.pip_cache
export TMPDIR="${RUNTIME}/tmp"
mkdir -p "$TMPDIR"

echo "[$(date -Is)] creating ${ENV_PREFIX} (python 3.10)"
"${CONDA_ROOT}/bin/conda" create -y -p "${ENV_PREFIX}" python=3.10 || true

PY="${ENV_PREFIX}/bin/python"
"$PY" -m pip install --upgrade "pip==25.2" "setuptools==78.1.0" wheel

echo "[$(date -Is)] installing locked policy requirements"
grep -v '^-e ' "$LOCK" > "${TMPDIR}/policy-lock.txt"
"$PY" -m pip install -r "${TMPDIR}/policy-lock.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu126

echo "[$(date -Is)] installing openwam (no deps, editable)"
"$PY" -m pip install -e "${REPO}" --no-deps --no-build-isolation

echo "[$(date -Is)] verifying"
"$PY" - <<'PYEOF'
import torch, transformers, diffusers, safetensors, websockets
print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("transformers", transformers.__version__, "diffusers", diffusers.__version__)
import openwam; print("openwam", openwam.__file__)
PYEOF
echo "[$(date -Is)] policy env done: ${ENV_PREFIX}"
