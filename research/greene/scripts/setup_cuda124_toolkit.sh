#!/usr/bin/env bash
# Greene deployment: standalone CUDA 12.4.1 toolkit prefix, used ONLY as
# CUDA_HOME for the cuRobo extension build.
#
# Why a separate prefix: `conda create ... -c nvidia cuda-toolkit=12.4.1` in
# setup_bench_env.sh resolved the metapackage at 12.4.1 but pulled 13.3
# sub-packages (nvcc 13.3), which torch's cpp_extension rejects against
# torch 2.4.1+cu124 (owam-curobo job 18756828). The frozen source machine used
# /usr/local/cuda-12.4. Installing from the version-locked label channel with
# --override-channels guarantees every CUDA component is 12.4.1, and leaves the
# patch-verified openwam-bench env untouched.
#
# gcc/g++ 12: nvcc 12.4 supports host gcc <= 13; openwam-bench carries gcc 15.
# The frozen setup does not record its host compiler — recorded as a deviation.
set -euo pipefail

CONDA_ROOT=/scratch/zz4330/conda
PREFIX="${CONDA_ROOT}/envs/cuda-12.4"
export CONDA_PKGS_DIRS=/scratch/zz4330/conda_pkgs
export TMPDIR=/scratch/zz4330/openwam-runtime/tmp
mkdir -p "$TMPDIR"

if [ ! -x "${PREFIX}/bin/nvcc" ]; then
  echo "[$(date -Is)] creating ${PREFIX}"
  "${CONDA_ROOT}/bin/conda" create -y -p "${PREFIX}" --override-channels \
    -c nvidia/label/cuda-12.4.1 cuda-toolkit
  echo "[$(date -Is)] host compiler gcc/g++ 12"
  "${CONDA_ROOT}/bin/conda" install -y -p "${PREFIX}" --override-channels \
    -c conda-forge "gcc_linux-64=12" "gxx_linux-64=12"
fi

echo "[$(date -Is)] verify"
"${PREFIX}/bin/nvcc" --version | tail -2
"${PREFIX}/bin/x86_64-conda-linux-gnu-g++" --version | head -1
bad=$(ls "${PREFIX}/conda-meta" | grep -E '^cuda-' | grep -vE -- '-12\.4[.-]' || true)
if [ -n "$bad" ]; then
  echo "non-12.4 CUDA packages present:"; echo "$bad"; exit 1
fi
echo "all cuda-* packages are 12.4.x"
