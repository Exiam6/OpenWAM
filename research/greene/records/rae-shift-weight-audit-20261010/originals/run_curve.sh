#!/usr/bin/env bash
# Reproduce launch.py's environment for a read-only sigma-curve evaluation.
# Usage: run_curve.sh <study_root> <variant> <checkpoint> <out.json>
set -euo pipefail

R=/home/zifanz4/openwam-runtime
SRC=/home/zifanz4/openwam-rae-research

export PYTHONPATH="$SRC"
export CUDA_VISIBLE_DEVICES=GPU-a4c607bb-f8f8-b999-19ea-0eb65290bb10
export CUDA_HOME="$R/cuda-toolkit"
export CPATH="$R/cuda-toolkit/targets/x86_64-linux/include"
export TORCH_CUDA_ARCH_LIST=12.0
export TMPDIR="$R/tmp"
export TORCH_EXTENSIONS_DIR="$R/cache/torch_extensions-sm120"
export TRITON_CACHE_DIR="$R/cache/triton-sm120"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PATH="$R/policy-env/bin:$R/cuda-toolkit/bin:/usr/bin:/bin"

cd "$SRC"
exec "$R/policy-env/bin/python" -u "$R/eval_sigma_curve.py" "$@"
