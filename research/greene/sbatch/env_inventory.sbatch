#!/bin/bash
#SBATCH --job-name=owam-envinv
#SBATCH --account=torch_pr_222_courant
#SBATCH --partition=l40s_courant
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/zz4330/openwam-runtime/logs/%x.%j.out
# cross-cluster-rootcause-20261003 §1: environment inventory (same env as official_baseline.sbatch; no policy, no scoring).
set -euo pipefail
export TZ="America/New_York" CUDA_HOME=/scratch/zz4330/conda/envs/cuda-12.4 TMPDIR=/scratch/zz4330/openwam-runtime/tmp
unset PYTHONPATH
OUT=/scratch/zz4330/OpenWAM/research/greene/records/torch-env-inventory-20261003.json
/scratch/zz4330/conda/envs/openwam-bench/bin/python -u /scratch/zz4330/OpenWAM/research/greene/cross-cluster/env_inventory.py > "${OUT}.tmp"
mv "${OUT}.tmp" "$OUT"; cat "$OUT"
