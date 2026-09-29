#!/bin/bash
#SBATCH --job-name=owam-e2e
#SBATCH --account=torch_pr_222_courant
#SBATCH --partition=l40s_courant
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/zz4330/openwam-runtime/logs/%x.%j.out
# One endtoend checkpoint on one L40S. Required: ROUTE, SEED, and either GATE=1 or CELLS='[[task,cond],...]'.
# See research/greene/endtoend-eval/. Finite; no retry.
set -euo pipefail
export TZ="America/New_York"
RUNTIME=/scratch/zz4330/openwam-runtime
kind=$([ "${GATE:-0}" = 1 ] && echo gate || echo eval)
export OPENWAM_RUN_DIR="${RUNTIME}/runs/endtoend-${kind}-${ROUTE}-seed${SEED}-${SLURM_JOB_ID}"
mkdir -p "$OPENWAM_RUN_DIR" "${RUNTIME}/tmp"
export TMPDIR="${RUNTIME}/tmp" HF_HOME=/scratch/zz4330/hf_cache
export TORCH_EXTENSIONS_DIR="${RUNTIME}/cache/torch_extensions" TRITON_CACHE_DIR="${OPENWAM_RUN_DIR}/triton"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_HOME=/scratch/zz4330/conda/envs/cuda-12.4
echo "node $(hostname) job ${SLURM_JOB_ID} start $(date -Is) run ${OPENWAM_RUN_DIR}"
nvidia-smi --query-gpu=name,uuid,pci.bus_id,memory.total --format=csv
cd /scratch/zz4330/OpenWAM/research/greene/endtoend-eval/scripts
if [ "$kind" = gate ]; then args=(--gate-only); else args=(--cells "$CELLS"); fi
set +e
/scratch/zz4330/conda/envs/openwam-policy/bin/python -u run_cell.py --route "$ROUTE" --seed "$SEED" "${args[@]}"
code=$?
echo "{\"returncode\": ${code}, \"time\": \"$(date -Is)\"}" > "${OPENWAM_RUN_DIR}/exit.json"
echo "finished $(date -Is) exit ${code}"
exit $code
