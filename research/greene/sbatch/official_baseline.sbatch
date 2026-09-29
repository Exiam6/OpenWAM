#!/bin/bash
#SBATCH --job-name=owam-official
#SBATCH --account=torch_pr_222_courant
#SBATCH --partition=l40s_courant
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/zz4330/openwam-runtime/logs/%x.%j.out
# Official-checkpoint reference baseline: released OpenWAM-Alpha-Sim-RoboTwin-Full,
# 3 tasks x 20 clean sealed scenes, L40S (sm_89, the frozen build arch).
# See research/greene/baseline-official/protocol.json. Finite; no retry.
set -euo pipefail
export TZ="America/New_York"
RUNTIME=/scratch/zz4330/openwam-runtime
export OPENWAM_RUN_DIR="${RUNTIME}/runs/official-baseline-${SLURM_JOB_ID}"
mkdir -p "$OPENWAM_RUN_DIR" "${RUNTIME}/tmp"
export TMPDIR="${RUNTIME}/tmp" HF_HOME=/scratch/zz4330/hf_cache
export TORCH_EXTENSIONS_DIR="${RUNTIME}/cache/torch_extensions" TRITON_CACHE_DIR="${OPENWAM_RUN_DIR}/triton"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_HOME=/scratch/zz4330/conda/envs/cuda-12.4
echo "node $(hostname) job ${SLURM_JOB_ID} start $(date -Is) run ${OPENWAM_RUN_DIR}"
nvidia-smi --query-gpu=name,uuid,pci.bus_id,memory.total --format=csv
cd /scratch/zz4330/OpenWAM/research/greene/baseline-official/scripts
set +e
/scratch/zz4330/conda/envs/openwam-policy/bin/python -u run_baseline.py
code=$?
echo "{\"returncode\": ${code}, \"time\": \"$(date -Is)\"}" > "${OPENWAM_RUN_DIR}/exit.json"
echo "finished $(date -Is) exit ${code}"
exit $code
