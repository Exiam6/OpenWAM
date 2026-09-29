#!/bin/bash
#SBATCH --job-name=owam-e2e-view
#SBATCH --account=torch_pr_222_courant
#SBATCH --partition=cs
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/zz4330/openwam-runtime/logs/%x.%j.out
# Builds the endtoend path view + remapped checkpoint configs; re-hashes all 9 checkpoints
# against the frozen sha256. Writes research/greene/records/endtoend-view.json.
set -euo pipefail
echo "node $(hostname) job ${SLURM_JOB_ID} start $(date -Is)"
/scratch/zz4330/conda/envs/openwam-policy/bin/python -u /scratch/zz4330/OpenWAM/research/greene/endtoend-eval/scripts/build_view.py
echo "finished $(date -Is)"
