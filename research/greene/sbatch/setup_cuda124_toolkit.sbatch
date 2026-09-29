#!/usr/bin/env bash
#SBATCH --job-name=owam-cuda124
#SBATCH --account=torch_pr_222_courant
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/zz4330/openwam-runtime/logs/%x.%j.out
#SBATCH --error=/scratch/zz4330/openwam-runtime/logs/%x.%j.out

# CPU node: the toolkit download is several GB and login nodes kill large downloads.
set -euo pipefail
export TZ="America/New_York"
echo "node $(hostname) job ${SLURM_JOB_ID} start $(date -Is)"
bash /scratch/zz4330/OpenWAM/research/greene/scripts/setup_cuda124_toolkit.sh
echo "finished $(date -Is)"
