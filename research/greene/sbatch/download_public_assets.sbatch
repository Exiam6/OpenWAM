#!/usr/bin/env bash
#SBATCH --job-name=owam-dl
#SBATCH --account=torch_pr_222_courant
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/zz4330/openwam-runtime/logs/%x.%j.out
#SBATCH --error=/scratch/zz4330/openwam-runtime/logs/%x.%j.out

# Public assets for the official-checkpoint reference baseline.
# Runs on a CPU node: the Greene login-node cgroup killed the unzip/download
# of RoboTwin's objects.zip, and a CPU job does not consume the 48-GPU
# Courant cap that the cuRobo/eval jobs queue against.
set -euo pipefail
export TZ="America/New_York"
echo "node $(hostname) job ${SLURM_JOB_ID} start $(date -Is)"
bash /scratch/zz4330/OpenWAM/research/greene/scripts/download_public_assets.sh
echo "finished $(date -Is)"
