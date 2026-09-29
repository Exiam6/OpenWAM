#!/usr/bin/env bash
#SBATCH --job-name=owam-delta-ref
#SBATCH --account=torch_pr_222_courant
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/zz4330/openwam-runtime/logs/%x.%j.out
#SBATCH --error=/scratch/zz4330/openwam-runtime/logs/%x.%j.out

# Delta transfer step 1: per-tensor SHA-256 index of the public released
# checkpoint (24.8 GB read). The small JSON goes to GPU02 for pack_delta.py.
set -euo pipefail
export TZ="America/New_York"
REF=/scratch/zz4330/OpenWAM/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full/checkpoint_step_118655.safetensors
OUT=/scratch/zz4330/openwam-runtime/delta/reference-index.json
echo "node $(hostname) job ${SLURM_JOB_ID} start $(date -Is)"
python3 /scratch/zz4330/OpenWAM/research/greene/delta/make_reference.py "$REF" "$OUT"
ls -l "$OUT"
echo "finished $(date -Is)"
