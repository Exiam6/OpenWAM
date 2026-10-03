#!/bin/bash
#SBATCH --job-name=owam-oidn
#SBATCH --account=torch_pr_222_courant
#SBATCH --partition=l40s_courant
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/zz4330/openwam-runtime/logs/%x.%A_%a.out
# Render-only OIDN probe (cross-cluster-rootcause-20261003): A=as deployed, B=denoiser none, C=OIDN 2.3.3 via LD_PRELOAD.
set -euo pipefail
V=(A B C);export VARIANT=${V[$SLURM_ARRAY_TASK_ID]}
export TZ="America/New_York" CUDA_HOME=/scratch/zz4330/conda/envs/cuda-12.4 TMPDIR=/scratch/zz4330/openwam-runtime/tmp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
RUNTIME=/scratch/zz4330/openwam-runtime;OUT=${RUNTIME}/runs/oidn-ab-${SLURM_ARRAY_JOB_ID}/${VARIANT}
unset PYTHONPATH
export EXPECTED_RENDER_PCI=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader) WAM_ROOT=$RUNTIME ROBOTWIN_PATH=${RUNTIME}/benchmarks/RoboTwin ROBOTWIN_RUNTIME_ROOT=${OUT}/runtime ROBOTWIN_ENABLE_PLANNER_FALLBACK=1
if [ "$VARIANT" = C ]; then O=${RUNTIME}/downloads/oidn-2.3.3.x86_64.linux/lib; export LD_PRELOAD=${O}/libOpenImageDenoise.so.2; export LD_LIBRARY_PATH=${O}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; fi
nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv,noheader
cd /scratch/zz4330/OpenWAM
/scratch/zz4330/conda/envs/openwam-bench/bin/python -u research/greene/cross-cluster/oidn_ab.py "$OUT"
