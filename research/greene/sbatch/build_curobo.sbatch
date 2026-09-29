#!/usr/bin/env bash
#SBATCH --job-name=owam-curobo
#SBATCH --account=torch_pr_222_courant
#SBATCH --partition=l40s_courant
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/zz4330/openwam-runtime/logs/%x.%j.out
#SBATCH --error=/scratch/zz4330/openwam-runtime/logs/%x.%j.out

# Build cuRobo @ d64c4b0 (v0.7.8) CUDA extensions into the openwam-bench env.
#
# Deviation from the frozen source machine, recorded deliberately:
#   benchmark-setup.json pinned build_arch 8.9 (its only GPU). Greene has
#   L40S (sm_89, identical to the frozen arch) and H200 (sm_90), so this is a
#   fat build for "8.9;9.0". Build-time arch selection changes no numerics;
#   sm_89 kernels here are the same source as the frozen build.
#   max_compile_jobs raised 2 -> 8 (node has 16 cores, 96 GB); build-time only.
set -euo pipefail
export TZ="America/New_York"

RUNTIME=/scratch/zz4330/openwam-runtime
ENV_PREFIX=/scratch/zz4330/conda/envs/openwam-bench
CUROBO_DIR="${RUNTIME}/benchmarks/RoboTwin/envs/curobo"

echo "=== cuRobo build ==="
echo "node      : $(hostname)"
echo "job       : ${SLURM_JOB_ID}"
echo "gpu       : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "started   : $(date -Is)"
echo "commit    : $(git -C "${CUROBO_DIR}" rev-parse HEAD)"

# CUDA 12.4.1 toolkit + gcc 12 live in a separate prefix (setup_cuda124_toolkit.sh):
# the bench env's own nvcc resolved to 13.3 and its gcc to 15 (job 18756828).
CUDA_PREFIX=/scratch/zz4330/conda/envs/cuda-12.4
export CUDA_HOME="${CUDA_PREFIX}"
export PATH="${CUDA_PREFIX}/bin:${ENV_PREFIX}/bin:${PATH}"
export CC="${CUDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CUDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="${CXX}"
echo "cxx       : $(${CXX} --version | head -1)"
export TORCH_CUDA_ARCH_LIST="8.9;9.0"
export MAX_JOBS=8
export TMPDIR="${RUNTIME}/tmp"
export PIP_CACHE_DIR=/scratch/zz4330/.pip_cache
export TORCH_EXTENSIONS_DIR="${RUNTIME}/cache/torch_extensions"
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR"

echo "nvcc      : $(${CUDA_HOME}/bin/nvcc --version | tail -2 | head -1)"

cd "${CUROBO_DIR}"
"${ENV_PREFIX}/bin/python" -m pip install -e . --no-build-isolation --no-deps -v 2>&1 | tail -40

echo "=== verify ==="
"${ENV_PREFIX}/bin/python" - <<'PYEOF'
import torch
print("torch", torch.__version__, "device", torch.cuda.get_device_name(0))
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
tensor_args = TensorDeviceType()
ik = IKSolver(IKSolverConfig.load_from_robot_config(
    RobotConfig.from_dict(cfg, tensor_args), None, rotation_threshold=0.05,
    position_threshold=0.005, num_seeds=20, self_collision_check=False,
    tensor_args=tensor_args, use_cuda_graph=False))
import curobo; print("curobo OK:", curobo.__file__)
PYEOF

echo "finished  : $(date -Is)"
