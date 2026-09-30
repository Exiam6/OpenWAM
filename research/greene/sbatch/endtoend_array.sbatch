#!/bin/bash
#SBATCH --job-name=owam-e2e-eval
#SBATCH --account=torch_pr_222_courant
#SBATCH --partition=l40s_courant
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:30:00
#SBATCH --output=/scratch/zz4330/openwam-runtime/logs/%x.%A_%a.out
# Frozen endtoend re-evaluation: array index i runs protocol.json cells[i] =
# [route, seed, task, condition] through endtoend_cell.sbatch (one server, one cell).
#   sbatch --array=0-80%6 research/greene/sbatch/endtoend_array.sbatch
# Finite; failures retained, never auto-retried.
set -euo pipefail
P=/scratch/zz4330/OpenWAM/research/greene/endtoend-eval/protocol.json
read -r ROUTE SEED CELLS < <(python3 -c "import json,sys;c=json.load(open('$P'))['cells'][int(sys.argv[1])];print(c[0],c[1],json.dumps([[c[2],c[3]]]).replace(' ',''))" "$SLURM_ARRAY_TASK_ID")
export ROUTE SEED CELLS GATE=0
echo "array ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}: ${ROUTE} seed${SEED} ${CELLS}"
exec bash /scratch/zz4330/OpenWAM/research/greene/sbatch/endtoend_cell.sbatch
