#!/usr/bin/env bash
#SBATCH --job-name=owam-e2e-progress
#SBATCH --account=torch_pr_222_courant
#SBATCH --partition=cs
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=4-12:00:00
#SBATCH --output=/scratch/zz4330/openwam-runtime/logs/%x.%j.out
# Auto-updates the research log while the endtoend eval array is live. Every 20 min:
# update_endtoend_progress.py --commit; one last update when the array leaves the queue.
#   sbatch --export=ALL,WATCH_ARRAY=<array job id> research/greene/sbatch/endtoend_watcher.sbatch
set -uo pipefail
export TZ="America/New_York"
U=/scratch/zz4330/OpenWAM/research/greene/scripts/update_endtoend_progress.py
while squeue -h -j "$WATCH_ARRAY" 2>/dev/null | grep -q .; do
  python3 "$U" --array "$WATCH_ARRAY" --commit
  sleep 1200
done
python3 "$U" --array "$WATCH_ARRAY" --commit
echo "watched array ${WATCH_ARRAY} finished; final update done $(date -Is)"
