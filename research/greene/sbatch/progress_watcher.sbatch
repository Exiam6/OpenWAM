#!/usr/bin/env bash
#SBATCH --job-name=owam-progress
#SBATCH --account=torch_pr_222_courant
#SBATCH --partition=cs
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=18:00:00
#SBATCH --output=/scratch/zz4330/openwam-runtime/logs/%x.%j.out
# Auto-updates the research log while a Greene run is live (scrontab is disabled
# on this cluster). Every 20 min: update_progress.py --commit (only the progress
# record + README block; pushes only if changed). One last update when the
# watched job leaves the queue, then exits.
#   sbatch --export=ALL,WATCH_JOB=<jobid>,WATCH_RUN=<run dir> sbatch/progress_watcher.sbatch
set -uo pipefail
export TZ="America/New_York"
U=/scratch/zz4330/OpenWAM/research/greene/scripts/update_progress.py
while squeue -h -j "$WATCH_JOB" 2>/dev/null | grep -q .; do
  python3 "$U" --run "$WATCH_RUN" --commit
  sleep 1200
done
python3 "$U" --run "$WATCH_RUN" --commit
echo "watched job ${WATCH_JOB} finished; final update done $(date -Is)"
