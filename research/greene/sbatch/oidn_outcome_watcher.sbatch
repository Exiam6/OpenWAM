#!/usr/bin/env bash
#SBATCH --job-name=owam-oidn-progress
#SBATCH --account=torch_pr_222_courant
#SBATCH --partition=cs
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/zz4330/openwam-runtime/logs/%x.%j.out
# Auto-commits the latest oidn-outcome progress + log tails every 10 min; one last update when the job leaves the queue.
#   sbatch --export=ALL,WATCH_JOB=<job id> research/greene/sbatch/oidn_outcome_watcher.sbatch
set -uo pipefail
export TZ="America/New_York"
U=/scratch/zz4330/OpenWAM/research/greene/scripts/update_oidn_outcome_progress.py
while squeue -h -j "$WATCH_JOB" 2>/dev/null | grep -q .; do
  python3 "$U" --job "$WATCH_JOB" --commit
  sleep 600
done
sleep 60
python3 "$U" --job "$WATCH_JOB" --commit
echo "watched job ${WATCH_JOB} finished; final update done $(date -Is)"
