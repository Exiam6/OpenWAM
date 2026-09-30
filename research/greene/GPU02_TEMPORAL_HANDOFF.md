# Torch / Greene handoff: do not duplicate GPU02 temporal evaluation

Captured **2026-09-30T16:45:25.164932-05:00**. GPU02 worker active: **True**.
**882/1080 complete**, 16 technical failures,
2 unfinished/in-flight directories, 180 never attempted; zero duplicate identities.
Current: learned-seed44 / place_object_basket / head_camera_yaw_5deg.
Authorized resume deadline: **2026-10-01T18:43:29.998112-05:00**; the September 29 deadline below is historical.
Every 30 minutes, the GPU02 systemd monitor checks and resumes eligible pending work.

## Execution ownership

**Do not submit the mean/learned temporal-pooling study on Torch/Greene.**
GPU02 physical GPU 5 owns all remaining cells of the 1080-cell matrix
(mean/learned × seeds 42/43/44 × 3 tasks × 3 conditions × 20 sealed scenes).
Do not retrain its six fixed checkpoints or replay completed, failed, or unfinished identities.
The GPU02 resume worker skips every attempted identity. Failures remain missing valid outcomes,
not policy failures; all attempted does not mean all valid.

This notice remains in force after the snapshot/deadline becomes old. Transfer requires an explicit
coordinated handoff, confirmation the GPU02 worker is stopped, and a fresh attempt inventory.
A Git snapshot is a coordination notice, not a distributed scheduler lock.

- [Machine-readable ownership](records/gpu02-temporal-ownership.json)
- [Latest captured progress](../../research/experiments/studies/temporal-gpu02-20260928/snapshots/20260930T164525/progress.json)
- [Attempt ledger, results and SHA256](../../research/experiments/studies/temporal-gpu02-20260928/snapshots/20260930T164525/attempt-ledger.json)
- Live root: `/home/zifanz4/openwam-runtime/temporal-new-20260928/resume-v1`
- Live inventory command on GPU02: `python3 /home/zifanz4/openwam-runtime/temporal-new-20260928/resume-v1/resume_state.py`
- Worker: `openwam-gpu5-worker.service`; timer: `openwam-gpu5-monitor.timer`.

## Separate cluster work

The Greene wan/svae/pca 1620-cell endtoend evaluation is a different registered study.
Its existing jobs are not duplicates of mean/learned temporal pooling and remain outside this notice.
The official checkpoint clean reference has already completed 60 episodes (58 successes), per
the cluster's committed record; do not relaunch it based on older GPU02 notes.
Do not merge outcomes across machines or claim an aggregate pooling benefit before a complete audit.
