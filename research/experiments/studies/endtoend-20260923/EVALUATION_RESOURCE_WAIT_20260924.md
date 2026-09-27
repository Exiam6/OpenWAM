# Finite evaluation resource wait

Recorded 2026-09-24 21:37 CDT, before any heldout policy evaluation.

The recovered PCA44 and SVAE44 trainings completed normally. Wan44 is still finishing. cm009 GPU1 (kagaram2) and GPU4 (bhavyaa2) remain occupied. The frozen evaluation queue asserts on occupied resources instead of waiting, so the dependent controller is held before leaving its not-ready training dependency loop.

This is an operational scheduling change under the existing authorization to finish the matched study. Scientific protocol, all frozen source bytes, checkpoints, evaluation groups, scene order, seeds, endpoints, statistics, and GPU-hour ceiling are unchanged. It starts no inference or training and repeats no experiment.

The one-shot helper uses a pidfd for controller PID1075445 after matching UID2093, exact argv and process start ticks2618082887. Only that Python controller receives SIGSTOP/SIGCONT. Its original outer timeout process remains active. Training jobs and other users' processes are untouched.

The finite helper checks original cm009 slots1–6. It requires three consecutive snapshots with memory below100MiB, utilization0, uncorrected ECC0 and no compute process. It then resumes the same controller; original cohort, stream, checkpoint and per-queue resource checks still apply. Idle snapshots are not cluster reservations, so later resource contention can still fail the original queue safely.

The wait ends no later than the original dependency deadline2026-09-25T19:11:52.028093-05:00. Original controller deadline2026-09-28T13:41:59.240315-05:00 and576GPUh ceiling remain unchanged. A user pause or deadline releases the controller back to its original dependency-loop guard, which rejects that condition. The helper also resumes the exact owned process in its exit handler; no manual process hunting or broad signals.

Validation passed: good/occupied/high-memory/nonzero-utilization/ECC/missing-slot predicates; pidfd stop/resume on a disposable child created by the test; exact live controller identity and39 prior frozen/recovery file hashes. Live execution verified controller stateT and a fresh resource-wait heartbeat, with Wan training still advancing.

Selfchecks must inspect evaluation-resource-wait-20260924/progress.json and exit.json. A stale original pipeline heartbeat is expected while controller_held=true and the exact controller isT; the resource-wait heartbeat must be fresh. Never launch another copy or treat this operational hold as a user pause. Do not rerun the completed recovery.

Outputs are under /data02/zifanz4/openwam-experiments/endtoend-20260923/evaluation-resource-wait-20260924. This is one finite dependency job, not a recurring timer. No scheduled wakeups were created or verified here.
