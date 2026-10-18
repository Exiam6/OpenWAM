# Per-seed pilot capacity queue

This CPU observer advances the three already registered VAE seeds into their
15-episode pilots as each full 20,000-step training job finishes successfully.
It invokes the unchanged corrected-protocol-v1/run_batch.py, which owns GPU
admission, exclusive claims, resource limits, pause handling and cleanup.

Before dispatch, the observer checks the final training parent result, immutable
result digest, fixed checkpoint path, strict reload, corrected protocol/runtime
lineage, actual initialization digest, complete remaining allocation window,
absence of any earlier pilot attempt, actual idle capacity and GPU lock. The
original parent independently verifies the checkpoint digest and all original
scientific gates before starting the pilot. The resource plan and wrapper files
are frozen before this observer starts and revalidated immediately before each
launch. At most one attempt per seed is permitted; the exclusive output directory
and retained batch/claim/output paths prevent automatic retries.

GPU preference is 3, 1, 2, 4. Active pilots reserve their selected GPU locally
while the original parent obtains its own exclusive lock and samples capacity.
Existing training jobs and all frozen scientific code remain unchanged. Pilot
failure, pause, or expiry stops new dispatch. An observer failure records active
bounded parent PIDs and preserves those independently supervised attempts.

The queue cannot release or launch full evaluation. Actual pilot videos must
still be inspected and the original measured-cost release must pass. This wrapper
changes only handoff timing; it changes no seeds, checkpoints, task scenes,
sampling, budgets, stopping rules or outcome criteria.

Validate with `python3 -m unittest -q test_pilot_queue`. Register a runtime snapshot
and `pilot-capacity-queue-v1-plan.json` before running the snapshot's
`pilot_queue.py --check-readiness`. Only then start the snapshot exactly once with
the policy environment's Python and preserve its process record and log.
