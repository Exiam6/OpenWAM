# Conditional dedicated allocation for the existing G and V endpoints

Registered before activation on 2026-10-18. This is a CPU resource amendment,
not a new scientific condition. The original G 900-scene endpoint, V 180-scene
endpoint, seeds, checkpoints, evaluation rules, data and numerical files remain
unchanged. G retains its 172800 occupied-second cap and V its 73800-second cap.
There are no retries of any scientific GPU attempt.

## Preconditions and admission

The original V extra-GPU admission expired before any GPU stage: its preflight
and scheduler are terminal, the stage list is empty and occupied GPU seconds
are zero. Preserve both original states and the original failure log, bound by
`extra-window-admission-expired.json`. Any prior V GPU stage or output rejects
this route. GPU0/5/6/7 are excluded; foreign processes are never changed.

Take the existing G `dedicated-dispatch.lock` exclusively. This prevents pending
held jobs from being claimed by the old CPU dispatchers; those dispatchers and
all existing scientific parents remain running. Check original reservations and
all registered file hashes before taking the lock. Do not edit held claims until
their sole actual admission. The immutable joint plan binds both registrations,
CPU source snapshots, original file inventory, and current parent identities.

Choose the first actually completed G43 lane among GPUs2/3/4. Its result must be
complete, its original parent gone, and its physical GPU idle for three polls
plus a final check: registered UUID, no compute processes, actual memory used
below256MiB, utilization0. Driver-reserved memory is recorded separately.
Forecast completion is never sufficient for admission.

Exactly three held G44 jobs run sequentially on this lane, using the original
`code/run_job.py`, 10800-second job caps, 45-second parent cleanup reserve and
40-second termination grace. Their order is wide, native_shift16, wide_shift16.
For each claim, require strictly more than `remaining*(10800+60)+60` seconds
before16:59UTC. The first admission must therefore be before07:55UTC, the second
before10:56UTC and the third before13:57UTC. Original occupied-time accounting
and the sum of completed occupation plus remaining full caps must fit48GPUh.

After the first G claim actually launches, record an immutable assignment of
that lane to G and the other three lanes to V. The V CPU preflight verifies the
coordinator identity and actual lock ownership before every admission check.
Any one actually idle assigned V lane can run serial cache/smoke/deployment/tiny
stages. Original current parents on other V lanes continue to completion; their
full-cap releases must still allow the entire V endpoint.

## V window and scientific gates

V preflight requires strictly more than9000+22560 seconds before16:59UTC;
its first GPU stage must start before08:13UTC. Also require22560 seconds after
the later of worst preflight completion and every reserved lane's current-parent
full-cap release. No scientific stage budget or tolerance changes. Full asset
hashing and every original numerical preflight command remain in force.

Use separate `dedicated-preflight-run.json` and `dedicated-preflight-logs`;
never overwrite the original capacity failure. The resource parent points all
GPU commands to the original V `code` directory. Full training still requires
independent review and a separately written release from the same cache,
gradient, reload, deployment, tiny-fit and measured-throughput predicates.
The new CPU release records its exact preflight path and evidence hash.

The existing registered resource batch wrapper can then use only the three
assigned V GPUs; it independently rechecks actual idle capacity and full caps.
All three fresh20000-step training seeds and all pilot/video/cost releases remain
required before full evaluation. No full training is launched by this amendment.

## Cancellation and evidence

If initial admission expires, release the global queue lock without creating
any scientific attempt. If V preflight fails, release still-held G jobs to the
original dedicated dispatchers and let the already-running G job finish under
its original cap. Do not terminate that G child merely because V failed.
Hard pause/deadline/scientific errors retain the original fail-stop behavior;
owned children are cleaned up, and failed records remain visible. After all
three G jobs finish, release the global queue lock normally.

Final reports retain both capacity histories, the joint plan and assignment,
CPU snapshots, coordinator provenance/state and hashed selected V preflight.
Unchanged endpoint, cost, media and actual visual-review gates still apply.
The V final-report snapshot is frozen separately; earlier snapshots remain.

CPU tests cover strict cutoff equality, future-lane bounds, actual flock
ownership, competing atomic claims, immutable reservation archives, prior GPU
work rejection, cancellation preserving a live child, memory accounting,
byte-identical numerical stage bodies, equivalent scientific release predicates
and rejection of altered report evidence. Run these before registration and
activation. Partial completion is not a scientific conclusion.
