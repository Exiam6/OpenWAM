# Preserve fresh capacity checks after a long queue hold

Resource amendment registered on2026-10-18 before either joint coordinator has
admitted a GPU job. The original joint CPU coordinator is waiting only. Its
original plan, code, process record and terminal state remain preserved.
No original G/V scientific file, active scientific parent, scope or budget changes.

The original dedicated dispatcher checks GPU capacity before taking the global
queue lock. A long joint reservation hold can make that sample stale. Unlocking
on V failure must not allow a pending job to launch from that earlier sample.

## Isolate the three unstarted claims

Before migration, commit and freeze this amendment, both updated CPU entrypoints,
the new dispatcher, and tests. Construct separate `dedicated-reservations-v3`
archives for exactly the three still-held G44 jobs. Each archive retains the
original job and binds the untouched original archive path/hash, while using
status `resource_held_v3` and token `joint-safe-dispatch-20261018-0636`. Bind the
new archive bytes in both the v3 handoff and the v2 joint resource plan.

Verify all four original scientific parents are still running, all dedicated
GPUs remain occupied, the original joint CPU coordinator has no child/claim or
assignment, and all three original held claims still match their archives.
Retire only that waiting CPU coordinator. Acquire the global dispatch lock,
recheck all three original claims, and replace their resource-reservation records
with the registered v3 archive bytes. Preserve all original archives and all
running/complete claims. Write a migration proof containing before/after hashes.
If any migration precondition fails, do not activate the replacement.

The old dispatchers ignore the v3 held status, so cannot consume those claims
or use stale probes after the lock opens. They continue supervising their current
scientific children unchanged and later wait for the existing jobs to complete.
The new joint coordinator accepts only the registered migrated reservations.
It retains the same serial-chain, VAE preflight, cutoff and cancellation logic,
with a distinct CPU state/process record and resource-plan-v2 hash.

## Fresh fallback admission

Four new v3 dedicated CPU dispatchers use distinct CPU locks/state files. They
share the same global dispatch lock with the new joint coordinator and recognize
only the migrated namespace. They cannot claim while the joint coordinator holds
that lock. After acquiring it, they discard every earlier capacity sample,
recheck the authorized full-cap window, and require three fresh idle polls plus
one final check. The actual launch uses that final physical-GPU sample and newly
checked allocation end. Changed reservations and prior attempt directories fail
closed. Each registered job can still be admitted only once.

On V preflight failure, the joint coordinator may release pending v3 reservations
without stopping its current G child. Only the new dispatchers can consume them,
with fresh admission checks. On normal completion, all held reservations have
already been consumed, so neither old nor new idle dispatchers can duplicate work.
VAE full training and evaluation continue to require their original independent
scientific release gates. The new CPU release records its resource-plan path,
allowing final reports to verify and retain both versions of resource history.

CPU tests demonstrate that the original dispatcher ignores migrated claims,
a busy GPU or expired window after lock acquisition refuses an otherwise valid
old sample, all four fresh samples are required, only the final sample is used,
and scientific attempt directories remain immutable. Existing reservation,
cancellation, numerical-stage equivalence and reporting tests remain required.
