# Interrupted native full row recovery

Question: can the identical native clean adjust_bottle 100 accepted episode row
finish after an observed shared-GPU interruption, without changing scientific
settings or exceeding the full stage's registered 280 GPU-hour envelope?
Baseline: ex8-d_nativefull_1021 at c6689da34daa701c7381df83eb7c766f057e8593,
released checkpoint and frozen official/RoboTwin/input hashes in its protocol.
Official tracking source was pulled on 2026-10-21 at 18:45 UTC and remains
782e3133f674449e2986754cfc5f7c4e204835a2, identical to that frozen runtime.
Compatibility: no upstream changes. No running/completed runtime is modified.

Hypothesis: interruption is an external resource event; an unchanged full row
can complete. Competing explanation: a native technical error. Preserve and
inspect all native logs; any unknown error stops this separate attempt.
Single intervention: a new bounded attempt of the interrupted row, under a new
scientific identity. No checkpoint, seeds, inference, planner, native task config
or candidate acceptance changes. Reuse the SHA-verified frozen native runner by
import, replacing only output root and admission checks for this one-row plan.
Native100, seed0, default optimizations, no fallback, same 3 cameras and20D IO.
Original23 partial accepted episodes stay retained and are all excluded from
scoring. New row starts from the same native seed sequence; no cherry picking.
All other99 rows stay assigned to ex8-d and are not repeated here.

Budget: 9000 seconds for this attempt, including model load, shutdown and video
validation. Deadline is the original ex8-d deadline, not extended. Charge the
original terminal worker's occupied_parent_seconds rounded UP to an integer,
plus all99 unchanged maximum allocations, plus9000 seconds. This must be <=
280*3600 before registration, admission and launch. Original caps and claims are
unchanged. This releases only unused reservation of a terminal failed attempt;
no running or pending allocation is reduced. No blind second retry is allowed.
Measured cost remains uncertain for94 task/mode rows; this is a cap, not a
completion guarantee. If another failure occurs, preserve and diagnose.

L0: runner diff limited to wrapper/admission/output, source hashes, one-row
registry and complete100-row fixed selector. L1-L3: original audited30-episode
readiness, ex8-d first-row technical gate, and native partial23 accepted episodes
with no native fatal classifier markers. Verify these before registering.
L4: one full100 row only, via existing dispatcher, with all pause paths enforced,
30min idle admission and priority gpu01 indices1/2 at any hour. Other cards only
Chicago01:00-09:30. Never preempt or alter other users' processes.

Primary metric: original full-suite equal-task clean/randomized means on exactly
100 rows/10000 episodes. Frozen selector uses this recovery row plus the original
99 rows, retaining failed parent in a separate exclusion ledger. No partial
headlines or policy failures inferred from missing episodes. Prediction: same
published range; practical local margin <=2pp is not statistical equivalence.
Report exact gaps separately. Secondary: native result/log/video integrity.
Stop on any pause, source drift, unknown native error, cap, deadline, insufficient
disk, incomplete counts/video. Outputs: plan and committed source, immutable
protocol, interrupted-attempt diagnosis, local registry, dispatcher job,
all native logs/videos and independently checked composite endpoint.
No representation superiority claim, no upload/publication, no scheduler edit.
