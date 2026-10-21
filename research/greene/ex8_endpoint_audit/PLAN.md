# Independent composite endpoint audit

CPU validation of the already registered ex8-d/ex8-e endpoint, with no new GPU
attempt or scientific intervention. Baseline cd7b4775e80d8c2e378c54ff9d9f8ca3e6b6613b.
Official tracking pull at 2026-10-21T19:00:43Z remains 782e3133; compatible with
both frozen attempts. Do not modify any frozen runtime, dispatcher or bot.

The recovery selector must continue to choose ex8-e clean adjust_bottle and the
other 99 ex8-d rows; retain the interrupted 23-episode parent as excluded.
Reuse the frozen integrity auditor for source, log, video and native result
checks, then independently parse native Success!/Fail! lines and cumulative
counters using the actual RoboTwin eval_policy.py as source of truth. Cross-check
observed partial rows without scoring them or confusing log progress with a
completed worker. Full headline requires all 100 rows and 10000 episodes.

Restore the original full-stage paired-task bootstrap reporting on this composite
endpoint: fixed seed 20261021, 10000 paired task draws, 50 tasks per draw, percentile
indices 249/9749. Report clean/randomized/mean gaps against 93.74/93.46/93.60 and
local <=2pp practical margin separately from exact equality. This describes
across-task variation, not author seed variance or statistical equivalence.
Full paper reproduction remains false pending scientific gap interpretation.

Budget: CPU only, <=5 minutes validation; zero added GPU hours. Test malformed
native counts, outcome disagreement, missing/duplicate task coverage and paired
bootstrap arithmetic before auditing live state. Outputs go only to self-check
evidence, never to running/completed experiment directories. Every write checks
global pause and registered pause paths. No upload or external communication.
