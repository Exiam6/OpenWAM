# Full native evaluation allocation, conditional on readiness

This is the L4 phase of ex8-a, with the same hypothesis, official checkpoint,
scientific configuration, two environments, 50 tasks and original seed-0 native
candidate screening specified in PLAN.md. Full reproduction is not yet achieved.

Start only after the fixed 30-episode pilot completes without technical errors,
its independent state/action/count audit passes, at least one clean rollout
succeeds, and a three-camera rollout has been visually checked. The first
adjust_bottle episode was checked at its beginning, midpoint and endpoint; the
native head video was fully decoded. Images show the right gripper grasping and
raising the bottle upright; the left wrist view is largely table/background.
There is no external whole-robot video. This limited visual check does not replace
full-suite numerical verification or video review when diagnosing task failures.

Full execution uses the official native default of 100 episodes, without the
pilot cap. Each task/mode has a separate server, simulator runtime and output
directory. Passive debug PNG recording returns to the official default false to
avoid redundant disk I/O; native per-episode head-camera videos remain enabled.
There is no model, observation, action, success-rule, planner, scene, step-limit
or inference-optimization change. All 30 pilot episodes are excluded from the
full score. Every full task restarts the prescribed seed-0 candidate sequence.

Full-plan cost gate: extrapolate all six completed pilot group wall times to
10,000 episodes and add 180 seconds per task/mode for repeated checkpoint hash,
server load and compilation (five GPU hours total). This estimate is not a
confidence bound: three pilot tasks do not cover every full-suite task length.
If this estimate exceeds the already registered 200 GPU-hour ceiling, stop
before registering L4 jobs and report the measured estimate. Do not change
inference settings or delete unsuccessful pilot episodes to lower the estimate.

If the gate passes, reserve at most 720,000 GPU seconds across 100 jobs: each
receives a 600-second startup/validation allowance, with the remaining 660,000
seconds proportional to its native per-task step limit. Round budgets down to
whole seconds. Long tasks thus receive more time without changing their native
step limit. These are resource ceilings, not new episode termination rules.
A timed-out/incomplete task is unscored and retained, never counted as a policy
failure, retried automatically, or silently dropped from the benchmark.
Seven-day deadline from registration. Existing idle-GPU admission rules apply.
Initially register gpu01 only, whose exact native runtime has passed the pilot;
gpu02 needs its own verified asset/environment manifest before admission.

Verify sequential episode counts and monotonically increasing accepted scene
seeds, success increments in {0,1}, exactly 100 decoded videos, and equality
between native _result.txt and independently parsed success counts. Full scores
require all 100 task/mode rows. Report every task, clean/randomized macro rates,
mean, differences from 93.74/93.46/93.60%, and task-bootstrap intervals. The local
2-percentage-point practical agreement margin is not an author-provided margin
or a guarantee of numerical equivalence. Retain the disclosed benchmark PyTorch
2.7.0 versus documented 2.4.1 difference. Do not resume RAE superiority claims
while full coverage, technical validity, or reproduction agreement is unresolved.
