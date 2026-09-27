# Next decision after the negative two-command probe

NOT LAUNCHED. The six fixed-command traces are complete, exit0, all twelve
commands changed measured qpos, and every paired recorded field agrees exactly.
This does not reproduce the original closed-loop observation divergence. Do not
relabel it a repair, suppress it, or rerun until a preferred result appears.

## What remains confounded
The probe removed the policy server and online inference, added passive physical
state/planner reads, and truncated preceding scenes to two commands. Its first
scene's requests agree with the oldround0 first two requests; later scenes agree
only at the initial request. These are context changes, not evidence for a
particular source of nondeterminism. Bottle velocities/solver caches were not
measured. Full closed-loop outcomes still agree across both observed repeats.

## Smallest context control to prepare on CPU
Keep the same passive recorder in both conditions. Consider only first scene
400000 (predeclared first, with no prior-scene history), two actions each, two
fresh server/process rounds. Within each round, compare recorded-command replay
while the original model server is resident with native live inference. Keep
same seed/prompt/initial requests and full32-action buffer. Record live outputs
and compare them with saved commands; differing commands are a finding, not an
exclusion criterion. Four short traces, no success-rate estimation or training.

This tests online generation/latency with model residency held present, but is
not a complete factorial test of all rendering/physics/timing effects. Fix trial
order and boundaries before outcomes. A negative result may justify documenting
limited reproducibility instead of extending simulation debugging indefinitely.
The useful contribution remains optional tracing plus honest measurement labels.

Before launch: freeze a distinct protocol, sources and commands, CPU-check the
live/fixed transport selection against the same native eval path and ensure no
policy/action difference is discarded. One idle L40S only, proposed10-minute cap.
The existing23:45:59CDT ceiling is not automatically extended; defer if fewer
than10 minutes remain when all gates pass. Never duplicate completed studies or
terminate another task. This is a draft, not a launched or completed experiment.

If the launch window closes, continue CPU-only extraction of a small optional
trace/check contribution and protocol documentation. Do not create a new timer,
new agent, additional GPU allocation or adapter sweep merely to fill the next tick.
