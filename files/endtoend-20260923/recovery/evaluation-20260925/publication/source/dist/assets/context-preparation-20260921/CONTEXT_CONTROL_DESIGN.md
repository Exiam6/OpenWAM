# Context-control design v2 — CPU prepared, NOT LAUNCH-READY

Supersedes only the unlaunched CONTEXT_PROBE_DRAFT.md design. No frozen completed
protocol, model, dataset or result is changed. Design fixed2026-09-21; runtime
integration, launcher and complete manifest still require a separate freeze.
The old23:45:59CDT allocation ceiling passed. No GPU allocation is extended.

## Question and contrast

Can online policy activity change repeatability of the observation/execution
path even when the executed commands are held fixed? The previous short probe
removed online inference and the server, added physical-state getters and
shortened predecessor episodes. Its negative result cannot isolate these factors.

Both new arms would keep a freshly initialized original model server resident
and execute the SAME saved commands from the FIRST scene400000, clean condition.
A resident_idle: server initialized/pinged/reset normally; no prediction calls.
B shadow_online: send each native payload once with predict_once and wait for the
response, but execute the saved command. Retain the complete online response,
real counter, latency and equality/difference to the saved action. Never replace
or discard commands or cases because the live action happens to differ.

The A/B contrast is the combined effect of online prediction, RPC and waiting,
NOT GPU load alone. No latency padding or outcome-dependent delay tuning. A
positive result does not locate physics/planning/rendering as the root cause;
a negative result does not exclude them or establish global determinism.

## Fixed scope and order for a future separately bounded pilot

First scene400000 only, no predecessor episode, clean/no perturbation. Each trial
has a fresh model server AND fresh simulator process. Proposed predeclared order:
A0, B0, B1, A1 (four trials, two commands each). Each uses the same checkpoint,
server seed42, simulation seed400000, prompt, initial observation expectations,
32-action synchronous executor and native20D-to16D conversion. No adapter.
B has two obs RPCs per trial: normally one32-action generation then one buffered
reply, NOT two independent generations. Thus eight executed commands total,
four online obs calls total, normally two generated chunks. No task success rate.
Only one pilot, no resampling until a desired result appears.

Saved source: results/repeatability-20260921/closedloop/r0-clean/
seed-400000-trace.jsonl first two rows, same source as the completed short probe.
New CPU source checks and immutable command hashes accompany this document.
Both arms use the SAME physical-state/planner getters and observation calls;
no extra get_obs, rendering or simulation steps. Identical startup health/reset
logic, including the native instruction-change reset before the first action.
No reset after commands begin; every trial requires a fresh process.

## Measurements and interpretation

Primary: exact native request, raw camera and EEF comparison at observation1,
which is after executing saved command0. Retain all four initial observations.
Secondary: actual qpos/qvel/root/link/object poses and planner inputs/results
before/after action0; action1 is retained to verify the same two-command boundary.
Native joint_action.vector is a drive target, not measured qpos. Left/right
articulation reads in this robot duplicate one whole-robot articulation.

Compare within-arm repeat pairs first, then both predeclared across-arm pairs.
Show all comparisons and actual response differences. With two repeats per arm,
report descriptive findings only, no causal or statistical significance claim.
Initial-control mismatches, server-counter mismatches, reset errors, ambiguous
network failures, or timeouts remain recorded; mark comparisons invalid where
needed, stop the affected trial, do not silently substitute extra trials.
There is no fabricated server step for idle responses: fixed_command_index is
local; real prediction counters exist only inside online_response. The old
TrajectoryTrace.predict assumes a server counter, so it cannot be reused unchanged.

## Prepared versus pending

Prepared scripts/context_trial_transport.py and six CPU checks in
scripts/check_context_trial_transport.py. Tests use mock transport and the real
native eval/converter; they create NO server, simulator, network or CUDA objects.
They check identical executed16D targets against saved source, idle zero calls,
shadow exactly-once requests, retention of unexpected response/counter, no input
or RNG changes, two-command bound, reset requirement and fail-stop after timeout.
This is preparation, not an experiment or evidence for either hypothesis.

Pending before any future launch:
1. Integrate a new context recorder that separates selected action/local index
   from actual server responses; freeze complete sources/configs/commands.
2. CPU-check native loop/reset/order, fresh-process runner, final summaries,
   abort/timeout paths and preservation of all failures. Existing6 tests do not
   certify these paths or real WebSocket behavior.
3. Establish a separate bounded allocation, idle-resource checks and hard wall
   cap from measured startup cost. Do not infer fit from GPU count or quietly
   reuse/extend the expired allocation. No launcher exists for this design yet.
4. Record residency/startup settings and elapsed intervals. Shared-node external
   activity and unobserved solver state remain limitations, even with ABBA order.

Stopping rule: once this one fixed-scope pilot finishes or hits its cap, retain
all data and stop. If the original divergence still does not reproduce, document
that limitation and return to representation/robustness work; do not expand
simulator debugging indefinitely. No repeated80evals or new weight selection.
