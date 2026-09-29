# Two-command planning/physical-state attribution probe, 2026-09-21

Freeze before new simulator outcomes. Six SHORT OPEN-LOOP traces: two fresh
simulator processes, each same ordered clean scenes400000,400001,400002. Reuse
first two20D outputs from stageC round0-clean, immutable commands independent of
new observations. Native ModelClient.step builds requests and converts actions;
only its transport returns saved commands. No policy server or model inference.
Keep native prompt/reset/ten-way language RNG and original scene replay. Initial
complete request hashes must match the six-input capture's clean records.

## Deliberate diagnostic changes, not a rerun of full policy evaluation
Use the native per-task override mechanism in process memory to cap this probe
at TWO actions, leaving the original400-step YAML and all prior source unchanged.
Native two-step success/failure printouts are not policy outcomes or success rates.
Round1 repeats the same three-scene order; each round has a fresh simulator
process. Later scenes have only two preceding commands per prior scene, unlike
stageC full rollouts. Do not claim identical full-rollout historical context.

## What is recorded, without changing native planning/execution
Reuse passive request/action recorder. For each of the two actions capture:
- pre-action three-camera/EEF/native drive-target vector and complete JSON request;
- before/after actual articulation qpos/qvel/root pose from native getters;
- before/after EEF-link global poses and bottle1/2 poses;
- exact native CuroboPlanner.plan_path arguments and returned status, positions
  and velocities for left then right arm, preserving original input/result objects;
- original20D command, exact converted16D target and counters.
No extra get_obs, renderer calls, planner calls, policy calls, physics steps or
seed reset. No physics/render/planner settings altered. This instrumented path
requires in-process separate left/right planners; otherwise stop and retain logs.
The snapshot covers specified measured fields, not every simulator hidden state.
Bottle velocities/contact solver caches are not recorded, so no global-state claim.

## CPU gates, fixed artifacts and resource bound
Before GPU: exercise actual native eval and the actual transformed RoboTwin loop
with fakes. Verify native action values/type, plan argument/result identity,
return values and counters, planner restoration, RNG preservation, distinct actual
qpos vs drive-target recording, fixed-source command fidelity and exactly two
calls in each of three scenes with original ten-choice RNG consumption.
Freeze protocol, source, original scene/capture/summary hashes and12 command-use
hashes (six unique commands replayed twice). Refuse overwriting prior results.

One idle cm001 L40S GPU2 under the existing gpu-cm001-2.lock; three samples with
memory<100MiB/utilization0/ECC0/no compute jobs before each process. Timeout is
at most TEN wall minutes, including setup, and must also end by the pre-existing
23:45:59CDT ceiling today. Defer if fewer than ten minutes remain at launch.
Do not terminate other tasks or extend the old study allocation. Preserve failures.

## Prespecified comparisons (all three pairs, no favorable selection)
Compare both commands in all three scene pairs, recording exact equality plus
max absolute numerical differences for each recorded field. At step0 and step1,
compare in order: measured pre-state, native planner inputs/outputs, measured
post-state, observed EEF/images at the next available control observation.
Report each arm separately and distinguish command drive targets from measured
positions. Record whether initial measured states differ despite matching policy
requests. Compare the new first two observed requests with stageC descriptively;
no exclusion if post-initial observations differ. Keep every trace and source.

Identical planner inputs with different returned paths supports inspecting the
planner/context. Equal returned paths with different observed post-state supports
examining initial/hidden state and physics execution. Equal recorded state with
different images supports inspecting rendering but cannot prove renderer fault.
Temporal ordering does not establish a specific library defect. Do not train,
select an adapter, change normalization or claim a policy improvement here.
