# ex4-f_nativecalibration_1017 — does restoring native calibration improve policy success?

Registration before GPU launch. Group ex4 asks why two tasks scored zero; this
experiment asks whether the verified execution-path fix transfers to learned
policies, before drawing conclusions about RAE representations.

Baseline: frozen ex3 runtime rae-factorial-20261010, source model e983676b,
native evaluation snapshot identified by eval-freeze.json; original results in
evaluation/fresh. Fix implementation parent 950e8661. Expert ex4-e L1 reproduced
old failure and fixed success on handover4101100, initial state exact, no planner
failures; CPU branch audit and independent URDF yaw predict the old offset.

H: restored ALOHA calibration raises handover success above zero. Competing
explanation: policy errors dominate even after execution is corrected. Single
intervention: relocate identical planner YAML while retaining the literal
aloha-agilex marker, selecting native base rotation. No accumulated target
compensation. No model retraining, checkpoint selection or simulator changes.

Held fixed: ex3 endpoint20000 checkpoints for raw768_native and svae48_native,
training seeds42/43/44; original native server (denoise10, horizon8), prompt,
episode RNG, physics, cameras, state/action normalization, native task limits
and success. The sample is the first five accepted scenes in each ex3 manifest,
in original order, selected before inspecting their outcomes. Historical counts
were inspected after choosing this rule; it is a development pilot, not an
unseen test. Full20 scenes remain frozen; no re-filtering after calibration.
Scene lists and all hashes are frozen in registration.json before launch.

Primary: handover success per arm (15 episodes each), plus paired win/loss
against original ex3 rows. Secondary: adjust control, place exploratory,
per-training-seed results, planner failures, latency and execution traces.
Prediction: >=2/15 handover successes for at least one arm. This supports
transfer on this pilot only. Zero for both arms leaves policy learning as a
remaining obstacle; it does not refute the separately verified executor bug.
One success or mixed results are undecided. Adjust loss >=20 percentage points
is flagged as regression, not hidden. No RAE ranking claim from this pilot.

L0: verify immutable input/code/checkpoint/scene hashes, remap regression tests,
expert L1, actual planner branch audit; compare evaluator byte-for-byte with ex3.
Both conditions use identical evaluator and server code. Only migration_render
differs. Old condition uses the original ex3 migration helper; fixed uses the
tested remapper and experiment-private YAML copies. Both YAML hashes match.

L1: training-scene smoke for every checkpoint and for both execution conditions:
finite 20D action, deterministic server reset, exact native step limits and
renderer identity. No scored scenes used in the smoke gate.
L2: fixed adjust first5 for seed42 in both arms. Require at least one success
per arm and zero infrastructure errors before continuing that arm. Existing
expert L1 supplies an independent successful handover execution control.
L3: old-path handover first scene per cell must reproduce its original failure,
then fixed handover first5 and place first5. Finish other seeds, including their
fixed adjust first5. Six old controls +90 fixed episodes total. Preserve every
failed attempt. No retries or overwrite. Inspect full paired video and measure
runtime before registering any larger experiment; no full900 launch here.

Budget <=6 occupied GPU-hours (two lanes, <=3 active hours each), deadline
2026-10-18 12:00 CDT. Estimate 3–5 GPU-hours from ex3 timings, no training.
GPU3/4, one new cell each; own-user occupants only, initial utilization <=85%,
>=32GiB free, three admission samples. Four CPU threads per subprocess. If
foreign jobs or insufficient capacity appear, wait before new cells; never
touch their processes. Resource review every30min. Stop on pause, time budget,
code/hash drift, failed integration/initial scene checks, control non-reproduction
or unexpected infrastructure failure. Diagnose in a new registered attempt.

Outputs: immutable code/environment/commands, per-scene JSON/frames/traces,
matched baseline extraction, per-seed summary, videos including failures,
scoped conclusion. This is a corrected execution benchmark version, separate
from ex3 and ex4c compensation. Future wide-head/shift evaluation requires a
separate registration after this pilot and completed expert validation.
