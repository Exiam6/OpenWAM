# Restore native handling of an expert-rejected candidate

Question: does randomized place_object_basket complete the same five accepted
native seed-0 episodes when the outer supervisor honors the official evaluator's
handling of the observed expert candidate rejection?

Baseline: ex8-a pilot protocol SHA256
44a564c62f6db851081157cdad9b78887ee48227ea8be58e974d697233f7e531,
run_native.py at bee8370, native checkpoint and runtime unchanged. The original
attempt stopped after 27 scored episodes (24 successes) at 07:40:44 UTC. Its
last group had 2/2 successes before an expert play_once candidate failed with
`AssertionError: target_pose cannot be None for move action.` The official
wrapper logs and rethrows; RoboTwin eval_policy catches this in expert screening,
increments the candidate seed, and continues. The outer supervisor incorrectly
classified this handled candidate exception as a fatal policy evaluation error.

Intervention: recognize only this exact logged place_object_basket.play_once
exception and native `error occurs !` catch marker as a rejected candidate. All
other tracebacks, planner prewarm failures, fallback, malformed records, missing
videos, budget exhaustion and process errors remain fatal. Preserve and count
the rejected-candidate log blocks. No candidate-seed substitutions or manually
selected scenes. No changes to official files, inference, planner, checkpoint,
normalization, cameras, success checks or native accepted-scene filtering.

Run only randomized place_object_basket, five accepted episodes from native
seed 0. Retain the old 2/2 as an interrupted diagnostic; never pool duplicates.
If complete, combine this five-episode group with the five complete ex8-a groups
for a 30-episode readiness report, retaining the original failure prominently.
The report is not the 10,000-episode official score. Prediction: the observed
candidate rejection is retained and skipped by native code, and five accepted
rollouts finish. If another exception occurs, stop for diagnosis; no auto retry.

Budget: one separate allocation, <=3600 seconds, deadline 2026-10-23T15:00Z.
Original attempt used 2388.86 seconds; original plus this recovery reserve is
<1.67 GPU hours, inside the initial 3-hour readiness budget. Existing idle/GPU
time-window policy remains in force. The full-suite 200-GPU-hour planning
ceiling is unchanged; no full dispatch from this recovery alone.

L0: inspect the native catch, regression-test actual and synthetic log blocks,
verify all frozen scientific inputs. L1-L3: finite 20D actions/state, five native
success-count entries and full-decoded videos, a preserved rejection count, then
independent combined coverage and cost audit. These gates cannot establish
reproduction of the paper numbers. A full-suite gap still requires diagnosis.
