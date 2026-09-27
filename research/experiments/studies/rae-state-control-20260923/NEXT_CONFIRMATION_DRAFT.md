# Next independent confirmation — draft, not launched

This document prepares future work. It does not extend the current study's
2026-09-23 01:15:12 CDT deadline, reserve GPUs or authorize a new time budget.
The completed equal-noise controls are exploratory on the exposed 60 scenes.
Selection of PCA48 as the simple prospective primary candidate is based on those
exposed results and must be disclosed; they cannot serve as its confirmation.

## Question and minimal intervention

Do fixed compact pretrained features preserve the observed goal/translation
advantage without the raw-feature gripper regression on previously unseen expert
scenes? Do not change any encoder, reducer, readout, normalization, alpha choice,
noise rule or metric after new scenes are observed. The intervention is the frozen
representation choice; no world/action policy is trained in this confirmation.

## Proposed design to freeze before any collection

- Three tasks in the existing order, 20 accepted expert-feasible complete scenes
  each, first 40 consecutive candidates at most per task. Proposed disjoint seed
  ranges start at 900000, 901000 and 902000. Check all prior manifests for collisions
  before freezing. Do not skip a candidate based on representation predictions.
- New directory and immutable manifest; the terminal goalaux56/60 cohort is not a
  source, and old60/80 are not replayed. Archive all expert failures, unstable
  scenes, infrastructure errors and partial worker outputs. Native expert plan
  and successful replay define the sampled population, not policy success.
- Same native camera/geometry/planner/proprio and current-frame causal encoding.
  Record actual Curobo planner types and rendering/GPU provenance. Missing assets
  or fallback planners fail the preflight before a collection claim.
- Reuse the already frozen equal-noise goal and state readouts. No new fitting,
  recalibration, hyperparameter search or winner selection. Same 12 state frames,
  initial-frame first-close XY target, clean and 3 noise draws each at .04/.10.
  Lock current target scaling, pixel/channel order and frame selection.
- All routes retained: both Wan normalizations, raw DINO, PCA48 and all three SVAE
  seeds (errors averaged within scene). Include cached-fit proprio-only state
  predictions. Distinguish 192 pooled visual inputs from raw3072.
- Primary pair: PCA48 versus native Wan; co-primary descriptive targets are
  noise.10 goal XY and short-state E. For an improvement claim, both paired
  two-sided95% bootstrap intervals must lie below zero. This intersection rule
  cannot be rescued by a better secondary route or noise severity. All individual
  scene differences and all contrasts remain public in the summary artifact.
- Translation and gripper groups plus clean error are mandatory endpoints, not
  optional diagnostic plots. A goal/state-E gain does NOT establish gripper
  noninferiority. Do not invent an acceptable gripper margin after seeing the new
  outcomes. Without an independently justified margin, report its difference and
  interval, and claim gripper improvement only if its full interval is below zero.
- Task-stratified scene bootstrap, 2000 draws, fixed seed20260922; never bootstrap
  individual frames/noise variants/SVAE seeds as independent scenes. These 60
  scenes may be insufficient for a precise gripper estimate; retain uncertainty.
- Freeze protocol, all code/environment/checkpoint/readout hashes and collection
  cap/deadline before launch. Score once only after the complete balanced60 and
  integrity gates pass. If incomplete at the new predeclared cutoff, archive it
  unscored; no automatic extension or smaller favorable substitute cohort.

## Feasibility and next budget

The prior complete native60scene collection took5651.85seconds (94.2minutes),
before feature extraction/scoring/audit. The original current window has less
than40minutes remaining when this draft is written. It cannot support a comparable
full cohort on the tested sequential path. A future allocation should allow about
three hours for checked environment setup, collection, scoring and independent
artifact audit, with a separately fixed collection cutoff and no more than one
idle ECC-clean GPU unless parallel collection is separately validated and bounded.
This is an estimate, not a throughput measurement on the proposed host.

## RAE and policy phase, only after the evidence supports it

The current DINO feature path is RAE-style but has no trained pixel decoder.
A complete RAE reproduction requires its matching frozen encoder and trained
reconstruction decoder, separate reconstruction evaluation, and compatible
world/action model integration. Training a decoder alone leaves the frozen
readout features unchanged and cannot explain a gain in these readouts.

For a controller claim, compare matched world/action training runs with aligned
feature normalization, spatial/temporal shapes, data, updates and policy inference
budget. Do not inject a new latent into an old policy checkpoint and call the
result an encoder comparison. Profile a bounded native training step first;
the earlier20minute profile timed out and did not establish two-GPU throughput.
RAE/PCA/SVAE policies require a separate realistic budget and fresh closed-loop
scenes. Viewpoint and true contact robustness remain untested.
