# Next diagnostic draft — separate planning from physical execution

Not launched. This does not change the frozen12-rollout study or extend its
original90-minute allocation. Finalize only after all six trajectory pairs and
post-run audits are reviewed. No extra policy evaluation/training in this draft.

## Uncertainty
Current completed clean pairs show different images/EEF at observation1, after
identical action0, while policy action differences first appear at replan32.
The saved joint_action vector is drive targets, not measured qpos. We cannot yet
locate the earliest difference inside planning, actual state, or rendering.

## Candidate minimal probe for a subsequent separately bounded study
Use the same first three clean scenes, fixed prompts and initial request hashes.
Replay each scene's first TWO recorded original20D actions through the native
20D-to16D conversion and take_action path, twice in fresh simulator processes.
Six short open-loop traces, not a new success-rate benchmark. No policy server,
new action generation, new scenes, model selection or additional closed-loop arm.

Before launch freeze a new protocol, the twelve fixed command hashes, source and
runtime budgets. Proposed ceiling: one idle L40S, ten wall minutes; do not infer
permission to extend the completed study's allocation. If timing/resource limits
cannot be satisfied, keep the draft pending rather than duplicating a launcher.

Record passively before and after each command:
- actual articulation qpos/qvel, observed EEF transforms, native drive targets;
- exact planner inputs and output status/position/velocity trajectories for each
  arm (retain numerical arrays, not only success flags);
- images and relevant actor poses/velocities where native getters are available;
- planning reset/seed path and process boundaries, with no reset behavior change.

Exercise the actual upstream call path in CPU fakes to verify wrapper argument,
return-object, RNG and counter preservation before GPU. Do not change planner
settings or physics determinism options in this attribution probe. Retain every
trace and compare in order: initial actual state, planner inputs/outputs, post-step
actual state, EEF, rendered pixels. Equal inputs with unequal outputs narrow the
component boundary; this still does not prove a specific library defect.

## What this enables
If the planning trajectories already differ, pursue planner context/reset or
numerical reproducibility with a separate default-off change. If planned paths
agree but actual state differs, investigate simulation execution. If actual state
and transforms agree while images differ, investigate rendering. Keep the latent
adapter frozen until the measurement baseline is understood. The useful minimal
contribution is a documented optional trace/check, regardless of whether an
accuracy-improving representation is found.
