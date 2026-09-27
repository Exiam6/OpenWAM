# Equal-noise goal-readout control

Completed 2026-09-23 00:05:20 CDT, exit0; one previously idle A40 released.
Training105episodes with7variants each: cached clean plus3independent draws at
sigma.04 and3at sigma.10. All21newtask-specific ridge fits used identical
scene-disjoint fivefold CV, training-only statistics and alpha grid. Ridge
penalty was scaled by7to preserve regularization per independent scene; a
synthetic repeated-clean fixture verified unchanged predictions (2.44e-13).
No encoder/reducer updates. Existing60scene evaluation vectors were reused,
with all tasks, clean/noisy conditions, both Wan normalizations and3SVAEseeds.
The epoch/sample count is not735independent scenes.

## Results: expert initial-image first-close waypoint XY NMSE

| Route | Clean-trained clean | Noise-adapted clean | Clean-trained noise.10 | Noise-adapted noise.10 | Noise-adapted noise.04 |
|---|---:|---:|---:|---:|---:|
| Wan48 native |0.2384|0.3685|182.6310|0.4148|0.3159|
| Wan48 per-token LN |0.2371|0.2762|314.5404|0.4323|0.3228|
| DINO raw768 |0.1197|0.0742|0.3832|0.1139|0.0893|
| DINO PCA48 |0.1351|0.0898|0.6421|0.1107|0.0931|
| DINO SVAE48 seed mean |0.1760|0.1030|0.6014|0.1160|0.0974|

Wan's previously enormous noisy error largely disappears after fair training
adaptation. Its original182.63versusDINO0.38gap cannot be described as intrinsic
representation information loss. The new intervention changes the training
mixture, training normalization and readout jointly; it does not separately
identify each one's causal contribution. Clean Wan performance regresses
(native+54.6percent, LN+16.5percent point estimates); those regressions remain.

A substantial DINO-associated advantage remains under equal augmentation.
At noise.10, PCA48 versus nativeWan48 changes error by−73.31percent, descriptive
paired95percent interval[−82.47,−62.32]; versus WanLN it is−74.40percent,
[−83.27,−63.87]. These are both192-dimensional pooled visual controls.
RawDINO andSVAE48 also have lower errors than both Wan routes; every one of
these comparisons has the same direction on each of the three tasks. Report all
contrasts, not only the best. RawDINO's pooled dimension3072is not matched to192.
PCA andSVAE errors are close; this run does not establish that a learned SVAE is
necessary or that PCA is superior in full world/action training.

## Validation and limits

Saved-artifact audit passed all21fits and63prediction conditions. Training
normalization matches exactly; ridge normal-equation residual1.02e-14;
expanded-affine prediction residual1.01e-15; episode-metric residual8.53e-14.
All7variants of a scene stay in one CV fold. No evaluation vectors or outcome
labels were accessed by the fitting stage. The complete run took47.25seconds,
including training-only encoding and cached-vector evaluation.

Three fixed training-only hardware checks passed predeclared tolerances.
Wan cached vectors were bit-identical; DINO/reducer features differed across
devices: max task RMS0.04597and max coordinate0.22464in clean-training standard
deviation units (limits0.05and0.5). This is approximate parity, not bit identity,
and must remain a numerical limitation. Only those three frames were checked.

The60evaluation scenes and noise severities were already exposed. All intervals
are descriptive, not independent confirmatory inference. This measures only
initial-image waypoint information, not actual contact, object pose, short-state
noninferiority or closed-loop policy success. Architecture/pretraining/objective
differences are confounded. DINO has no trained pixel decoder in this path; full
RAE reproduction remains incomplete. No56/60partial cohort or old80policyreplay.
The original Sep23 01:15:12CDTdeadline remains fixed.

## Next uncertainty

Within the unchanged01:15:12CDTdeadline, prospectively freeze the short-horizon state counterpart of the equal-noise control: identical105training episodes and12fixed current frames per scene, all frozen representations, noise0.04/.10 draws, episode-disjoint CV and alpha_eff=7*alpha relative to the original12-frame-per-scene fit (not84*alpha). Reuse already extracted initial-frame noisy vectors from this run and all original clean/evaluation vectors; only encode missing training-frame variants. Retain all translation/gripper, clean/noisy, task and seed results. This resolves whether the goal-information advantage also preserves short-state information. No world/action-policy gain is claimed; no expired56/60 or old80 replays. No launch follows automatically from this report; freeze and validate code/resources first.
