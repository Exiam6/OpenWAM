# Supplemental descriptive analysis —2026-09-22, before fresh scoring

Primary protocol/checkpoints/selection/success criteria are unchanged. This is a
post-development explanatory analysis, NOT an additional primary hypothesis or
candidate-selection rule. No new fitting, extraction or simulator use.

Noise error decomposition uses exact saved predictions and the same training-only
label scales. For each coordinate let e=(clean_prediction-target)/scale and
q=(noisy_prediction-clean_prediction)/scale. Then noisy squared error minus clean
squared error is exactly2*e*q+q*q. Average within translation/gripper, episode,
reducer seed, then task in the original equal-weight order. q*q measures movement
of this frozen readout's prediction under noise; it is not proof of irreversibly
lost latent information or a unique causal mechanism. Verify the decomposition
identity numerically for every condition, publish all sources/compressors/seeds.

The12hour plan requested gripper-change phases but did not freeze a numerical
predicate in the initial execution protocol. Explicitly fill that reporting gap
here as supplementary, after aggregate development results were seen and before
fresh scoring: native gripper-state change over rawframe+4 has
max(abs(future_gripper-current_gripper))>0.001; all other rows are stable. This
threshold is a fixed numerical guard, not tuned to outcome or phase size. Current
gripper is the last2coordinates of16Dproprio; future is the last2labels of8Dtarget.
Publish both phases and per-task frame/episode counts; zero-sized phases are
unavailable, not dropped silently. Interpreting phases requires available label
range; do not call gripper change physical contact. No phase-specific improvement
can replace the whole-dataset primary result. This addition is not described as
preregistered before development.

Existing raw/PCA/SVAE spatial-shuffle and visual-mismatch controls are also fully
reported. Their development-only status and the missing fresh spatial controls,
missing wrist-trained within-view baseline and missing contact labels remain
explicit. New sources or loss sweeps require a separate prospective study.
