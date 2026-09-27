# Development readout freeze, before development control scoring

Shared3-task reducers remain fixed. Each task receives its own identically
budgeted readout, representing a task-conditioned policy without adding task
identity to the frozen visual encoding. Per-task train35/dev5old episodes;
no new expert confirmation data in this development run.

Input current head only: normalized current-frame48channel grid pooled2x2,
plus16current proprio values. Raw768channel controls have larger input capacity,
labelled diagnostics, not equal-budget winners. Future achieved translation6D
and gripper2D labels at rawframe+4. Each group has its own train-episode5fold ridge
alpha1e-4,.01,1,100. Standardization is refit in each fold. Both groups get equal
weight, all tasks equal weight. Save all coefficients and decisions BEFORE scores.
Each frame has3paired noise draws for sigma0.10(primary),0.04(secondary), added
after resize to[0,255] RGB and clipped/uint8; identical RNG across source arms.
Per-episode noise/frames averaged; seeds42/43/44 averaged within episode before
stratified within-task paired bootstrap2000. These15development episodes are NOT
45independent seed replicates and not new confirmation. All seed outcomes kept.

Spatial shuffle independently permutes the four2x2cells per sample with fixed
seed777+sampleindex, preserving exact channel values; visual mismatch cycles
within-task episodes at equal sampled-frame ordinal, preserving proprio/labels.
Both are diagnostic interventions, not physical perturbations.

Development gate only: primary noisyEmean improves and cleanEmean degradation
<=5%; final strict>=10%plusCI and clean3%noninferiority belong to new confirmation.
MLP capacity check still required: two hidden layers128,ReLU,AdamWlr1e-3,
weight_decay1e-4,batch64,500optimizersteps,seed42,train-only normalization,
group-balanced standardized MSE; no early-stopping/model-size tuning.
K4 stays sole candidate even if L6/PCA wins a secondary comparison.

Pre-score numerical gate: re-encode each development conditioning image repeated
9times so the backbone batch geometry matches clean9frame extraction; use only
its first output. Include a zero-noise path and require all raw/PCA/SVAE pooled
vectors to match cached clean exactly before fitting/scoring. This is a numerical
control, not another perturbation strength for selecting a favorable result.

Launcher preflight amendment before vectors/scoring: after all9fits finished
exit0, the first readout launcher saw1MiB,11%utilization,ECC0 and stopped before
Python. No vector generation or scoring occurred. Retain readout.log; v2 checks
for no compute processes and memory<100MiB/ECC0, then waits up to20seconds for
utilization to reach0. It does not proceed on an occupied card or restart a fit.

Readout implementation correction, before scoring: vector stage failed when
opening first development image because cached path was already relative to
experiment root (begins data/), while readout prepended another data/. No vectors,
coefficients or development results were saved. Failed exit/log preserved.
readout_v2.py uses root/manifest_path and checks all manifest image paths before
recomputing unfinished vectors. No new fits, data changes, thresholds or scoring
choices. Prior9reducer checkpoints reused exactly, original readout.py unchanged.
