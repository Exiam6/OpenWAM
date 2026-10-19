# Matched Wan VAE baseline: static preparation

Status: CPU preparation only, before any new GPU launch. This is a new research
line, not an amendment to the active calibrated factorial or official control.

Question: under identical demonstrations, minibatches, trainable trunk, training
steps and calibrated evaluation, does the raw DINO representation improve closed-loop
success relative to the pretrained native Wan VAE? The existing five-arm study
contains no Wan VAE arm. The published official policy differs in size and training
history and serves only as a functional control.

Proposed intervention: raw768 DINO latent -> Wan VAE48 latent. Keep the 12-layer,
1024-wide video trunk, 512-wide action backbone, 20D state/action, 20000-step final
checkpoint, native heads, video/action shifts5, batch3 x accumulation2, optimizer,
seeds42/43/44 and independent paired action-noise streams. Same-shaped trainable
parameters must match the existing per-seed initialization exactly; input/output
projections necessarily differ. Frozen encoder pretraining differs by definition.
Equal examples/steps do not imply equal FLOPs, parameter count or wall time; report
these costs separately. Do not import any trained policy weights from the larger
published checkpoint.

Baseline: the frozen raw768_native seed42/43/44 checkpoints in
`/home/zifanz4/openwam-runtime/rae-factorial-20261010`, with corrected execution
results from ex4-g_calibratedfactorial_1017 once completed and independently audited.
The same data cache contains6120 train and304 development windows. Use its exact
metadata order, raw RGB frame indices, normalization, text tensors, action/proprio
labels and minibatch schedules. Keep all failed collection attempts in the ledger;
do not add demonstrations or select scenes based on the current outcomes.

Proposed endpoint: the same20 frozen development scenes per task across three
training seeds, three tasks,180 episodes per arm. Primary is task-balanced paired
raw-DINO minus Wan-VAE success. A practical raw-DINO benefit requires at least5pp,
positive differences in at least two of three training seeds, and a paired95%
bootstrap lower bound above zero. Upper bound<=0 refutes a positive benefit on this
cohort; otherwise undecided. Use10000 crossed seed/within-task scene resamples,
RNG921018, preserving arm pairing. Compact DINO-SVAE48 is a secondary dimensionality
control; no best-arm selection. Known development scenes cannot establish unseen
benchmark generalization. Three training seeds cannot establish broad robustness.

Competing explanations: both representations may be limited by training scale or
an unresolved serving/data issue; the ongoing official functional control and
corrected factorial inform whether a full new comparison is worth its cost.
Positive/negative results cannot isolate the effect of encoder pretraining from
latent representation, nor rank methods at the larger official scale.

Current gate and budget: **zero GPU-hours authorized by this preparation plan**;
CPU-only inspection is limited to three wall hours with at most four CPU threads.
No model training, GPU encoder cache, or simulator launch is part of this stage.
Before any GPU work, commit a separate concrete admission/cost amendment defining
cache, tiny-training and full-training hard caps within the existing allocation
and deadline, informed by completed G/H evidence and real throughput. Never extend
the current G/H budgets or occupy their resources to fill a utilization graph.
If the complete endpoint cannot fit, preserve preparation and state that limit.

Static deliverables: independently hash all source trajectories and the old cache;
prove every cached label and initial state against normalized raw states; freeze
all6424 RGB window indices; prove train/dev disjointness and shape contracts; replay
all three fixed minibatch schedules; inventory frozen VAE tensors and common
initialization artifacts. GPU-stage gates, if admitted later: exact cache/deploy
preprocessing, native first-frame causality, paired noise/initialization, finite
nonzero gradients and save/load consistency, easy-case behavior, measured cost,
then a fixed full endpoint. No retries, overwritten attempts or checkpoint selection.

Stop static preparation on hash mismatch, missing data, split leakage, inconsistent
labels or nonmatching schedules. Diagnose the preserved evidence before declaring
readiness. Existing jobs, caches and checkpoints remain unchanged.
