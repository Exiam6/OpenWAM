# Proposed follow-up: goal-aware compression with a state-retention requirement

Status: proposal only. No auxiliary compressor has been trained or evaluated.
This is a new intervention study, not an amendment to the completed layer study.
The original deadline remains 2026-09-23 01:15:12 America/Chicago; no extension.

## Question and evidence

Can a small training-only goal auxiliary loss improve visual goal information in
an L12-derived 48-channel SVAE without sacrificing short-horizon state readout?
The completed experiment showed that K4 improves the initial-waypoint endpoint
but worsens noisy state readout. This motivates a joint requirement rather than
promoting K4 as a general replacement. It does not establish irreversible
information loss, policy benefit, or superiority over a conventional VAE/RAE.

## Proposed matched intervention

Keep the frozen DINOv3 L12 source, SVAE architecture and 48-channel interface.
Compare reconstruction-only continuation against reconstruction plus a small
auxiliary loss, initialized from the same completed L12 checkpoints for each of
seeds 42, 43 and 44. Both arms receive identical examples, orders, augmentation,
optimizer schedules and numbers of updates; the original uncontinued checkpoint
is a descriptive anchor, not the equal-budget treatment control.

The auxiliary head consumes only the posterior mean from the initial head image.
Its target is the first downward gripper-crossing event's active-arm native XY,
using the already audited definition. It is not an object pose or contact label.
Use a task-specific training-only head and training-only target normalization.
No event-frame image, later state, future action, or test label enters the encoder.
The native reconstruction and KL objective remain active on the original sampled
training clips. Remove the auxiliary head at inference. An unchanged latent shape
does not imply compatibility with a released policy: changed weights alter the
latent distribution, requiring matched policy adaptation before a control claim.

A candidate budget is 2,000 continuation updates per arm and seed. The exact
auxiliary weight, head/pooling, first-frame sampling frequency, optimizer resets,
KL schedule, gradient clipping and numerical precision must be chosen using only
training information and written into a separate frozen protocol before launch.
There is no loss-weight sweep authorized by this document; no value is frozen yet.

## Implementation checks before any new GPU training

1. Native `SVAE.forward` exposes `mu`; reuse this computation for the auxiliary
   head rather than sampling another latent. With auxiliary weight zero, verify
   identical native loss, random-number state, encoder gradients and one update.
2. Verify finite nonzero encoder gradients from the auxiliary objective alone;
   decoder-only gradients are not evidence that the representation is learning.
3. Confirm sample-to-task/episode alignment and frame zero on training data. Check
   that perturbing later cached frames cannot change the initial latent. Do not
   infer input causality solely from using a future event as a supervised target.
4. Keep preprocessing and color conventions unchanged: the native dataset writes
   RGB arrays directly through OpenCV JPEG encoding, and existing decoding follows
   that contract. Do not introduce a conventional BGR swap without an actual test.
5. Profile a bounded training-only run on an idle device; measure both arms' memory
   and throughput. A short native-policy profile is not this compressor profile.

Update: a separate bounded CPU FP32 contract check passed on one existing
training frame and the seed42 checkpoint. Zero-weight native loss/RNG/gradient/
AdamW update equality, nonzero auxiliary encoder gradients, and zero future-input
gradients passed. See ../goalaux-20260922/attempt2/contract-result.json. This is
not a full data-alignment audit or a GPU BF16/throughput check; those remain
pending. No new compressor has been fitted and no benefit has been measured.

## New confirmation data and analysis

All 60 scenes from the completed confirmation are now exposed development data
for this follow-up. They cannot be relabeled as fresh or used to select a favorable
loss weight and then support a new confirmatory claim.

Before any new fit, freeze a new consecutive candidate range (proposed starts
800000, 801000, 802000), first 20 expert-feasible scenes per task, at most 40
attempts per task, at most two collection hours. Check collisions and all native
collector settings first. Preserve rejected scenes and reasons. All old 80 policy
evaluations and the stopped context pilot remain untouched.

The primary decision must jointly require goal improvement and state retention.
A proposed rule is a negative upper paired 95% confidence bound for relative
noisy goal error and a less-than-5% upper bound for noisy and clean state error.
Freeze the final thresholds, clean-goal retention check, task-regression rule and
multiplicity treatment before training. Use task-stratified paired episode
bootstrap, average paired noise replicas and compressor seeds within episodes,
and report every task, translation/gripper component, source and comparator.
Fixed ridge is primary; fixed MLP is a secondary check. Keep raw and PCA controls;
proprio-only and constant-goal baselines remain visible. No new endpoint may
silently replace a failed primary.

Stop if checks, sample counts, resource caps or joint criteria fail. Do not select
new seeds, a best checkpoint, a favorable subset or another loss weight afterward.
Insufficient sample size or an interval crossing a margin is inconclusive, not a
successful noninferiority result.

## Resource and contribution boundaries

At most four simultaneous GPUs, 48 total GPU-hours, 24 CPU threads and 80 GB new
cache across the original study window; subtract resources already used. Recheck
actual ownership and idle resources immediately before launching a bounded job.
If fitting, collection, scoring and audit cannot fit the remaining original
window, record the unlaunched stage rather than extending the deadline.

The useful upstream deliverables already available are the optional sample-cache
provenance patch and auditable representation diagnostics. Any auxiliary-loss
patch remains experimental until this separate study passes; a closed-loop claim
additionally needs matched policy adaptation and new policy rollouts. Do not
submit or message upstream on the basis of this proposal alone.
