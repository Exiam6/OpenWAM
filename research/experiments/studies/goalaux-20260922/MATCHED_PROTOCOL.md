# Prospective matched goal-auxiliary representation study

This is a separate intervention study motivated by the completed layer study.
It does not amend that study's failed primary endpoint. Freeze this document and
implementation/input hashes before retaining any new fit or collecting/scoring
new confirmation scenes. Original deadline: 2026-09-23 01:15:12 America/Chicago.

## Question and fixed intervention

Does initial-waypoint supervision improve a compressed L12 representation while
preserving short-horizon achieved-state information? The two arms are native
reconstruction-only continuation and reconstruction plus a goal auxiliary with
fixed weight 0.1. No weight sweep or post-result switch of primary endpoint.

Use the same frozen DINOv3 B/16 L12 features and native 768-to-48 SVAE architecture.
For each seed 42,43,44, both arms start from the same completed 2,000-step L12
checkpoint and receive exactly 2,000 additional updates. Compare to the matched
continued baseline, not merely the earlier uncontinued checkpoint. Retain the
final checkpoint only. Check equal sample-sequence hashes for the two arms.

Training episodes are exactly the original 35 per task (105 total) from the
frozen layer-study manifest. Load only these files. Reconstruction samples one
of 420 cached clips per task per update, giving batch shape 3x768x3x24x20.
The auxiliary input is only that sampled episode's frame-zero feature, shape
3x768x1x24x20. The native codec works per frame; no later feature is an auxiliary
input. Supervision is the earliest >0.5 to <=0.5 gripper-crossing active-arm XY,
with the original left-arm tie break, not contact or object-pose ground truth.
Targets are standardized using all 35 training episodes within each task.

The auxiliary head uses the posterior mean, per-token channel layer normalization
with eps=1e-6, fixed 2x2 spatial pooling, and a task-specific Linear(192,2) head.
This normalization explicitly matches the already frozen evaluation readout
interface: the earlier engineering prototype omitted it. Before retaining any
fit, check finite nonzero encoder gradient through this aligned path under BF16
and deterministic algorithms. The auxiliary head is training-only and removed
at inference. An unchanged latent shape does not prove released-policy compatibility.

Native loss is standardized-feature reconstruction MSE plus cosine loss plus
1e-4 times native mean KL. No new KL warmup. AdamW is reset identically in both
arms: lr=1e-4, betas=(0.9,0.99), weight_decay=1e-4; lr is multiplied by
min(1,step/20)*(0.95+0.05*cos(pi*step/2000)). Clip global gradient norm to 1.0,
including the auxiliary heads where active. Both arms construct the same heads
and optimizer parameter list; zero-weight heads receive no gradients. Use the
same per-seed CPU sample generator and reset neural RNG to that seed after
model/head creation. BF16 CUDA autocast with FP32 parameters, TF32 disabled,
deterministic algorithms enabled, CUBLAS_WORKSPACE_CONFIG=:4096:8.

## Prospective confirmation and fixed analysis

The previous 60 confirmation scenes are now exposed development data for this
new intervention. Do not use them for confirmatory inference or model selection.
No new development scoring is required to choose among these two fixed arms.

Collect task-specific consecutive candidate seeds beginning at 800000,801000,
802000 for adjust_bottle,handover_block,place_object_basket respectively. Before
collection, verify that none were already generated and retain the collision
check. Accept the first 20 expert-feasible scenes per task, with at most 40
attempts each and at most two collection hours within the original deadline.
Use the same native task config, robot, camera, planner, rendering and typed
failure classification as the preceding collection. No learned policy selects
scenes. Retain all rejected attempts and reasons. Infrastructure failures stop
the collector; fix only through a recorded amendment, without retrying completed
attempts or extending the original deadline. Fewer than 20 complete scenes per
task means confirmation incomplete; report it, do not replace seeds posthoc.

Freeze fitted reducer and readout hashes before viewing any confirmation scores.
For each task, fit only the original 35 training episodes. Use the previous
per-token-normalized, 2x2 spatial visual vector (192D). State inputs add the
current 16D proprioception; targets are displacements of both end positions from the current frame
to raw frame+4 (6D), and both gripper values at frame+4 (2D). Use 12 previously
specified uniformly spaced current-frame positions per episode. Normalize each
output with training standard deviation and compute E as equal weighting of
translation and gripper group NMSE, first averaging within episode and then task.
Goal inputs use only frame-zero visual vectors; the target is the audited XY
definition above. Goal E is mean normalized XY squared error.

Primary readouts: task-specific grouped ridge, alphas [1e-4,0.01,1,100], five
training-episode folds fixed by seed 20260922, independent translation/gripper
alpha selection for state. Use the unchanged layer-study normalization and
fitting implementation. Secondary: fixed 128x128 ReLU MLP,500updates,batch64,
AdamW lr1e-3,wd1e-4,seed42, matching the existing state/goal readout procedures.
No architecture/alpha/MLP-seed tuning based on confirmation scores.

Primary corruption is pixel Gaussian sigma0.10, clipped to valid image range,
three copies per input paired across arms and seeds. Clean and sigma0.04 are
also fixed. Preserve prior preprocessing, image color convention, frame
selection and deterministic noise-seeding rule. Zero-noise feature parity must
pass before scoring; raw labels, source assets and train/test identities must
be audited. No future frame may enter an evaluated current-frame prediction.

Average replicas, frames (state only), and three compressor seeds WITHIN episode.
Use 2,000 paired task-stratified bootstrap resamples of the 20 episodes per task,
seed20260922, and percentile95% intervals of relative aggregate E change. The
independent unit is one of 60 scenes, not a frame, noise replica or training seed.

## Joint primary acceptance, fixed before any new fit

ALL conditions must hold for goal_auxiliary versus reconstruction_only ridge:

1. Noisy goal E point estimate improves at least10%, and its paired95% upper
   confidence bound for relative change is below0%.
2. Noisy AND clean state E each have a paired95% upper bound below+5%.
3. Clean goal E has a paired95% upper bound below+5%.
4. No task's noisy goal or noisy state point estimate regresses by more than10%.

This is an intersection decision: failing any requirement means no accepted
combined improvement. Do not promote a secondary endpoint after failure. A
confidence interval crossing a noninferiority margin is not evidence of
preservation. Always report all tasks, both state groups, all seeds and both
readout families, including unfavorable results. Individual intervals are not
simultaneous guarantees for every descriptive comparison.

Keep proprio-only, constant-goal, frozen L12 raw768/PCA48 and the uncontinued
L12 codec as descriptive anchors with training-only readouts. They do not become
alternative primary candidates. Do not claim superiority to all representations,
full RAE-versus-VAE evidence, or policy benefit from this diagnostic study.

## Bounds and archival

GPU preflight v1 passed the zero-weight exact contract but then failed because
adaptive_avg_pool2d CUDA backward cannot run deterministically. V2 uses equivalent
fixed-grid block means, passes CPU forward/gradient equivalence and GPU forward/
deterministic-backward checks, and completes timing without replaying the passed
zero-weight check. Original failures and device-relocation records are retained.

Training: one currently idle, ECC-clean L40S,4CPUthreads,30minute cap for all six
fits including data loading; stop at the earlier original study deadline. No
other user's task may be terminated. Original aggregate caps remain4GPUs,48GPUh,
24CPUthreads,80GBnewcache; no automatic extension. Training and collector bounds
are separate but both count inside the original12hour window. Profile timing is
model-compute timing only and must not be reported as full end-to-end runtime.

If any fit, check, collection count, resource or joint endpoint fails, preserve
all attempted outputs and report failure/incompleteness. Continue only unstarted
stages or explicitly documented repair of failed work; never rerun old80policy
evaluations or the stopped context pilot. No policy adaptation, new policy rollout,
upstream PR or maintainer message is authorized by this implementation protocol.
