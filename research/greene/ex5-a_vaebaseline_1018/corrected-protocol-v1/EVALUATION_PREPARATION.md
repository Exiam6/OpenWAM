# Matched evaluation implementation preparation

Status: CPU preparation; no VAE checkpoint, GPU reference prediction or rollout
exists. The separate GPU admission amendment is still required.

The corrected native evaluator, training-scene simulator smoke, renderer,
calibration mapping and task configuration are copied byte-for-byte from the
verified G study. There is no change to the success rule, physics, instructions,
20D state/action representation,10-step denoising,8-action execution horizon,
per-episode600s limit or the60distinct scene seeds. The new route label is
vae48_native. The template audit freezes15files; GPU registration must additionally
carry the complete engine, benchmark, scene and model asset inventory inherited
from the established study, together with the new runtime snapshot.

Each seed42/43/44 first evaluates the firstfive fixed scenes of each task
(15evaluations), then the remaining15per task (45evaluations). The final endpoint
is180unique evaluations. Pilot and remaining stages share a10800-second per-seed
occupied budget; pilot is additionally capped at3600s. Loading, smoke, reference
prediction and cleanup count. An actual video/cost release is required per seed;
policy failures remain valid outcomes and never select scenes or checkpoints.
All15pilot pairs must have complete videos, with the first-scene contact sheet
for each task actually reviewed. The remaining stage uses the same step20000
checkpoint. No retries or overwrite.

Before scored rollouts, the model receives each frozen scene's original PNG
camera inputs and initial state twice, with the same per-episode inference seed.
Its first8actions must repeat exactly. The simulator's first8actions are then
checked against that reference within1e-6, separately from the initial-state,
all-three-camera, action-seed, native-step-limit and instruction audits. These
references check deterministic serving without simulator feedback; they use the
same policy/server implementation and are not an independent policy implementation.
Failed partial reference files and logs are retained. Native image/cache agreement
is separately checked by deployment_contract.py before full training.

The batch supervisor starts allthree fixed training seeds; each completed seed
can independently enter its pilot, video review and remaining evaluation while
other seeds continue earlier stages. Completed child lanes release their locks
immediately. This avoids a barrier at the slowest seed without dropping any seed
from the180-evaluation endpoint. Actual process lifetimes are charged by the parent;
GPU occupancy is checked before each admission, and each child has its own budget.

The8.5-hour full-cap scheduling envelope needs an additional15-minute allowance
for per-seed video review. The preflight and training resource gates now reserve
that time. Availability remains conditional; do not claim that a future GPU will
remain free. If the measured complete endpoint cannot fit, do not reduce scenes,
seeds or training steps to make it fit.

CPU evidence: all60float32 initial-state hashes match the original raw-native
seed42 cohort. Tests challenge the validator with a previously audited real
rollout fixture and corrupted camera/state/action/seed values. Separate real CPU
child tests cover completion, failure, pause/deadline refusal, timeout/grandchild
cleanup, health failure and concurrent batch accounting. Resource admission is
mocked in the batch tests; none of these tests validates VAE GPU behavior.

Pilot evidence and continuation gate are prepared in make_pilot_pairs.py and
release_evaluation.py. All 15 Raw/VAE pairs use audited result and trace hashes,
matching training seed, scene and initial state. The three actual camera views
are shown for both policies, aligned by action index; shorter episodes hold their
final frame. Every video is decoded completely. The first scene of each task has
six sampled action indices for actual manual review, recorded with preview hashes.
This does not claim exhaustive visual inspection or an external whole-robot view.

Before the remaining 45 evaluations, the gate charges the parent's actual pilot
process lifetime. For each task it projects all 15 remaining episodes at the full
native action cap, using the median measured action-loop wall time per action
with a 10% reserve, plus maximum measured setup time. It adds the greater of 300s
or measured pilot non-episode overhead, and requires another 60s below the
remaining shared 10800s budget. Frame modification times measure the loop only;
inconsistent or nonfinite timings fail admission. This is a resource estimate,
not a guarantee or a success-rate extrapolation. A refusal preserves results and
requires diagnosis; no budget increase, truncated endpoint or automatic retry.

CPU tests exercise real short video encoding/decoding, shorter-episode alignment,
corrupted preview rejection, full native action caps despite early fixture stops,
and invalid/nonfinite timing rejection. Historical Raw timing is only a sanity
input for this calculation, never evidence about VAE runtime or task performance.
