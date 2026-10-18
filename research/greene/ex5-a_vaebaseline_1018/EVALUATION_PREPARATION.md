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
