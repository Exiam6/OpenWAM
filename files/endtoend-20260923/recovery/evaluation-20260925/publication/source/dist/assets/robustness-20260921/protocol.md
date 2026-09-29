# Published-policy robustness and coordinate-preserving restoration pilot

Frozen before this round's outcomes, 2026-09-21. This is a small simulation pilot,
not the official 100-episode-per-task benchmark or a new RAE architecture.

## Fixed baseline
Published DINO/S-VAE checkpoint revision af1595c8ee54955116abbc0b0ffa17fa48deb691,
OpenWAM 7c5861e45cfe1339a0323f0e0b03a3316c37971c, RoboTwin
0aeea2d669c0f8516f4d5785f0aa33ba812c14b4. pick_dual_bottles, demo_clean,
unseen instruction templates, original expert-feasibility filter, 400 steps,
10 diffusion steps, seed 42 per generated chunk, compile disabled, original cache.
Run five expert-feasible starts from simulator seed 200000, separately for:
(1) clean; (2) independent Gaussian RGB noise, sigma=0.10 of the 0..1 range,
head camera only, clipped/rounded to uint8 before original preprocessing;
(3) simultaneous front camera replaces head, wrists/proprio/prompt unchanged.
Noise uses a local RNG keyed by simulator seed and control step; never mutate
NumPy's global RNG, simulator observations, expert checks, or language selection.
Record actual seeds, instructions, initial observations, outcome and action count.
Assert paired seeds/instructions/initial clean image hashes before comparisons.
Camera replacement is a large fixed camera transfer, not small viewpoint jitter.

## Offline adapter (train/validation only before selection)
Reuse pick_dual_bottles' fixed 35/5/10 episode split (permutation seed 42), with
12 evenly spaced current frames from the same complete 33-raw-step windows.
Only original published DINO + original published S-VAE + final feature LayerNorm
latents are targets. Verify a standalone encoder against the full policy before
using it. Freeze all original weights and statistics. Extract clean/noisy pairs
with sigma .10 at the head camera only; 2 independent train corruptions, one
validation corruption. Test episodes remain unopened for fitting/selection.
Compare identity, a zero-init residual Conv1x1(48,48), and zero-init residual
Conv1x1(48,96)-GELU-Conv1x1(96,48). 2,000 steps each, seeds 42/43/44, AdamW lr .001,
weight_decay .0001, batch 8, clean identity loss weight 1. No sweep. Apply only
observed frame zero; future slots pass through unchanged. Train in float32;
cast output back to original encoder dtype at inference. No policy finetuning.
Select architecture on mean validation restoration MSE across the three seeds.
Eligibility: noisy-to-clean MSE <=90% of identity and clean-output distortion
<=10% of identity noisy-to-clean MSE. If none qualifies, stop without deployment.
Deploy seed 42 of the selected architecture (predeclared, not best seed).

## Frozen confirmation, only after eligible adapter selection
On five expert-feasible starts from 300000, run clean/noise x identity/adapter
(20 rollouts). Do not select adapter, noise strength or checkpoint from these
outcomes. Confirm coordinate and zero-init equivalence before interventions;
record adapter hook call counts and maximum change to verify it was exercised.
Report all paired outcomes and exact discordant-pair test; five episodes only
permit descriptive/uncertain conclusions. Preserve all failures and videos.
Offline validation improvement without closed-loop gain is a negative result for
policy benefit. Do not call gripper transitions contact labels or claim contact
robustness. No repeated weights/noise tuning after confirmation.

## Resource / integrity bounds
At most one healthy idle L40S for policy+sim and one for extraction/small training.
Preflight UUID/process/ECC check and own-process cleanup, preserve other users'
jobs. 45-minute timeout per five-episode condition. Correct success denominators
from recorded episode outcomes (retain the known buggy native result file).
Source hashes and checkpoint provenance saved with results. All checks and
operational fixes logged, no silent exclusion of failed evaluated episodes.

## Operational correction before condition comparisons (2026-09-21 13:31 CDT)
Source inspection found Python random.shuffle/choice in RoboTwin's instruction
list generation, while setup only seeds NumPy/Torch. The initial clean-only
engineering attempt is invalid for paired comparison and is retained separately.
The wrapper now scopes Python random to the actual simulator seed around
instruction generation and restores its prior state afterward. Rerun all three
baseline conditions and use this correction for confirmation. No outcome-driven
model, corruption, selection, episode-count or seed-range change. Offline adapter
selection completed before this discovery and remains frozen.
