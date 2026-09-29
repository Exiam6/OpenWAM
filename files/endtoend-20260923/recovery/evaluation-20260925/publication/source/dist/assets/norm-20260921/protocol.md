# Frozen normalization follow-up — 2026-09-21

Written before any outcomes of this round. Hypothesis: the trained residual
adapter's output mean/scale shift can affect control despite lower latent MSE.
This is a follow-up motivated by a known clean failure, not an independent
confirmation of the original hypothesis. No new adapter fitting or selection.

## Interventions
Use the exact previously selected MLP seed42, SHA256
f0d0cf8eeae4b87e3d16a742c8733e05b82ec737a445dec6cd18efedd3092586.
Four fixed arms: identity; renorm (original latent through the encoder's own
parameter-free LayerNorm again); adapter; adapter_renorm (adapter followed by
that same LayerNorm). Normalize only the current observed frame, never future
slots. Reuse original BF16 inference and eps=1e-6. Identity renorm controls for
numerical effects of repeated normalization. No interpolation or hyperparameter
sweep. Full published weights, policy settings and simulator stack unchanged.

## Offline diagnostic, no eligibility/selection gate
Reuse the previous validation latent cache (five demonstration episodes),
report all four arms' clean distortion, noisy-to-clean MSE, token means/RMS,
and episode-level paired uncertainty. These measurements do not choose the
intervention or decide whether to run it. Leave the ten held-out demonstration
episodes unopened: the new evidence will be fresh closed-loop scenes.

## Closed loop
Task pick_dual_bottles/demo_clean, unseen instructions, 400 actions, diffusion
seed42 per chunk, ten denoising steps, original cache, no compilation, no policy
finetuning. Same healthy GPU2 as prior experiment.

1. Regression diagnostic: replay ONLY the previously observed failing scene
   300003, all four arms, clean observation. This is an explicitly selected
   debugging example, not part of the fresh primary success rate.
2. Establish a fresh manifest from the FIRST TEN expert-feasible starts beginning
   at seed400000, using the original policy, clean observations and seeded
   instruction generation. Record all ten, independent of success. Those ten
   reference rollouts establish feasibility only; exclude their outcomes from
   the primary comparison, because repeated expert filtering consumes RNG.
3. Replay the frozen ten scenes and exact prompts in all four arms x clean / head
   RGB noise sigma=.20 = 80 primary rollouts. Sigma .20 is fixed here, doubling the
   prior .10 stress that reached a success ceiling; it is OUTSIDE the adapter's
   .10 training corruption and is not a noise-level sweep. Keep wrists, proprio
   and prompts unchanged. Use the same per-seed/per-step local noise RNG.
   Verify reference image/proprio hashes before every first action. Do not rerun
   the expert filter. All four arms have identical seed/prompt lists. Preserve
   all failures, videos, actual initial policy inputs, and native result files.

Operational order: identity server establishes ten references, freezes manifest,
then historical diagnostic and two primary conditions; renorm, adapter, then
adapter_renorm each run historical plus both primary conditions. No adapting the
plan from intermediate outcomes. Expected 94 total rollouts, 80 primary + 4
historical + 10 feasibility-reference. No exclusions for policy failures.

## Analysis / decision
Primary comparisons: adapter_renorm vs adapter, separately for clean and noise;
report paired wins/losses, exact two-sided discordant-pair test, Wilson intervals,
and all individual outcomes. Also compare renorm vs identity and each adapter
arm vs identity. Diagnostic scene results are separate and descriptive.
A rescued historical failure alone is not generalization evidence. No overall
benefit claim if a method only reduces MSE or a clean/noise tradeoff persists.
With ten scenes per arm, uncertainty remains large and one task cannot establish
RAE superiority, contact robustness, or broad manipulation improvement.

## Integrity / resources
One GPU, serial inference, advisory flock and idle/process/ECC preflight before
loading. At most 60 minutes per ten-episode condition. Stop on integrity errors,
record and disclose any operational amendments before resuming. Never overwrite
prior round results. Freeze and hash exact scripts/protocol before GPU execution.
