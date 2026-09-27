# Frozen diagnostic: action repeatability before another representation change

Prepared 2026-09-21, before any new capture/inference outcomes. This is a diagnostic
on reused scenes, not independent confirmation or a new success-rate benchmark.

## Fixed inputs and interventions
- Use exactly the first three entries of norm-20260921/fresh-scenes.json:
  400000, 400001, 400002, in that order. Preserve each exact instruction and the
  original ten-way NumPy instruction-choice consumption, despite a 3-scene subset.
- Capture the initial observation using the existing upstream setup/replay loop,
  then stop without executing any policy action. No expert refiltering. Verify
  original head RGB and EEF proprio hashes and instruction before accepting inputs.
- Save the actual ModelClient.step payloads, produced using all three native camera
  images, original prompt formatting, state casting and lossless PNG encoding.
  Produce clean and existing sigma=.20 head-only noise at action step0. Capture
  clients use an in-memory recording transport; no synthetic action is executed.
  The capture-only upstream failure counters are not policy evaluation results.
- Four arms remain identity, renorm, adapter, adapter_renorm with frozen previous
  weights, LayerNorm eps1e-6, BF16, and intervention on observed frame0 only. Do not
  change the published encoder/DiT/action model, caches, denoising settings or seed.

## Fixed sequence
A. One process loads the original full server. Identity: run the six payloads in
clean-scene order then noise-scene order, twice (12 generated chunks).
B. Stop A normally, start a fresh process on the same GPU. Identity: same six once
(6 chunks), then renorm twice, adapter twice, adapter_renorm twice (36 chunks).
Total54 generated chunks. Reset the original executor before every request. Use
server.predict on exact saved JSON payloads, original preprocessing and full
engine.generate; capture its full physical-unit chunk and the final first action.
Log the effective config, explicit process/repetition/input order, payload and
latent hashes, first action, finite checks, chunk shape and duration. Do not claim
an engine chunk equals a post-projection executed trajectory. No model selection.

## Fixed analyses
- Identity within-process A repeat1 vs repeat0; processB vs A repeat0 and repeat1.
- Each candidate within B repetition1 vs0; each candidate vs B identity on the same
  payload. All comparisons retained, no threshold tuning or cherry-picking.
- Exact byte equality, max absolute difference, RMSE and mean absolute difference
  on full raw generated chunks, and separately on final first actions. Groups:
  xyz indices0:3/10:13; rot6d3:9/13:19; gripper9/19. Rotation6D errors are parameter
  errors, not angular errors. Also report gripper sign changes at the policy's
  actual binary threshold when applicable. These metrics are not task success.
- Verify read-only tracing returns the original object and does not change random
  states; test replay's 3-scene subset against the actual upstream control loop and
  original ten-way instruction draw before capture.

## Optional stageC gate, not part of the54 requests
Only after validating the saved requests, repeat original-policy clean/noise twice
on the same three scenes: at most12 closed-loop episodes,400-step cap. Separate
protocol/source freeze for observational action tracing before execution. Preserve
order and restart boundaries. A matching first image/state alone cannot prove the
whole rollout is deterministic; find the first divergent action/observation.

## Resources and provenance
At most one L40S, cm001 GPU2 UUID GPU-025b7fc9-8f25-5467-5c28-8e59884ab6d3,
using the existing gpu-cm001-2.lock. Three idle samples: memory<100MiB, utilization0,
uncorrected ECC0, no compute process. Defer if busy, never kill others.90-minute
wall/GPU allocation ceiling including optional stageC; currently no stageC launch.
Separate results/repeatability-20260921; preserve previous frozen results and code.
Source hashes and manifests must be saved before launch; failed attempts retained.
No additional training, hyperparameter search, data download or public PR posting.
