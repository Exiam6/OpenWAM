# RAE-informed OpenWAM contribution plan

2026-09-21. Research proposal; no new policy improvement has been demonstrated. The observation adapter is an untrained structural prototype. Keep it separate from the completed held-out-evaluator contribution patch.

## What is already known

- Original RAE uses a frozen pretrained representation encoder with a learned image decoder. If the encoder is frozen, improving only that decoder does not change the features consumed by a policy. RAE image-generation results do not establish robot-control robustness. Source: https://arxiv.org/html/2510.11690v1 and https://github.com/bytetriper/RAE .
- OpenWAM Study already compares DINOv3 and V-JEPA representations, both directly and compressed with S-VAE to 48 channels. Figure 7 uses a randomly initialized Wan-shaped backbone to isolate encoder priors; compressed DINO approaches Wan-VAE, while the direct high-dimensional alternatives perform worse. This is not a controlled test against the full pretrained OpenWAM-alpha recipe. Source: https://arxiv.org/pdf/2609.07398#page=12 .
- RAEv2 combines several late encoder layers while preserving channel width. Its action-conditioned navigation experiment evaluates future-video prediction, not a robot's closed-loop success. Multi-layer fusion is prior work; our possible contribution is a controlled compression/robustness evaluation and a useful OpenWAM integration. Source: https://arxiv.org/html/2605.18324v1 .
- Our completed three-task pilot: S-VAE reduces feature reconstruction error substantially relative to PCA-48, but clean state-change probe gains are small and task-dependent. This motivates measuring control information separately. It does not prove pretrained features help or hurt compared with an unpretrained encoder.

## Hypotheses to separate

H1: Last-layer semantic features omit some local geometry available in intermediate layers.
H2: Compression discards useful information already present in the frozen encoder.
H3: Observation corruption displaces otherwise useful features away from the policy's training distribution.
H4: A representation change breaks the policy's latent coordinate system, even with unchanged channel count.

Use raw versus compressed versions of each feature source to distinguish H1/H2; paired clean/noisy observations for H3; original latent targets and unchanged-policy rollouts for H4. These contrasts are diagnostic, not a causal isolation of pretraining from all other design choices.

## Candidate A: last-layer versus last-four-layer DINO features

Change only feature selection initially: K=1 versus the sum of the last four transformer-layer features, followed by train-only normalization and the same 48-channel S-VAE. Keep token positions, image preprocessing and causal temporal pooling unchanged. No larger backbone or pixel decoder.

Implementation gate: K=1 must exactly reproduce the original `last_hidden_state` output. Inspect the Hugging Face model's final LayerNorm: entries in `hidden_states` need not equal the post-normalized final output. Use an explicit original-output K=1 branch and record the K=4 normalization convention. Do not accidentally include an embedding state or class/register tokens in the spatial grid.

Initial matrix: three existing tasks; identical episode splits; seeds 42/43/44; original S-VAE loss and fixed 2,000-step screening budget. K=1 caches and runs may be reused only after numerical equivalence checks. Add K=4 PCA-48 and raw-feature diagnostics; include raw K=1. Fit per-source normalization and PCA on train only. Compare common-target control readouts, not reconstruction MSE across different feature targets. Use the same spatial pooling, standardized output targets, probe regularization search, data and parameter reporting; raw versus compact probes have different input sizes and are not a policy-capacity-matched comparison.

Measures: clean and corrupted held-out state-change readouts; event-conditioned errors; optional relative end-effector/object geometry only where valid pose labels can be recovered. Include a proprioception-only baseline to quantify the extra contribution from vision. Keep the clean-trained readout frozen under each perturbation; report both each method's absolute error and its clean-to-perturbed degradation.

Limitation: a new compressor creates a new latent coordinate system. Equal 48-channel shapes do not make it compatible with the released DINO policy, let alone the Wan-VAE alpha policy. If this candidate wins offline, quantify alignment to the old latent coordinates before deciding whether policy adaptation is affordable.

## Candidate B: preserve the original latent coordinates under noise

Freeze the released DINO encoder, S-VAE and policy. After the original compressor, apply a zero-initialized residual adapter to the observed latent frame only:

    z = original_encoder(observation)
    z_observed_new = z_observed + residual(z_observed)
    residual = Conv1x1(48,96) -> GELU -> Conv1x1(96,48)

There are 9,360 trainable parameters. The future latent slots are unchanged. At initialization the result is exactly the original tensor. This is not itself a new RAE.

Train on same-state pairs: clean image versus noisy/brightness-altered image. Targets are detached, original clean latents; add a clean-identity penalty. Freeze original encoder statistics. Use identity/no adapter, a zero-initialized linear 1x1 residual, and this two-layer residual as controls. Start with one noise family; select training strength and identity coefficient using validation episodes before unlocking the test comparison. Generate independent corruptions for train/validation; the existing perturbation cache contains test episodes only and must never be used to fit the adapter.

The adapter can reduce perturbation-induced displacement; it cannot recover information absent from the clean compressed target. A per-token MLP may also be too weak to recover spatially destroyed evidence. Neither limitation justifies silently expanding the model before the simple controls are evaluated.

Implementation status: `rae_observation_adapter.py` and three CPU tests exist. Tests passed: exact identity initialization and parameter count; current-frame-only modification without input mutation; frozen target/input gradients with trainable residual gradients. No optimizer run, no policy integration, no new success-rate result.

## Data available and genuine gaps

A read-only schema audit of all 150 downloaded HDF5 demonstrations (50 per task) found simultaneous head/front/left/right RGB and camera calibration, but no contact/force/touch/depth fields. Point clouds were empty in the sampled episode 0 of each task. Frame counts are 134-174, 273-315 and 228-293 for adjust_bottle, handover_block and place_object_basket respectively.

A useful real-view diagnostic is to replace the head image with the simultaneous front image while preserving both wrist views, robot state and labels. This is a fixed camera-pair transfer test, not a smooth viewpoint sweep. Train on head-view episodes; evaluate held-out paired head/front episodes with a frozen readout. Do not impose per-patch feature equality between different viewpoints without geometric correspondence; image coordinates refer to different rays. Calibration alone does not provide the missing depths.

Gripper transitions and end-effector motion can define event proxies. They are not contact labels. Genuine contact-phase claims require simulator contact events or a manually audited subset. Choose event rules using training data only and report coverage and per-event counts. Sparse uniform clips may undersample transitions; define and freeze any additional event-centered sampling before testing.

## Order of work and decision gates

1. Establish clean inference and paired perturbation rollouts with one released DINO/S-VAE policy. Verify asset/environment versions, action/state scaling and observation layout. Record policy diffusion seeds as well as simulator initial-state seeds. A rollout smoke test is not the official full benchmark.
2. In parallel with that baseline work, run the offline representation contrasts above on one available healthy L40S. No multi-node training is needed for this screening. Prior 1.04 GB peak allocation measured reducer training only, not DINO extraction or full-policy inference. Profile the full policy before deciding whether an additional GPU is needed for rendering.
3. Train Candidate B first if noise is the main published-policy failure mode. Prioritize Candidate A if raw/intermediate representations retain useful geometric information that the existing bottleneck loses. Do not combine both changes in the first comparison.
4. Choose the candidate on validation episodes; report all test tasks/seeds and paired episode uncertainty. Advance only with consistent corruption improvements and an explicitly reported clean-performance trade-off. A smaller reconstruction error alone is insufficient. If results remain ambiguous, contribute the evaluator and diagnosis without changing model defaults.
5. For one selected candidate, use the same checkpoint, paired simulator starts, diffusion noise and inference settings in closed loop. Report clean and perturbed success, confidence intervals, time-to-failure and contact-related failure categories where valid labels exist. Treat a small subset as a pilot; retain the upstream full-benchmark protocol for reproduction claims. No claim of improvement until these runs exist.

## Reviewable contribution boundary

First contribution: existing held-out evaluation entry point, plus a separately reviewable perturbation and event-bucket extension when validated. Second contribution only if supported: an optional, default-off adapter or layer-selection option with reproducible ablations. The completed evaluator patch remains independent of this proposal.

## Documentation correction

Released `num_frames=33` denotes 33 raw HDF5 time steps. With `video_stride=4`, this gives nine sampled images, matching our pilot. The previous public description of 33 sampled images was incorrect and has been corrected. Actual differences are sparse complete-window coverage (12 per episode), three-task scope, short reducer training and offline evaluation. Numerical results are unchanged. Source: https://github.com/OpenWAM-Official/OpenWAM/blob/7c5861e45cfe1339a0323f0e0b03a3316c37971c/openwam/dataloader/robotwin.py#L356 .
