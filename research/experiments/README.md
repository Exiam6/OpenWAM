# OpenWAM representation pilot

Execution root: cm006:/vol13/zifanz4/openwam-experiments
Public report: https://embodied-research-notebook.exiamzifan.chatgpt.site
Pinned upstream: 7c5861e45cfe1339a0323f0e0b03a3316c37971c

## Scope
This is an offline representation and compression experiment on one RoboTwin 2.0 simulated task, adjust_bottle. It is not an OpenWAM policy rollout, a closed-loop success-rate reproduction, or an equivalent reproduction of Figure 7.

Only the frozen DINOv3 and S-VAE tensors are extracted from the official, public, revision-pinned OpenWAM Study checkpoint using validated HTTP byte ranges. Encoder state dict loading is strict. The original upstream DinoV3VideoEncoder, SVAE, and multiview assembly implementation are imported without editing. The minimal pilot environment does not install the full WAM/DeepSpeed/robot simulator stack.

## Data
- Official aloha-agilex_clean_50.zip, SHA-256 checked against the HF LFS metadata.
- 50 episodes, deterministic 35/5/10 train/validation/test split, split seed 42.
- 12 evenly spaced, complete 33-step windows per episode, 600 total.
- Video stride 4: 9 frames, native 384x320 three-camera composition.
- DINO features: 768 channels, 3 latent times, 24x20 patches.
- PCA and newly trained S-VAE statistics use only the training episodes.
- Current-frame future-invariance check passed; regression probes use only current visual features.

## Experiments
Baseline: raw DINO, PCA-24/48/96, fresh S-VAE-48 at 500 steps, published S-VAE-48 as contextual reference only.
Follow-up: newly initialized S-VAE-48, fixed 2000-step budget, seeds 42/43/44. The follow-up was chosen after viewing the pilot, so it is exploratory and reuses the same episode split. A new-task holdout will be needed for confirmatory claims.

The published S-VAE may have seen all these episodes during its original training. It must not be ranked as an independently held-out method against locally trained PCA/S-VAE.

Probe: linear ridge regression on current-frame 2x2 pooled features after per-token LayerNorm, optionally plus current proprioception. Target is achieved end-effector XYZ displacement four raw time steps ahead (6 dimensions), plus future achieved gripper state (2 dimensions). It is not the simulator controller action. Feature and target scaling are fit on train only; ridge alpha selected on validation only. Test reports standardized MSE, translation RMSE, gripper RMSE, and an episode-cluster bootstrap interval. The target is deliberately limited and is not a full action representation.

## Reproduce
Use the dedicated environment and explicitly assigned GPU. The initial driver is scripts/run-pilot.sh; it checks GPU memory, utilization, active processes and uncorrected ECC, takes an experiment lock, and enforces a 45-minute timeout. scripts/run-seeds.sh performs the fixed-budget seed extension; launch under a 20-minute timeout as recorded in the activity log. Both are host-specific launchers and must be adapted if moving GPUs.

Upstream asset revisions, source hashes and package versions are recorded in experiment-provenance.json and requirements-lock.txt. Data manifests, raw outputs, training curves and model states remain under results/. Cache files should be reused rather than extracted again.

## Next decision
Add held-out tasks and randomization, then check whether stronger reconstruction predicts better policy behavior. Full WAM training remains a separate memory/throughput gate; this small model experiment does not establish that a 5B training run fits on the same GPU.
