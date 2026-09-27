# Minimal-contribution experiment protocol — 2026-09-21

Written before the new multi-task results were inspected. Exploratory, not a registered confirmatory study.

## Candidate contributions, in order
1. Deterministic held-out S-VAE evaluation with separate condition-frame and pooled-target metrics. Preserve the upstream model and checkpoint format.
2. A small, optional condition-frame loss weight, only if paired experiments support it. Default behavior must stay exactly unchanged.
3. Document compression quality versus a frozen state-change probe and corruption sensitivity. No claim about closed-loop policy improvement.

## Fixed design
- Upstream OpenWAM commit: 7c5861e45cfe1339a0323f0e0b03a3316c37971c (verified current main).
- Frozen published DINOv3 encoder from the prior pilot. This is the Study representation branch, not OpenWAM-alpha's default Wan VAE.
- Tasks: adjust_bottle (previously explored), handover_block and place_object_basket (new tasks).
- 50 clean demonstrations per task, 35/5/10 episode train/validation/test split, fixed seed 42; 12 complete clips per episode. No random clip split.
- Train-only normalization and PCA; no published S-VAE in fair comparisons.
- Baseline PCA-48; fresh S-VAE-48 with condition weight 1 or 4, seeds 42/43/44, 2000 steps, identical initialization/minibatches per paired seed. Model architecture and inference unchanged.
- Candidate weighted objective: [T * baseline_loss + (w-1) * first_frame_loss] / (T+w-1). Weight applies to MSE/cosine/KL; free-bits remains 0. No additional action supervision.
- Report the fixed final checkpoint, not the best test score; validation used only to fit ridge probe regularization. Log reconstruction every 200 steps.
- Primary diagnostics: held-out standardized reconstruction MSE separately on current conditioning frame and pooled future target frames; clean visual+proprio achieved-state-change probe normalized MSE.
- Probe target: 4-raw-step achieved end-effector displacement (6D) and future achieved gripper (2D), NOT controller commands.
- Frozen clean-trained probe evaluated also under input Gaussian noise sigma=10/255 and brightness factor=0.6. Perturb current input image only; keep targets/proprio fixed. These are synthetic image corruptions, not LIBERO-Plus or simulated camera viewpoint changes.
- Per-task episode-cluster paired bootstrap; report seed variation, and all task outcomes. No seed/task cherry-picking or claim from overlapping unpaired confidence intervals.
- Do not treat windows as independent trials. Use identical corrupted images across representation variants.
- One healthy, idle L40S; scripts must preflight memory/utilization/ECC and stop on errors. No full WAM training.

## Decision rule
A useful diagnostic can be contributed even if weighting fails. Recommend the weight option only if improvement in condition reconstruction repeats across tasks without a material deterioration of pooled-target reconstruction or control probes. Otherwise keep it as a negative result and propose the evaluation-only patch.

## Numerical-control amendment (before interpreting corruption results)
Initial corruption encoding processed one frame per DINO batch, while the clean cache used nine. A three-clip audit found single-frame raw-feature MSE about 4e-5, whereas a repeated-nine-frame control matched the original conditioning features within 3e-8. Re-encode BOTH perturbations with nine frames and assert matching clean features for EVERY held-out clip. Retain initial outputs as an audit trail; final analyses use only `evaluation-matched-batch`. No training objective, task, seed, step budget, target, or corruption strength changes.

## Sampling-description correction (2026-09-21)
The prior report incorrectly interpreted released `num_frames=33` as 33 sampled video frames. In `openwam/dataloader/robotwin.py:356-363`, it is the raw HDF5 window length; `video_stride=4` produces 9 video frames. Our temporal sampling matches that configuration. The real differences are sparse complete-window sampling, task coverage, short reducer training and offline probes. No data, training, or numerical result was changed.
