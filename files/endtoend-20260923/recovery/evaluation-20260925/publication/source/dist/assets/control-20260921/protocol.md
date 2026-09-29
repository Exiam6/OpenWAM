# Control-supervised compression experiment (2026-09-21)

Status: fixed before launch. This is exploratory offline evaluation, not a published-policy or closed-loop result.

- Same three tasks, 50 episodes/task, 35/5/10 episode split and 12 clips per episode as the completed study. Old held-out episodes now constitute a development benchmark; fresh episodes are needed for confirmatory claims.
- Same frozen DINO features, train-only normalization/PCA metadata, S-VAE-48 architecture and original reconstruction/cosine/KL terms.
- Factor: auxiliary loss weight 0, 0.1 or 1; three paired seeds 42/43/44; final checkpoint at 2,000 steps. Exactly 27 runs; no best-step or best-seed selection.
- Auxiliary input: 2x2 spatially pooled, channel-normalized posterior mean of CURRENT frame and train-standardized CURRENT robot state. Small MLP: (192 + state_dim) -> 128 -> 8. No dropout.
- Target: train-standardized achieved end-effector translation delta over four raw steps (6 values), plus achieved gripper state at that horizon (2 values). These are not controller commands. No future latent or future state is an auxiliary input.
- Freeze DINO. Update compressor and auxiliary head; separate gradient clipping/optimizers ensure zero-weight head does not alter baseline clipping. Preserve Gaussian RNG across head initialization. The auxiliary head is never used as the reported evaluation probe.
- Checks before training: future input cannot alter current latent; auxiliary gradients reach current latent and encoder only; weight zero reproduces original gradients exactly; fitted statistics ignore all test states/labels.
- Re-run all nine zero-weight controls and record maximum weight difference to previous original-loss checkpoints. Any difference is disclosed, not silently treated as exact replication.
- Independent clean ridge probes: train on train episodes; select regularization on validation. Use the same pooling and preprocessing for raw DINO and compressed features, while disclosing different probe input sizes.
- Choose ONE global lambda by mean validation MSE ratio to its paired zero-weight baseline across tasks/seeds. Save the decision before reading new test metrics; all test methods/seeds are then reported for transparency.
- Evaluation: clean, matched-batch Gaussian noise sigma 10/255, brightness factor 0.6, fixed task-local visual derangement and proprioception-only. Existing corruption caches are test-only and never used for fitting. No claim of real viewpoint or contact robustness in this round.
- Interpret visual derangement as a dependence diagnostic with mismatched observations, not a realistic sensor perturbation. Compare paired episode losses with confidence intervals, reporting training-seed variation separately.
- Retain reconstruction metrics and clean-vs-corruption trade-offs. New latent coordinates require explicit policy alignment/adaptation before closed-loop conclusions.
- One healthy idle L40S, bounded 6 CPU threads. Check ECC and competing GPU processes during execution; stop our own job if another job appears. No other user jobs are interrupted.

## Pre-training numerical amendment
The first preflight failed: adding a literal zero-weight auxiliary autograd branch changed BF16 gradient accumulation by up to 6.1035e-5. No training runs had started. Use an explicit zero-weight branch returning the original loss; train its reference head on detached latents with a separate backward call. Re-run exact gradient checks before starting. Keep the failed preflight log as an audit trail.

## Isolated numerical control and correction
Repeating the ORIGINAL loss backward with restored RNG and no auxiliary branch also produced a 6.1035e-5 gradient difference on this GPU. Therefore the first discrepancy cannot be attributed specifically to the auxiliary branch. Enable PyTorch deterministic algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8 for every new condition, then require exact same-regime zero-weight gradients. All nine original-loss controls are retrained under this regime; comparisons with old nondeterministic checkpoints are diagnostic, not assumed identical. The explicit zero branch remains for clarity. No optimization runs had started during these checks.

## Deterministic pooling implementation
PyTorch rejected adaptive_avg_pool2d CUDA backward under deterministic mode. For the fixed 24x20 token grid, implement the same four non-overlapping 2x2 spatial cells using reshape and mean, which has deterministic backward. Check forward agreement with the reference pool at atol/rtol 1e-6 before training, and apply this implementation consistently to every new auxiliary head and independent probe. This changes neither the intended pooling cells nor latent width.

## Additional evaluation registered during training, before test metrics
After the fixed 27-run study and its validation-only selection, add a separate same-state camera-pair diagnostic: replace only head_camera with front_camera, retaining both wrist inputs, state and labels. Evaluate every method/seed without changing or refitting any model, selecting coefficients, or tuning on this condition. This is fixed camera transfer, not continuous viewpoint jitter. It does not provide contact labels.

Also validate the S-VAE's CURRENT output at the same B=2,T=3 compression batch geometry as clean cached evaluation. For current-only perturbations, duplicate the current raw feature into three frame slots and discard the latter two; the model has no cross-frame attention. Check all test current vectors against the original clean vectors, and quantify any original T=1 versus T=3 numeric difference. Recompute perturbation metrics at matched geometry before interpretation if needed. Retain original evaluation.json files; final reporting uses evaluation-matched.json exclusively after this check passes. Re-encode original head frames alongside front replacements so camera effects have a matched clean control even if encoder numerics differ.

## Explicitly post-hoc readout-group diagnostic
After observing the primary held-out results, the candidate improved translation but worsened gripper RMSE. Without any representation training or coefficient reselection, fit separate translation and gripper ridge readouts, each choosing regularization on validation only. Preserve joint-probe results as primary; save this investigation separately as group-probe-diagnostic.json. It cannot be treated as a preregistered confirmatory experiment.
