# Reproduce the control-supervised compression study

This is an exploratory per-task offline representation study, not a released-policy evaluation.

1. Reproduce the previous night study using its published script package and REPRODUCE_NIGHT.md. This provides the pinned upstream modules, minimal Python environment, three tasks, DINO weights, episode manifests, feature/PCA caches, old original-loss checkpoints, and matched-DINO-batch corruption caches under results/night-20260921.
2. Place this package's scripts under the same project root. Preserve CONTROL_PROTOCOL.md and requirements-lock.txt. Do not mix code/configuration revisions in one results/control-20260921 directory. Use a fresh run directory/root for a new revision, retaining old outputs for audit.
3. Select one idle healthy GPU according to cluster rules; set CUDA_VISIBLE_DEVICES to its UUID. Set CUBLAS_WORKSPACE_CONFIG=:4096:8 and OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=6 before starting Python. A checks-only run is available as `python scripts/control_study.py --checks-only`.
4. Run `python -u scripts/control_study.py`. It runs the checks, 27 fixed 2000-step fits, independent validation probes, global coefficient selection, then initial test diagnostics. It periodically aborts its own process if another GPU compute process appears or uncorrected ECC is detected. Do not bypass these checks to displace another user's job.
5. After complete.json exists and that process exits, run `python -u scripts/control_robustness.py` on the same or another explicitly selected idle healthy GPU. It adds paired camera observations and matches compressor batch geometry. It changes no checkpoint, selected coefficient or fitted readout.
6. Run `python scripts/summarize_control.py` (CPU; no GPU work). Final analysis requires robustness-complete.json and consumes evaluation-matched.json, not initial evaluation.json.

The zero-weight auxiliary head trains on detached current latents with a separate backward pass. All conditions use deterministic algorithms and identical spatial pooling. Checkpoint files contain architecture/stats and can be loaded by the original S-VAE shell; that does not imply compatibility with an existing WAM policy. The task-specific latent coordinates have changed.

The 50 episodes per task are split 35/5/10 with 12 complete windows each. Test episodes were previously inspected in earlier studies and are reused here as a development benchmark. Fresh episodes, a shared multitask setting and closed-loop policy execution are required for stronger claims. No contact labels were available in this experiment.

Optional post-hoc diagnostic: `python scripts/control_group_probe.py` is CPU-only (set CUDA_VISIBLE_DEVICES to an empty string if desired). It fits group-specific ridge readouts on training episodes and chooses their regularization on validation; it preserves the primary joint-probe results and adds group-probe-diagnostic.json. This additional analysis was motivated by observed gripper regressions and is explicitly exploratory.
