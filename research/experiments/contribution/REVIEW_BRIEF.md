# Proposed contribution: held-out S-VAE reconstruction diagnostics

The standalone trainer reports aggregate training metrics but has no separate held-out evaluation entrypoint. For external encoders, the first latent frame conditions the policy and later frames are temporally pooled targets; a single aggregate hides that distinction.

This patch adds a separate evaluator for existing checkpoints and feature shards. It preserves the checkpoint's training normalization, uses deterministic posterior means, weights errors by feature-element count, and reports conditioning / pooled-target errors plus relative target-energy errors. It does not modify training, models, or default configurations.

Motivation: an exploratory three-task, three-seed, fixed-budget DINO/S-VAE study found much larger reconstruction improvements over PCA than gains in a frozen achieved-state-change probe. A candidate conditioning-loss weight did not consistently help. The patch therefore contributes measurement capability, not an unsupported policy-improvement claim.

Validation: five focused CPU tests, Ruff, real trained-checkpoint/native-shard smoke, and clean application to pinned upstream. Full upstream suite remains to be run in the full dependency environment before a PR is submitted. No PR or maintainer message has been sent.

See the public report and NIGHT_PROTOCOL.md for all task outcomes, short-clip and offline limitations, and numerical-control corrections.
