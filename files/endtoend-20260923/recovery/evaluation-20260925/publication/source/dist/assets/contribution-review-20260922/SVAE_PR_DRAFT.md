Title: Add held-out S-VAE reconstruction diagnostics by temporal role

The standalone S-VAE trainer reports aggregate training metrics, which cannot
show whether conditioning-frame reconstruction trades off against pooled video
targets on held-out episodes. Add a separate evaluator for existing checkpoints
and native feature shards. It preserves checkpoint normalization and reports
deterministic, element-weighted MSE and target-energy-relative errors for both
roles, without changing training or inference defaults.

Five focused CPU tests cover weighting, single-frame/empty/non-finite input,
shard dimensions, checkpoint normalization and relative errors. They also pass
in a 17-test combined check with two independent optional evaluation patches;
Ruff passes. A real-checkpoint/native-shard smoke previously passed with a leaf
bootstrap. The documented CLI in the full upstream environment and full upstream
suite remain unverified. The patch makes no claim of policy improvement or
RAE superiority. Episode disjointness must be supplied by the data manifest.
