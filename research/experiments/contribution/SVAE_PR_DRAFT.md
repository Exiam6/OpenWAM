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
Ruff passes. The documented CLI now also passes with normal package imports in the existing
policy environment, an actual trained checkpoint and three fixed clips packaged
in native shard format: CPU batch sizes2and1 agree within1e-6 and input hashes
remain unchanged. The full upstream suite remains unrun. The patch makes no claim of policy improvement or
RAE superiority. Episode disjointness must be supplied by the data manifest.
