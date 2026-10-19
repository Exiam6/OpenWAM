# Original RGB trajectory recovery

The first CPU audit (source db5f665) stopped on a missing local raw trajectory.
Its failed record remains in the original runtime input-audit.json. The latent
cache SHA256 matched the frozen dataset audit. Only nine of841 raw trajectories
were copied to this host during prior diagnostics; this is a migration gap,
not evidence that the original demonstrations or latent cache are corrupt.

A read-only query to the already configured training host
shenlong-gpu-02.cs.illinois.edu confirmed all841 source trajectories in the same
collection path. Copy exactly the frozen audit's filenames into the new study's
source-data directory with rsync, capped at100MiB/s. Do not change the shared
collection, replace missing episodes, add scenes or re-encode inputs yet.

After transfer, run the CPU audit under attempt-data-restored using its explicit
trajectory-root option. Preserve the original failed attempt and compare each
copied file against the original recorded SHA256 before any VAE encoder use.
No GPU work or experiment criterion changes are introduced by this data recovery.
