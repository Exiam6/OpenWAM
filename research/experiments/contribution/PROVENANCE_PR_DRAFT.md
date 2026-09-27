Title: Preserve sample provenance in opt-in S-VAE feature collection

Offline S-VAE feature shards currently lose the association between shuffled
feature rows and their source episodes/windows. Add optional per-row dataset
identity, available episode/window metadata, and a SHA256 digest of the actual
preprocessed encoder input, including dtype and shape. The default shard format
and model/training behavior stay unchanged; existing feature-only readers can
ignore the extra basic-type metadata.

Four focused CPU tests pass, including the real collector loop with a stub
encoder/dataset: shuffled row alignment, a final partial batch, shard flushing,
provenance enabled/disabled, BF16 byte hashing and invalid alignment. Ruff passes.
This is not a live distributed/GPU collection test or the full upstream suite.
Input hashing has opt-in CPU-transfer overhead; the digest is not an automatic
split-leakage check, an augmentation replay recipe or an encoder-checkpoint hash.
