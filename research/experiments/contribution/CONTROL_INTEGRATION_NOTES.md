# Optional control-supervision integration: bounded review notes

The current study is an external exploratory driver. It does not modify the upstream model or trainer. No policy improvement is claimed before closed-loop validation.

At pinned OpenWAM 7c5861e45cfe1339a0323f0e0b03a3316c37971c, collect_svae_features.py writes shards with only features, raw_dim and encoder_name (line 219). The standalone trainer's shard dataset returns feature tensors only. Therefore adding a task-supervision objective upstream also needs correctly aligned labels and split/provenance metadata; it is not merely a one-line loss addition.

Keep two review scopes distinct:
1. Existing independent reconstruction evaluator: checkpoint/shard compatible, no label-schema change, no model/default changes.
2. Only after positive validated evidence: optional label sidecar keyed to exact collected sample IDs; provenance and split checks; optional auxiliary training head; explicit lambda=0 baseline; independent probes and latent compatibility checks. Backward compatibility with existing unlabeled caches is required.

The sidecar cannot recover label alignment from shard tensor order unless the collector records the exact dataset indices and augmentations. For achieved-state targets, verify temporal horizon, coordinate frame, state representation, valid masks and current-versus-future indexing. Future observations must not be available to the auxiliary head. Do not invent contact labels from gripper values.

A per-task readout improvement in this pilot would motivate shared multi-task and new-episode validation, not immediate changes to released checkpoints. Changing S-VAE weights changes the latent target space seen by the world/action model; equal dimensions do not preserve policy compatibility.
