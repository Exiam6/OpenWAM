# Experimental temporal compression candidate

Not a validated improvement or submitted PR. Native frame-grid helper plus optional dinov3_temporal encoder;3072 frozen-after-fit kernel parameters; identical48channel/causal4frame interface. Zero logits exactly recover native mean pooling. Constructor, strict checkpoint sidecar and reload supported. See ../../studies/temporal-20260926/protocol.json for fixed train/closed-loop plan. Runtime source under /data02 is frozen by study manifests.

Eight contract checks passed, native pretrained mean parity passed, finite gradient through frozen S-VAE passed, both arms native checkpoint roundtrip and future-group isolation passed. Three matched training seeds and1080 held-out rollouts are pending. Do not infer control improvement from fitting error.

The review patch omits context lines: validate/apply with `git apply --unidiff-zero temporal-pool.patch` against the recorded native source. This avoids whitespace warnings on blank context lines inside the patch artifact.
