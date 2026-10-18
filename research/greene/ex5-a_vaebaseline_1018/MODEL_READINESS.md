# Matched VAE control: model construction readiness

All three real CPU model constructions passed explicit frozen-weight loading and
paired trainable-initialization checks. This establishes construction and input
pixel contracts; no VAE encode, policy forward, gradient step or CUDA allocation
was executed. GPU-stage validation and compute admission remain pending.

| Training seed | Exactly paired trainable tensors | Dimension-specific exclusions | Trainable parameters |
|---|---|---|---|
| 42 | 676 | 3 | 295035604 |
| 43 | 676 | 3 | 295035604 |
| 44 | 676 | 3 | 295035604 |

The excluded tensors are only the latent input/output projections whose shapes
necessarily differ between768 and48 channels. All other trainable tensors copy
the corresponding original per-seed initialization exactly. The frozen VAE comes
from the previously verified extracted asset and the frozen text encoder from
the same source as the raw-native baseline. No larger official policy weights
were loaded. The actual model uses the Wan VAE external-encoder wrapper with
from_scratch=true; its native-VAE placeholder is released and frozen weights
are loaded into video_encoder._m explicitly.

Each task's first development RGB window was checked against native Wan pixel
preprocessing: all three comparisons were exact with shape1x3x9x384x320 and BF16
dtype. This checks pixels only, not latent encoding or causal forward behavior.
All three model builds ran against the same immutable engine source used by
current evaluation. GPU encoding, shared-noise loss behavior, gradients,
serialization of a trained checkpoint and deployment consistency remain gates.

[Full model audit](/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018/model-cpu-audit.json) · [Execution provenance](/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018/model-audit-provenance.json) ·
[Data readiness](/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018/CPU_READINESS.md) · [Plan](/home/zifanz4/openwam-vae-control/research/greene/ex5-a_vaebaseline_1018/PLAN.md)
