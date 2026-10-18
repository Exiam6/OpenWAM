# Matched VAE control: CPU readiness

The original training inputs are now available and independently matched on this
host. This establishes data readiness, not a trained baseline or a representation
result. GPU-stage budget and admission remain pending; no GPU was initialized.

| Check | Verified evidence |
|---|---|
| Original trajectories recovered | 841 files, 22704727809 bytes |
| Train / development episodes | 765 / 76 |
| Frozen input windows | 6424 |
| Maximum action / initial-state difference | 0.0 / 0.0 |
| Minibatch schedules replayed exactly | 3 training seeds |
| Frozen VAE tensors strictly matched and serialized | 196 |
| Frozen VAE artifact size | 1409402328 bytes |

The initial audit stopped because only nine raw trajectories had been migrated.
That failed attempt remains [recorded](/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018/input-audit.json). The complete source
collection was read from the configured training host into an isolated directory;
every copied file matched the original recorded SHA256. No demonstrations were
replaced or added. The latent cache also matched its original hash. The corrected
attempt verified the same frame indices, all action/proprio labels, train/dev
separation and the exact pre-existing per-seed schedules.

The VAE extraction contains only frozen pretrained encoder/decoder tensors, with
no published policy weights. Its file SHA256 is `8b3051a79685605bbf97c32f30b98c60ce3c6c744e6b8adb971efcaee55b1267`. The proposed
small-model template retains the raw-native architecture and training recipe;
only encoder selection, dimension-specific input/output projections and output
paths differ. A template is not a successful model build or training run.

Next gates: implement explicit frozen VAE loading and matched RGB preprocessing;
validate shared trainable initialization and temporal causality; obtain a concrete
staged compute budget after current G/H evidence; run cache/training smoke and
measure cost before any full fixed-endpoint comparison. Equal training steps
must not be described as equal FLOPs. The published larger official policy is
not the matched RAE baseline.

[Plan](/home/zifanz4/openwam-vae-control/research/greene/ex5-a_vaebaseline_1018/PLAN.md) · [Full input audit](/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018/attempt-data-restored/input-audit.json) ·
[Window manifest](/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018/attempt-data-restored/window-manifest.json) ·
[VAE extraction](/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018/vae-extraction.json) · [Configuration diff](/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018/template-audit.json)
