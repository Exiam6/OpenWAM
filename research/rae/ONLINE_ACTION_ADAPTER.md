# Online action-grounded DINO ablation

This isolated worktree adds a causal residual adapter to the calibrated DINOv2
encoder. Its output projection starts at zero, so the initial representation is
exactly the frozen seed-42 DINO baseline. A 128-channel internal bottleneck
learns a width-preserving residual over three *past/current* latent frames.
The DiT input/output width (768) and token grid remain unchanged.

During training, frozen DINO features are computed under the existing
`prepare_inputs()` no-grad path. `compute_loss()` applies the adapter with
autograd, detaches its output for future video noise and targets, and uses the
live first latent as the clean TI2V conditioning frame. A boundary hook routes
only the weighted action-loss gradient from that current latent into the
adapter; the shared world/action trunk still receives both losses. Inference
applies the saved adapter inside `batch_encode()`. The adapter refuses to train
without action labels, a TI2V clean first frame, or trainable parameters.

This is **not** a reproduction of ReWAM's TRB: it uses DINOv2 Base, retains
the 768-channel latent grid, and does not implement ReWAM's 2x2x2 compression,
multi-layer DINOv3 features, or causal running-statistics normalization. It is
a controlled test of online action-only representation gradients in OpenWAM.

Build an initialization from the same foundation donor and training-only
feature statistics as the matched raw-DINO run:

The shared Python environment has an editable install of the frozen main
worktree, so commands from this branch must set `PYTHONPATH` explicitly:

```bash
export PYTHONPATH=/root/zifan/OpenWAM-online-action
/root/zifan/openwam-envs/policy/bin/python research/rae/build_grafted_checkpoint.py \
  --feature-stats /root/zifan/openwam-runs/rae_feature_stats_stride16.pt \
  --online-action-adapter-hidden-dim 128 \
  --seed 42 \
  --output /data/zifan-openwam-rae/online_action_adapter_20261025/graft_seed42
```

The builder removes `video_backbone.video_encoder` from the broad freeze list;
the DINO ViT remains frozen by the encoder itself. The graft must be checked
for identical shared weights and reset projections against seed-42 raw DINO
before a 1,000-step run. Run the five CPU checks in
`test_online_action_adapter.py` and a short real-model training step with
gradient checkpointing and DeepSpeed ZeRO-2 before launching the full job.
Run the focused checks with
`python -m research.rae.test_online_action_adapter` from this worktree.
The extra `autograd.grad()` traversal can increase step time and its behavior
under the full distributed stack has not yet been verified. Do not add this
candidate to the native success-rate claims until it has a SHA-verified final
checkpoint and matched closed-loop evaluation.

The existing seed-42/43/44, width-matched, and offline action-informed queues
run from their frozen worktrees. This branch does not alter those jobs.

## Registered initialization (2026-10-25)

The seed-42 adapter graft above completed at
`/data/zifan-openwam-rae/online_action_adapter_20261025/graft_seed42`.
Weight SHA256: `8521d59e54b50bea0b4a43385b8ddf2384921624ee4a29f24a28dedfdf18d8e4`.
Its manifest confirms the same donor, DINO weights, train-only feature stats,
and 1,889 matched shared tensors as the earlier calibrated seed-42 graft.
The new graft has 235 non-donor tensors versus 229 previously: the six extra
tensors are the adapter's three convolution weights and biases. Both grafts
reset the same visual projections with seed 42+1009 using the unchanged
`reset_visual_projections` helper. The new config sets adapter width 128 and
removes `video_backbone.video_encoder` from the freeze list. This initialization
has not yet undergone a real DeepSpeed training step or any evaluation.
