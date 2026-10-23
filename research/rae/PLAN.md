# RAE visual representation study (preparation; no training run registered)

Source: OpenWAM `782e3133f674449e2986754cfc5f7c4e204835a2`.
Worktree: `research/rae-visual-20261023`. The released RoboTwin-Full checkpoint
and its 10,000-episode native evaluation remain the reproduction baseline.

## Question and competing explanations

Does replacing the reconstruction-trained Wan video VAE latent with a frozen,
pretrained visual representation improve closed-loop RoboTwin success when the
training data, model trunk, action head, optimizer, budget, and evaluation are
matched? Apparent gains could instead come from a wider DiT input/output head,
different latent scale or noise schedule, changed token count, extra decoder
training, or different demonstrations. Record and control each difference.

## Proposed method

1. Use the existing `DinoV3VideoEncoder` interface as the representation
   encoder. Pin a public DINOv3 snapshot, hash every weight file, freeze the
   encoder, and keep the three RoboTwin camera order and frame timing fixed.
2. Fit per-channel latent normalization on the **training split only**. Keep
   the existing per-token normalization as a separate baseline; do not silently
   change it in a resumed run.
3. Train a pixel decoder from frozen DINO patch features on training videos to
   make a representation autoencoder. Validate reconstruction on held-out
   demonstration videos. The decoder must not consume actions or test frames.
4. Adapt the OpenWAM DiT input/output projections to the representation's
   channel and patch shape, keeping the common trunk and action head as equal
   as possible. Initialize both matched VAE and RAE runs from the same recorded
   source weights and seed. Do not compare a short RAE fine-tune directly with
   the released, fully trained VAE checkpoint as a causal result.
5. Train the matched VAE and RAE runs with the same demonstrations, steps,
   optimizer, batch size, action loss, inference steps, and checkpoint rule.
   Run separate head-width and noise-shift ablations only after the base pair.

## Gates

- Reproduction: checkpoint hash, native three-camera RT, cuRobo, and complete
  released-checkpoint evaluation must be audited before claiming improvement.
- Data: freeze train/dev/test demonstration manifests and content hashes;
  inspect episode and scene overlap; fit statistics only on train.
- Numerical: one batch with finite latents, actions and losses; check shapes,
  nonzero intended gradients, save/reload equality, and raw video/action loss
  at matched noise/timestep/dtype.
- Engineering: a bounded small-data overfit run is an implementation check,
  excluded from the final endpoint.
- Scientific: at least three matched training seeds, fixed task/scene pairs,
  per-task success and technical failure rates, and paired confidence intervals.
  A trained model is not evidence of a visual-representation benefit until this
  comparison is complete.

The evaluation hardware, data manifests, GPU time limit, stop rule, exact DINO
snapshot, and model initialization hashes must be recorded in a separate run
registration before allocating a training GPU. No GPU run is registered here.

## Related primary work

- Zheng et al., *Diffusion Transformers with Representation Autoencoders*,
  arXiv:2510.11690; reference implementation `bytetriper/RAE` (MIT), inspected
  at `a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041`.
- Jiang et al., *Rethinking Representations for World-Action Modeling*,
  arXiv:2609.38163. Its feature calibration, temporal bottleneck, and
  action-grounded shaping are candidate **separate** ablations, not part of the
  first RAE/VAE comparison.
