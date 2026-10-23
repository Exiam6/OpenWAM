# RAE visual representation study

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

1. Start with the public `facebook/dinov2-base` snapshot at
   `f9e44c814b77203eaa57a6bdbbd535f21ede1415` (model SHA256
   `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`).
   The DINOv3 candidate requires manual access approval. Freeze DINOv2,
   preserve the three RoboTwin camera order and frame timing, pad only the
   patch border, and resample its 14-pixel grid to the released model's
   16-pixel latent grid. Record that spatial resampling as a method choice.
2. Fit per-channel latent normalization on the **training split only**. The
   7,985-frame stride-16 pass yielded statistics at
   `/root/zifan/openwam-runs/rae_feature_stats_stride16.pt`, SHA256
   `c582c0c8dca54f0944d4d5a99dd1fc34a25948bd61313ba17ba61fd92d54b838`.
   Raw channel means span -5.407 to 4.874 and standard deviations span
   0.122 to 6.419. The calibrated DINOv2 path stores the mean/std as frozen
   checkpoint buffers; the earlier per-token normalization stays a separate
   ablation. Do not change normalization in a resumed run.
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
  The bounded ten-task archive selection is frozen in `DATA_SELECTION.md`.
- Numerical: one batch with finite latents, actions and losses; check shapes,
  nonzero intended gradients, save/reload equality, and raw video/action loss
  at matched noise/timestep/dtype.
- Engineering: a bounded small-data overfit run is an implementation check,
  excluded from the final endpoint.
- Scientific: at least three matched training seeds, fixed task/scene pairs,
  per-task success and technical failure rates, and paired confidence intervals.
  A trained model is not evidence of a visual-representation benefit until this
  comparison is complete.

The stage-one decoder pilot was registered at
`/root/zifan/openwam-runs/rae_decoder_20261023_seed42/registration.json`.
It trained for 1,000 steps on train-only frames; the selected step-1,000
checkpoint reached dev normalized L1 0.12870. A single test read after
selection gave RGB MAE 0.02844 and PSNR 23.15 dB over 1,002 sampled frames.
These are reconstruction metrics, not robot-task improvement.
That first decoder used per-token-normalized features. A calibrated RAE
requires a separately trained decoder on the calibrated features.
The matched 1,000-step calibrated decoder reached dev L1 0.12991 and test
RGB MAE 0.02880 / PSNR 22.75 dB on the same 1,002-frame test sampling.
Calibration did not improve this pixel-reconstruction metric. Its effect on
WAM denoising and task success remains unmeasured; do not infer either from
decoder reconstruction.
Both stage-one decoders were trained on single frames. The current DINOv2
WAM encoder has `pixel_decode=False`; these pilots do not yet reconstruct a
9-frame generated clip from its 3 temporally pooled latent frames. A temporal
decoder is required before reporting generated-video quality.

The released RoboTwin-Full checkpoint was trained on RoboTwin and may contain
the ten selected tasks or episodes. It is therefore the **evaluation
reproduction target only**, not the donor for this controlled comparison.
The matched VAE/RAE pilot will instead use the released
`OpenWAM-Alpha-Pretrain-Foundation-Model` checkpoint, revision
`52df4e66c82c5c8b480adcc8d01f4db7415dfb56`, weight SHA256
`180a02653118b0f96da28a9cae9ec7b4c1c1e6cd0e4608b56c7b4884f8001d3d`.
The released pretraining mixture includes simulation data; episode-level
membership has not been audited, so this choice removes the known downstream
RoboTwin fine-tune overlap but does not prove zero pretraining overlap.
The RAE graft retains every shared trunk tensor and the action head by
name and shape, removes the native VAE, loads frozen DINOv2, and resets the
two visual projections. The VAE control resets those same projections with
the same per-run RNG seed. Width still differs (48 versus 768 channels), so
projection capacity is a separate required ablation before attributing gains
to representation quality. Register each WAM GPU run separately.

This host has CUDA A100s but currently cannot initialize SAPIEN's required
Vulkan RT renderer. Native closed-loop reproduction and the final paired
success-rate evaluation remain gated on a working RT host.

The matched single-seed pilots are registered under
`/root/zifan/openwam-runs/wam_pilot_20261023`. A detached
`research/rae/continue_pilot.py` process waits for the running RAE training,
verifies its final checkpoint, releases only the SHA-verified reproducible
graft weight file for disk space, and trains the VAE control. It then uses
`eval_teacher_forced.py` on the same 100 dev windows with four noise draws
each and `compare_teacher_forced.py` for per-task paired action-loss deltas.
The state file is `continuation_state.json`; any failure stops the sequence
and preserves logs. The video losses have different latent targets and are
not compared across variants.

## Related primary work

- Zheng et al., *Diffusion Transformers with Representation Autoencoders*,
  arXiv:2510.11690; reference implementation `bytetriper/RAE` (MIT), inspected
  at `a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041`.
- Jiang et al., *Rethinking Representations for World-Action Modeling*,
  arXiv:2609.38163. Its feature calibration, temporal bottleneck, and
  action-grounded shaping are candidate **separate** ablations, not part of the
  first RAE/VAE comparison.
