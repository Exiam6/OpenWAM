# Interpretation rules for this pilot

- The published policy already uses pretrained DINO plus its trained S-VAE. This
  study tests an optional restoration residual, not RAE versus an unpretrained
  encoder or the full Wan-VAE OpenWAM-alpha recipe.
- The adapter fits original, post-normalization latent coordinates. Equal channel
  count alone would not make a separately trained compressor policy-compatible.
- Validation restoration MSE averages tokens/channels. It need not weight
  task-relevant object geometry or control-sensitive directions appropriately.
- If baseline success saturates at 5/5, equal adapter success is not proof of no
  benefit; it means this pilot provides no demonstrated success-rate gain.
- If clean behavior regresses, do not relax the predeclared distortion criterion
  or tune against these confirmation seeds. Record the regression and design a
  separate action-preservation study with new confirmation seeds.
- The front-camera image is a substantial change in projection, object scale
  and visible crop. It is a stress test of the whole policy, not a causal
  measurement of information destroyed only by the S-VAE bottleneck.
- Per-condition Wilson intervals and paired exact tests at n=5 do not establish
  multi-task generalization. These are operational and hypothesis-screening runs.
- Language RNG scoping is independently justified by the actual upstream source
  and CPU reproduction, regardless of whether the adapter helps. The OpenWAM
  patch is opt-in; the external RoboTwin patch is a separate review alternative.
