# Representation contribution decision, 2026-09-21

This is a reasoning note, not a new experiment authorization, frozen protocol,
or selected model. It does not change the active12-rollout stageC study.

## What the completed studies justify
- Published policy already uses pretrained DINO features plus S-VAE compression.
  Our existing runs do not compare matched RAE vs VAE policies. We cannot claim
  pretraining is superior from those runs alone.
- The selected published-coordinate residual adapter reduces validation latent
  restoration error by about35%, but the small closed-loop studies establish no
  net policy benefit. Training already contains a clean identity penalty with
  weight1; proposing that same penalty again would not address a new hypothesis.
- Repeating final LayerNorm changes actual first actions even without fitting.
  BF16 casts and downstream sensitivity make coordinate/numeric preservation an
  explicit requirement. Do not assume an extra normalization is an identity map.
- Earlier control-supervised compressors improve translation probes but have
  gripper tradeoffs. Gripper-transition windows are not contact annotations.
- The54 generated chunks establish repeatability only at six saved initial
  inputs; the current12 closed-loop repeats check where later trajectories diverge.

## Decision tree after the frozen closed-loop study
1. If observations diverge before actions, investigate the first execution or
   sensing difference using the saved full requests, EEF/joints, server20D and
   native16D target. Separate three-camera differences from proprio changes.
2. If actions diverge while requests match, inspect the32-action buffer position
   and the complete input/context at its last generation boundary. A request
   made during cached execution is not the input that generated that action.
3. If full repeats agree, preserve that reproducibility evidence. It does not
   resolve historical seed300003 or establish global determinism. The next
   representation question is alignment to control-sensitive directions, not
   another normalization/weight sweep on the already examined scenes.

## Small candidate worth assessing, not yet running
Keep the published DINO/S-VAE/policy frozen and the residual adapter architecture
fixed. Before proposing any new fitted adapter, inspect whether latent restoration
improves a frozen control readout, separated into translation, rotation parameters
and gripper. A train-only readout could later supply an action-aware auxiliary
loss in the published latent coordinates; this is distinct from the earlier
per-task S-VAE compressor studies and must be described as a proxy, not the policy.

A future protocol would have to freeze training/validation episodes, readout,
loss scaling, final checkpoint and resource cap before outcomes; compare with the
unchanged previous latent-MSE adapter and identity, not select a favorable seed.
Require clean preservation and group-specific control diagnostics before a new
closed-loop confirmation on fresh scenes. Any proxy improvement still requires
closed-loop validation. No claim that this candidate is already better.

## Mergeable contribution available without a positive result
An opt-in evaluator can expose the gap between latent MSE, action change and
closed-loop outcome with fixed scene/prompt replay and passive trace comparison.
The existing denominator correction and scoped-language-RNG patches already have
focused tests and drafts; retain them independently of representation claims.
A concise reproducibility diagnostic with all negative results is preferable to
adding an unvalidated adapter to the default inference path. No PR has been sent.
