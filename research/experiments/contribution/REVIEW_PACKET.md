# OpenWAM representation contribution — review packet

Update 2026-09-22: normal S-VAE CLI import/evaluation gate has passed on a real
checkpoint and fixed3clip fixture in the existing policy environment. Full upstream
suite remains unrun. CONTROL_READOUT_FINDINGS.md reports the new fixed-adapter
control-readout tradeoff; current patch scope remains reconstruction only.
The original review details below are retained with their original test scope.

Decision: prioritize the standalone held-out S-VAE evaluator. It adds 237 lines
across four files at eb31a97c4baa7dc5c4511a14f1f6c9b32795a5fd, against pinned
upstream 7c5861e45cfe1339a0323f0e0b03a3316c37971c. Model, training, checkpoint
format and default inference are unchanged. The contribution is measurement
capability; current results do not support enabling a new representation module.

## Three independent changes, not one required bundle

| Priority | Change | Commit | Scope | Review status |
|---|---|---|---|---|
| 1 | Held-out S-VAE reconstruction evaluator | eb31a97 | 4 files / +237 lines | Metrics reviewed; 5 CPU tests; normal CLI package bootstrap remains to verify in full upstream environment |
| 2 | Opt-in scene-seeded language sampling | 4cace2d | 3 files / +134 lines | 6 CPU cases; still verify accepted scenes and actual prompts for each paired study |
| 3 | Opt-in request/action trace comparison | d1273ca | 4 files / +363 lines | 6 CPU cases; recorded RPC boundary is not measured robot motion |

All three apply together to the pinned base in a new isolated source tree.
Combined targeted tests: 17 passed; Ruff passed; CUDA hidden; two CPU threads.
This checks combined source/import compatibility, not the full upstream suite
or live model/simulator integration. No patch was modified during this review.

The earlier real checkpoint/native-shard S-VAE smoke used an isolated leaf-module
bootstrap. It does not prove that the documented CLI imports successfully under
the complete upstream dependency stack. Preserve this gap before PR submission.
The initial tooling command in this review failed because the experiment venv has
no pip; the existing isolated pytest/Ruff tools were then used. Both logs remain.

## What the existing evidence supports

- Three-task reconstruction study: original S-VAE reconstruction MSE about
  45–50% below PCA; clean achieved-state-change probe differences are small and
  mixed. Weighting the first frame by four did not consistently help. Fixed
  final 2000-step checkpoints; three seeds; ten held-out episodes/task; episode
  split 35/5/10. These are offline achieved-state targets, not controller commands.
- The same episodes were subsequently reused for control-supervision development.
  Those follow-ups are not independent confirmation. Translation and gripper
  changes must be reported separately; gripper transitions are not contact labels.
- Published-coordinate adapter validation restoration MSE improved about 35%.
  The later frozen 80-rollout study found clean/noise success counts:
  identity 10/10,7/10; renorm 10/10,9/10; adapter 9/10,9/10;
  adapter+renorm 10/10,8/10. No reliable net benefit is established; small n,
  one task, and trajectory repeatability limit interpretation. All arms retained.
- No matched RAE-versus-VAE policy comparison has been run. Published DINO/S-VAE
  already uses pretrained features. Do not relabel a restoration residual as RAE.
- Later repeatability/attribution work does not increase the representation sample
  size. The context pilot stopped incomplete at 1/4 trials after a port preflight
  failure. It yields no A/B finding and will not be automatically restarted.

## Acceptance boundaries

For the evaluator, keep training mean/std, posterior-mean evaluation, element
weighting, separate conditioning/pooled-target errors, and target-energy-relative
errors. Shard paths alone do not establish episode disjointness; retain the
collection manifest. Relative MSE is not held-out-centered R-squared. A one-frame
cache has null pooled-target metrics. These outputs do not predict success rates.

Next software gate: exercise the documented CLI in a dependency-complete upstream
environment using a fixed existing small checkpoint/shard fixture, without GPU or
new model fitting. Do not mark the full upstream suite as passing from 17 tests.

Next research gate: inspect existing adapter-cache sample identities, split and
original HDF5 timestamps before designing a frozen control readout. This resolves
whether the same clean/noisy/restored observations can be aligned with current
state and future achieved-state targets without leakage. A valid train-only
readout would test whether restoration helps control-related directions; it is
not the deployed policy. Existing validation episodes are development data.
No new adapter loss, weight sweep or closed-loop run is authorized by this note.
A later experiment needs its own predeclared bounds and fresh confirmation plan.

No upstream PR or maintainer message has been sent. The ready artifact is a local
review packet and public evidence, not an accepted community contribution.
