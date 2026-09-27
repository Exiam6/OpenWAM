# Frozen-coordinate restoration / control-readout development diagnostic

Prepared 2026-09-22 before fitting this readout or inspecting its evaluation scores.
This is a NEW bounded CPU offline diagnostic, not another rollout or adapter fit.
Run exactly once after CPU gates and source/input freeze; keep all conditions.

Question: does the unchanged published-coordinate restoration adapter improve a
clean-trained achieved-state readout under the cached noise? Global latent MSE
and group-specific readout error may disagree. This does not compare RAE vs VAE.

Data: previously audited pick_dual_bottles adapter-cache. 420 unique clean frames
from35 training episodes;60 validation frames from5 episodes (6,11,19,22,35).
No demonstration test episodes opened. Two training noise replicas are NOT extra
clean training samples; train the readout on unique clean samples only. Inputs
are current-frame48-channel latents (T=1), pooled2x2 ->192 values, plus current
16D achieved proprioception. Targets: achieved translation delta at rawframe+4
(6D) and achieved gripper at that frame (2D). No rotation/contact/command target.
Raw frame indices are not physical timestamps. Cache manifest/code alignment was
checked; no retrospective byteproof of original per-image feature provenance.

Model: two independently fitted linear ridge readouts, visual+proprio and
proprio-only. Feature and target mean/std fitted on training rows only, std floor
1e-6, intercept represented by centering. Minimize mean squared standardized
error plus alpha times squared coefficients. Select alpha from [1e-4,.01,1,100]
using5 episode folds made from seed20260922 permutation of35 training IDs.
Fit normalization separately in each training fold. Select minimum mean fold MSE
(equal weight for8 standardized target coordinates); ties retain grid order.
Record CV scores and chosen alpha BEFORE opening validation target arrays for
scoring. Fit final readout on all420 clean training frames. Never refit it for
noise, restored representations, or validation results.

All reported conditions: proprio-only; visual clean; visual noisy (existing sole
validation noise replica); each existing MLP adapter seed42,43,44 on clean and
noisy latents. Primary adapter is preselected deployment seed42;43/44 are fixed
sensitivity checks, not candidates for a new deployment choice. Verify selected
checkpoint bytes equal MLP42. Also report one clean visual mismatch control:
cyclically replace validation episode with the next sorted episode, matching
ordinal sampled frame; retain original state/target. This is not realistic noise.
No renormalization, new loss, fitting of adapter, or architecture search.

Report all8 coordinate errors plus translation normalized MSE/RMSE in mm and
gripper normalized MSE/RMSE. Record per-episode errors; paired bootstrap over the
5 episodes (2000 common resamples, seed20260922), not over60 correlated windows.
For adapters report clean change vs visual clean and noisy change vs visual noisy,
and full latent restoration MSE against same clean latents. Save all predictions,
readout coefficients/scalers, selections, checksums and logs. Do not interpret
small-sample percentile intervals as independent confirmation or broad coverage.

Validity gates: fold episode disjointness; held-out label/feature mutation cannot
change training-only fit/selection; ridge arithmetic on known data; zero-initial
adapter preserves input exactly; bootstrap units are episodes. Match cached BF16
input/output contract, FP32 adapter arithmetic on CPU; this is not a GPU numeric
parity claim. Model/encoder/S-VAE frozen throughout; no policy generation.

Budget: one process,2 CPU/BLAS threads,CUDA invisible,300-second wall cap, no
shared-environment changes. Failure stops run; no changing protocol to improve
outcomes. Resource check before launch. Scientific experiment must not be rerun
by later selfcheck ticks merely because it has completed.

Limits: these5 validation episodes already chose adapter architecture, and the
published representation may have seen these demos. This is developmental reuse,
not an independent held-out test. Readout capacity/linear accessibility and
proprioception matter; error changes do not isolate lost information or imply
policy success. One task, sparse12frames/episode, one validation noise draw/frame.
If no consistent translation-and-gripper gain appears, do not launch another
adapter loss sweep from these outcomes. If a promising signal appears, require a
separate fresh-episode protocol and eventual closed-loop verification; never
promote an alternative training seed from this diagnostic alone.
