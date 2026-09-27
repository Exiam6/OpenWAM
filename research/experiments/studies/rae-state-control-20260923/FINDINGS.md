# Equal-noise short-state control: improvement with a component tradeoff

Completed 2026-09-23 00:24:31 CDT, exit 0. One A40, 386.32 seconds; GPU released
and checked at 0 MiB / 0% / 0 uncorrected volatile ECC errors, no compute process.
The original study deadline remains 2026-09-23 01:15:12 CDT.

## Controlled comparison

105 training scenes, 12 fixed frames per scene, 7 variants per frame (clean plus
3 draws each at pixel sigma .04/.10). 8,820 rows represent 105 independent scenes.
All 1,260 clean vectors and 630 already computed noisy initial-frame vectors were
reused. Only 6,930 missing noisy training images were encoded. All existing
60-scene evaluation vectors were reused without another encoder pass.
42 scene-grouped ridge fits (7 routes x 3 tasks x translation/gripper), matching
fivefold CV, training-only normalization and base-alpha grid. Effective alpha is
7x the base alpha relative to the original 12-frame clean fit, not 84x. Both
Wan normalizations, all tasks/conditions and all three SVAE seeds are retained.
No encoder or compression weights were changed. SVAE errors are averaged within
the same scene; seeds are not counted as independent scenes.

## Results: short-horizon state E (lower is better)

E is the equally weighted mean of training-variance-normalized translation and
gripper MSE, then equally averaged over scenes and tasks. It predicts future
end-effector deltas and gripper state, not contact forces or commanded actions.

| Route | Clean | Noise .04 | Noise .10 |
|---|---:|---:|---:|
| Wan48 native | .311144 | .301808 | .308657 |
| Wan48 per-token LN | .317889 | .302171 | .307048 |
| DINO raw768 | .138145 | .126476 | .137719 |
| DINO PCA48 | .174708 | .158600 | .173212 |
| DINO SVAE48 seed mean | .171989 | .160043 | .173965 |
| Cached proprio-only reference | .347911 | .347911 | .347911 |

At noise .10, PCA48 versus noise-adapted native Wan changes E by -43.88%,
descriptive scene-paired 95% interval [-52.42, -33.87]; versus Wan LN it changes
by -43.59% [-52.33, -33.03]. PCA48 and Wan each have 192 pooled visual inputs;
DINO raw has 3,072. The matched low-dimensional result is not attributable simply
to the raw DINO readout having more inputs. Architecture/objective/pretraining
still differ, so this does not isolate a causal effect of pretraining.

Noise adaptation also rescues Wan's clean-trained state error: 37.338 -> .3087
(native), 26.538 -> .3070 (LN) at sigma .10. Therefore the original massive noisy
gap cannot be interpreted as irreversible loss of control information. Wan's
clean E worsens by about 5.7% in both normalizations. PCA/SVAE clean improvement
versus their own clean-trained versions is about 5.9%/5.6%, with intervals crossing
zero. No uniformly beneficial training intervention is established.

## Do not hide the gripper tradeoff

Noise .10, averages over three tasks; each column is a normalized MSE group.

| Route | Translation | Gripper | Gripper change vs native Wan |
|---|---:|---:|---:|
| Wan48 native | .596071 | .021243 | reference |
| Wan48 per-token LN | .592837 | .021259 | +0.08% |
| DINO raw768 | .240619 | .034819 | +63.91% |
| DINO PCA48 | .327466 | .018959 | -10.75% |
| DINO SVAE48 seed mean | .329050 | .018879 | -11.13% |
| Cached proprio only | .675046 | .020776 | -2.20% |

Raw DINO has the best aggregate and translation error but worse gripper error
than Wan on ALL three tasks (+102.38%, +114.20%, +39.04%). More representation
dimensions are not automatically better for every target under this readout.
The role of finite-sample readout regularization versus information content is
unresolved; do not claim compression causally removes nuisance information.

PCA/SVAE have lower translation and gripper point errors than both Wan variants
on each task. However PCA's aggregated gripper change vs native Wan has a
post-hoc descriptive interval [-25.09%, +9.84%], and SVAE [-24.93%, +8.86%].
These intervals include worsening: a reliable gripper benefit or noninferiority
has NOT been established. This subgroup summary does not replace the frozen
aggregate endpoint or promote exploratory intervals to confirmatory inference.

The cached proprio-only reference was declared before the state results.
PCA's total E is 50.21% below proprio-only, indicating that the joint visual/state
readout adds predictive value in this setup. The gripper story remains mixed:
PCA/SVAE gripper errors on adjust_bottle are +3.56%/+2.29% above proprio-only;
their average gripper changes are -8.74%/-9.13%, with intervals crossing zero.
Raw DINO's average gripper error is +67.59% above proprio-only. The comparison
reuses the original frozen proprio-only fit, with no new fit or heldout calibration.

## Independent arithmetic verification and limitations

All 42 fits and 63 prediction conditions passed a separate saved-artifact audit:
training statistics match exactly; normalized ridge normal-equation residual
6.51e-13, expanded affine prediction difference 2.33e-15, per-episode group
metric difference 6.22e-15. All 630 reused initial noisy vectors are bit-identical.
Frozen inputs/code remained unchanged, folds are scene-disjoint, and no evaluation
vectors or labels were accessed during fitting. The efficient eigensolver matched
the original direct solve on a training-independent synthetic fixture to 3.02e-14.

The supplemental proprio summary first stopped on a bit-exact aggregate comparison.
Diagnosis found only 1.11e-16 to 4.44e-16 differences from clean versus triplicated
noise reductions. The corrected check permits 1e-14 absolute roundoff. Every raw
score and the failure diagnosis remain; no model, result or primary audit changed.

Across-device DINO parity remains approximate: prior three-training-frame check
had max RMS .04597 and max coordinate .22464 in training-SD units, below the
predeclared .05/.5 limits. Wan was exact. This is not a full hardware invariance
study. The 60 evaluation scenes and perturbation strengths were already exposed;
all intervals are descriptive. These are offline expert-trajectory readouts, not
learned-policy rollouts, independent confirmation, measured contact reasoning,
novel-camera robustness, causal evidence of pretraining, or complete RAE training.
DINO's frozen-feature path still lacks the pixel decoder of a complete RAE.

## What this can contribute

A paired representation benchmark with a native Wan baseline, strict image-range
and causal-first-frame checks, an equal-noise-training control, train-only scene CV,
PCA dimension control, cached proprio baseline, and separate gripper/translation
reporting. The biggest methodological correction is that clean-trained noise
collapse is not itself proof of information loss. The practical candidate for
further validation is compact pretrained features; learned SVAE necessity is not
demonstrated because PCA is similarly competitive. Do not change policy defaults.

## Next uncertainty

Archive the equal-noise goal/state findings and prepare an independent confirmation protocol with frozen preprocessing, all routes, both Wan norms, clean/noisy conditions, and separate translation/gripper endpoints. The remaining original window cannot accommodate the measured roughly94-minute native60scene collection plus scoring; do not launch a knowingly over-budget cohort or shrink it after observing results. No extension past01:15:12CDT, no expired56/60collection reuse, no old80replay. Independent sample validation and fullRAE/world-action training remain future work, not completed gains. Finish the small reproducibility/input-range documentation contribution and preserve the deadline.
