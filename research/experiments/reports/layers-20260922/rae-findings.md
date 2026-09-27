# Matched Wan baseline — exploratory result

Completed 2026-09-22 23:52 CDT. One A40, 135.6 seconds for the matched probe,
18 new ridge readouts, 105 training scenes and the entire old exposed 60-scene
layer-study cohort. Two Wan normalizations, both endpoints and all three noise
conditions retained. Existing DINO outputs were reused without inference or refits.
This does not complete the independent goalaux56/60 cohort, replicate full RAE,
isolate pretraining causally, or demonstrate closed-loop policy improvement.

The interface preflight passed on eight training frames from one episode:
strict official Wan weights, expected48x24x20 current-frame latent, exact causal
first-frame and cache-reset equality, finite pixel decoding. Nine-frame encoding
peak allocated memory1.964GB. The matched probe exited0 and released its GPU.
Separate saved-artifact audit checked36prediction conditions and18CV choices;
expanded-affine prediction residual3.55e-15, episode-metric residual1.82e-12.

## Observed error (lower is better)

| Representation | State clean | State noise.10 | Goal clean | Goal noise.10 |
|---|---:|---:|---:|---:|
| Wan48 native scale |0.2944|37.3384|0.2384|182.6310|
| Wan48 per-token LN |0.3007|26.5377|0.2371|314.5404|
| DINO raw768 |0.2169|0.7943|0.1197|0.3832|
| DINO PCA48 |0.1857|0.3682|0.1351|0.6421|
| DINO SVAE48, three-seed mean |0.1821|0.3736|0.1760|0.6014|

Full noise.04 and per-task results, both Wan normalization contrasts, and
unadjusted descriptive bootstrap intervals are retained in probe-result.json.
They are not confirmatory intervals on an independently collected new cohort.
The goal is first-close expert waypoint XY, not object pose or actual contact.
The state score weights translation and gripper groups equally.

## Interpretation and remaining confound

DINO-derived representations support markedly more robust clean-trained ridge
readouts in this setup. PCA48 versus Wan48 keeps the pooled visual dimension at192;
rawDINO uses3072 and is a different capacity comparison. PCA48 and SVAE48 have
similar short-state noise errors; rawDINO retains lower goal error than either
compressor. This suggests a compression tradeoff, not proof that SVAE is required.

The enormous Wan noise degradation requires an explanation before promoting any
99-percent relative reduction as a representation benefit. A separately declared
post-hoc diagnostic used only saved vectors and fixed coefficients, with no
correction, retraining or model inference. Wan noisy feature RMS in training
standard-deviation units is5.6–8.0 for state and9.6–26.5 for goal, versus roughly1
for clean. For rawDINO the corresponding ranges are1.40–1.42 and2.10–2.56.
Across both Wan normalizations, tasks and the noise.10 condition,96.7–99.93percent
of noise-induced output-shift energy is the task-wide mean component. This is
energy of noisy-minus-clean predictions, NOT the fraction of total prediction
error, and does not establish that information is irreversibly lost.

Thus current evidence supports a strong noise distribution shift/readout bias
mechanism. It does not establish RAE superiority under equally noise-adapted
training. A fair next control trains every route with identical training-only
noise augmentation and retains all clean/noisy outcomes. Heldout average offsets
must not be used to calibrate the readouts. Full RAE additionally needs a trained
pixel decoder; training only that decoder cannot improve an unchanged frozen
encoder's features. No original protocol or deadline was modified.

## Next uncertainty

Before original01:15:12CDTdeadline, prospectively fix a bounded training-noise control for the initial-image goal endpoint: same105training scenes, identical clean/noise mixtures and episode-CV across Wan native/LN and cached-DINO-derived raw/PCA/SVAE routes. Estimate all normalization/augmentation/calibration from training data only. Reuse existing old60 evaluation feature vectors and retain their exploratory label; do not use heldout means to correct scores. This distinguishes robustly accessible information from the current clean-trained linear-readout distribution shift. No new run is launched by this report; freeze inputs/code/resources first. Expired56/60 and old80 remain untouched.
