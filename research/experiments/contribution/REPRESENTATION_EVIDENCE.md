# Update: equal-noise state/goal evidence — 2026-09-23T00:34:01.562350-05:00

Equal-noise short-state control completed00:24:31 exit0,42fits/63prediction conditions independently audited;105train/exposed60eval. Noise.10stateE: Wan native0.308657/LN0.307048, DINOraw0.137719/PCA480.173212/SVAE480.173965, cachedproprio0.347911. PCAvsnativeWan -43.88% descriptive95[-52.42,-33.87]; same pooled192D visual input. RawDINO gripper+63.91% vsWan despite besttotalE. CompactPCA/SVAE gripper pointgains~11%, intervals crosszero; adjust_bottlegripper+3.56%/+2.29%vsproprio. Wan cleanregression~5.7%retained. No independent/fullRAE/closedloopgain. GPUreleased; no pause; originaldeadline unchanged. PublicSites lookup again failed projectnotfound; portableHTML updated.

Small contribution: paired native-Wan/pretrained-feature offline benchmark, including matched noise adaptation, PCA dimensionality control, component-wise gripper checks and cached proprio reference. Do not replace encoder/policy defaults. A separate documentation-only patch corrects WanVideoVAE.batch_encode input range to[-1,1], matching native preprocessing; source remains frozen/unmodified.

---

# Update: equal-noise representation controls — 2026-09-23T00:10:25.810495-05:00

Equal-noise goal control completed00:05:20 exit0;21newridgefits/63predictionconditions independently audited.105training scenes,7matched variants, per-scene regularization, cached exposed60eval. Wan noise.10goalE182.631→0.41479(native),314.540→0.43234(LN); DINOraw/PCA48/SVAE48afteradaptation0.11391/0.11069/0.11603. Wan cleanEworsens0.23838→0.36845and0.23708→0.27619; retained. DINOadvantage survives same augmentation and matched192D PCA control; not proof of fullRAE, causalpretraining, shortstate or policygain. Across-device DINO parity approximate (maxRMS0.04597trainSD), within predeclared bound. GPUreleased. See studies/rae-noise-control-20260922/FINDINGS.md. Public publication again failed: Sites project not found; local HTML updated.

Potential small contribution: an opt-in encoder/readout benchmark with paired perturbations, scene-grouped CV, per-scene regularization, PCA dimension control, full positive/negative reporting and validated native preprocessing. No default policy/encoder change or fullRAE claim is justified yet.

---

# OpenWAM representation contribution: evidence and scope

This packet separates measured representation results, an unfinished intervention,
and a reviewable engineering contribution. It does not propose changing the
released default representation on the present evidence.

| Question | Evidence | Defensible conclusion |
| --- | --- | --- |
| Does K4-SVAE48 improve short-horizon state readout over L12-SVAE48? | Completed 60-scene paired confirmation; noisy ridge error +45.675%, 95% interval [+38.023%, +53.588%]; fixed MLP +16.932%. | The fixed primary improvement criterion failed. |
| Does the same K4 change help initial-waypoint information? | Secondary endpoint on those same scenes: noisy ridge error −36.265%, 95% interval [−48.001%, −22.567%]; MLP −24.541%, with task-specific regressions. | A task/endpoint-dependent tradeoff, not an additional independent replication or a general control gain. |
| Is direct head-to-wrist failure attributable solely to the representation? | Matched wrist-readout baselines substantially reduce the head-readout mismatch; all 90 paired diagnostic conditions retained. | Readout training domain is a material confound; different visibility still prevents a pure geometric-invariance claim. |
| Does goal supervision fix the tradeoff? | Six equal-budget paired models and training-only readouts completed; new collection stopped at the fixed deadline with 20/20/16 accepted scenes. | Unanswered. No new confirmation score or partial-subset substitute. |
| Does a pretrained representation beat pixel VAE/RAE alternatives or improve the released policy? | No matched cross-backbone/codec family study or completed policy adaptation for these new representations. | Not established. |

## Small contribution ready for local review

The opt-in S-VAE sample-provenance patch preserves the association between shuffled
cached rows and their source episodes/windows, with a hash of the preprocessed
encoder input. This supports inspection of split identity and frozen-data studies;
it is not itself an automatic leakage detector or a representation improvement.
See PROVENANCE_PR_DRAFT.md and svae-sample-provenance.patch. Four targeted CPU
checks and Ruff passed earlier; they were not rerun in this archival step.
The default cache format/behavior is unchanged when provenance is disabled.
No actual distributed GPU collection validation or upstream submission is claimed.

## Research material that needs packaging before upstream integration

The complete layer comparison, all task/seed/readout conditions, scene-level paired
statistics, proprio-only and constant-goal controls, and camera-matched baselines
are retained in the portable report and source artifacts. These address evaluation
blind spots observed in this checkout: reconstruction quality alone does not
establish which control-relevant variables remain decodable. The current scripts
still contain study-specific paths and are not presented as an upstream-ready API.

Noisy goal improvement and state preservation must be tested jointly. Spatial
waypoint labels use the expert's first downward gripper-threshold crossing; they
are not object-pose or physical-contact ground truth. Frames, noise replicas and
compressor seeds are averaged within scene, not counted as independent samples.

Native collection also exposed copied absolute-asset-path issues and a null-grasp
sentinel leading to an assertion. The failure lineage and narrow, source-checked
outcome annotations are retained as reproducibility evidence. They are not a
reason to change simulator behavior or silently drop unrelated errors.

## Remaining uncertainty

The question is still whether a fixed auxiliary objective can retain both kinds
of information on an independent, sufficiently complete cohort. A future study
would need a prospectively fixed collection plan with measured full-pipeline
feasibility, unchanged matched-training controls and joint acceptance rules.
It must not relabel the expired 56-scene run as a successful confirmation, select
new scenes by model scores, or promote a favorable endpoint after the fact.
This packet schedules no new compute and sends no maintainer message or PR.
