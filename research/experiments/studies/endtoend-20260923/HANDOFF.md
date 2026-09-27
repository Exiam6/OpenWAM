# Active validation handoff

The latest direct instruction authorizes full new representation training and validation. Active study: endtoend-20260923. Old layers/confirmation deadlines remain unchanged and do not govern this new explicitly authorized study. Do not replay old80 or expired56; do not create agents/timers.

At19:52CDT2026-09-23, six native training runs oncm009GPU1–6 were at412–460 optimizer updates of6000, with finite logged losses. Seeds42/43 are active; seed44 is queued behind seed42 onslots1–3. Same training IDs/prompts/augmentation strengths checked across825/852 matched microbatches for seeds42/43. Source frozen; do not edit running training or compressor code.

cm001GPU5 collects adjust_bottle thenhandover_block; GPU7 collects place_object_basket. Snapshot19/50bottle,14/50basket, handover notstarted. Fixed collectiondeadline07:11:52CDT2026-09-24; atmost100candidates/task, no efficacy-based replacement. No new learned-policy outcomes yet.

Two finite dependency processes verified in tmux:
- `ow-e2e-smoke`: waits forcollectionslot7 release; one development-scene simulator smoke onthat sameGPU, profile-v3PCAcheckpoint. Output `/data02/zifanz4/openwam-experiments/endtoend-20260923/simulation-preflight/`. Actual GPU smoke remains pending, not passed.
- `ow-e2e-pipeline`: waits for all9trainings,all150fresh experts,and simulator smoke; verifies fullpaired training streams, checkpoint hashes and cohort identity; then launches six boundedcm009evaluation queues and completes fixed analysis/audit. Output `.../pipeline/`. Waitdeadline is frozeninpipeline-plan.json. It is a finite job dependency chain, not a new timer. No automaticfailed-job retry.

The evaluation code is syntax/CPUcontract-tested but its real simulator gate remains pending. If that preflight fails, preserve every failure and diagnose before releasing heldout outcomes. Technical corrections may get a newly versioned preflight/code freeze before any new efficacy outcomes; do not alter scientific protocol or silently overwrite gate evidence. Never restart an already attempted rollout automatically.

Native reconstruction tests for allthree encoders passed. Training/deploy multiview pixels exactlymatch. Independent prior60offline audit passed; none ofthis establishes policygain. Reduced295M primary comparison isPCA vsnativeSVAE; Wansecondary. CurrentPCA hasno pixeldecoder and isnotfullRAE. A supported small-model result stillneeds prospectivefull-sizevalidation.

Useful commands (runoncm001):
- `python3 /home/zifanz4/openwam-experiments/scripts/endtoend/update_status.py` reads artifacts and updates status/selfcheck/localHTML, no launches.
- Inspect `studies/endtoend-20260923/progress.json`, `.../handoff-verification.json`, actual `/data02/.../training/*/metrics-rank0.jsonl`, and everyexit.json.
- Never edit files frozen by `pipeline-plan.json` while the dependent jobs are waiting/running without recording a prospective technical revision and preserving oldattempts. The gate intentionally refuses mismatched source.

Draft community contribution: `contributions/dinov3-pca/optional-dinov3-pca.patch`,208lines, adds optionalencoder/config/CPUtest/docs; appliescleanly tolocalsourcesnapshot. Contractpassedvia direct invocation;pytest isnotinstalledinpolicy-env andthe initialpytestattempt ispreservedasfailed. No upstreamsubmission.

Public report remainsblocked: existingSitesprojectlookup returnedNOT_FOUND404again19:54CDT. Local HTML at`reports/endtoend-20260923/index.html` and`/home/zifanz4/research-reports/dist/endtoend.html`. No publicversion orreplacementsite wascreated. Initialbounds8concurrentGPUs/576GPUh; allheavydatais/data02. At19:52free/data02~295GiB, /home~294MiB; avoidheavy/homeoutputs.
