# Heldout evaluation actually running — 2026-09-27T12:32:09.036757-05:00

Active recovery: studies/temporal-recovery-v2-20260927. Both mean/learned seed42 passed33distinct real WS prompt cache checks (capacity32; post-eviction reference action exact) and all3native development tasks. Full development rollouts completed139/136steps; these reused development results are not test evidence. First fixed heldout group adjust_bottle/clean started12:30:50CDT on both routes, scene1100000. New output root and original fixed Sep29deadline recorded in active-study.json; no old80/270replay. Run report_progress.py to refresh progress from actual artifacts.

Next: Continue finite paired evaluation across3seeds×3tasks×3conditions×20scenes×2routes=1080. Require complete outcomes and independent raw audit before any benefit claim. Preserve any technical failure without automatic retry; each job obeys disk/ECC/lock/pause/deadline checks.

---

# Explicit closed-loop recovery — 2026-09-27T12:26:17.764674-05:00

User confirmed prompt recovery of closed-loop evaluation. Active study: studies/temporal-recovery-v2-20260927, separate recovery-plan.json and launch.json; outputs on alternate user-owned data volume with136GiB free at admission. Two finite lanes launched12:24:43, model loaders alive; deployment gates are in progress, not yet a completed heldout rollout. Existing six final temporal checkpoints and1080matrix unchanged. Fixed Sep29 17:48:49deadline unchanged. Read report_progress.py and progress.json for current output counts.

Independent recovery source contains only documented deployment/operational amendments: cache eviction repair, checkpoint parent-directory+explicit final filename, alternate output paths, multi-prompt WS parity and development gates, signal/disk/deadline guards. Original224 frozen sources unchanged. Initial recovery v1 failed before model load because old serve entrypoint passed a file to directory API; both failures retained, zero outcomes. v2 all6 checkpoint argument contracts checked before launch.

Next: Require33real training-only WS prompts, exact post-eviction reference action and native development rollout gates before heldout scenes. This resolves deployment correctness; only the full paired1080matrix can test policy benefit. No automatic retry on a new failure. Old270/5400 and6 failures stay untouched; old80/layers never replayed. No new timer or subagent.

---

# Recovery verified — 2026-09-27T12:14:40.031720-05:00

All six temporal trainings finished4000microsteps/2000updates, exit0; last completed01:22:02CDT. Full paired streams42/43/44 match; existing final-checkpoint audit passed and224 frozen source hashes remain unchanged. Temporal closedloop0/1080: both seed42 evaluation admission attempts exited1 at01:24 before child creation, followed by terminal lane exits. The bare AssertionError does not definitively log its source; timing and current disk shortage are consistent with the32GiB admission guard, not evidence that these attempts hit prompt-cache eviction.

Prior endtoend:270/5400 completed,6 infrastructure failures; all6 queues and controller exit1, last02:22:47CDT. No complete benefit comparison. Resources now~16.8GiB free on data volume (below32GiB temporal and20GiB prior admission limits),~1.2GiB home and~45MiB root. No training restarted or failed evaluation retried. No pause markers; original deadlines unchanged. Control-SVAE remains planned/unlaunched.

Next: Preserve all results and failed starts; check storage recovery and separately versioned multi-prompt deployment gates before any prospective recovery. This distinguishes operational readiness and inference equivalence from policy-success benefit. No automatic failed-queue replay or deadline extension.

---

# Accepted next experiment: control-supervised S-VAE — 2026-09-26T19:38:58.247267-05:00

User explicitly asked to add this representation-learning study. Design: studies/control-svae-20260926/PLAN.md. Status: planned, not launched; numeric launch recipe and absolute deadline must be frozen after training-only technical checks. Two matched reducers, three paired seeds, original temporal mean and48channels; only auxiliary control gradient differs. Future four-recorded-frame deltaXYZ6/future gripper2 are labels, never inputs or contact ground truth. Fresh sealed60scenes; planned1080rollouts; primary clean task success.

Next: audit label time/units, historical data exposure, current-only inputs and reducer gradient paths; profile within existing resource limits, then freeze the independent launch protocol. No need for another user authorization to perform that preparation. Current temporal-20260926 lanes continue under unchanged Sep29 deadline, prior evaluation under unchanged Sep28 cutoffs. This plan addition does not start a GPU job or change any existing experiment.

---

# Current priority: learned temporal compression — 2026-09-27T00:07:29.932367-05:00

Four of six matched trainings completed: seeds42/43 mean/learned each4000microsteps/2000updates exit0; seed44 running mean1921/learned1868. Full42/43 and shared44 streams match;224 frozen files unchanged. All six encoder parity gates passed.0/1080 closedloop; no policy benefit claim. Deadline remains Sep29 17:48:49 CDT.

Prior endtoend has180 completed rollouts/5400 and4 infrastructure failures: queues1/3/4/6 exit1,2/5 still waiting. Text-cache eviction remains in frozen sources. Isolated cache integration (studies/cache-integration-20260927) completed00:05:16 exit0 in72.08s, GPU5 released. Five first20D action predictions exactly match across original cold/hit and fixed cold/hit/post-eviction. One training frame, no heldout scene, no WebSocket or full simulator check. This verifies a limited repair path, not success-rate improvement or completed evaluation recovery.

Next: Prepare separately versioned cache-recovery and multi-prompt deployment checks, preserving frozen code, partial results and old deadlines. Continue control-SVAE timing/exposure/gradient preflight before its own numeric freeze.

Control-SVAE read-only label availability audit passed105episodes/23204 valid t+4 pairs; physical time horizon remains unverified (collection may insert boundary frames), no representation training launched. Latest storage snapshot: root0.084GiB/home1.829GiB/data0266.072GiB free; use data volume and check bounds before large writes. No new timer, subagent, old80/layers replay or deadline extension.

---

<!-- END_TO_END_CURRENT_BEGIN -->
# Current end-to-end validation — 2026-09-27T12:12:34.345362-05:00

All nine fixed trainings completed exit0 (12000 microsteps / 6000 optimizer updates). All three seed-paired batch streams match. No training is queued or still running. The 150 fresh expert trajectories passed the frozen file/image/pose exclusion check; all nine full final-checkpoint payloads were hashed and the evaluation-ready gate passed: True.

The old six-slot resource barrier expired Sep25 19:11:52 CDT. Its waiter and controller exited1; original records/deadlines remain unchanged. Under the existing complete-validation authorization and direct continue, the separate first-start recovery began19:46:45. It changes operational admission only: each original cm009GPU1-6 queue independently waits for three idle observations. Same scientific code, scenes, routes/seeds, final checkpoints, conditions, inference and statistics; no completed experiment repeated.

Current snapshot: 6/6 finite queues started; 6 admitted, 0 still active, 6 terminal. Completed learned-policy rollouts 270/5400; technical failures 6; action diagnostics 0/48600. Status: evaluation_partial_infrastructure_failure. Queue start is not GPU reservation or policy inference. Primary PCA-vs-SVAE benefit remains unestablished unless the full matrix and independent audit pass. Model scale remains295M, not5B; PCA is not fullRAE.

Absolute evaluation cutoff Sep28 10:41:59 CDT; overall stage cutoff Sep28 13:41:59 CDT, unchanged from the previous outer-controller deadline. Each queue also retains the original232800-second execution cap. Failure/partial results retained without automatic retry. No layers-window restart, old80 or expired56 replay, new agents, new recurring timers or unrelated process termination.

Next: Inspect the complete terminal evidence and retain every failure. No automatic replay or deadline extension. A primary benefit claim requires all 5400 rollouts plus the independent audit.

The existing public site is accessible again; actual publication outcome is in studies/endtoend-20260923/publication.json. Local report: reports/endtoend-20260923/index.html. Keep the existing3-hour monitor; the delayed00:00 message is not backfilled as a completed selfcheck. Read the recovery plan and live controller/queue records at every check; refresh this snapshot with scripts/endtoend/report_evaluation_recovery_20260925.py, not the old frozen update_status.py.

<!-- END_TO_END_CURRENT_END -->

# New end-to-end study authorized; independent audit passed — 2026-09-23T18:49:16.856826-05:00

The user explicitly authorized complete training and closed-loop validation. Active study: studies/endtoend-20260923 (PLAN.md and authorization.json). All old frozen protocols, raw outcomes and deadlines are preserved. The following historical closed-window restrictions do not prohibit this newly authorized study.

The separate bounded CPU audit completed exit0: 60 raw scenes, 135 predictions, 99 component intervals; maximum metric residual 7.55e-14. The independent offline PCA-vs-Wan goal/state evidence is now numerically audited; it still does not establish PCA superiority over native S-VAE or any policy/RAE benefit. Audit: studies/endtoend-20260923/confirmation-audit.json.

Next: validate the optional PCA encoder and native checkpoint parity; construct matched 105-episode training data/statistics and common trainable initialization for Wan48/SVAE48/PCA48; profile before fixing the full training and fresh-scene evaluation budget. Runtime artifacts use /data02. No new training job is claimed launched yet. Existing public Sites report remains unavailable (last verified 404).

---

# Independent60scenes completed; window closed; audit pending — 2026-09-23T15:53:49.370099-05:00

All60newscenes collected (20/20/20;78attempts); all3collectors exit0. Frozen feature/scoring job completed12:20:42 exit0, wellbefore unchanged15:38:51CDTdeadline. No ownedstudyGPUworker remains; cm009GPU1/2/3 andrainierGPU4 checked1MiB/0%/ECC0at15:49. No newexperiment afterdeadline.

Stored scorerflags co-primarypass: PCA48vsnativeWan noise.10goalerror-73.05% (paired95[-79.60,-65.72]),stateE-39.11%([-48.70,-27.30]). Gripper-12.33%([-26.12,+5.91])remainsuncertain; rawDINOgripper+79.65%([42.65,126.73]). Fullmatrix andtaskdifferences retained. These are pending independent numericalaudit, not finalauditedbenefit or fullRAE/closedloopclaims. 370frozenfilehashesverifiedunchanged; pipelineidentity/paritypassed.

RecoveredA40featurepipeline12:07usingprivateglibccompatibility; originalfailedL40Sparity andpre-Pythonrainierstartupattempt retained. Automaticapprovalservice usagefailure prevented12:08records/reportwrite andsubsequentsupervision until15:49. Backgroundjobscompletednormally. audit.py was preparedbutnotexecutedbeforedeadline; itsguardunchanged. See studies/confirmation-20260923/FINDINGS.md.

Next: New4hwindow closed15:38:51CDT with all60scenes and frozen scoring completed12:20:42. No new experiment, refit, inference, rescore or automaticdeadlineextension. Preserve old56unscored/old80unchanged and allnewfailures. Independent numerical audit scripts/confirmation/audit.py was prepared but notexecuted beforedeadline because toolapproval service exhaustedusage; its original deadline guard is preserved. Next required validation is a separately authorized bounded CPUartifact audit of rawlabels,135savedpredictions and99pairedintervals; then package minimal representation-baseline and component-reporting contribution. Current completed-run numbers remain provisional pendingthat audit; no fullRAE/closedloop claims.

PublicSiteslookup15:51stillfailed404projectnotfound. LocalHTMLandartifactsupdated; no newsiteor publicsuccessclaim.

---

# Original12hour budget window CLOSED — 2026-09-23T01:19:04.719517-05:00

Ended at 2026-09-23T01:15:12-05:00 without extension. Last experiment finished00:24:31; no owned study worker remains. Layer primary failed; goalaux56/60 remains incomplete/unscored; equal-noise goal/state controls completed and audited, with compact-feature advantages and unresolved gripper tradeoffs on exposed60. No completeRAE or closedloopgain claimed. See studies/layers-20260922/WINDOW_CLOSURE.md and the local HTML. Public lookup01:17still returns Sites project not found.

Next: The original12hour window ended2026-09-23T01:15:12-05:00. No new experiment, training, collection, inference, confirmation scoring or automatic budget extension is authorized by later timer messages. Preserve completed and negative results, keep expiredgoalaux56/60 unscored, and do not repeatold80. The next scientific uncertainty is whether compact-feature goal/translation gains and gripper tradeoffs reproduce on new scenes. The prepared independent60scene draft estimates about3hours with one idle ECC-cleanGPU and requires a separately fixed, newly authorized experiment budget before implementation/launch. FullRAE decoder/world-action training and policy gains remain untested. Subsequent selfchecks should reconcile terminal state quietly unless a material change occurs.

---

# Equal-noise short-state control completed — 2026-09-23T00:34:01.562350-05:00

Equal-noise short-state control completed00:24:31 exit0,42fits/63prediction conditions independently audited;105train/exposed60eval. Noise.10stateE: Wan native0.308657/LN0.307048, DINOraw0.137719/PCA480.173212/SVAE480.173965, cachedproprio0.347911. PCAvsnativeWan -43.88% descriptive95[-52.42,-33.87]; same pooled192D visual input. RawDINO gripper+63.91% vsWan despite besttotalE. CompactPCA/SVAE gripper pointgains~11%, intervals crosszero; adjust_bottlegripper+3.56%/+2.29%vsproprio. Wan cleanregression~5.7%retained. No independent/fullRAE/closedloopgain. GPUreleased; no pause; originaldeadline unchanged. PublicSites lookup again failed projectnotfound; portableHTML updated.

Next: Completed goal/state controls, independent audits, local report, confirmation draft and documentation patch are archived in c4255fa. No new experiment fits inside the remaining original01:15:12CDTwindow on the validated full60scene collection path. Preserve all terminal results and check only for material state changes or deadline closure. The next scientific uncertainty is independent-scene reproducibility of compact-feature goal/translation gains and gripper tradeoffs; the prepared new60scene draft is not launched and would require a separate fixed budget. No old80 or expired56/60 replay, no extension.

---

# Equal-noise goal control completed — 2026-09-23T00:10:25.810495-05:00

Equal-noise goal control completed00:05:20 exit0;21newridgefits/63predictionconditions independently audited.105training scenes,7matched variants, per-scene regularization, cached exposed60eval. Wan noise.10goalE182.631→0.41479(native),314.540→0.43234(LN); DINOraw/PCA48/SVAE48afteradaptation0.11391/0.11069/0.11603. Wan cleanEworsens0.23838→0.36845and0.23708→0.27619; retained. DINOadvantage survives same augmentation and matched192D PCA control; not proof of fullRAE, causalpretraining, shortstate or policygain. Across-device DINO parity approximate (maxRMS0.04597trainSD), within predeclared bound. GPUreleased. See studies/rae-noise-control-20260922/FINDINGS.md. Public publication again failed: Sites project not found; local HTML updated.

Next: Within the unchanged01:15:12CDTdeadline, prospectively freeze the short-horizon state counterpart of the equal-noise control: identical105training episodes and12fixed current frames per scene, all frozen representations, noise0.04/.10 draws, episode-disjoint CV and alpha_eff=7*alpha relative to the original12-frame-per-scene fit (not84*alpha). Reuse already extracted initial-frame noisy vectors from this run and all original clean/evaluation vectors; only encode missing training-frame variants. Retain all translation/gripper, clean/noisy, task and seed results. This resolves whether the goal-information advantage also preserves short-state information. No world/action-policy gain is claimed; no expired56/60 or old80 replays. No launch follows automatically from this report; freeze and validate code/resources first.

---

# Matched Wan baseline completed — 2026-09-22T23:56:27.370049-05:00

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


New result: DINO representations have much lower noisy clean-trained ridge error than Wan; very large Wan distribution shift and shared output bias explain why this is not yet a fair noise-adapted/fullRAE benefit claim. All conditions retained, independent arithmetic audit passed. See studies/rae-baseline-20260922/FINDINGS.md.

Next: Before original01:15:12CDTdeadline, prospectively fix a bounded training-noise control for the initial-image goal endpoint: same105training scenes, identical clean/noise mixtures and episode-CV across Wan native/LN and cached-DINO-derived raw/PCA/SVAE routes. Estimate all normalization/augmentation/calibration from training data only. Reuse existing old60 evaluation feature vectors and retain their exploratory label; do not use heldout means to correct scores. This distinguishes robustly accessible information from the current clean-trained linear-readout distribution shift. No new run is launched by this report; freeze inputs/code/resources first. Expired56/60 and old80 remain untouched.

Public report remains unpublished: existing Sites project lookup returned project not found. Local HTML updated, no replacement site.

---

# User steering — prioritize direct RAE comparison — 2026-09-22T23:28:11.609894-05:00

User asked “不用rae吗”. The prior next-experiment answer focused on SVAE auxiliary
supervision and omitted a direct RAE baseline. Verified official RAE definition
and local OpenWAM code. DINOv3 raw features are already measured; SVAE48 adds a
semantic bottleneck; current native DINO path has no pixel decoder. Decoder-only
training with a frozen encoder cannot itself improve those unchanged features.
Scope written in studies/RAE_FOLLOWUP_SCOPE.md. No new experiment launched,
frozen protocol altered, deadline extended or previous result reinterpreted.

Next: Prepare the direct Wan pixel-VAE / frozen DINOv3 high-dimensional representation / same-DINOv3 SVAE48 comparison described in studies/RAE_FOLLOWUP_SCOPE.md. Reuse existing raw-feature evidence without relabeling it a complete RAE reproduction. Audit training-only preprocessing/temporal alignment and available baseline assets, then freeze any new runnable experiment before launch. Goal-auxiliary remains a secondary matched intervention; expired56/60confirmation stays archived/unscored; original01:15:12CDTdeadline and all resource caps retained.

---

# Selfcheck — archived outcome unchanged; log transport corrected — 2026-09-22T19:33:12.709614-05:00

Trigger2026-09-22T19:30:01.718953-05:00; handling began19:30:25CDT.
Actual remote check2026-09-22T19:31:07.946252-05:00: accepted counts20/20/16,70distinct
candidates started/69completed, collectorexit1 and finalworkerexit124 unchanged.
Both old dependent scorer exits remain1; no score summary, confirmation-v3,
owned worker or goalaux tmux session. No pause markers. No experiment launched.
GPU now829MiB/0%/0ECC, process1511221 belongs to jw116; our job released it,
and this is another user's later allocation. No process was signaled.

One archival issue resolved: seven worker logs contained carriage returns that
initial read_text transport normalized. All seven normalized hashes exactly
match the old local text copies; remaining42files match raw source hashes.
Added byte-exact copies and a49filecanonical terminal-archive-index.json, preserving
the original text archive. No remote modification or HDF5/model/score change.
A note corrects the previous overly broad '49 raw files unchanged' wording.
This is an archival transport correction, not a new scientific result or rerun.

Scientific status remains incomplete/unscored. Existing local report adds the
transport note; public scientific result unchanged, no deployment attempted.
Latest publication error remains the19:21recorded Sites404, not a new lookup.
No timer, subagent, protocol change, budget extension or user decision required.

Next: Keep this expired confirmation archived and unscored. At a later selfcheck compare terminal metadata using studies/goalaux-20260922/terminal-archive-index.json (canonical raw copies), inspect exits/owned-worker presence, and notify only a meaningful change. Do not repeat the 56-HDF5 audit or any completed fit/evaluation. The GPU is now occupied by another user and must not be treated as reserved or interrupted. The unresolved scientific question remains joint goal improvement plus state preservation; no valid answer is available from the incomplete cohort. Publication remains blocked by the previously observed Sites404; no new publication result this turn. Collection deadline19:15:55Sep22 and overall deadline01:15:12Sep23 unchanged.

---

# Selfcheck — collection deadline reached; confirmation incomplete — 2026-09-22T19:21:18.931171-05:00

Trigger 2026-09-22T19:15:01.385636-05:00; handling began19:15:20CDT.
Remote final snapshot19:16:02 and archival check2026-09-22T19:17:48.667533-05:00 verify
56/60 accepted scenes (20/20/16), 70 distinct candidates started, 69 completed,
3 unstable and10expert-infeasible outcomes retained. Last candidate802024 timed
out with workerexit124; parentexit1 written2026-09-22T19:15:56.641086-05:00.
The original19:15:55.548938collection deadline held. No owned worker/tmux job
remains; GPU1MiB/0%/0ECC. No fresh model scores or confirmation-v3 launch.

Archived49raw metadata/log files unchanged under fresh-v4-terminal. Terminal
CPU/file audit passed all56accepted hashes (834104320totalHDF5bytes), native
expert plan/replay, actualCurobo/renderprovenance, consecutive seed/candidate
bounds and completion beforedeadline. Cross-cohort identity/label/prediction
gates did not execute and are not claimed passed. Frozen protocol JSON and
Markdown hashes match. No inference, refit, old80replay, deadline extension or
other-task termination. No pause markers were present.

INCOMPLETE_CONFIRMATION.md records the terminal interpretation without changing
raw outcomes. Contribution/REPRESENTATION_EVIDENCE.md distinguishes the completed
layer-study tradeoff, unmeasured auxiliary benefit, and opt-in provenance patch.
Portable local HTML/status/evidence updated. Existing Sites project lookup again
returnedNOT_FOUND404; no deployment, publicversion update or replacementsite.

Next: The goalaux confirmation branch is terminal/incomplete: retain all 56 accepted scenes, 70 distinct started candidates, six trained models and frozen readouts. Do not restart collection, score the partial cohort, refit, extend the original bounds or reinterpret incompleteness as benefit/no benefit. Scientific uncertainty remains whether fixed goal supervision jointly improves noisy initial-waypoint information and preserves state information; this run cannot answer it. The completed layer-study evidence and opt-in provenance patch are available for review in contribution/REPRESENTATION_EVIDENCE.md. Subsequent selfchecks should only verify unexpected changes or address the existing publication blocker; no duplicate GPU job/timer/agent. A future scientific experiment would need its own prospective protocol and cannot silently complete this expired cohort. Original overall study deadline remains 2026-09-23T01:15:12-05:00.

---

# Selfcheck — existing collector healthy; confirmation still pending — 2026-09-22T19:09:46.679527-05:00

Trigger 2026-09-22T19:00:01.918398-05:00; processing began 19:04:35 CDT.
Actual remote snapshot 2026-09-22T19:08:35.408263-05:00: 52/60 accepted scenes (20 adjust_bottle,
20 handover_block, 12 place_object_basket); 64 completed distinct candidates,
52 accepted / 3 unstable / 9 expert-infeasible. Active basket seed 802019,
PID 1503203 alive, worker log updated 19:07:47. Sole collector tmux session
alive; collector exit not yet written. GPU memory 1837 MiB / 0% utilization /
0 uncorrected volatile ECC at the sampled instant. Confirmation-v2 exit 1 is
the previously recorded dependency failure; no fresh score summaries exist.
No pause markers found locally or in the checked runtime locations.

Reviewed the existing collector bound: the parent limits each owned worker to
min(600 seconds, original remaining time), terminates only its own process group
on timeout, and the launcher has a 30-second cleanup grace. The grace is not an
extension of the data-collection budget. No runtime code, frozen protocol,
scientific criterion, seed order or task was changed in this check. No fits,
completed candidates, old 80 policy evaluations or tests were repeated.

Next: At the next actual check, inspect fresh-v4 terminal exit, progress, run.log and GPU release against the ORIGINAL collection deadline 2026-09-22T19:15:55.548938-05:00. If incomplete, archive the incomplete cohort and do not calculate partial confirmatory scores or extend/restart collection. Only if all 60 scenes and exit 0 are verified may the unchanged frozen identity/parity/scoring/audit stages be routed to fresh-v4 and new confirmation-v3, with routing frozen first and original study deadline 2026-09-23T01:15:12-05:00 retained. This resolves whether the prespecified independent confirmation cohort exists; it does not yet establish a representation benefit.

No new substantive result, failure or decision in this check; ordinary collection
progress remains local. No public deployment attempted or claimed; the previous
Sites 404 remains unresolved. No new timer, agent or other job launched.
Evidence: studies/goalaux-20260922/selfcheck-20260922T190835.json.

---

# Selfcheck continuation — second native null-grasp site; 49/60 retained — 2026-09-22T19:03:53.302286-05:00

Still processing18:45trigger; original study/collection deadlines unchanged.
Fresh-v3 collected two additional accepted scenes, then exited1at802013 because
the same native(None,None)grasp sentinel arose at the later basket-grasp call,
line122. The intentionallyfirst-grasp-only classifier did not match and stopped.
Confirmation-v2 exited1at dependencygate without identity/parity/scoring. Preserve
both failures; no completed seedincluding802013repeated.

Recorded NULL_GRASP_ALL_CALLS_AMENDMENT.md at18:59:53. NativeAST enumerates all three
grasp_actor calls in this task:73,112,122; all use the same alreadyverified native
null-sentinel body. Metadata-only classification checks pass allthree and reject
other methods/errors/sites/tasks/nonzero exits. No simulator/native source or
numerical scoring/acceptance criterion changed. Original errors remain annotated
in the consolidated candidate manifest. Fresh-v4 froze/launched19:00:39, retained
59completed candidates/49accepted scenes and resumed802014 under ORIGINALdeadline
19:15:55.548938CDT. No scorerrelaunched while the cohort remains incomplete.

Snapshot2026-09-22T19:01:55.506504-05:00: counts{'adjust_bottle': 20, 'handover_block': 20, 'place_object_basket': 9};60candidatescompleted,49accepted,3unstable,
8expert-infeasible including2source-proven null-grasp classifications. Active
802015process/logalive; sole tmuxcollectorv4alive. GPU845, 3, 0; no newscore.
Remaining time is limited; full60completion is not assured and noextensionallowed.

Routing metadata harness first inspection failed on torch.inference_mode wrapper
globals(KeyError O), without executing any function. Originalfailure preserved;
inspect.unwrap fixed only that harness and remainingfunction checks passed18:57:48.
The frozen pipeline code was unchanged. Evidence:all-calls-current-check.json,
null-grasp-all-calls-check.json,confirmation-routing-import-check.json, terminal
fresh-v3 andconfirmation-v2 snapshots. No prior evaluation repeated.

Next: Inspect fresh-v4 actual logs, manifest and exit first; only collector wam-goalaux-collect-v4 is running. confirmation and confirmation-v2 both stopped at dependency gate, no scoring executed. Do not mistake these old exits for current collector failure. Original collection deadline19:15:55CDT Sep22 is binding: if incomplete, archive as incomplete with no partial confirmatory score or extension. ONLY if full60 and collector exit0, route original frozen gates/scorer/auditor to fresh-v4 and a new confirmation-v3 output, freeze directory routing, run unstarted stages on idle ECC-clean GPU within original study01:15:12CDT Sep23 deadline. Existing original numerical code/models/thresholds unchanged; no completed seed/evaluation replay.

Local HTML updated with both repair records and stopped-scorer state. This turn's
Sites lookup returnedNOT_FOUND404; no publicdeployment or replacementsite.

---

# Selfcheck — native null-grasp failure classified; retained 47/60 scenes — 2026-09-22T18:55:49.071657-05:00

Trigger18:45:02processed18:45:25. No pause; original studydeadline01:15:12CDT
Sep23 and collectiondeadline19:15:55CDT Sep22 unchanged. Collector fresh-v2 exited1
at basket802008; dependent confirmation alsoexited1 BEFORE any identity/parity/
extraction/scoring stage. Selected GPU released1MiB/0%/0ECC. Original failures,
tracebacks, outcome and both exits retained under fresh-v2-terminal and
confirmation-original-failure in this controlrepo; runtime originals unchanged.

Source-proven cause: first basket grasp returns(None,None), then nativegrasp_actor
constructs Action(move,None). This cannot be an expert-feasible trajectory.
Exact native AST/Action assertion CPUcheck passed; validbranches and failed-plan
shortcircuit unchanged; negative classifier tests reject other errors/tasks/phases,
nonzero exits and acceptedepisodes. Native sourcehashes verified. The scientific
protocol permits recorded failed-work repair; NULL_GRASP_AMENDMENT.md fixes only
this exact source/traceback outcome classification, preserving original_outcome.
No planner/simulator/task source, feasibility acceptance, numerical scoring or
scientific threshold changed. No representation/policy score informed this repair.

Fresh-v3 froze/launched18:50:32, retained all54completed candidate attempts and47
accepted scenes in place, started at802009; no completed seedincluding802008replayed.
Originalcollectiondeadline unchanged. Confirmation-v2 routes only directories;
original scoring routines/readouts/checkpoints unchanged,69filesfrozen18:51:41.
Only previouslyunstarted downstream stages queued; waiting allocatesnoGPU.

Current verified snapshot2026-09-22T18:53:32.552933-05:00:counts{'adjust_bottle': 20, 'handover_block': 20, 'place_object_basket': 7},attempts56,
outcomes{'accepted': 47, 'unstable_scene': 3, 'expert_infeasible': 6}. Active802011worker/logalive; both existingcontinuation
jobsalive, no exits/newsummary. GPU5502, 0, 0. Two newlyattempted candidates after
repair were ordinarynative expert-infeasible and remain recorded. No new benefit
claim. Evidence:repair-current-check.json,null-grasp-check.json and both amendments.

Next: Read actual G2 results/goalaux-20260922/fresh-v3 and confirmation-v2 logs/exits (supersede stopped fresh-v2/confirmation). Native null-first-grasp classifier repair recorded before continuation; original failure retained, no completed seed replay. Continue existing jobs only. Need full60 before unchanged identity/parity/scoring; no partial subset scores. If remaining scenes do not finish by ORIGINAL19:15:55CDT collection deadline, record incomplete and do not extend. The unresolved question remains joint goal improvement and state preservation; overall study deadline01:15:12CDT Sep23 unchanged.

Local HTML/public-safe evidence updated. Existing Sites get_site still404; no
publicversion update, replacementsite, newtimer or agent created.

---

# Selfcheck — first task collected; 36/60 fresh scenes — 2026-09-22T18:27:48.913895-05:00

Trigger17:45:01CDT processed18:24–18:27CDT. Original study deadline01:15:12CDT
Sep23 and collector deadline19:15:55CDT Sep22 unchanged; no pause markers.
Snapshot 2026-09-22T18:25:02.837950-05:00: adjust_bottle20/20,handover_block16/20,basket0/20.
41candidate attempts complete:36accepted,3typed unstable_scene,2expert_infeasible.
All completed candidate exits0; actual native CuroboPlanner and correct GPU PCI
records verified for all36accepted scenes. Active801017worker/log alive; selected
GPU2201MiB/0%/0uncorrectedECC. Collector and bounded dependent scorer bothalive,
no terminal exits or confirmation scores. No completed evaluation was repeated.
Read-only storage snapshot: isolatednativecopy17.75GB and newstudyresults1.28GB;
these are this intervention's directories, not an estimate of the whole cluster.
Evidence: studies/goalaux-20260922/selfcheck-20260922T182503.json,provenance-progress-20260922T1825.json.

Judgment: first planned task reached20/20, collection remains healthy; no new
scientific performance conclusion. No protocol, seed, resource or timer change.
Next: Continue the two existing jobs; first task now complete20/20, second16/20, third not started. Remaining uncertainty: whether fixed goal auxiliary improves independent-scene noisy goal readout while preserving clean/noisy state readout. Wait for full60, then frozen identity/parity/scoring and independent audit. No partial-cohort score, new fit or duplicate launch. Collection still stops19:15:55CDT Sep22; overall study ends01:15:12CDT Sep23 even if messages arrive late.

Local HTML/report counts updated. Existing Sites project lookup retried and still
returns projectnotfound; no publicversion updated or replacementsite created.

---

# Selfcheck — new matched confirmation collecting (8/60) — 2026-09-22T17:34:18.326944-05:00

Snapshot 2026-09-22T17:31:57.612601-05:00. Resumed; no pause markers. Original study window unchanged:
13:15:12CDT Sep22 to01:15:12CDT Sep23. Delayed16:15/16:45triggers coalesced in this
active turn; no new timer, subagent, completed experiment replay, or task termination.
Old80policy evaluations, completed layer results, and six matched codec fits unchanged.

New readout fits exit0:105original train episodes/1260clips,54ridge groups and36MLPs.
Independent saved-data/statistics/folds/normal-equation/NumPy-forward audit passed;
maximum relative ridge residual 1.75e-15.
Training-only interface checks cover69new/anchor state and goal model paths.

Copied isolated native environment to Group2 because no idle ECC-clean Group1L40S.
273native source/config hashes verified. Empty-scene native rendering/CUDA binding
passed exit0. First seed800000 then stopped before actions: copied robot YAML still
referred to Group1 collision assets and triggered Mplib fallback. Actual-backend guard
prevented collecting altered data. Initial failure/exit1 retained. Permitted recorded
failed-work repair changed only two copied YAML absolute path prefixes, verifying
URDF/collision geometry hashes; no completed expert trajectory or fit repeated.
COLLECTION_PATH_REPAIR.md records the continuation before it ran. Same seed800000
initialization repaired under fresh-v2; real CuroboPlanner and render PCI pass each scene.

As of snapshot: {'adjust_bottle': 8, 'handover_block': 0, 'place_object_basket': 0}; completed distinct candidates8,
current collector exitNone. One L40S used,4threads; GPU snapshot 881, 9, 0.
Collector originaldeadline 2026-09-22T19:15:55.548938-05:00 remains unchanged.
Bounded confirmation job is already queued, no GPU during dependency wait;58files
frozen before scores. It requires all60scenes, content identity against210previous
scenes, and exact encoder/codec zero-noise parity. One-hour post-allocation cap,
original studydeadline; no refits. No new confirmation metrics have been observed.

Next: Read G2 results/goalaux-20260922/fresh-v2 and confirmation actual logs/exit codes. Bounded dependent job already waits for all60 scenes, then identity/parity/extraction/scoring/independent CPU audit; do NOT create a duplicate launcher. Next uncertainty: whether goal auxiliary improves noisy initial-waypoint information while preserving both clean/noisy state information on independent scenes. Joint ridge criteria fixed; no benefit claim until complete audited result. Collection deadline19:15:55CDT Sep22 and study deadline01:15:12CDT Sep23 unchanged. Any further infrastructure failure stops; preserve artifacts and repair only with recorded evidence, never replay completed work.

Publication retry: existing Sites project stillnotfound. Local HTML and public-safe
artifacts updated; publicversion26 NOTupdated. Existing Sites account clarification
pending; no replacementsite or duplicate timer created.

---

# Selfcheck — GPU contract passed; matched intervention 6/6 fits — 2026-09-22T16:37:38.606396-05:00

No pause marker; fixed study deadline2026-09-23T01:15:12-05:00 unchanged.
Completed layer-study results and old80policy evaluations are unchanged.
GPU BF16 native/zero-aux loss,RNG,80gradients and oneAdamW update were exact.
The first actual GPU run then failed at nondeterministic adaptive-pool backward.
A recorded continuation replaced only that operation with equivalent fixed-grid
means, passed CPU forward/gradient and GPU forward/deterministic backward checks,
and completed within the original10minute preflight window. Originalfailure
retained; the already passed zero-weight check was not replayed. Model-compute
medians25.42ms baseline,33.96ms auxiliary; this excludes full data IO.
Occupied and ECC-error devices were excluded before execution; no task terminated.

Separate prospective protocol MATCHED_PROTOCOL.md frozen before any retained fit:
L12 baseline continuation versus goalaux0.1;3matched seeds42/43/44;2000updateseach;
training-only105episodes; auxiliary normalization matches the frozen readout.
Its aligned BF16 gradient check passed before fitting. Currentcompleted6/6,
launcherexit=0; independentcheckpoint/sampling audit=True; singleidleL40S,4threads,30mincap. No performance benefit
or new confirmation score has been observed. Goal improvement and state retention
must both pass the frozen joint criteria on NEW scenes, not the old60episodes.

Next: Six final checkpoint/paired-sampler audits already passed: do not rerun them. Prepare/freeze/check fresh-collection code and data-identity gate for the already fixed800000/801000/802000ranges. Fit/readout hashes must be frozen before new confirmation scoring. No completed evaluation replay, no loss/seed sweep; the previous60scenes are exposed development. Original study deadline01:15:12CDT Sep23 unchanged.

Public Sites lookup still404; local report updated; originalpublicversion26 stale.
Existing single15mincron unchanged; no new timer or claim of timely idle wakeup.

---

# Selfcheck — GPU contract passed; matched intervention 1/6 fits — 2026-09-22T16:32:35.988128-05:00

No pause marker; fixed study deadline2026-09-23T01:15:12-05:00 unchanged.
Completed layer-study results and old80policy evaluations are unchanged.
GPU BF16 native/zero-aux loss,RNG,80gradients and oneAdamW update were exact.
The first actual GPU run then failed at nondeterministic adaptive-pool backward.
A recorded continuation replaced only that operation with equivalent fixed-grid
means, passed CPU forward/gradient and GPU forward/deterministic backward checks,
and completed within the original10minute preflight window. Originalfailure
retained; the already passed zero-weight check was not replayed. Model-compute
medians25.42ms baseline,33.96ms auxiliary; this excludes full data IO.
Occupied and ECC-error devices were excluded before execution; no task terminated.

Separate prospective protocol MATCHED_PROTOCOL.md frozen before any retained fit:
L12 baseline continuation versus goalaux0.1;3matched seeds42/43/44;2000updateseach;
training-only105episodes; auxiliary normalization matches the frozen readout.
Its aligned BF16 gradient check passed before fitting. Currentcompleted1/6,
launcherexit=None; singleidleL40S,4threads,30mincap. No performance benefit
or new confirmation score has been observed. Goal improvement and state retention
must both pass the frozen joint criteria on NEW scenes, not the old60episodes.

Next: Await the bounded six-fit launcher exit code, then verify all checkpoint hashes and paired sample sequences. Next uncertainty is whether goal supervision improves new-scene readout without state regression; only the fixed prospective cohort can answer it. Never restart completed fits or score the old60as fresh. Original deadline01:15:12CDT Sep23 unchanged.

Public Sites lookup still404; local report updated; originalpublicversion26 stale.
Existing single15mincron unchanged; no new timer or claim of timely idle wakeup.

---

# Selfcheck — auxiliary CPU contract passed — 2026-09-22T16:02:10.232384-05:00

Completed layer-study results are unchanged; all final scoring exit codes are0.
New bounded CPU check: attempt1 exited1 before any parameter update because the
G2 fixture's relative data path differed on G1; attempt2 fixed only this path,
kept raw image hash verification and passed in6.23s with2threads and noGPU.
For one training frame/seed42 checkpoint, lambda0 preserves native loss, RNG,
80parameter gradients and one AdamW update exactly. Auxiliary loss gives finite
nonzero encoder/head gradients; decoder gradients are absent. Changing later
features leaves the initial mean exact; gradients to future input are zero.
This is CPU FP32 engineering evidence, not improved representations or policy.
No new compressor checkpoint or held-out scoring; no completed experiment replay.
Original deadline01:15:12CDT Sep23 and all resource caps remain unchanged.

Next: Freeze a separate bounded GPU BF16 parity and real-shape throughput preflight, then check current idle resources. Only after those checks, freeze the full matched L12 reconstruction-only versus goal-auxiliary protocol, including joint goal-improvement/state-retention criteria, before any new compressor fitting or fresh-scene collection. The previous60confirmation episodes are exposed development for this follow-up; no winner/weight sweep or completed-evaluation replay. Original deadline2026-09-23T01:15:12-05:00 remains unchanged.

Sites get_site again404; local report updated, publicversion26 remains stale.
Single existing15mincron retained. The delayed13:30message was observed at15:51;
this verifies delayed delivery, not reliable15min autonomous reasoning cadence.
See studies/goalaux-20260922/attempt2/contract-result.json and PATH_AMENDMENT.md.

---

# Latest — audited representation study complete; next intervention unlaunched — 2026-09-22T15:42:18.120468-05:00

Original study window13:15:12CDT Sep22 to01:15:12Sep23, no automatic extension.
Fresh data COMPLETE:72consecutive-range attempts,60accepted,3unstable,9expert
infeasible. Three tasks20episodes each; no model-based selection or replacement.
Primary confirmation exit0,384prediction conditions independently recomputed,
720rawlabels/image hashes verified;60fresh vs150existing identity gates passed.
K4SVAE vsL12SVAE noise E:+45.675% ridge (95%+38.023,+53.588),+16.932%MLP.
Primary acceptance FALSE. Raw-source advantage reverses after48Dcompression on
this state-readout endpoint. No policy adaptation or newclosed-loop rollout.

Secondary initial-goal probe (fixed after development, before fresh scoring):
K4SVAE noise E:-36.265% ridge (95%-48.001,-22.567),-24.541%MLP.
Independent60raw-event/105train-sample/270prediction/6CI audit passed.
MLP handover_block regresses+10.912%; low-noise MLP interval crosses0.
Same60episodes as primary; not another independent replication or universalgain.
This is initial-image to firstclose-event XY, not physicalcontact/objectpose.
Wrist-trained baselines complete0: head-to-wrist failure substantially reduces
when fitting same-budget wrist readouts;90conditions retained, no reducer refit.
Supplemental phase/decomposition v1/v2 failed;v3completed0, original artifacts
preserved. Floating arithmetic/JSON fixes did not alter primary predictions.

Local report: reports/layers-20260922/index.html; portable NumPy audits pass.
Sites get_site still404 at15:08; originalpublicversion26 is stale, NOT updated.
No new Site or credential workaround. Pending user connector-access question.
Optional native cache provenance patcha030f01:4CPUtests/Ruffpass; no upstreamPR.

Next: Prepare a separate prospective matched L12 goal-auxiliary compressor study and lambda=0 parity/causal-gradient checks on training data only. Do not change completed protocols, promote K4 as a universal replacement, reuse these60episodes as fresh, or start a loss sweep. Freeze/check any new bounded experiment before launch within original deadline.
See PROPOSED_GOAL_AUX_FOLLOWUP.md. No goal-aux codec training launched yet.
All old80rollouts and stopped context pilot remain untouched. Existing single
15mincron remains; post-resume queued delivery is not independently demonstrated
as an idle-task reasoning wakeup. Some local status writes were delayed by
automatic approval timeouts; narrow sequential operations restored access.

---

# Active study — development audited, fresh confirmation pending — 2026-09-22T14:14:29.430766-05:00

Explicit resume remains active. Original study deadline01:15:12CDT September23.
All9 compression fits completed exit0; grouped ridge and fixed MLP completed
exit0, independently audited282+192prediction conditions. All15zero-noise
source/compressor paths exactly match. K4-SVAE primary worsens noise.10 E by
45.773% ridge and14.790%MLP on15reused development episodes. Primary gate FAILED;
no conditional policy adaptation and no promotion of secondary winners.
Raw768 K4 improves aggregate readout while compressed48D worsens, a descriptive
source/compression interaction requiring independent confirmation, not policy gain.

Fresh collector v2 running: {'adjust_bottle': 20, 'handover_block': 4, 'place_object_basket': 0}, 27attempts.
Keep absolute15:25:23CDT deadline and fixed attempt order; do not rerun oldseeds.
Next freeze reducer/readout hashes, verify inference parity and data identity,
then score the complete3x20freshmanifest once, no refitting. See
studies/layers-20260922/CONFIRMATION_PROTOCOL.md and progress.json.
Native profile trial2 ended124 after68microsteps/8optimizer updates, incomplete.
SingleGPU timing is insufficient for planned2GPU adaptation feasibility claims.

Existing single15minselfcheckcron restored, delivery queued but post-resume
reasoning wakeup not independently observed. Old80rollouts and stopped context
pilot unchanged; no new agents. Sites project404 and empty site listing block
publication; publicversion26 still says paused and is NOT current. Local results
are authoritative until access is restored; pending user connection question.

---

# Active study — explicit user resume 2026-09-22T13:47:26.457542-05:00

The user said resume, continue. The single15minselfcheck cron is restored;
13:30tick ran and queued a message, but post-resume reasoning delivery is not
yet independently observed. No duplicate timer created. The previous PAUSED
entries below are historical, superseded by this explicit resume.

Read studies/layers-20260922/launch.json, progress.json and PROTOCOL.md.
Window13:15:12CDT September22 to01:15:12September23, no automatic extension.
First-hour interface check passed:K1exact, no future leakage,150episode schema.
Remote compression/readout logs are at the Group2 results/layers-20260922 tree;
cm004 tmux wam-layers-compression, then wam-layers-readout. Wait for exit codes;
never restart a completed or failed launcher without preserving original state.
Localcm001 results/layers-20260922/fresh has two original accepted records and
one nativeUnStableError; v2 continues unattempted seeds, originalexit1 retained.
See FRESH_CLASSIFICATION_AMENDMENT.md; absolute collection deadline15:25:23CDT.
Native profile trial1failedbeforetraining at sidecar export; trial2on1GPU runs
with20mincap. Read NATIVE_PREFLIGHT_AMENDMENT.md. No closed-loop gain established.
No old80rollouts/context probes restarted. No new subagents. Public update pending.

---

# Current priority — selfchecks PAUSED; 12-hour plan only — 2026-09-22T01:40:39.818352-05:00

The user explicitly requested pausing selfchecks and planning the next12hours.
The tagged cron entry was commented at01:21CDT; paused.json guard was verified:
manual invocation exits0 without changing latest.json, and no queued tick was
pending at that check. Do NOT resume timers or act on stale selfcheck messages
without an explicit user instruction. No new experiment launched this turn.

Read plans/REPRESENTATION_12H_PLAN.md for the new research priority:
fixed DINO L12/L6/K4 sources × PCA48/S-VAE48;3tasks,9small fits,
new independent episodes, group-balanced readouts and spatial/proprio controls.
The only primary candidate is K4-S-VAE vs L12-S-VAE, not a posthoc winner.
Real OpenWAM adaptation and160new paired rollouts are conditional on measured
throughput, interface parity and predeclared development gates. New latent
coordinates require equal adaptation of both policy arms. T0 is a future actual
launch, not the planning timestamp. No old80rollouts or stopped context pilot
will be repeated. No full RAE/VAE policy comparison is claimed.

Resource checks were read-only. FQDN checks succeeded; short aliases initially
failed DNS and that failed snapshot is retained separately. No GPUs reserved.
Maximum future concurrency4L40S,48GPUhours and12hwall; fresh preflight required.
Detailed node/slot snapshots remain local. Public plan excludes infrastructure
identifiers after auto-review rejected the first publication attempt for that
reason. Revised public artifacts passed explicit identifier and link/hash checks.

Sanitized public plan version26 deployed successfully at06:41:14UTC:
https://embodied-research-notebook.exiamzifan.chatgpt.site/plan12h.html
Site source 64e0005b6af3ae9d3a12a2f17084521998a82286; deployment appgdep_6ab2230409088191a27a3f999811a901.
All existing page validators plus50 plan/index local links and2 plan artifact
hashes passed. Public artifacts contain no internal node/account/path/connection
identifiers. Historical version25 results remain accessible, unchanged.
The plan is complete. No new experiments are running from this turn.

---

# Current state — frozen control readout complete and published, 2026-09-22T01:18:17.661140-05:00

Publicversion25 succeeded06:16:37UTC:
https://embodied-research-notebook.exiamzifan.chatgpt.site/readout.html
Site source9d8c038f6763912f3b71eaf5ce6e8fa2345b03c1, deployment
appgdep_6ab21d3cf4948191bb6c7c1ab6b9d236. Experiment source6687e266;
analysis/report sourceaee36a1. All10 predeclared conditions retained, exit0.

New CPU experiment: unchanged existing adapter and clean-trained frozen linear
readout,35train episodes/420unique frames,5previously-used validation episodes/
60frames. Five training-episode CV folds choose visual+proprio alpha1 and
proprio-only0.01 before evaluation. Sources/inputs frozen01:07:24CDT after5CPU
gates; launched01:07:50, completed01:07:52, within300s cap;2threads,CUDA invisible.
No adapter/encoder/policy fit;2final ridge readouts plus fixed training-onlyCV.
Primary remains deployment seed42;43/44 are sensitivity checks, never selected.

Primary noisy: latent restorationMSE-34.72%, overall readoutNMSE-11.04%,
translation-11.84%, gripper+5.82%. Clean overall+2.64%, translation+2.64%,
gripper+2.51%. Noisy translation improves and gripper worsens in mean for all3
seeds; clean overall error increases for all3. Primary noisy-gripper episode
bootstrap interval crosses0. Same5episodes reused across seeds, not15 independent
samples. No claim of statistically proved broad degradation or policy benefit.
Proprio-only gripperNMSE0.03729 is below visual+proprio clean0.07744: fixed-readout
regularization/capacity can contribute; cannot infer irreversible S-VAE loss.
The consistent translation+gripper gain condition was NOT met. No new loss sweep.

Audit independently recomputed all10conditions/60predictions each, group metrics
and2000 paired episode bootstraps; training target scaling, selection-before-
scoring and11frozen source/input hashes verified. Original80rollouts unchanged.
Development reuse only:5validation episodes previously selected adapter; published
representation may have seen demos. No RAE/VAE ablation, rotation/contact targets,
controller-command or closed-loop claims. CPU adapter numeric parity with GPU not
established. Full source/coefficients/predictions/per-episode results published;
110local links/33asset hashes plus old reports checked. Static figure inspected.

Software gap closed: documented S-VAE CLI runs with normal package imports on an
existing trained checkpoint and3fixed feature clips in native shard format, CPU
batches2and1, both exit0 and metrics agree1e-6. Checkpoint/input hashes unchanged;
existing policy-env unchanged. Full upstream suite not run; fixture is software
validation only. Preflight480clean/900noisy sample-reference mapping,35train/5val
split, rawframe+4 label alignment checked; no demonstration test episode opened.
Original extraction lacks per-image hashes, so historic feature byteprovenance
not proven. Current sample/image/label hashes retained. Frame rows are not time.

Current contribution remains independent237-line held-out reconstruction tool;
new readout script/result is supplementary research evidence, not that patch's
functionality. PR draft updated for successful normal CLI. No PR/message sent.
Resource snapshot01:15:19 GPU2 6MiB/0%, unrelated tmux exp untouched. No GPU jobs,
new timers/agents, other task termination, context retry or completed study rerun.
Single15min monitor unchanged; old monitor integrity is not new-study validation.

---

# Current state — representation contribution review published, 2026-09-22T00:58:27.064034-05:00

Public version24 succeeded05:57:12UTC:
https://embodied-research-notebook.exiamzifan.chatgpt.site/review.html
Site source6ad25b332faca72b16e91740cd4bdbdd74080271; deployment
appgdep_6ab218b17858819193f9fe5241f541bc. Private source2ee2426.

Returned to representation objective. Prioritize held-out S-VAE evaluator
(eb31a97,4files,+237lines), with scene-seeded language (4cace2d,3files,+134)
and optional boundary trace (d1273ca,4files,+363) as independent contributions.
No patch/model/default/frozen runtime was changed. First combined-source review:
all3 apply to pinned7c5861e,17 targeted CPU tests pass in1.67s, Ruff passes.
CUDA hidden,2CPU threads; original summaries and patch files hashed unchanged.
Initial tool setup failed because venv lacks pip; existing isolated test tools
were used instead, failed log retained. Not a full upstream suite or live test.
S-VAE documented CLI bootstrap in full upstream environment remains unverified;
prior real-checkpoint smoke used isolated leaf loading. No upstream PR/message.

Consolidated all3 task outcomes and all80 prior rollouts without new evaluation.
S-VAE vs PCA reconstructionMSE improves45–50%, clean achieved-state probe change
small/mixed; pretrained advantage not isolated. Adapter restoration improvement
~35% has no demonstrated reliable closed-loop net gain. Old follow-up episode
reuse is developmental, not fresh confirmation. No RAE-vs-VAE or contact claim.
Review packet, PR draft, three independent patches,17-test logs and all limits
published;88 local links and16 artifact hashes verified plus prior report checks.

Context pilot remains stopped/incomplete: A0 two commands, B0 failed port
preflight before launch, B1/A1 not run; original exit1. No restart/extra trials.
Resource snapshot00:56:36CDT GPU2 6MiB/0%, only unrelated tmux exp; no GPU job
started, no tasks terminated, no agents. Single existing15min cron unchanged;
latest00:45 monitor integrity true for old study, not context completion.

Next: close documented CLI integration gap without training; separately audit
adapter-cache sample/split/time alignment for a prospective train-only frozen
control readout. See contribution/REVIEW_PACKET.md. Any later new experiment
requires its own fixed plan/budget; do not rerun80 or tune against old outcomes.

---

# Current state — incomplete context pilot stopped and published, 2026-09-22T00:42:44.645417-05:00

Publicversion23 deployed: https://embodied-research-notebook.exiamzifan.chatgpt.site/context-pilot.html
Site sourceefe4e41209a90d4363d4f42cfbfd318118e9293c, deployment
appgdep_6ab214f0899481918f7b34c5f2a51777 succeeded05:41:15UTC.

Pilot source7accd7b428fd1152f33d4db795103df77b61d765, frozen00:26:40CDT;
7 lifecycle/CLI CPU checks and14 prior context checks passed. Pilot launched
00:28:26.608 outer; source verification finished00:28:42.299. A0 resident_idle
completed2 fixed commands (zero prediction RPCs), server ready104.627seconds,
trial150.922seconds. B0 passed3 idle-GPU checks but its ordinary bind port18848
preflight failed Errno98 Address already in use. No B0 server/client launched;
B1 and A1 not run. Inner+outer exit1, stopped00:31:40.936, internal178.636s,
within600second cap. NO scientific A/B comparison is available. Do not call this
a negative online-activity result or infer no effect from an incomplete pilot.

CPU ephemeral loopback reproduction confirmed a potential preflight defect:
TIME_WAIT with no listener causes plain bindErrno98, while SO_REUSEADDR bind
succeeds. Original18848 had no entries at later inspection. Failure-time socket
state was NOT captured; exact original cause remains unproved. No frozen source
was changed or pilot restarted. See CONTEXT_PILOT_POSTMORTEM.md and diagnosticJSON.

Audit passed:55 runtime files/checkpoint,6 protocol/input files unchanged;
2 complete requests,6 decoded images,4 planner calls, saved20D/16D commands,
and one2-frame video verified. Every planned trial/pair retained, incomplete
pairs marked unavailable. All36 public artifact hashes and95 local links checked.
Current resource snapshot: 2, GPU-025b7fc9-8f25-5467-5c28-8e59884ab6d3, 0 %, 6 MiB, 0
Only unrelated tmux: exp: 3 windows (created Sat Sep 19 20:14:57 2026)
Own pilot groups cleaned, no other task stopped. No agents/new timers. Prior80
and completed repeatability/attribution studies untouched; no weights trained.

STOP this context pilot per frozen rule. Do not rerun missing trials or extend
simulator debugging merely because a selfcheck arrives. Next: return to existing
representation/robustness evidence and prepare the smallest reviewable community
contribution with explicit data-reuse, negative findings and generalization limits.
Port-preflight correction, if later needed, belongs in a separate CPU-tested
change and not a hidden amendment/restart of this frozen scientific run.

---

# Bounded context pilot launched — 2026-09-22T00:28:26.608189-05:00

New study results/context-pilot-20260922, tmux wam-context. Source7accd7b,
55 runtime files and checkpoint hash frozen00:26:40CDT. Independent600s outer
wall cap (source verification included), internal595s work cutoff+cleanup reserve.
A0,B0,B1,A1, firstscene400000, two fixed commands each, one L40S GPU2. Seven
lifecycle/runner CPU gates plus14 earlier context gates pass. No old study rerun.
Any failed trial stops pilot; no substitute or adaptive extension. Results pending.
Only own dedicated Popen groups are cleaned; unrelated tmux exp left running.

Next: inspect logs and exits, retain partial failures, compare initial/action0
states/planners and observation1. This tests online activity jointly, not isolated
GPU-load or representation benefit. Publish all planned rows even if not run.

---

# Current state — context recorder/CPU integration published, 2026-09-22T00:15:17.511577-05:00

Publicversion22 deployed successfully: https://embodied-research-notebook.exiamzifan.chatgpt.site/context.html#integration
Site sourceb7d89b524e1ce01015a3942b4a493abe85deb93e, deployment
appgdep_6ab20e8cc9408191ace4ea2d3ed20e37 succeeded05:13:55UTC.
Preparation sourcec5db731c9652519c25029a9e8c00439b65c5fbaf.

Added ContextTrace/bind_context_interface. Selected fixed20D commands/local
indices remain separate from genuine online responses. No synthetic server step.
Valid different online actions are retained, not executed/excluded. Wrong counters
and initial mismatches are saved then abort before motion. No completed/frozen
study changed. Timing labels now separate client RPC interval from hashing/copy
interval; neither represents pure GPU compute. Original exceptions preserved.

8 new CPU integration gates +6 transport regression gates pass (14 total).
Real pinned RoboTwin loop with FAKE environment: one setup/close, two resets,
two observations/actions, fixed seed/prompt and same language RNG consumption.
Real WSPolicyClient connects only to a loopback STUB server; counts checked and
actual socket timeout sends one obs, never retries/continues. Fake planner/state
checks are not real simulation or model validation. Initial import-path failure
retained; explicit runtime PYTHONPATH fixed bootstrap. All logs/source are public.
Runtime artifacts: results/context-integration-20260922. ready_for_gpu_launch=false.

Prior closedloop/attribution/saved-response replay exit0 and complete logs verified.
No model/GPU experiment started this turn. GPU/tmux point snapshot saved:
2, GPU-025b7fc9-8f25-5467-5c28-8e59884ab6d3, 0 %, 6 MiB, 0
exp: 3 windows (created Sat Sep 19 20:14:57 2026)
No other task terminated; own temporary loopback test sockets/threads closed.
One existing15min monitor unchanged, no new agent/timer. Old23:45:59 allocation
expired; this CPU preparation does not extend it.

Next: complete actual client lifecycle, fresh model+sim process runner and hard
wall-cap/cleanup gates. Then freeze full runtime configs/sources and consider only
the independently bounded predeclared4-trial ABBA pilot on freshly idle resources.
Do not label source freeze or launch ready before those checks. Uncertainty is
still whether online activity changes first-observation repeatability with fixed
commands. No new root-cause/RAE/control-benefit conclusion, no repeated80evals.

---

# Current state — context design CPU-prepared, 2026-09-22T00:00:53.588391-05:00

Publicversion21 successfully deployed: https://embodied-research-notebook.exiamzifan.chatgpt.site/context.html
Site source982bf19449f4ba85410b0106518ecad525e4215f, deployment
appgdep_6ab20b47cc8c8191b40ffaef42a1951d succeeded04:59:57UTC.

Delayed23:45 selfcheck reconciled version20 and completed logs: full closedloop,
attribution and saved-response CPU replay all exit0; no experiments rerun.
Direct resource check this turn: GPU2 6MiB/0%/0ECC; unrelated tmux exp retained.
No new GPU tasks, timers, agents or termination. Old23:45:59 ceiling is expired.

New preparation source69b219be4e6702ec826757650a8d01852e066088:
CONTEXT_CONTROL_DESIGN.md and private context_trial_transport/check scripts.
Both proposed arms execute same two saved firstscene400000 commands and keep
original model resident. resident_idle sends no predict; shadow_online performs
one no-retry prediction per native request, records its full response, then
returns the saved command. Distinct local index, no fake server prediction step.
Contrast includes inference/RPC/waiting jointly, not isolated GPU-load effect.
Four proposed trials ABBA, two commands each; no actual trial launched.

Six NEW CPU tests pass (actual native eval/converter, fake network client):
identical16D saved targets, zero vs once-only predictions, unexpected response
retained, original input/RNG preserved, bound/reset enforcement, timeout fail-stop.
These differ from the earlier6 generic tracing tests. No model/simulator/CUDA or
live WebSocket validation. CPU results/context-preparation-20260921; ready_for_gpu_launch=false.

Pending: integration with a new trace separating selected vs actual responses,
CPU tests of native loop/reset/order and runner/timeout/abort, full source/config
freeze, and a separately bounded future allocation. No launcher exists yet.
Do not interpret this preparation as fresh experimental or RAE-benefit evidence.

---

# Current state — local contribution validated and published, 2026-09-21T23:50:44.007467-05:00

Public version20 successfully deployed: https://embodied-research-notebook.exiamzifan.chatgpt.site/contribution.html
Site source47aee3c79099c6521d6271a70530ad9d31a3aa59,
deploymentappgdep_6ab208db50ec81919139e9bc59dc87f6 succeeded04:49:38UTC.

Local contribution/policy-trace commitd1273ca685d7f65b3754843b5fe42a210ecdcfa0
at /home/zifanz4/openwam-trace-contribution, upstream base7c5861e. Four files,
363 added lines (utility/tests/docs/link); opt-in, no default runtime changes.
6 unit tests pass, including actual RoboTwin eval 20D-to-16D target parity;
Ruff/diff checks pass, patch applies to pinned base. No upstream PR submitted.

CPU saved-response replay complete, exit0: all18 existing traces/1881 requests,
1899 source hashes checked before/after unchanged. All6 full-rollout pairs first
differ request/image/state at index1 and action at index32; all3 short fixed-command
pairs equal. No new policy inference or simulation. Not independent experiments,
no root-cause/fix claim, no RAE superiority or new control-benefit claim.
Assets: results/policy-trace-replay-20260921; see TRACE_CONTRIBUTION.md.

GPU context-control probe DEFERRED. The old23:45:59 allocation ceiling has passed;
no automatic extension or new GPU launch. Latest direct GPU2 snapshot6MiB/0%/0ECC;
only unrelated tmux exp remains. Closedloop, attribution and CPU replay exit0.
The23:45 cron snapshot reported transient GPU2 use5794MiB (not attributed here),
so do not infer continuous idleness or availability from the later point sample.
One existing15min cron preserved; no new agents/timers or task termination.

Next uncertainty: why full-rollout observation divergence disappears in the short
fixed-command probe. Prepare only a separately frozen, context-matched first-scene
protocol and CPU gates, explicitly accounting for online inference load/timing,
model residence, episode ordering and added getters. Do not launch the old draft
as if these were already controlled. Quantify any residual confounds before a
new bounded GPU allocation. No repeated80evals or new adapter selection.

---

# Current state — attribution complete and published, 2026-09-21 23:32 CDT

Public version19 successfully deployed:
https://embodied-research-notebook.exiamzifan.chatgpt.site/attribution.html
Sourceb356f64d394caec597b3c1b9342c58b1b4917778, deployment04:32:33UTC.
All6 two-command traces/12 commands/24 planning calls retained. Repeated fixed
commands produced identical planner paths, measured pre/post-state and requests.
The original full-rollout divergence did NOT reproduce; cause remains unresolved.
All commands physically changed qpos; this was not a zero-action capture.
42 source hashes,36 images,12 conversions and6 two-frame videos audited.
GPU job exited23:24:13CDT after134.895s, within600s bound and old23:45:59 ceiling.

No publication or experiment remains running. Current context-control draft is
NOT launched. CPU preparation may continue; no GPU launch if fewer than10min
remain before23:45:59 after all gates. Do not extend automatically. Otherwise
prepare a minimal optional tracing contribution from existing evidence. No new
model selection, adapter fitting, duplicated evaluations, agents or timers.
The23:30 cron snapshot correctly includes attribution exit0 and deduplicated wake.

---

# Two-command probe COMPLETE — 2026-09-21 23:31 CDT

All6 short traces/12 fixed commands completed, exit0, launcher134.895seconds.
Both repeats are bit-identical in every recorded request/image/EEF, measured
qpos/qvel/root/link/object pose, drive target and both planners' inputs/outputs.
All12 commands change measured qpos; all24 planner calls succeeded. This is a
NEGATIVE attribution result: it did not reproduce the original full closed-loop
observation divergence. It does not prove a fix or exclude physical/planner noise.

Audit passes:42 source files,12 native conversions,36 PNG images,24 planner calls,
6 two-frame videos. Initial requests all match; both probe repeats match old
stageC round0 at step1 only for firstscene400000, not later scenes. Execution
context changed (no online server, shortened predecessors, passive reads added).
No frozen study was altered or repeated. GPU2 idle after completion; unrelated
tmux exp retained. Public attribution.html built; native publication in progress.

Next: CPU-prepare a small context-matched live-versus-recorded transport control,
with model resident and firstscene400000 only; see CONTEXT_PROBE_DRAFT.md. It is
NOT launched and needs a new freeze/check. Existing23:45:59 deadline remains;
if fewer than10 minutes remain after gates, defer GPU work and prepare the small
optional trace/check contribution on CPU. No further adapter training or tuning.

---

# Two-command attribution probe running — 2026-09-21 23:22 CDT

Separate frozen protocol ATTRIBUTION_PROTOCOL.md, source0af6132. Six two-command
open-loop traces, no new policy inference and no full-rollout repeats. Native
per-task step override2 exists only in probe process memory; original400-step
files untouched. Source commands are stageC round0-clean first two per scene.
Nine CPU gates pass. cm001GPU2, original flock, tmux wam-attribution. New hard
600-second cap ends23:31:58CDT, also within previous23:45:59 ceiling; no extension.
Runtime results/attribution-20260921. Inspect launcher/round logs and exit-code.txt.

Next uncertainty: compare measured initial qpos/qvel, exact planner inputs/paths,
and measured post-state. Prior joint_action.vector is drive targets; this probe
adds actual physical-state getters. No extra planning, get_obs or physics calls.
All three scenes/two repeats will be retained, regardless of equality or failure.
Public version18 remains /trajectory.html; probe publication pending results.

---

# Current state — complete and published, 2026-09-21 23:12 CDT

StageC all12 trajectories complete; audits pass and public Siteversion18 deployed.
https://embodied-research-notebook.exiamzifan.chatgpt.site/trajectory.html
Source28bd99c70445d6748d20b1e52e4ec8c497868132; deploy succeeded04:12:00UTC.
All12 videos/lightweight traces and first-divergence requests are public. Homepage
now links to the latest report and replaces the stale unevaluated-policy label.

Conclusion:6/6 pairs differ first in images/EEF/request at index1, then actions
at index32. Both rounds clean3/3/noise2/3; all paired outcomes agree. Planning vs
physical execution vs rendering is not isolated. Native joint vector is drive
targets, not measured qpos. All1869 requests and5607 camera images were audited.
Jobs exited, GPU2 idle. Completed80/54/12 experiments must not be repeated.
Next study remains a draft only; no further GPU job or adapter training started.
Existing15-minute cron remains single, with one deduplicated pending self-check.

---

# StageC complete — 2026-09-21 23:10 CDT

All12 original-policy trajectory repeats finished, exit0. All6 pairs first differ
in raw cameras/EEF/request at index1, and first differ in server20D and env16D
actions at index32 (the next full32-action generation boundary). Their first32
actions match despite differing observations. All12 first actions match the
prior saved-request diagnostic. Both rounds: clean3/3, noise2/3; every paired
outcome agrees. Clean400002 takes105 vs106 actions; all other lengths agree.
No complete recorded trajectory pair is bit-identical. This supports investigating
execution/observation before blaming initial-input model randomness; it does NOT
isolate physics/rendering/IK, prove global determinism, or prove adapter benefit.

Post-run audit:1869 requests,5607 decoded camera images, all1869 native20D-to16D
conversions and proprio payloads exact;33 source hashes and12 videos checked.
First EEF xyz differences range0.089–0.656 micrometres. Treat numerical equality
and task outcomes separately. Original80 rollouts and54 chunks not repeated.

Measurement clarification: native joint_action.vector (saved as joint_state)
contains joint DRIVE TARGETS, not measured qpos. EEF gripper fields are commanded
values, not contact observations. Pinned source verified; no runtime/protocol
change. See closedloop/measurement-semantics.md.

GPU job/tmux exited; GPU2 returned6MiB/0%/0ECC. StageC finished around23:09, within
original23:45:59 deadline; no extra allocation or other task termination.
Local trajectory.html report built and checked, including all12 videos and light
traces; native publication in progress. Existing public version17 remains valid.

Next: a NOT-LAUNCHED draft for two-command open-loop probes separating planner
paths from actual qpos/EEF/rendering. Must freeze/check a distinct bounded scope;
no automatic extension of this study or duplicate launcher. Continue static/CPU
work on the probe if needed. See NEXT_DIAGNOSTIC_DRAFT.md and REPRESENTATION_DECISION.md.
Do not train another adapter yet or select weights from these reused outcomes.

---

# StageC running — 2026-09-21 22:51 CDT

Twelve original-policy trajectory repeats launched under tmux wam-trajectory,
cm001 GPU2, original flock; strict three-sample idle preflight passed. Source freeze
commit1dcbd407; protocol TRAJECTORY_PROTOCOL.md. Nine fresh CPU checks pass, including
actual native eval argument/return preservation and actual RoboTwin400-action cap.
Runtime results/repeatability-20260921/closedloop. Launcher starts22:48:40CDT and
has3438 seconds remaining in the original budget; hard deadline23:45:59CDT.
No80-rollout or54-chunk reruns, new training, candidates or subagents.

Next uncertainty: first raw observation/request/action divergence between two full
original-policy repeats. All12 outcomes and traces must be retained and audited;
success alone does not establish repeatability. Public report stillversion17,
/repeat.html covers completed54 fixed-input chunks; stageC publication pending.
Cron22:15 message actually delivered and processed at22:35; later ticks dedupe
one pending self-check. Monitor now includes stageC progress without a new timer.

---

# Current follow-up complete — fixed-input action repeatability

2026-09-21 22:30 CDT: all54 preregistered generated chunks are complete, exit0.
Source/input checks and the independent54-file NPZ/latent/first-action audit pass.
Six saved inputs (3 reused scenes × clean/noise) produce byte-identical full chunks
and first actions within a process and across the fresh identity restart. Each
intervention's repeated outputs are also byte-identical. All three interventions
change actions on all6 inputs. Repeating LayerNorm alone changes a clean first-action
position coordinate by up to4.203mm. First-action gripper changes are zero on these
initial states; no contact-phase inference follows from that.

This establishes repeatability only for these initial requests, not full simulation
trajectories. It does not prove which mechanism caused the previous success/failure
changes. Next: stageC, at most12 original-policy full trajectories with passive
input/action traces and a new protocol/source freeze. No stageC job has started.

Two engineering attempts are retained: one generated chunk before a request
mutation guard failed; zero chunks during an aborted stale-test-gate restart.
Fresh per-request JSON decoding matches native WebSocket behavior.8 corrected CPU
checks pass; amendments are recorded. Main54 comparison sequence and frozen arms
remain unchanged; diagnostic inputs were reused, not recaptured or selected.

15-minute cron is active. The22:15 tick queued the next self-check; this substantive
turn was triggered by the initial queued monitor check. Queue delivery and actual
agent processing have been demonstrated; no extra timers or subagents were created.
Latest scheduler state: /home/zifanz4/.local/state/openwam-selfcheck/.

Public report version17 is live: https://embodied-research-notebook.exiamzifan.chatgpt.site/repeat.html
Published2026-09-22T03:34:31Z, source528bea2b1c682ae53665c9ea045bef7f59c60936.
Preserve /norm.html results and do not restart completed launchers.
Source protocol REPEATABILITY_PROTOCOL.md. All diagnostic GPU jobs exited; GPU2
returned to6MiB/0% utilization/0ECC. Cron22:15 and22:30 ticks verified; pending
checks are deduplicated while this task is active.

---

# Current status — 2026-09-21 22:05 CDT

The normalization study is COMPLETE; this section supersedes all historical
running-status entries below. All 80 primary plus 14 reference/diagnostic rollouts
passed the frozen source/pairing/counter/video audit. All six exit markers are 0.
Resume source hashes and resource amendment hash were independently rechecked.
Our normalization GPU jobs have exited; do not rerun their launchers.

| Arm | Clean | Head noise sigma=.20 |
|---|---:|---:|
| identity | 10/10 | 7/10 |
| renorm | 10/10 | 9/10 |
| adapter | 9/10 | 9/10 |
| adapter_renorm | 10/10 | 8/10 |

No established net improvement: postnorm vs adapter rescues one clean failure but
adds one noisy failure (both prespecified paired p=1). Renorm and adapter vs
identity have noisy paired p=.5. Ten scenes/condition in one task are exploratory.
All four historical replays fail400; earlier original success did not reproduce.
No direct RAE-vs-VAE or contact-robustness conclusion.

Final public report: https://embodied-research-notebook.exiamzifan.chatgpt.site/norm.html
Site version16, source5c00df56f1fe1c0325dc802b2a96d95a2f1375fb, deployment succeeded
2026-09-22T03:04:39Z. All94 videos, protocol, raw summaries and reproduction scripts
are published. Local link and archive checks passed. Sites plugin helper vanished
from cache mid-turn; a credential-in-memory static workflow performed source
verification, push and archive preparation, then native save/deploy succeeded.

Next: quantify action repeatability and first divergence before more adapter
training. See NEXT_STEPS.md for fixed diagnostic scope and resource budget.

15-minute monitor configured in the existing user crontab on cm001 (survives SSH
logout); the provisional user systemd timer is disabled to avoid duplicate ticks; local state is in
/home/zifanz4/.local/state/openwam-selfcheck/. First audit ran22:03:52CDT and queued
self-check01a0c711-e024-7250-88d0-12291b16d56f to this existing task. The earlier delivery probe was subsequently received and processed in this task.
Queue delivery is now verified; an actual cron-triggered reasoning cycle is still
pending. See delivery-probe-ack.json; do not equate queued with executed.

---

# OpenWAM experiment status — 2026-09-21

The previous reconstruction-weighting study is completed. The explicitly requested control-supervision study is also completed; see below. The group-supervision follow-up and a published-policy closed-loop smoke test have also completed. All GPU jobs started in this turn have stopped.

## Completed follow-up: grouped targets + independent task
- Preregistered GROUP_PROTOCOL.md; 28/28 training runs completed, exit0, 27.28 minutes total /25.23 training minutes, peakTorch1.086GB. Driver source hashes match launch protocol.
- Actual execution cm001 GPU0, root /data02/zifanz4/openwam-experiments. One full2000-step numerical reuse audit matched the earlier control checkpoint exactly (maxdiff0). Then18 old-task validation runs and9 fresh-task runs.
- Group2 cm002 GPU5 became occupied before preflight twice; both attempts stopped before training. Inputs/environment copied to Group1 /data02 because /home was nearly full. No existing environment was mutated.
- Validation selected50:50 translation/gripper loss groups; passed all fixed old-task gates.75% gripper failed translation gate. No old-task test re-evaluation.
- Fresh task pick_dual_bottles: selected group vs original auxiliary gives gripper RMSE-4.37% (paired normalizedMSE CI negative) and translation+0.99% (CI crosses0). Versus original loss: translation-12.65%; gripper-8.16%, but gripper CI crosses0.
- Important remaining trade-off:19/120 gripper-transition clips; gripper RMSE .14249 original loss, .15190 original auxiliary, .14807 balanced. Balanced remains+3.91% vs original loss on this subset. No claim that all control information improves; no further weight search.
- Selection hash verified at test unlock, exactly9 fresh result rows and10 test episodes per seed. Fixed-seed paired episode bootstrap only, not all uncertainty.
- Published policy full checkpoint23.64GB and all structural artifacts downloaded with SHA256 checks. Separate policy-env uses original copied torch2.7/cu126 packages plus inference dependencies; original experiment env unchanged.
- Two real training observations passed published-policy inference: finite20D actions,16D execution conversion, peakTorch23.916GB; compileoff,10denoisingsteps,DiTcacheon.
- Separate benchmark-env torch2.4.1/cu124, SAPIEN3.0.0b1, cuRobo0.7.8, mplib.2.1, warp1.13.0. Renderer and planning imports verified on selected device; robot/object assets verified; no point-cloud pipeline.
- Published policy closed-loop smoke completed5/5 successes, seeds100000–100004, clean pick_dual_bottles,400-step limit, unseen instruction templates, no plannerfallback, zero setup exceptions, clientexit0. Five videos preserved. This tests the original policy and infrastructure, NOT the newly trained compressors; compileoff and5episodes differ from official benchmark protocol.
- Confirmed reporting bug: OpenWAM cap wrapper uses5 actual episodes but pinned RoboTwin main writes successes/100. Actual5/5 run wrote.05. CPU reproduction and minimal RoboTwin patch prepared;5/default100/emptyoverride checks pass, gitapply--check passes; patch not applied to active evaluation, no PR sent.
- Public report: https://embodied-research-notebook.exiamzifan.chatgpt.site/groups.html . Full small artifacts and videos: public-results/groups-20260921.
- Post-run GPUs0/2 were idle (5/6MiB,0% utilization,0ECC). The temporary localhost policy server was shut down by its owning launcher.

## Current experiment: control-supervised compression
- Run: results/control-20260921; protocol CONTROL_PROTOCOL.md; driver scripts/control_study.py.
- Completed on cm007 GPU 5 (GPU-a778817b-6e28-cbcd-289c-fc1bf8d9f1dd), tmux openwam-control-20260921. Group2 root /vol13/zifanz4/openwam-experiments.
- Log: logs/control-20260921.log. Check results/control-20260921/exit-code.txt and complete.json before declaring completion.
- Fixed 27 runs: three tasks, three seeds, auxiliary weights 0/0.1/1, 2,000 steps. All nine baselines rerun with deterministic settings.
- All pretraining checks passed after isolating CUDA nondeterminism and replacing adaptive pooling backward with equivalent deterministic cell means. Failed preflight logs retained; no training ran during failed checks.
- Global coefficient selected using validation only before new test evaluation; old held-out split remains a development benchmark, not fresh confirmation.
- Completed results are published at https://embodied-research-notebook.exiamzifan.chatgpt.site/control.html .
- Completed all 27 runs, main exit code 0; paired-camera/matched-batch diagnostic exit code 0. All 360 re-encoded head frames and all tested current vectors match exactly; original T=1 vs T=3 reducer difference also 0 in this run.
- Validation selected lambda=1 before test evaluation. Mean per-task relative MSE changes: clean -34.97%, noise -26.25%, brightness -37.38%, paired front camera -37.00%. Basket noise paired CI crosses zero.
- Trade-offs: translation RMSE improves 16-22%; gripper RMSE worsens 1%, 26%, 9%; feature reconstruction worsens 10-12%. Post-hoc separate-group ridge fits still show gripper RMSE +9.54%, +13.70%, +13.37%; no new representation training or coefficient selection.
- Candidate front/head error ratios remain about 4.9-8.5x. No contact or closed-loop performance claim; per-task compressors and reused development split only.
- Main run 1375.26 s, training total 1245.69 s, added view/numeric diagnostic 86.08 s; peak PyTorch 1.0858 GB. Post-run GPU 1 MiB, 0% utilization, 0 uncorrected ECC.
- Final local/remote public-results/control-20260921 includes summary.json, probe-results.svg/png and group-probe-diagnostic.json. Report renderer: build_control_report.py. Core run source is preserved as launched; source hash recorded in protocol.json.

## Completed experiment
- Fixed source: OpenWAM 7c5861e45cfe1339a0323f0e0b03a3316c37971c (verified current main).
- Tasks: adjust_bottle, handover_block, place_object_basket; each 50 simulated demonstration episodes, 35/5/10 episode split, 600 short clips.
- Frozen DINOv3, train-only PCA-48, fresh S-VAE-48 with condition weights 1 and 4; 3 paired seeds, fixed final 2000-step checkpoints: 18 training runs.
- All training completed successfully. Matched-batch corruption reevaluation also completed successfully. All 360 held-out clean re-encodings exactly matched the fp16 feature cache.
- Weight 4 vs original: condition MSE +0.4% to +1.8%; pooled-target MSE +6.5% to +11.1%; clean probe mean +1.1% to +4.6%. Some confidence intervals cross zero. Noise probes improved on two tasks and worsened on one, so there is no universally beneficial change.
- Original S-VAE vs PCA: reconstruction MSE about 45-50% lower, with much smaller/mixed clean probe differences. This is offline evidence, not closed-loop policy performance.
- Resources: one L40S; 662.54 s total reducer training; 895.56 s initial run including extraction/evaluation; 90.06 s matched-batch reevaluation; peak PyTorch allocated 1.0424 GB. Downloads/installation are outside those times.
- Execution: cm005, GPU 0 (GPU-d5dc6850-099b-e2da-8b07-a438cb854fdd), shared data at /vol13/zifanz4/openwam-experiments. Post-run GPU: 1 MiB, 0% utilization, 0 uncorrected volatile ECC errors.
- First cm006 GPU4 launch was safely aborted by the preflight after another job claimed the device. No other jobs were interrupted.

## Prepared contribution
- Local worktree: /home/zifanz4/openwam-svae-contribution; branch contribution/svae-heldout-diagnostics; commit eb31a97.
- 4 files, 237 insertions: standalone held-out evaluator (110 lines), tests (108), guide (17), README link (2).
- Does not change training, model code, inference or default configurations. Reports conditioning/pooled-target MSE and target-energy-relative errors.
- Five focused CPU tests and Ruff pass. Actual trained checkpoint + native held-out shards smoke evaluation passes (isolated leaf-module bootstrap in the minimal experiment environment).
- Patch applies cleanly to pinned upstream. Full upstream test suite not run; no upstream PR or maintainer message sent.

## Artifacts
- Local public-results/night-20260921: summary.json, split-manifests.json, scientific figures.
- Remote results/night-20260921: all features, final checkpoints, training curves, initial and corrected evaluations.
- Final analyses use evaluation-matched-batch exclusively; original single-frame corruption results remain for audit only.
- Public report: https://embodied-research-notebook.exiamzifan.chatgpt.site/night.html

## Limits / next scope
- Sampling correction: official num_frames=33 means RAW steps. With video_stride=4, both our study and the released configuration use 9 sampled video frames. Our actual sampling limitation is 12 complete windows per episode rather than exhaustive windows including masked tails. Numerical results unchanged.
- DINO/S-VAE Study branch, not OpenWAM-alpha's Wan-VAE representation.
- Fixed short-budget training, 3 tasks and 10 held-out episodes per task; no convergence claim or closed-loop success claim.
- Recommend the evaluation-only patch first. Future research can validate a second encoder, and then closed-loop policy relevance. Do not launch more experiments automatically merely to find a positive result.

## RAE follow-up (2026-09-21)
- Official RAE/RAEv2 and OpenWAM representation comparison reviewed. Proposal: multi-layer DINO versus coordinate-preserving observation denoising; see proposals/RAE_PLAN.md.
- Data schema audit found paired head/front RGB but no contact-force/event labels.
- Untrained 9,360-parameter adapter prototype: 3 CPU tests passed. No new GPU training or policy evaluation launched in this follow-up.
- Public follow-up: https://embodied-research-notebook.exiamzifan.chatgpt.site/rae.html

## 2026-09-21 13:59 CDT — paired policy robustness in progress

- Preregistered ROBUSTNESS_PROTOCOL.md before new experiments (commit 83dd1f0).
- Six restoration fits complete: original published latent coordinates, identity
  vs linear vs MLP; MLP chosen on validation, 35.0% lower restoration MSE,
  clean distortion 4.6% of baseline noise MSE. Deployment seed 42 fixed.
- Full-policy vs standalone encoder check: max absolute difference 0.
- Completed baseline: clean 5/5, sigma=.10 head-noise 5/5, front-camera transfer
  0/5; actual seeds 200001–200005. Paired initial image/state hashes and prompts
  match. Front view has substantial projection/crop differences.
- Twenty fresh confirmation rollouts queued/running via tmux robust-confirmation,
  actual starting seed 300000; clean/noise x original/selected adapter. GPU2,
  localhost 18848, automatic own-server cleanup. Do not duplicate these jobs.
- GPU0 parallel attempt failed preflight because other jobs claimed the GPU;
  zero confirmation episodes executed there. Reverted to original serial runner.
- An initial unpaired clean-only engineering attempt is preserved separately.
  Python language RNG was unseeded upstream; all paired runs scope it to the
  scene seed and restore outer RNG. No model/strength selection used rollouts.
- OpenWAM contribution worktree: /home/zifanz4/openwam-robustness-contribution,
  branch contribution/robotwin-paired-language, commit 4cace2d, default-off option
  ROBOTWIN_SCENE_SEEDED_LANGUAGE=1. Six tests + Ruff pass; actual upstream language
  generator matches the independently tested RoboTwin patch. PR draft only.
- Public Site version 11 is a clearly marked progress snapshot, not final results.
  https://embodied-research-notebook.exiamzifan.chatgpt.site/robustness.html
- Completion: runtime scripts/summarize_robustness.py verifies all 35 paired
  rollouts and hashes. Then build_robustness_report.py packages all 35 videos,
  selected small weights, full metrics and patches; publish to the same Site.

## 2026-09-21 14:17 CDT — confirmation pairing correction

The first confirmation attempt accepted 300002 in clean but skipped it in noise
before corruption. Stopped before any adapted-policy rollout and preserved 5
clean + 2 completed noise episodes plus partial logs under
results/robustness-20260921/invalid-refiltering-attempt. This is not policy failure.

Final confirmation now uses fixed-scenes.json: the first clean reference's five
expert-feasible seeds and exact generated prompts, independent of its success
outcomes. All four arms rerun. replay_confirmation_client.py keeps the upstream
action loop, disables only repeated feasibility filtering, replays the exact
scene/prompt list, and asserts original image/state hashes before first action.
The audited CPU test executes the real transformed loop and verifies fixed seed
order, prompts and original NumPy choice consumption. First actual replay has
passed initial hashes and completed successfully. See fixed-scene-amendment.md.

Active tmux: robust-fixed-confirm. Launcher run-fixed-confirmation.sh; GPU2,
localhost 18848. Old launchers are inactive and should not be restarted. Final
summary checks fixed manifest SHA, source hashes, all initial state/prompt/image
pairs, counters and video count. This protocol correction must remain visible in
the final website and summary; do not describe it as unchanged preregistration.

## 2026-09-21 — final paired robustness study complete

Supersedes the running-status notes above. All 35 final rollouts completed and
all seven conditions passed scene/prompt/initial-state pairing, counters, source
hashes and video-count checks. All our GPU jobs have exited.

- Baseline: clean 5/5, head RGB noise sigma=.10 5/5, front-camera replacement 0/5.
- Fresh fixed-scene confirmation: original clean/noise 5/5 and 5/5; selected
  adapter clean/noise 4/5 and 5/5. Seed 300003 is an added clean failure: original
  94 actions and success, adapter 400 actions and failure. Each arm has only five
  scenes; no statistically established general performance decline or gain.
- Validation restoration MSE decreases 35.0% averaged over three training seeds,
  34.7% for the deployed seed42 model. This is not a policy improvement claim.
- Five completed language-unpaired and seven completed refiltering-attempt
  rollouts remain separately archived, never included in the final 35. The
  fixed scene protocol amendment was made before any adapted-policy rollout.
- Final report preserves all 35 videos, exact initial policy input examples,
  the selected 9,360-parameter weights, hashes, scripts and protocol amendments.
- OpenWAM default-off language RNG patch: six CPU tests, Ruff and actual upstream
  language-generator integration passed. A local PR draft exists; no upstream
  submission or maintainer message was sent.
- Report: https://embodied-research-notebook.exiamzifan.chatgpt.site/robustness.html

Final public Site version 12 deployed successfully at 2026-09-21 19:49:26 UTC.
Site source commit: 3b63cce0d01245f0c59e4be037581fea3ffd33cc.
Deployment: appgdep_6ab18a3d9aa08191990d2c10a57cd80b.
All 35 video checksums match runtime originals; 117 local links checked.

## 2026-09-21 15:15 CDT — normalization follow-up running

Preregistered NORM_PROTOCOL.md, commit ec378de; implementation/source freeze
8a7c27e. Four frozen arms: identity, renorm, adapter, adapter_renorm. Ten fresh
scenes from seed400000, clean and sigma=.20 head noise: 80 primary rollouts.
Also ten expert-feasibility references and four selected historical diagnostics
(seed300003), excluded from primary success rates. No new fitting/selection.

Offline five-validation-episode cache complete: adapter MSE .108439, postnorm
.111726; clean distortion .007568 vs .007191. Postnorm RMS returns to ~1.
This is diagnostic, not control benefit. CPU intervention and replay checks pass.

Runtime root: /data02/zifanz4/openwam-experiments/results/norm-20260921.
GPU2 only. tmux wam-norm owns sequential rollout pipeline, wam-norm-audit waits
then audits all 94 rollouts and builds an isolated final static report.
Do not start duplicate runs or alter frozen sources. Read study-exit-code.txt,
postprocess-status.json, launcher.log and per-condition logs for progress.
On failure, preserve every artifact and disclose any correction before resume.

Public Site version13 is a progress snapshot:
https://embodied-research-notebook.exiamzifan.chatgpt.site/norm.html
Source9a22f9f6a51c469962749e6522afd9633fa32f2f; deployment succeeded
2026-09-21 20:11:44 UTC (appgdep_6ab18f7a2ed88191b4586fe2787d8197).
When complete, inspect the audit and scientific results, then publish the final
page via the native Sites workflow. The background auditor does not publish.

Historical diagnostic update 15:20 CDT: the original policy now fails seed300003
after400 actions despite identical initial head/state/prompt and zero latent
change. This case was replayed individually (previously third in five), after
ten reference episodes in a new server wrapper. Cause not isolated; do not
label this proven simulator nondeterminism or stable adapter-caused regression.
Both report pages explicitly disclose the follow-up; no primary settings changed.
The first fresh primary replay has passed initial hashes and is running.

Public progress update version14 deployed successfully at2026-09-21 20:23:25 UTC.
Source35a3a73cfc6c9e58cc8351a40c7d4ffc0d1aef50; deployment
appgdep_6ab192375828819182d1275f23ece015. Includes historical baseline failure
video and caveat on both current/prior report pages; snapshot has3/80 primary
episodes, current live job had5/80 primary (identity clean5/5) at last check.
All10 frozen runtime source hashes still match. Both tmux jobs healthy.
Final publication remains pending the complete paired experiment and audit.

## 2026-09-21 17:33 CDT — 60/80 audited, last arm resumed

At17:24 inspection found the original job had stopped at16:27 before loading
adapter_renorm: strict idle preflight saw233MiB graphics usage from another
same-user G1 task, with no GPU2 compute job. Earlier60 primary+3 historical+10
reference episodes were complete and are all audited in partial-audit.json.
Original interruption/failed-auditor logs remain in interruption-20260921T1627.

Completed clean/noise: identity10/10,7/10; renorm10/10,9/10; adapter9/10,9/10.
Both noise comparisons vs identity have2 wins/0 losses, paired p=.5. No stable
benefit claim. Historical scene300003 failed for all three completed arms.

Resume only adapter_renorm on sameGPU2, original model/protocol/source unchanged.
New helper narrowly permits a verified small same-user graphics-only context;
actual17:27:59 preflight passed fully idle (6MiB,0%,no processes), because the
other process ended naturally. No other work stopped. Source freeze, amendment
and actual idle-start evidence are retained.

Current tmux wam-norm-resume and wam-norm-resume-audit. New launcher writes
resume-exit-code.txt; after all arms complete it explicitly supersedes the
archived failure with combined study-exit-code0. New watcher waits for resumed
completion, then invokes the original frozen audit and isolated final builder.
Inspect resume-launcher.log, resume-postprocess.log and postprocess-status.json.
Do not restart old all-arm launcher or duplicate completed conditions.

Public Site version15 deployed22:33:17 UTC, source
066ee688a8c972d0e3c27a932f4bcb903f27399b, deployment
appgdep_6ab1b0a4d03c8191bb791c2d1b42a284. Snapshot shows audited60/80 and73 videos,
with explicit interruption and recovery record. Final native publication still
requires checking complete results; the watcher builds locally only.
